"""Stage 6.4b — Modern decoder-only block-level probe orchestrator.

Runs the block-level obfuscated forward against a real (best-effort)
LLaMA / TinyLlama / Qwen / Qwen2 block or, when the model cannot be
loaded locally / over the network, a synthetic LLaMA-shape block. Emits
a JSON-safe report covering model loading, block spec, plain-reference
construction, and obfuscated allclose for every (use_pad × bundle)
combination requested.

This wraps :mod:`pllo.hf_wrappers.modern_decoder_block_wrapper`. The
wider system default mode (``nonlinear_mode``) remains ``"trusted"`` and
the default mitigation bundle remains ``"fresh_perm_only"``. The probe
exposes the wrapper for both bundles so the report can publish per-bundle
allclose plus the full bundle metadata that Stage 5.3e introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.architectures.architecture_registry import (
    DEFAULT_ARCHITECTURE_MODELS,
    MODERN_DECODER_FAMILY_MAP,
)
from pllo.hf_wrappers.modern_decoder_block_wrapper import (
    ModernDecoderBlockWeights,
    ObfuscatedModernDecoderBlockWrapper,
)
from pllo.model_zoo.modern_decoder_spec import (
    ModernDecoderBlockSpec,
    inspect_modern_decoder_block,
    spec_to_dict,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    normalize_mitigation_bundle,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderLoadConfig:
    """Best-effort model loading. Defaults skip network access entirely."""

    model_id: str | None = None
    attempt_real_model_load: bool = False
    allow_synthetic_fallback: bool = True
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = False


@dataclass
class ModernDecoderBlockProbeConfig:
    output_dir: str = "outputs"
    load: ModernDecoderLoadConfig = field(default_factory=ModernDecoderLoadConfig)
    batch_size: int = 2
    seq_len: int = 8
    # Synthetic-fallback block shape.
    synthetic_hidden_size: int = 64
    synthetic_intermediate_size: int = 128
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 16
    synthetic_rope_base: float = 10000.0
    # Sweep settings.
    use_pad_values: tuple[bool, ...] = (False, True)
    mitigation_bundles: tuple[str, ...] = VALID_MITIGATION_BUNDLES
    nonlinear_mode: str = "compatible_islands"
    seed: int = 2026


# ---------------------------------------------------------------------------
# Model loading (best-effort, never raises)
# ---------------------------------------------------------------------------


def _import_transformers_or_none():
    try:
        import transformers  # noqa: F401

        return transformers
    except Exception:  # noqa: BLE001
        return None


def _try_load_real_block(
    load: ModernDecoderLoadConfig,
) -> dict[str, Any]:
    """Return a structured ``model_loading`` dict; never raises."""
    base_record = {
        "load_status": "skipped",
        "resolved_model_id": None,
        "model_family": None,
        "model_class": None,
        "load_error": None,
        "fallback_used": False,
        "candidates_tried": [],
    }
    if not load.attempt_real_model_load:
        base_record["load_status"] = "synthetic_only"
        base_record["load_error"] = (
            "attempt_real_model_load=False (default); pytest runs the"
            " synthetic fallback to avoid network downloads."
        )
        base_record["fallback_used"] = bool(load.allow_synthetic_fallback)
        return base_record
    transformers = _import_transformers_or_none()
    if transformers is None:
        base_record["load_status"] = (
            "synthetic_only" if load.allow_synthetic_fallback else "skipped"
        )
        base_record["load_error"] = "transformers package not importable"
        base_record["fallback_used"] = bool(load.allow_synthetic_fallback)
        return base_record
    candidates = (
        (load.model_id,)
        if load.model_id
        else DEFAULT_ARCHITECTURE_MODELS["modern_decoder_only"]
    )
    base_record["candidates_tried"] = list(candidates)
    auto = transformers.AutoModelForCausalLM
    failures: list[str] = []
    for mid in candidates:
        try:
            kwargs: dict[str, Any] = {}
            if load.local_files_only:
                kwargs["local_files_only"] = True
            model = auto.from_pretrained(mid, **kwargs)
            model.eval()
            family = MODERN_DECODER_FAMILY_MAP.get(mid, "unknown")
            base_record.update(
                {
                    "load_status": "loaded",
                    "resolved_model_id": mid,
                    "model_family": family,
                    "model_class": type(model).__name__,
                    "load_error": None,
                    "fallback_used": False,
                }
            )
            base_record["_model_obj"] = model   # consumed internally
            return base_record
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mid}: {type(exc).__name__}: {exc}")
    base_record["load_status"] = (
        "synthetic_only" if load.allow_synthetic_fallback else "skipped"
    )
    base_record["load_error"] = "; ".join(failures)
    base_record["fallback_used"] = bool(load.allow_synthetic_fallback)
    return base_record


# ---------------------------------------------------------------------------
# Synthetic block spec helper
# ---------------------------------------------------------------------------


def _synthetic_spec_and_weights(
    config: ModernDecoderBlockProbeConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[ModernDecoderBlockSpec, ModernDecoderBlockWeights]:
    spec = ModernDecoderBlockSpec(
        model_family="synthetic_modern_decoder",
        model_class="SyntheticLlamaBlock",
        block_path="synthetic.layers.0",
        block_index=0,
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_intermediate_size,
        num_attention_heads=config.synthetic_num_attention_heads,
        num_key_value_heads=config.synthetic_num_key_value_heads,
        head_dim=config.synthetic_head_dim,
        norm_type="rmsnorm",
        activation_type="swiglu",
        position_encoding_type="rotary",
        attention_variant=(
            "mha"
            if config.synthetic_num_key_value_heads == config.synthetic_num_attention_heads
            else (
                "mqa"
                if config.synthetic_num_key_value_heads == 1
                else "gqa"
            )
        ),
        q_proj_path="synthetic.layers.0.self_attn.q_proj",
        k_proj_path="synthetic.layers.0.self_attn.k_proj",
        v_proj_path="synthetic.layers.0.self_attn.v_proj",
        o_proj_path="synthetic.layers.0.self_attn.o_proj",
        gate_proj_path="synthetic.layers.0.mlp.gate_proj",
        up_proj_path="synthetic.layers.0.mlp.up_proj",
        down_proj_path="synthetic.layers.0.mlp.down_proj",
        input_norm_path="synthetic.layers.0.input_layernorm",
        post_attention_norm_path="synthetic.layers.0.post_attention_layernorm",
        rope_base=config.synthetic_rope_base,
        notes=["synthetic fallback; no real model weights were loaded"],
    )
    weights = ModernDecoderBlockWeights.from_synthetic(
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_intermediate_size,
        num_attention_heads=config.synthetic_num_attention_heads,
        num_key_value_heads=config.synthetic_num_key_value_heads,
        head_dim=config.synthetic_head_dim,
        dtype=dtype,
        device=device,
        rope_base=config.synthetic_rope_base,
        seed=config.seed,
    )
    return spec, weights


def _real_spec_and_weights(
    model, model_id: str | None, dtype: torch.dtype, device: torch.device
) -> tuple[ModernDecoderBlockSpec, ModernDecoderBlockWeights] | None:
    """Try to inspect + extract; return ``None`` if anything goes wrong."""
    try:
        spec = inspect_modern_decoder_block(model, model_id=model_id)
    except Exception:  # noqa: BLE001
        return None
    # Walk to the block module.
    block_container = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        block_container = model.model.layers
    elif hasattr(model, "layers"):
        block_container = model.layers
    if block_container is None:
        return None
    block = block_container[spec.block_index]
    try:
        weights = ModernDecoderBlockWeights.from_hf_block(block, spec)
    except Exception:  # noqa: BLE001
        return None
    # Cast / move so the obfuscated path sees consistent dtype/device.
    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(dtype=dtype, device=device).contiguous()

    return (
        spec,
        ModernDecoderBlockWeights(
            hidden_size=weights.hidden_size,
            intermediate_size=weights.intermediate_size,
            num_attention_heads=weights.num_attention_heads,
            num_key_value_heads=weights.num_key_value_heads,
            head_dim=weights.head_dim,
            rope_base=weights.rope_base,
            input_norm_weight=_to(weights.input_norm_weight),
            input_norm_eps=weights.input_norm_eps,
            w_q=_to(weights.w_q),
            b_q=_to(weights.b_q) if weights.b_q is not None else None,
            w_k=_to(weights.w_k),
            b_k=_to(weights.b_k) if weights.b_k is not None else None,
            w_v=_to(weights.w_v),
            b_v=_to(weights.b_v) if weights.b_v is not None else None,
            w_o=_to(weights.w_o),
            b_o=_to(weights.b_o) if weights.b_o is not None else None,
            post_attention_norm_weight=_to(weights.post_attention_norm_weight),
            post_attention_norm_eps=weights.post_attention_norm_eps,
            w_gate=_to(weights.w_gate),
            b_gate=_to(weights.b_gate) if weights.b_gate is not None else None,
            w_up=_to(weights.w_up),
            b_up=_to(weights.b_up) if weights.b_up is not None else None,
            w_down=_to(weights.w_down),
            b_down=_to(weights.b_down) if weights.b_down is not None else None,
        ),
    )


# ---------------------------------------------------------------------------
# Probe entry point
# ---------------------------------------------------------------------------


_CAVEATS = [
    "Block-level integration; not a full model-level wrapper.",
    "No generation / decode_step / KV cache runtime is implemented.",
    "RoPE is handled using post-RoPE per-head masking; mask-before-RoPE"
    " dense commutation is not assumed.",
    "If synthetic fallback is used, results do NOT come from real Qwen / TinyLlama weights.",
    "RMSNorm and residual alignment use orthogonal residual mask N_res so"
    " both branches stay in the same mask space; γ is folded into adjacent"
    " Linear weights.",
    "Inherits Stage 5.4 mitigation requirements (fresh permutation +"
    " dense sandwich + boundary pad).",
    "This is not a real TEE measurement.",
    "This is not formal security.",
]


def run_modern_decoder_block_probe(
    config: ModernDecoderBlockProbeConfig,
) -> dict[str, Any]:
    """Run the block-level probe and return a JSON-safe report."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.load.dtype == "float32" else torch.float64
    device = torch.device(config.load.device)

    load_record = _try_load_real_block(config.load)
    real_pair = None
    if load_record["load_status"] == "loaded":
        model = load_record.pop("_model_obj", None)
        real_pair = _real_spec_and_weights(
            model, load_record["resolved_model_id"], dtype, device
        )
        if real_pair is None:
            load_record["load_status"] = (
                "synthetic_only"
                if config.load.allow_synthetic_fallback
                else "skipped"
            )
            load_record["load_error"] = (
                "loaded model but inspection / extraction failed"
            )
            load_record["fallback_used"] = bool(
                config.load.allow_synthetic_fallback
            )
    # Strip private payload if it lingered.
    load_record.pop("_model_obj", None)

    if real_pair is None:
        spec, weights = _synthetic_spec_and_weights(config, dtype, device)
        source = "synthetic_fallback"
    else:
        spec, weights = real_pair
        source = "real_model"

    # Sample input.
    torch.manual_seed(config.seed + 1)
    x = torch.randn(
        config.batch_size, config.seq_len, spec.hidden_size,
        dtype=dtype, device=device,
    )

    # Sweep (bundle × use_pad × nonlinear_mode).
    per_run: list[dict[str, Any]] = []
    for bundle in config.mitigation_bundles:
        bundle = normalize_mitigation_bundle(bundle)
        for use_pad in config.use_pad_values:
            wrapper = ObfuscatedModernDecoderBlockWrapper(
                weights,
                dtype=dtype,
                device=device,
                use_pad=bool(use_pad),
                nonlinear_mode=config.nonlinear_mode,
                mitigation_bundle=bundle,
            )
            y_recovered, report = wrapper.forward(x)
            per_run.append(
                {
                    "mitigation_bundle": bundle,
                    "use_pad": bool(use_pad),
                    "nonlinear_mode": config.nonlinear_mode,
                    "block_index": spec.block_index,
                    "model_family": spec.model_family,
                    "source": source,
                    **report,
                }
            )

    all_allclose = all(r["allclose"] for r in per_run)
    return {
        "config": {
            **{k: v for k, v in asdict(config).items() if k != "load"},
            "load": asdict(config.load),
        },
        "model_loading": load_record,
        "source": source,
        "block_spec": spec_to_dict(spec),
        "per_run": per_run,
        "summary": {
            "source": source,
            "model_family": spec.model_family,
            "block_index": spec.block_index,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_attention_heads": spec.num_attention_heads,
            "num_key_value_heads": spec.num_key_value_heads,
            "head_dim": spec.head_dim,
            "norm_type": spec.norm_type,
            "activation_type": spec.activation_type,
            "position_encoding_type": spec.position_encoding_type,
            "attention_variant": spec.attention_variant,
            "rope_base": spec.rope_base,
            "nonlinear_mode_default": "trusted",
            "default_mitigation_bundle": DEFAULT_MITIGATION_BUNDLE,
            "mitigation_bundles_evaluated": list(config.mitigation_bundles),
            "all_runs_allclose": bool(all_allclose),
            "online_extra_matmul_count": 0,
            "implemented_block_level": True,
            "full_runtime_integrated": False,
            "rmsnorm_status": "orthogonal_island_with_gamma_folded_into_qkv",
            "rope_attention_status": "rope_post_mask_only",
            "gqa_status": "per_kv_head_mask_with_repeat_kv",
            "swiglu_status": "compatible_island_paired_permutation",
        },
        "caveats": list(_CAVEATS),
    }


__all__ = [
    "ModernDecoderBlockProbeConfig",
    "ModernDecoderLoadConfig",
    "run_modern_decoder_block_probe",
]
