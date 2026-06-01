"""Stage 6.4c — Modern decoder model-level wrapper probe orchestrator.

Drives :class:`pllo.hf_wrappers.modern_decoder_model_wrapper.ObfuscatedModernDecoderModelWrapper`
across (use_pad × mitigation_bundle × nonlinear_mode) and emits a
JSON-safe report covering full forward, prefill/decode_step, greedy
generation, and KV cache invariants.

Real model loading is opt-in. Default config falls back to a synthetic
LLaMA-shape model so pytest never touches the network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.architectures.architecture_registry import (
    DEFAULT_ARCHITECTURE_MODELS,
    MODERN_DECODER_FAMILY_MAP,
)
from pllo.experiments.modern_decoder_block_probe import (
    _try_load_real_block,
    ModernDecoderLoadConfig,
)
from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    ModernDecoderModelWeights,
    ObfuscatedModernDecoderModelWrapper,
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
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderModelWrapperConfig:
    model_id: str | None = None
    attempt_real_model_load: bool = False
    allow_synthetic_fallback: bool = True
    local_files_only: bool = False
    nonlinear_mode: str = "compatible_islands"
    mitigation_bundle: str = DEFAULT_MITIGATION_BUNDLE
    use_pad: bool = True
    max_layers: int | None = 2
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 2026
    collect_traces: bool = False
    # Synthetic-fallback shape.
    synthetic_vocab_size: int = 64
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8
    # Sweep settings.
    batch_size: int = 1
    prompt_length: int = 6
    max_new_tokens: int = 3
    mitigation_bundles: tuple[str, ...] = VALID_MITIGATION_BUNDLES
    use_pad_values: tuple[bool, ...] = (False, True)


# ---------------------------------------------------------------------------
# Loading (re-uses Stage 6.4b helper)
# ---------------------------------------------------------------------------


def _resolve_weights(
    config: ModernDecoderModelWrapperConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[ModernDecoderBlockSpec, ModernDecoderModelWeights, dict[str, Any], str]:
    """Return ``(spec, weights, model_loading, source)``.

    Synthetic spec is materialised when loading fails or
    ``attempt_real_model_load=False``.
    """
    load = ModernDecoderLoadConfig(
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        device=config.device,
        dtype=config.dtype,
        local_files_only=config.local_files_only,
    )
    load_record = _try_load_real_block(load)
    if load_record["load_status"] == "loaded":
        model = load_record.pop("_model_obj", None)
        try:
            spec = inspect_modern_decoder_block(
                model, model_id=load_record["resolved_model_id"]
            )
            weights = ModernDecoderModelWeights.from_hf_model(
                model, spec=spec, dtype=dtype, device=device,
                max_layers=config.max_layers,
            )
            source = _source_label_for(load_record["resolved_model_id"])
            return spec, weights, load_record, source
        except Exception as exc:  # noqa: BLE001
            load_record["load_status"] = (
                "synthetic_only"
                if config.allow_synthetic_fallback else "skipped"
            )
            load_record["load_error"] = (
                f"loaded model but model-level extraction failed: {exc}"
            )
            load_record["fallback_used"] = bool(
                config.allow_synthetic_fallback
            )
    load_record.pop("_model_obj", None)

    # Synthetic fallback.
    spec = ModernDecoderBlockSpec(
        model_family="synthetic_modern_decoder",
        model_class="SyntheticLlamaModel",
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
        rope_base=10000.0,
        notes=["synthetic fallback; no real model weights were loaded"],
    )
    weights = ModernDecoderModelWeights.from_synthetic(
        vocab_size=config.synthetic_vocab_size,
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_intermediate_size,
        num_attention_heads=config.synthetic_num_attention_heads,
        num_key_value_heads=config.synthetic_num_key_value_heads,
        head_dim=config.synthetic_head_dim,
        num_layers=max(1, config.max_layers or 2),
        dtype=dtype, device=device, seed=config.seed,
    )
    return spec, weights, load_record, "synthetic_block"


def _source_label_for(model_id: str | None) -> str:
    if model_id is None:
        return "synthetic_block"
    family = MODERN_DECODER_FAMILY_MAP.get(model_id)
    if model_id == "hf-internal-testing/tiny-random-LlamaForCausalLM":
        return "tiny_random_llama_model"
    if family == "qwen_like":
        return "qwen_like_model"
    if family == "tinyllama":
        return "tinyllama_model"
    if family == "llama_like":
        return "llama_like_model"
    return "real_model"


# ---------------------------------------------------------------------------
# Probe entry point
# ---------------------------------------------------------------------------


_CAVEATS = [
    "Model-level wrapper smoke; not a real TEE deployment.",
    "Real wall-time is not measured.",
    "Only greedy generation is implemented.",
    "Beam search / top-k / top-p are not implemented.",
    "RoPE scaling variants are not fully implemented unless explicitly supported.",
    "Qwen / TinyLlama real loading is opt-in; pytest stays synthetic.",
    "No LoRA training path is implemented.",
    "Security remains proxy-evaluated, not formal.",
    "Inter-layer hidden states are recovered to plain space between blocks;"
    " the attacker view is still constrained by Stage 6.4b's intra-block"
    " masks and the LM-head dense / vocab masks.",
]


def run_modern_decoder_model_probe(
    config: ModernDecoderModelWrapperConfig,
) -> dict[str, Any]:
    """Run the model-level smoke and return a JSON-safe report."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)
    spec, weights, load_record, source = _resolve_weights(config, dtype, device)
    input_gen = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    input_ids = torch.randint(
        0, weights.vocab_size,
        (config.batch_size, config.prompt_length),
        generator=input_gen,
    ).to(device=device)

    runs: list[dict[str, Any]] = []
    for bundle in config.mitigation_bundles:
        bundle = normalize_mitigation_bundle(bundle)
        for use_pad in config.use_pad_values:
            torch.manual_seed(config.seed + 100)
            wrapper = ObfuscatedModernDecoderModelWrapper(
                weights, dtype=dtype, device=device,
                use_pad=bool(use_pad),
                nonlinear_mode=config.nonlinear_mode,
                mitigation_bundle=bundle,
                collect_traces=config.collect_traces,
            )
            torch.manual_seed(config.seed + 200)
            logits, fwd_report = wrapper.full_forward(input_ids)
            torch.manual_seed(config.seed + 300)
            pf_out = wrapper.prefill(input_ids)
            torch.manual_seed(config.seed + 400)
            next_t = pf_out["logits_recovered"][:, -1, :].argmax(dim=-1)
            step_out = wrapper.decode_step(
                next_t, pf_out["kv_cache"],
                position=int(input_ids.shape[-1]),
                plain_layer_caches=pf_out["plain_layer_caches"],
            )
            torch.manual_seed(config.seed + 500)
            g_out = wrapper.greedy_generate(
                input_ids, max_new_tokens=config.max_new_tokens
            )
            runs.append({
                "mitigation_bundle": bundle,
                "use_pad": bool(use_pad),
                "nonlinear_mode": config.nonlinear_mode,
                "source": source,
                "full_forward": fwd_report,
                "prefill": pf_out["report"],
                "decode_step": step_out["report"],
                "greedy_generate": g_out["report"],
            })

    summary = {
        "source": source,
        "model_family": spec.model_family,
        "num_layers_used": len(weights.layers),
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "head_dim": spec.head_dim,
        "attention_variant": spec.attention_variant,
        "vocab_size": int(weights.vocab_size),
        "nonlinear_mode": config.nonlinear_mode,
        "mitigation_bundles_evaluated": list(config.mitigation_bundles),
        "use_pad_values": list(config.use_pad_values),
        "max_new_tokens": int(config.max_new_tokens),
        "all_full_forward_allclose": bool(
            all(r["full_forward"]["logits_metrics"]["allclose"] for r in runs)
        ),
        "all_prefill_allclose": bool(
            all(r["prefill"]["logits_metrics"]["allclose"] for r in runs)
        ),
        "all_decode_top1_match": bool(
            all(
                r["decode_step"]["logits_metrics"]
                and r["decode_step"]["logits_metrics"]["top1_match_rate"] == 1.0
                for r in runs
            )
        ),
        "all_generation_exact_match": bool(
            all(r["greedy_generate"]["sequence_exact_match"] for r in runs)
        ),
        "online_extra_matmul_count": 0,
        "implemented_model_level": True,
        "full_runtime_integrated": False,
        "modern_decoder_generation_status": "greedy_generation_implemented",
        "modern_decoder_kv_cache_status": "implemented",
    }
    return {
        "config": asdict(config),
        "model_loading": load_record,
        "source": source,
        "block_spec": spec_to_dict(spec),
        "input_ids_shape": list(input_ids.shape),
        "per_run": runs,
        "summary": summary,
        "caveats": list(_CAVEATS),
    }


__all__ = [
    "ModernDecoderModelWrapperConfig",
    "run_modern_decoder_model_probe",
]
