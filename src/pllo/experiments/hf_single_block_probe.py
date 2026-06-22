"""Stage 6.6 -- HF/ModelScope LLaMA/Qwen single-decoder-layer probe.

Validates the Stage 6.6 adapter on either a local checkpoint
(``local_files_only=True``, no network) or a randomly-initialised tiny HF
decoder layer (no checkpoint, no download). Compares the masked wrapper
against a plain reference computed from the *extracted* weights.

If transformers is unavailable (or the requested family's layer class is
not importable, or a given local path does not exist), the probe returns a
clean ``skipped`` status instead of crashing.

Single decoder layer only -- no tokenizer, embedding, LM head, sampling, or
generation loop. No formal/cryptographic/semantic security is claimed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.hf_wrappers.llama_qwen_single_block import (
    extract_hf_single_block_weights,
    generate_hf_single_block_masks,
    has_transformers,
    hf_single_block_masked_decode,
    hf_single_block_masked_prefill,
    infer_config_from_hf_layer,
    make_random_hf_decoder_layer,
)

_REQUIRED_STATEMENT = (
    "This stage validates a real HuggingFace-style LLaMA/Qwen decoder layer "
    "adapter using extracted weights. It does not validate a full model, "
    "tokenizer, embedding, LM head, sampling, or end-to-end generation."
)

_CAVEATS = [
    "single decoder layer only",
    "extracted-weight reference, not full HF model generation",
    "no tokenizer",
    "no embedding masking",
    "no LM head masking",
    "no sampling",
    "no full generation loop",
    "no NTK/YaRN RoPE scaling unless already supported by the local config",
    "RoPE-compatible masks preserve pair partition",
    "SwiGLU paired permutation exposes operator-compatible invariants",
    "attention scores remain visible in the current design",
    "KV cache masks are reused within a generation session",
]


@dataclass
class HFSingleBlockProbeConfig:
    model_family: str = "llama"  # or "qwen2"
    local_model_path: str | None = None
    layer_index: int = 0
    batch_size: int = 1
    seq_len: int = 8
    decode_steps: int = 2
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2029
    mask_family: str = "pairwise_complex_scaling"
    use_random_layer_if_no_path: bool = True


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _skipped(config: HFSingleBlockProbeConfig, reason: str,
             status: str = "skipped") -> dict[str, Any]:
    return {
        "stage": "6.6_hf_single_block",
        "experiment": "hf_single_block_probe",
        "status": status,
        "reason": reason,
        "model_family": config.model_family,
        "no_network_download": True,
        "local_files_only": True,
        "config": asdict(config),
        "statement": _REQUIRED_STATEMENT,
    }


def _load_local_layer(config: HFSingleBlockProbeConfig) -> tuple[Any, Any]:
    """Load a single decoder layer from a LOCAL path (no network)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.local_model_path, local_files_only=True,
        torch_dtype=torch.float32)
    model.eval()
    # LLaMA/Qwen2 store decoder layers at model.model.layers.
    base = getattr(model, "model", model)
    layers = base.layers
    layer = layers[config.layer_index]
    return layer, model.config


def run_hf_single_block_probe(
    config: HFSingleBlockProbeConfig,
) -> dict[str, Any]:
    if not has_transformers():
        return _skipped(config, "transformers is not installed",
                        status="skipped_transformers_unavailable")

    # Resolve the decoder layer + its model config.
    if config.local_model_path is not None:
        if not os.path.isdir(config.local_model_path):
            return _skipped(
                config,
                f"local_model_path does not exist: {config.local_model_path}")
        try:
            layer, model_config = _load_local_layer(config)
            source = "local_checkpoint"
        except Exception as exc:  # noqa: BLE001 - report cleanly, never crash
            return _skipped(config, f"local load failed: {exc!r}",
                            status="skipped_local_load_failed")
    else:
        if not config.use_random_layer_if_no_path:
            return _skipped(config, "no local_model_path and random disabled")
        try:
            layer, model_config = make_random_hf_decoder_layer(
                config.model_family, seed=config.seed)
            source = "random_config"
        except Exception as exc:  # noqa: BLE001
            return _skipped(config, f"random layer unavailable: {exc!r}",
                            status="skipped_layer_class_unavailable")

    dtype = _dtype(config.dtype)
    device = torch.device(config.device)
    block_cfg = infer_config_from_hf_layer(
        layer, model_config, dtype=dtype, device=config.device,
        mask_family=config.mask_family)
    weights = extract_hf_single_block_weights(layer, dtype, config.device)
    masks = generate_hf_single_block_masks(block_cfg, seed=config.seed)

    g = torch.Generator(device=device).manual_seed(config.seed)
    x = torch.randn(config.batch_size, config.seq_len, block_cfg.hidden_size,
                    generator=g, dtype=dtype, device=device)

    pre = hf_single_block_masked_prefill(
        x, weights, block_cfg, masks, decode_steps=config.decode_steps)

    cache_tilde, cache_plain = pre["cache_tilde"], pre["cache_plain"]
    decode_metrics: list[dict[str, Any]] = []
    for step in range(config.decode_steps):
        position = config.seq_len + step
        x_new = torch.randn(config.batch_size, 1, block_cfg.hidden_size,
                            generator=g, dtype=dtype, device=device)
        dec = hf_single_block_masked_decode(
            x_new, cache_tilde, cache_plain, weights, block_cfg, masks,
            position)
        m = dec["metrics"]
        decode_metrics.append({
            "step": step, "position": position,
            "output_max_abs_error": m["output_max_abs_error"],
            "cache_append_key_max_abs_error": m["cache_append_key_max_abs_error"],
            "cache_append_value_max_abs_error":
                m["cache_append_value_max_abs_error"],
            "allclose": m["allclose"],
        })
        cache_tilde, cache_plain = dec["cache_tilde"], dec["cache_plain"]

    decode_allclose = all(d["allclose"] for d in decode_metrics)
    all_allclose = bool(pre["metrics"]["allclose"] and decode_allclose)

    return {
        "stage": "6.6_hf_single_block",
        "experiment": "hf_single_block_probe",
        "status": "ok",
        "model_family": config.model_family,
        "model_type": block_cfg.model_type,
        "source": source,
        "no_network_download": True,
        "local_files_only": True,
        "statement": _REQUIRED_STATEMENT,
        "config": asdict(config),
        "block_config": {
            "model_type": block_cfg.model_type,
            "hidden_size": block_cfg.hidden_size,
            "intermediate_size": block_cfg.intermediate_size,
            "num_heads": block_cfg.num_heads,
            "num_key_value_heads": block_cfg.num_key_value_heads,
            "head_dim": block_cfg.head_dim,
            "rope_theta": block_cfg.rope_theta,
            "rms_norm_eps": block_cfg.rms_norm_eps,
            "attention_bias": block_cfg.attention_bias,
            "mlp_bias": block_cfg.mlp_bias,
        },
        "prefill_metrics": pre["metrics"],
        "decode_step_metrics": decode_metrics,
        "prefill_allclose": pre["metrics"]["allclose"],
        "decode_allclose": decode_allclose,
        "allclose": all_allclose,
        "metadata": {
            "stage": "6.6_hf_single_block",
            "model_style": "hf_llama_qwen_single_decoder_layer",
            "security_status":
                "operator_compatible_leakage_reduction_not_semantic_security",
            "semantic_security_claimed": False,
            "formal_security_claimed": False,
            "cryptographic_security_claimed": False,
            "no_intermediate_tee": True,
            "no_network_download": True,
            "no_hf_dependency": False,
            "mask_family": block_cfg.mask_family,
            "residual_mask_family": "orthogonal",
            "rmsnorm_mode": "orthogonal_core_affine_folded",
            "attention_mode": "rope_gqa_complex_scaling",
            "mlp_mode": "swiglu_paired_permutation",
            "selector_lifted_swiglu_default": False,
            "caveats": _CAVEATS,
        },
        "limitations": _CAVEATS,
    }


__all__ = ["HFSingleBlockProbeConfig", "run_hf_single_block_probe"]
