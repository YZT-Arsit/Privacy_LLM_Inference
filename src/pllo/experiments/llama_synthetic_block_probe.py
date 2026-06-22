"""Stage 6.5 -- LLaMA/Qwen-like synthetic decoder block probe.

Runs the full synthetic decoder block (RMSNorm -> RoPE-GQA attention ->
residual -> RMSNorm -> SwiGLU MLP -> residual) under operator-compatible
masks, for both an MHA case (num_heads == num_key_value_heads) and a GQA
case, across prefill and multi-step decode. Reports per-stage max abs
errors against the plain reference and the end-to-end invariant
``y_tilde == y_plain @ n_res``.

Synthetic, CPU-only, correctness-first. No HF/ModelScope model loading, no
GPT-2 wrapper, no embeddings/LM-head/sampling, no NTK/YaRN RoPE scaling.
No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.llama_synthetic_block import (
    SyntheticLlamaBlockConfig,
    generate_block_masks,
    init_synthetic_llama_block_weights,
    llama_block_masked_decode,
    llama_block_masked_prefill,
)

_REQUIRED_STATEMENT = (
    "This synthetic block validates end-to-end correctness of a LLaMA/"
    "Qwen-like decoder layer under operator-compatible masks. It does not "
    "claim semantic security and does not load a real HF model."
)


@dataclass
class LlamaSyntheticBlockProbeConfig:
    batch_size: int = 2
    seq_len: int = 8
    decode_steps: int = 3
    hidden_size: int = 32
    intermediate_size: int = 64
    num_heads: int = 4
    num_key_value_heads: int = 2
    mask_family: str = "pairwise_complex_scaling"
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2028


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _to_block_config(
    cfg: LlamaSyntheticBlockProbeConfig, num_kv: int,
) -> SyntheticLlamaBlockConfig:
    return SyntheticLlamaBlockConfig(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        decode_steps=cfg.decode_steps,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        num_key_value_heads=num_kv,
        rope_base=cfg.rope_base,
        rms_norm_eps=cfg.rms_norm_eps,
        mask_family=cfg.mask_family,
        dtype=_dtype(cfg.dtype),
        device=cfg.device,
        seed=cfg.seed,
    )


def _run_case(
    probe_cfg: LlamaSyntheticBlockProbeConfig, num_kv: int,
) -> dict[str, Any]:
    cfg = _to_block_config(probe_cfg, num_kv)
    cfg.validate()
    device = torch.device(cfg.device)
    g = torch.Generator(device=device).manual_seed(cfg.seed)

    weights = init_synthetic_llama_block_weights(cfg, g)
    masks = generate_block_masks(cfg, g)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size,
                    generator=g, dtype=cfg.dtype, device=device)

    pre = llama_block_masked_prefill(x, weights, masks, cfg)

    cache_tilde = pre["cache_tilde"]
    cache_plain = pre["cache_plain"]
    decode_metrics: list[dict[str, Any]] = []
    for step in range(cfg.decode_steps):
        position = cfg.seq_len + step
        x_new = torch.randn(cfg.batch_size, 1, cfg.hidden_size, generator=g,
                            dtype=cfg.dtype, device=device)
        dec = llama_block_masked_decode(
            x_new, cache_tilde, cache_plain, weights, masks, cfg, position)
        m = dec["metrics"]
        decode_metrics.append({
            "step": step, "position": position,
            "output_max_abs_error": m["output_max_abs_error"],
            "cache_append_key_max_abs_error": m["cache_append_key_max_abs_error"],
            "cache_append_value_max_abs_error":
                m["cache_append_value_max_abs_error"],
            "allclose": m["allclose"],
        })
        cache_tilde = dec["cache_tilde"]
        cache_plain = dec["cache_plain"]

    decode_allclose = all(d["allclose"] for d in decode_metrics)
    return {
        "num_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "prefill_metrics": pre["metrics"],
        "decode_step_metrics": decode_metrics,
        "prefill_allclose": pre["metrics"]["allclose"],
        "decode_allclose": decode_allclose,
        "allclose": bool(pre["metrics"]["allclose"] and decode_allclose),
    }


def run_llama_synthetic_block_probe(
    config: LlamaSyntheticBlockProbeConfig,
) -> dict[str, Any]:
    gqa = _run_case(config, config.num_key_value_heads)
    mha = _run_case(config, config.num_heads)
    all_allclose = bool(gqa["allclose"] and mha["allclose"])

    return {
        "stage": "6.5_llama_synthetic_block",
        "experiment": "llama_synthetic_block_probe",
        "status": "ok",
        "statement": _REQUIRED_STATEMENT,
        "config": asdict(config),
        "gqa": gqa,
        "mha": mha,
        "all_allclose": all_allclose,
        "metadata": {
            "stage": "6.5_llama_synthetic_block",
            "model_style": "llama_qwen_like_synthetic",
            "no_hf_dependency": True,
            "no_intermediate_tee": True,
            "mask_family": "pairwise_complex_scaling",
            "residual_mask_family": "orthogonal",
            "rmsnorm_mode": "orthogonal_core_affine_folded",
            "attention_mode": "rope_gqa_complex_scaling",
            "mlp_mode": "swiglu_paired_permutation",
            "selector_lifted_swiglu_default": False,
            "security_status":
                "operator_compatible_leakage_reduction_not_semantic_security",
            "caveats": [
                "Synthetic tensor-level block, not real HF LLaMA/Qwen wrapper",
                "Embedding, LM head, sampling, and tokenizer are not covered",
                "RoPE-compatible masks preserve pair partition",
                "SwiGLU paired-permutation exposes operator-compatible "
                "invariants",
                "No formal cryptographic or semantic security claim",
            ],
        },
        "limitations": [
            "Synthetic block only; not a real HF/ModelScope LLaMA/Qwen wrapper.",
            "No embedding, LM head, sampling, or tokenizer.",
            "No RoPE scaling variants (NTK/YaRN).",
            "CPU-only float64; no formal, cryptographic, or semantic security.",
        ],
    }


__all__ = [
    "LlamaSyntheticBlockProbeConfig",
    "run_llama_synthetic_block_probe",
]
