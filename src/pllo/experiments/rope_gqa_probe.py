"""Stage 6.4 / 6.4.1 -- RoPE-compatible masked GQA/MHA attention probe.

Synthetic, CPU-only, correctness-first. Runs the masked RoPE attention
prefill + decode for an MHA case (num_heads == num_key_value_heads) and a
GQA case (num_heads > num_key_value_heads), for two RoPE-compatible mask
families:

* ``pairwise_rotation``     -- orthogonal baseline; preserves per-pair norm.
* ``pairwise_complex_scaling`` -- preferred family; changes per-pair
  magnitude but preserves the RoPE pair partition.

Stage 6.4.1 adds leakage *proxy* metrics. These are NOT a security proof.
We do not send RoPE/attention/nonlinear layers into a TEE; the goal is
correctness-preserving, operator-compatible leakage reduction, not
standalone semantic security. RoPE-compatible masks are accepted as a
weaker local mask family because arbitrary dense masks do not commute with
RoPE.

Not a HuggingFace LLaMA/Qwen wrapper; q/k/v/o weight folding, RoPE scaling
variants (NTK/YaRN), embeddings, LM head, and full generation are out of
scope for this stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.gqa_attention import (
    generate_gqa_rope_masks,
    masked_rope_gqa_attention_decode,
    masked_rope_gqa_attention_prefill,
)
from pllo.ops.rope import (
    build_rope_cache,
    make_pairwise_complex_scaling_mask,
    make_pairwise_rotation_mask,
    rope_commutation_error,
)

_REQUIRED_STATEMENT = (
    "RoPE-compatible masks are correctness-preserving and reduce direct "
    "leakage, but they are weaker than dense masks. They are used because "
    "no intermediate TEE is allowed."
)

_PRESERVED_STRUCTURE = [
    "RoPE pair partition",
    "No cross-pair dense mixing inside RoPE-compatible region",
    "Attention scores are preserved by construction",
    "KV cache masks are reused within a generation session",
]


@dataclass
class RopeGQAProbeConfig:
    batch_size: int = 2
    seq_len: int = 8
    decode_steps: int = 3
    hidden_size: int = 32
    num_heads: int = 4
    num_key_value_heads: int = 2
    rope_base: float = 10000.0
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2027
    mask_family: str = "pairwise_complex_scaling"
    run_rotation_baseline: bool = True
    run_complex_scaling: bool = True
    run_leakage_proxy: bool = True
    leakage_num_samples: int = 128


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _orthogonal(dim: int, g: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype))
    return q


# ---------------------------------------------------------------------------
# Leakage proxy utilities (Stage 6.4.1)
# ---------------------------------------------------------------------------


def pair_norms(x: torch.Tensor) -> torch.Tensor:
    """Per adjacent-pair L2 norm: ``[..., D] -> [..., D//2]``."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.sqrt(pairs[..., 0] ** 2 + pairs[..., 1] ** 2)


def pearson_corr_flat(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12,
) -> float:
    """Pearson correlation over flattened tensors (float64)."""
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b / denom).item())


def nearest_neighbor_matching_accuracy(
    plain_features: torch.Tensor, visible_features: torch.Tensor,
) -> float:
    """Fraction of visible rows whose nearest plain row (L2) is its own index.

    Both inputs are ``[N, p]``. Used as a crude linkability proxy.
    """
    dist = torch.cdist(
        visible_features.to(torch.float64), plain_features.to(torch.float64))
    nearest = dist.argmin(dim=1)
    target = torch.arange(visible_features.shape[0],
                          device=visible_features.device)
    return float((nearest == target).to(torch.float64).mean().item())


def run_rope_leakage_proxy(
    config: RopeGQAProbeConfig, mask_family: str | None = None,
) -> dict[str, Any]:
    """Leakage *proxy* metrics for no_mask / rotation / complex-scaling.

    Not a security proof. Quantifies that rotation preserves per-pair norm
    exactly (so it does not reduce pair-norm linkability) while
    complex-scaling perturbs per-pair magnitude with per-session scales,
    reducing direct cross-session pair-norm matching. Both still preserve
    the RoPE pair partition and are weaker than dense masks.
    """
    dtype = _dtype(config.dtype)
    device = torch.device(config.device)
    head_dim = config.hidden_size // config.num_heads
    n = config.leakage_num_samples

    # Deterministic synthetic token/head vectors.
    gx = torch.Generator(device=device).manual_seed(config.seed + 777)
    x = torch.randn(n, head_dim, generator=gx, dtype=dtype, device=device)
    plain_pn = pair_norms(x)

    def _mask(family: str, seed_offset: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(config.seed + seed_offset)
        if family == "pairwise_rotation":
            return make_pairwise_rotation_mask(head_dim, dtype, device, g)
        return make_pairwise_complex_scaling_mask(head_dim, dtype, device, g)

    def _metrics(family: str) -> dict[str, float]:
        if family == "no_mask":
            vis1 = x
            vis2 = x
        else:
            # Two independently sampled per-session masks.
            vis1 = x @ _mask(family, 1001)
            vis2 = x @ _mask(family, 2002)
        vis1_pn = pair_norms(vis1)
        vis2_pn = pair_norms(vis2)
        return {
            "pair_norm_correlation_same_session":
                pearson_corr_flat(plain_pn, vis1_pn),
            "cross_session_pair_norm_correlation":
                pearson_corr_flat(vis1_pn, vis2_pn),
            "nearest_neighbor_matching_accuracy_pair_norm":
                nearest_neighbor_matching_accuracy(plain_pn, vis1_pn),
        }

    return {
        "no_mask": _metrics("no_mask"),
        "pairwise_rotation": _metrics("pairwise_rotation"),
        "pairwise_complex_scaling": _metrics("pairwise_complex_scaling"),
        "leakage_proxy_is_not_security_proof": True,
        "preserved_structure": list(_PRESERVED_STRUCTURE),
        "num_samples": n,
        "head_dim": head_dim,
        "feature": "pair_norm",
    }


# ---------------------------------------------------------------------------
# Correctness case
# ---------------------------------------------------------------------------


def _run_case(
    cfg: RopeGQAProbeConfig,
    num_heads: int,
    num_key_value_heads: int,
    mask_family: str = "pairwise_complex_scaling",
) -> dict[str, Any]:
    dtype = _dtype(cfg.dtype)
    device = torch.device(cfg.device)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    hidden = cfg.hidden_size
    head_dim = hidden // num_heads
    kv_dim = num_key_value_heads * head_dim

    def rn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=dtype, device=device) * scale

    x = rn(cfg.batch_size, cfg.seq_len, hidden)
    w_q, b_q = rn(hidden, num_heads * head_dim), rn(num_heads * head_dim)
    w_k, b_k = rn(hidden, kv_dim), rn(kv_dim)
    w_v, b_v = rn(hidden, kv_dim), rn(kv_dim)
    w_o, b_o = rn(num_heads * head_dim, hidden), rn(hidden)
    n_out = _orthogonal(hidden, g, dtype)

    masks = generate_gqa_rope_masks(
        num_heads, num_key_value_heads, head_dim, dtype, device, g,
        mask_family=mask_family,
    )
    max_pos = cfg.seq_len + cfg.decode_steps + 1
    cos, sin = build_rope_cache(max_pos, head_dim, cfg.rope_base, dtype, device)

    # Standalone RoPE commutation check (family-representative mask).
    probe_x = rn(cfg.batch_size, num_heads, cfg.seq_len, head_dim)
    if mask_family == "pairwise_rotation":
        m_single = make_pairwise_rotation_mask(head_dim, dtype, device, g)
    else:
        m_single = make_pairwise_complex_scaling_mask(head_dim, dtype, device, g)
    rope_comm_err = rope_commutation_error(probe_x, m_single, cos, sin)

    pre = masked_rope_gqa_attention_prefill(
        x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, n_out, masks, cos, sin,
    )

    def mx(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    prefill_metrics = {
        "score_max_abs_error": pre["score_max_abs_error"],
        "prob_max_abs_error": mx(pre["probs_plain"], pre["probs_tilde"]),
        "v_aggregation_max_abs_error": mx(
            pre["av_tilde"], pre["expected_av_tilde"]),
        "output_max_abs_error": mx(pre["out_tilde"], pre["expected_out_tilde"]),
        "prefill_cache_key_max_abs_error": mx(
            pre["cache"]["key_rope_tilde"], pre["expected_cache_key_tilde"]),
        "prefill_cache_value_max_abs_error": mx(
            pre["cache"]["value_tilde"], pre["expected_cache_value_tilde"]),
        "rope_commutation_q_error": pre["rope_commutation_q_error"],
    }

    # Decode steps.
    cache = pre["cache"]
    decode_steps: list[dict[str, Any]] = []
    cache_key_err = 0.0
    cache_value_err = 0.0
    for step in range(cfg.decode_steps):
        position = cfg.seq_len + step
        x_new = rn(cfg.batch_size, 1, hidden)
        dec = masked_rope_gqa_attention_decode(x_new, cache, position)
        k_err = mx(dec["appended_key_tilde"],
                   dec["expected_appended_key_tilde"])
        v_err = mx(dec["appended_value_tilde"],
                   dec["expected_appended_value_tilde"])
        o_err = mx(dec["out_tilde"], dec["expected_out_tilde"])
        decode_steps.append({
            "step": step, "position": position,
            "output_max_abs_error": o_err,
            "key_max_abs_error": k_err,
            "value_max_abs_error": v_err,
        })
        cache_key_err = max(cache_key_err, k_err)
        cache_value_err = max(cache_value_err, v_err)
        cache = dec["cache"]

    all_errors = (
        [rope_comm_err]
        + list(prefill_metrics.values())
        + [s["output_max_abs_error"] for s in decode_steps]
        + [cache_key_err, cache_value_err]
    )
    allclose = all(e <= 1e-8 for e in all_errors)

    return {
        "mask_family": mask_family,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rope_commutation_max_error": rope_comm_err,
        **prefill_metrics,
        "decode_steps": decode_steps,
        "cache_append_key_max_abs_error": cache_key_err,
        "cache_append_value_max_abs_error": cache_value_err,
        "allclose": allclose,
    }


def run_rope_gqa_probe(config: RopeGQAProbeConfig) -> dict[str, Any]:
    nh = config.num_heads
    nkv = config.num_key_value_heads

    correctness: dict[str, dict[str, Any]] = {}
    if config.run_rotation_baseline:
        correctness["pairwise_rotation"] = {
            "mha": _run_case(config, nh, nh, "pairwise_rotation"),
            "gqa": _run_case(config, nh, nkv, "pairwise_rotation"),
        }
    if config.run_complex_scaling:
        correctness["pairwise_complex_scaling"] = {
            "mha": _run_case(config, nh, nh, "pairwise_complex_scaling"),
            "gqa": _run_case(config, nh, nkv, "pairwise_complex_scaling"),
        }

    # Backward-compatible top-level mha/gqa use the default mask family.
    default_family = config.mask_family
    if default_family not in correctness:
        # Default family disabled; fall back to whichever ran.
        default_family = next(iter(correctness))
    mha = correctness[default_family]["mha"]
    gqa = correctness[default_family]["gqa"]

    all_allclose = all(
        case["allclose"]
        for fam in correctness.values()
        for case in fam.values()
    )

    leakage_proxy = (
        run_rope_leakage_proxy(config) if config.run_leakage_proxy else None
    )

    return {
        "stage": "6.4.1_rope_gqa_complex_scaling",
        "experiment": "rope_gqa_probe",
        "status": "ok",
        "security_status":
            "operator_compatible_leakage_reduction_not_semantic_security",
        "no_intermediate_tee": True,
        "statement": _REQUIRED_STATEMENT,
        "config": asdict(config),
        "mha": mha,
        "gqa": gqa,
        "all_allclose": bool(all_allclose),
        "correctness": correctness,
        "leakage_proxy": leakage_proxy,
        "mask_structure": {
            "rope_pairwise_commuting": True,
            "default_mask_family": default_family,
            "q_mask": "per-query-head inverse-transpose of mapped KV mask",
            "k_mask": "per-kv-head RoPE-compatible block mask",
            "v_mask": "per-kv-head RoPE-compatible block mask",
            "gqa_supported": True,
            "same_cache_mask_within_session": True,
        },
        "metadata": {
            "default_mask_family": "pairwise_complex_scaling",
            "leakage_caveats": [
                "RoPE pair partition is preserved",
                "No cross-pair dense mixing inside RoPE-compatible region",
                "rotation mode preserves per-pair norm",
                "complex-scaling mode changes per-pair norm but preserves "
                "pair structure",
                "KV cache requires same mask within a generation session",
                "attention scores remain visible to the GPU in this probe",
                "This is not a semantic-security proof",
            ],
        },
        "limitations": [
            "Synthetic tensor-level probe, not a HF LLaMA/Qwen wrapper.",
            "q_proj/k_proj/v_proj/o_proj weight folding is not yet "
            "integrated (masks applied at the Q/K/V tensor level).",
            "RoPE scaling variants (NTK/YaRN) are not implemented.",
            "Real model embedding, LM head, sampling, and full generation "
            "are not covered.",
            "RoPE-compatible masks are weaker than dense masks; dense masks "
            "should be restored before/after RoPE-constrained regions in "
            "later block integration.",
            "CPU-only; no formal, cryptographic, or semantic security is "
            "claimed.",
        ],
    }


__all__ = [
    "RopeGQAProbeConfig",
    "nearest_neighbor_matching_accuracy",
    "pair_norms",
    "pearson_corr_flat",
    "run_rope_gqa_probe",
    "run_rope_leakage_proxy",
]
