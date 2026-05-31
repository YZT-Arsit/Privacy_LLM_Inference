"""Stage 6.4 — Grouped / Multi-Query Attention KV-shape probe.

Validates that the per-head mask scheme used in Stage 5.0 / Stage 6.1 also
works for grouped query attention (``num_kv_heads < num_query_heads``).

Setup (synthetic):

* ``Q  : [B, q_heads, S, D]``
* ``K  : [B, kv_heads, S, D]``
* ``V  : [B, kv_heads, S, D]``
* ``q_heads % kv_heads == 0`` (each kv-head services ``group_size``
  q-heads).
* One mask pair per **kv head**:
    ``N_K[k]`` and its inverse-transpose ``N_K[k]^{-T}``.
* Per **q head** ``i``: ``N_Q[i] = N_K[i // group_size]^{-T}``.

Invariants verified:

* ``N_Q[i] N_K[i // group_size]^T = I`` per q-head.
* ``Q_tilde K_tilde_rep^T ≈ Q K_rep^T`` (score path).
* ``AttnProb V_tilde_rep ≈ (AttnProb V_rep) N_V_rep`` (value path).
* Mask dimension is ``head_dim``, NOT ``hidden_size`` and NOT
  ``num_heads``.

This is a **tensor-level** probe. No KV cache runtime, no generation,
no LM head.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pllo.masks.mask_generator import generate_invertible_matrix


@dataclass
class GqaProbeConfig:
    batch_size: int = 2
    num_query_heads: int = 4
    num_kv_heads: int = 2
    seq_len: int = 8
    head_dim: int = 16
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 2026


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads along the head axis: ``[B, kv, S, D] → [B, kv*n_rep, S, D]``.

    Matches HF LLaMA / Qwen ``repeat_kv`` semantics: the ``j``-th kv head
    is repeated ``n_rep`` times in adjacent positions.
    """
    if n_rep == 1:
        return x
    B, K, S, D = x.shape
    return x.unsqueeze(2).expand(B, K, n_rep, S, D).reshape(B, K * n_rep, S, D)


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def _allclose_metrics(
    expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float
) -> dict[str, float]:
    diff = (actual - expected).abs()
    ref_norm = float(expected.norm().clamp_min(1e-30).item())
    return {
        "max_abs_error": float(diff.max().item()),
        "relative_l2_error": float(
            ((actual - expected).norm() / max(ref_norm, 1e-30)).item()
        ),
        "allclose": bool(torch.allclose(expected, actual, atol=atol, rtol=rtol)),
    }


def run_gqa_probe(config: GqaProbeConfig) -> dict[str, Any]:
    """Run the synthetic GQA / MQA probe and return a structured report."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)
    atol, rtol = _atol_rtol(dtype)

    if config.num_query_heads % config.num_kv_heads != 0:
        return {
            "config": asdict(config),
            "status": "skipped",
            "reason": (
                f"num_query_heads {config.num_query_heads} not divisible by"
                f" num_kv_heads {config.num_kv_heads}; GQA requires divisibility."
            ),
        }

    B = config.batch_size
    Hq = config.num_query_heads
    Hk = config.num_kv_heads
    S = config.seq_len
    D = config.head_dim
    group_size = Hq // Hk

    Q = torch.randn(B, Hq, S, D, dtype=dtype, device=device)
    K = torch.randn(B, Hk, S, D, dtype=dtype, device=device)
    V = torch.randn(B, Hk, S, D, dtype=dtype, device=device)

    K_rep = repeat_kv(K, group_size)
    V_rep = repeat_kv(V, group_size)
    scores_plain = Q @ K_rep.transpose(-2, -1) / math.sqrt(D)
    probs_plain = F.softmax(scores_plain, dim=-1)
    av_plain = probs_plain @ V_rep

    # ---- One mask pair per kv head; N_Q is derived per q head. ----
    N_K_list: list[torch.Tensor] = []
    N_K_inv_list: list[torch.Tensor] = []
    for _ in range(Hk):
        N_K, N_K_inv = generate_invertible_matrix(D, dtype, device)
        N_K_list.append(N_K)
        N_K_inv_list.append(N_K_inv)
    N_V_list: list[torch.Tensor] = []
    N_V_inv_list: list[torch.Tensor] = []
    for _ in range(Hk):
        N_V, N_V_inv = generate_invertible_matrix(D, dtype, device)
        N_V_list.append(N_V)
        N_V_inv_list.append(N_V_inv)

    N_K_stack = torch.stack(N_K_list, dim=0)      # [Hk, D, D]
    N_V_stack = torch.stack(N_V_list, dim=0)      # [Hk, D, D]

    # Per-q-head N_Q[i] = N_K[i // group_size]^{-T} ⇒ N_Q N_K^T = I.
    N_Q_per_q: list[torch.Tensor] = []
    constraint_errs: list[float] = []
    eye = torch.eye(D, dtype=dtype, device=device)
    for i in range(Hq):
        k_idx = i // group_size
        N_Q_i = N_K_inv_list[k_idx].transpose(-2, -1)
        N_Q_per_q.append(N_Q_i)
        err = float((N_Q_i @ N_K_list[k_idx].transpose(-2, -1) - eye).abs().max().item())
        constraint_errs.append(err)
    N_Q_stack = torch.stack(N_Q_per_q, dim=0)      # [Hq, D, D]

    # ---- Score-path invariant. ----
    Q_tilde = Q @ N_Q_stack.unsqueeze(0)            # [B, Hq, S, D]
    K_tilde = K @ N_K_stack.unsqueeze(0)            # [B, Hk, S, D]
    K_tilde_rep = repeat_kv(K_tilde, group_size)    # [B, Hq, S, D]
    scores_tilde = (
        Q_tilde @ K_tilde_rep.transpose(-2, -1) / math.sqrt(D)
    )
    score_metrics = _allclose_metrics(scores_plain, scores_tilde, atol, rtol)

    # ---- Value-path invariant (per q-head N_V_rep). ----
    V_tilde = V @ N_V_stack.unsqueeze(0)            # [B, Hk, S, D]
    V_tilde_rep = repeat_kv(V_tilde, group_size)    # [B, Hq, S, D]
    probs_tilde = F.softmax(scores_tilde, dim=-1)
    av_tilde = probs_tilde @ V_tilde_rep            # [B, Hq, S, D]
    # Expected: AttnProb V_tilde_rep = (AttnProb V_rep) N_V_rep per-q-head.
    N_V_per_q = torch.stack(
        [N_V_list[i // group_size] for i in range(Hq)], dim=0
    )
    expected_av_tilde = av_plain @ N_V_per_q.unsqueeze(0)
    value_metrics = _allclose_metrics(expected_av_tilde, av_tilde, atol, rtol)

    return {
        "config": asdict(config),
        "status": "ok",
        "attention_variant": (
            "mqa" if Hk == 1 else ("mha" if Hk == Hq else "gqa")
        ),
        "shapes": {
            "Q": list(Q.shape),
            "K": list(K.shape),
            "V": list(V.shape),
            "K_rep": list(K_rep.shape),
            "V_rep": list(V_rep.shape),
            "scores_plain": list(scores_plain.shape),
        },
        "mask_dimension": int(D),
        "mask_is_per_head_not_hidden_size": True,
        "mask_is_per_head_not_num_heads": True,
        "group_size": int(group_size),
        "qk_constraint_max_error_per_q_head": float(max(constraint_errs)),
        "score_path": score_metrics,
        "value_path": value_metrics,
        "allclose": bool(score_metrics["allclose"] and value_metrics["allclose"]),
        "limitations": [
            "Tensor-level probe; no KV cache runtime is implemented.",
            "No generation path is exercised.",
            "Synthetic Q/K/V — not extracted from a real Qwen / LLaMA layer.",
        ],
    }


__all__ = ["GqaProbeConfig", "repeat_kv", "run_gqa_probe"]
