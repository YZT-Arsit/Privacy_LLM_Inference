"""Stage 6.4 — RoPE-aware attention probe for modern decoder-only models.

Two probes:

* **Probe A (required) — post-RoPE masking invariant.** Apply RoPE to
  plain Q / K first, then right-multiply by per-head Q/K masks that
  satisfy ``N_Q N_K^T = I``. Verify that the score
  ``Q_tilde K_tilde^T`` recovers the plain ``Q_rope K_rope^T``.

* **Probe B (feasibility / negative result) — pre-RoPE mask commutation.**
  Check whether ``RoPE(Q N) == RoPE(Q) N`` for three mask families:
  ``dense_invertible``, ``orthogonal``, and a constructed
  ``block_diagonal_rotation`` mask that is per-pair 2D rotation in the
  same plane as RoPE. Dense and generic orthogonal masks are expected
  NOT to commute; only the planar block-rotation mask should commute.
  This is reported as a feasibility note — system correctness only
  requires Probe A.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.ops.compatible_masks import generate_orthogonal


def _rope_freqs(
    head_dim: int, seq_len: int, base: float, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """LLaMA / Qwen convention: ``cos``/``sin`` shape ``[seq_len, head_dim]``.

    The angle for pair ``j ∈ [0, head_dim/2)`` and position ``p`` is
    ``p / base ** (2j / head_dim)``. The first ``head_dim/2`` channels are
    the "real" parts, the last ``head_dim/2`` channels are the "imag" parts
    (this matches HF LLaMA / Qwen's ``rotate_half`` convention).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    positions = torch.arange(seq_len, dtype=torch.float64, device=device)
    theta = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)   # [S, half]
    cos = theta.cos().repeat(1, 2).to(dtype)
    sin = theta.sin().repeat(1, 2).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply LLaMA / Qwen-style RoPE to ``x``.

    ``x`` shape: ``[..., seq_len, head_dim]``. Common callers pass
    ``[batch, heads, seq_len, head_dim]`` (Q / K layout). ``head_dim`` must
    be even. ``positions`` defaults to ``torch.arange(seq_len)`` along the
    second-to-last axis. Returns a tensor with the same shape.
    """
    head_dim = x.shape[-1]
    seq_len = x.shape[-2]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    if positions is None:
        positions = torch.arange(seq_len, device=x.device)
    if positions.shape[-1] != seq_len:
        raise ValueError(
            f"positions length {positions.shape[-1]} must match seq_len {seq_len}"
        )
    cos, sin = _rope_freqs(head_dim, seq_len, base, x.dtype, x.device)
    # Broadcast to x's leading dims by inserting unit dims for [..., S, D].
    extra_dims = x.ndim - 2
    for _ in range(extra_dims):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


def _generate_block_diagonal_rotation_mask(
    head_dim: int, dtype: torch.dtype, device: torch.device, seed: int = 0
) -> torch.Tensor:
    """``N`` that commutes with RoPE: pairwise 2D rotation in the SAME planes.

    LLaMA's RoPE pairs channel ``j`` with channel ``j + head_dim/2``. Any
    2D rotation in that same plane commutes with RoPE's rotation. We sample
    one fresh angle per pair (``head_dim / 2`` angles total) and assemble
    the corresponding block. The result is an orthogonal matrix that
    commutes with ``RoPE(·)`` for every fixed sequence position.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    g = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.empty(half, dtype=torch.float64).uniform_(
        -3.14159, 3.14159, generator=g
    )
    c = angles.cos().to(dtype)
    s = angles.sin().to(dtype)
    N = torch.zeros(head_dim, head_dim, dtype=dtype, device=device)
    # Real-imag pair (j, j + half) gets a 2D rotation in that plane.
    for j in range(half):
        N[j, j] = c[j]
        N[j + half, j + half] = c[j]
        N[j, j + half] = -s[j]
        N[j + half, j] = s[j]
    return N


# ---------------------------------------------------------------------------
# Config + probes
# ---------------------------------------------------------------------------


@dataclass
class RopeProbeConfig:
    batch_size: int = 2
    num_heads: int = 4
    seq_len: int = 8
    head_dim: int = 16
    base: float = 10000.0
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 2026


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def _allclose_metrics(
    expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float
) -> dict[str, float]:
    diff = (actual - expected).abs()
    max_err = float(diff.max().item())
    ref_norm = float(expected.norm().clamp_min(1e-30).item())
    rel_l2 = float(((actual - expected).norm() / max(ref_norm, 1e-30)).item())
    allclose = bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
    return {
        "max_abs_error": max_err,
        "relative_l2_error": rel_l2,
        "allclose": allclose,
    }


def run_rope_probe(config: RopeProbeConfig) -> dict[str, Any]:
    """Run both RoPE probes and return a structured report."""
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)
    atol, rtol = _atol_rtol(dtype)

    if config.head_dim % 2 != 0:
        return {
            "config": asdict(config),
            "status": "skipped",
            "reason": f"head_dim {config.head_dim} is odd; RoPE requires even head_dim.",
        }

    B, H, S, D = (
        config.batch_size,
        config.num_heads,
        config.seq_len,
        config.head_dim,
    )

    Q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    K = torch.randn(B, H, S, D, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Probe A — post-RoPE masking invariant (REQUIRED).
    # ------------------------------------------------------------------
    Q_rope = apply_rope(Q, base=config.base)
    K_rope = apply_rope(K, base=config.base)
    scores_plain = Q_rope @ K_rope.transpose(-2, -1)

    # Per-head mask pair with N_Q N_K^T = I.
    n_k_per_head = []
    n_q_per_head = []
    for _ in range(H):
        N_K, N_K_inv = generate_invertible_matrix(D, dtype, device)
        # Choose N_Q = N_K^{-T} ⇒ N_Q N_K^T = N_K^{-T} N_K^T = I.
        N_Q = N_K_inv.transpose(-2, -1)
        n_k_per_head.append(N_K)
        n_q_per_head.append(N_Q)
    N_K = torch.stack(n_k_per_head, dim=0)   # [H, D, D]
    N_Q = torch.stack(n_q_per_head, dim=0)
    constraint_err = (
        (N_Q @ N_K.transpose(-2, -1))
        - torch.eye(D, dtype=dtype, device=device)
    ).abs().max().item()

    Q_tilde = Q_rope @ N_Q.unsqueeze(0)      # [B, H, S, D]
    K_tilde = K_rope @ N_K.unsqueeze(0)
    scores_tilde = Q_tilde @ K_tilde.transpose(-2, -1)
    probe_a_metrics = _allclose_metrics(scores_plain, scores_tilde, atol, rtol)

    # ------------------------------------------------------------------
    # Probe B — pre-RoPE mask commutation (feasibility / negative result).
    # ------------------------------------------------------------------
    # All three families use a single ``[D, D]`` mask applied to Q before RoPE.
    families: dict[str, torch.Tensor] = {
        "dense_invertible": generate_invertible_matrix(D, dtype, device)[0],
        "orthogonal": generate_orthogonal(D, dtype, device),
        "block_diagonal_rotation": _generate_block_diagonal_rotation_mask(
            D, dtype, device, seed=config.seed
        ),
    }
    expected_per_family = {
        "dense_invertible": "expected_failure",
        "orthogonal": "expected_failure",
        "block_diagonal_rotation": "expected_to_commute",
    }
    probe_b: dict[str, dict[str, Any]] = {}
    for name, N in families.items():
        # Compare RoPE(Q N) vs. RoPE(Q) N — single-head Q for clarity.
        Q_single = Q[:, 0, :, :]   # [B, S, D]
        lhs = apply_rope(Q_single @ N, base=config.base)
        rhs = apply_rope(Q_single, base=config.base) @ N
        metrics = _allclose_metrics(rhs, lhs, atol, rtol)
        probe_b[name] = {
            "mask_family": name,
            "expected_behavior": expected_per_family[name],
            **metrics,
            "commutes": bool(metrics["allclose"]),
        }

    return {
        "config": asdict(config),
        "status": "ok",
        "qk_constraint_error": float(constraint_err),
        "probe_a_post_rope_masking_invariant": {
            "scores_shape": list(scores_plain.shape),
            "requirement": "scores_tilde ≈ scores_plain after Q/K masking",
            **probe_a_metrics,
        },
        "probe_b_pre_rope_mask_commutation": {
            "requirement": (
                "Feasibility/negative-result probe. RoPE(Q N) ?= RoPE(Q) N"
                " holds iff N commutes with the RoPE block rotation."
            ),
            "per_family": probe_b,
            "rope_mask_compatibility_notes": (
                "Only masks that act as 2D rotations in the same planes as"
                " RoPE commute with RoPE. Generic dense and generic"
                " orthogonal masks DO NOT commute. The required path uses"
                " probe A (post-RoPE masking)."
            ),
        },
        "limitations": [
            "RoPE is handled conservatively by masking after RoPE in the"
            " required probe.",
            "Mask-before-RoPE commutation is only a feasibility / negative"
            " result.",
            "This is not a real TEE measurement.",
        ],
    }


__all__ = [
    "RopeProbeConfig",
    "apply_rope",
    "run_rope_probe",
]
