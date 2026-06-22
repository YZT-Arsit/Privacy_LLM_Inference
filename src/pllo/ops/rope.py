"""RoPE primitives with mask-commuting pair-wise rotation masks.

Stage 6.4 (RoPE-compatible masked attention) needs a RoPE convention
whose 2D rotations act on *adjacent* feature pairs ``(2i, 2i+1)``, so a
block-diagonal mask ``M = blockdiag(R(a_1), ..., R(a_{head_dim/2}))``
(each block a 2D rotation) commutes with RoPE. Because all 2D rotations
commute (``SO(2)`` is abelian), for any pair-wise rotation mask ``M``:

    RoPE(x @ M) == RoPE(x) @ M.

This module is self-contained and uses the adjacent-pair convention
internally (it does NOT reuse the half-split convention in
``experiments/rope_probe.py``; that module is left untouched).

Conventions:
* row-vector; head tensors shaped ``[batch, heads, seq_len, head_dim]``;
* ``head_dim`` must be even;
* ``rotate_half([x0, x1, x2, x3, ...]) = [-x1, x0, -x3, x2, ...]``;
* ``x_rope = x * cos + rotate_half(x) * sin`` with ``cos`` / ``sin``
  built so each frequency is repeated across its adjacent pair.

CPU-only, float64-friendly. No CUDA, no transformers, no new deps.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "apply_rope",
    "build_rope_cache",
    "is_pairwise_complex_scaling_mask",
    "make_pairwise_complex_scaling_mask",
    "make_pairwise_complex_scaling_masks",
    "make_pairwise_rotation_mask",
    "make_pairwise_rotation_masks",
    "pairwise_complex_scaling_inverse",
    "rope_commutation_error",
    "rotate_half",
]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair rotate-half: ``[x0,x1,x2,x3,...] -> [-x1,x0,-x3,x2,...]``."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"head_dim must be even, got {x.shape[-1]}")
    x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    rotated = torch.stack((-x1, x0), dim=-1)
    return rotated.reshape(*x.shape)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` each ``[seq_len, head_dim]`` (adjacent-pair).

    Each of the ``head_dim/2`` frequencies is repeated across its adjacent
    pair so that ``cos = [c0, c0, c1, c1, ...]`` aligns with
    :func:`rotate_half`.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    device = torch.device(device)
    half = head_dim // 2
    inv_freq = base ** (
        -(torch.arange(0, half, dtype=dtype, device=device) * 2.0) / head_dim
    )  # [half]
    t = torch.arange(seq_len, dtype=dtype, device=device)  # [seq_len]
    freqs = torch.outer(t, inv_freq)  # [seq_len, half]
    # Repeat each frequency across its adjacent pair: [f0, f0, f1, f1, ...].
    emb = freqs.repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply RoPE to ``x`` ``[..., seq_len, head_dim]``.

    ``cos`` / ``sin`` are ``[max_seq, head_dim]`` caches. When
    ``position_ids`` (shape ``[seq_len]``) is given, the rows are gathered
    by position (used for decode at an arbitrary absolute position).
    """
    seq_len = x.shape[-2]
    if position_ids is None:
        cos_sel = cos[:seq_len]
        sin_sel = sin[:seq_len]
    else:
        cos_sel = cos.index_select(0, position_ids)
        sin_sel = sin.index_select(0, position_ids)
    # Broadcast to [1, 1, seq_len, head_dim].
    while cos_sel.dim() < x.dim():
        cos_sel = cos_sel.unsqueeze(0)
        sin_sel = sin_sel.unsqueeze(0)
    return x * cos_sel + rotate_half(x) * sin_sel


def make_pairwise_rotation_mask(
    head_dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Block-diagonal rotation mask ``[head_dim, head_dim]``.

    ``M = blockdiag(R(a_1), ..., R(a_{head_dim/2}))`` with each block
    ``[[cos a, -sin a], [sin a, cos a]]``. ``M`` is orthogonal and commutes
    with the adjacent-pair RoPE.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    device = torch.device(device)
    half = head_dim // 2
    if generator is not None:
        angles = torch.rand(half, generator=generator, dtype=dtype,
                            device=device) * (2.0 * torch.pi)
    else:
        angles = torch.rand(half, dtype=dtype, device=device) * (2.0 * torch.pi)
    cos_a = angles.cos()
    sin_a = angles.sin()
    M = torch.zeros(head_dim, head_dim, dtype=dtype, device=device)
    idx = torch.arange(half, device=device)
    two_i = idx * 2
    M[two_i, two_i] = cos_a
    M[two_i, two_i + 1] = -sin_a
    M[two_i + 1, two_i] = sin_a
    M[two_i + 1, two_i + 1] = cos_a
    return M


def make_pairwise_rotation_masks(
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Stack of ``num_heads`` pair-wise rotation masks ``[num_heads, D, D]``."""
    masks = [
        make_pairwise_rotation_mask(head_dim, dtype, device, generator)
        for _ in range(num_heads)
    ]
    return torch.stack(masks, dim=0)


# ---------------------------------------------------------------------------
# Pair-wise complex-scaling masks (Stage 6.4.1)
# ---------------------------------------------------------------------------
#
# A pair-wise rotation mask preserves each 2D pair's norm exactly. A
# pair-wise *complex-scaling* mask uses blocks ``s * R(alpha)``:
#
#     [[a, -b],
#      [b,  a]]   with a = s cos(alpha), b = s sin(alpha), s > 0.
#
# Each block is still 2D and acts only inside its adjacent feature pair, so
# it commutes with the adjacent-pair RoPE exactly like a rotation block
# (``s`` is a scalar and ``R(alpha)`` commutes with RoPE; ``SO(2)`` is
# abelian). Unlike rotation, the block is NOT orthogonal: ``M^{-1} != M^T``.
# The closed-form inverse of a block is ``1/(a^2+b^2) * [[a, b], [-b, a]]``,
# itself a complex-scaling block with scale ``1/s`` and angle ``-alpha``.
#
# This is a strictly weaker local mask family than dense masks; it is used
# only because arbitrary dense masks do not commute with RoPE. It is NOT a
# semantic-security construction.


def make_pairwise_complex_scaling_mask(
    head_dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    scale_low: float = 0.25,
    scale_high: float = 2.0,
    min_abs: float = 1e-6,
) -> torch.Tensor:
    """Block-diagonal complex-scaling mask ``[head_dim, head_dim]``.

    ``M = blockdiag(s_1 R(a_1), ..., s_{D/2} R(a_{D/2}))`` with each block
    ``[[a, -b], [b, a]]``, ``a = s cos(alpha)``, ``b = s sin(alpha)``,
    ``s in [scale_low, scale_high]``, ``alpha in [0, 2pi)``. ``s`` is
    clamped so each block has ``a^2 + b^2 = s^2 >= min_abs`` (invertible).
    Commutes with the adjacent-pair RoPE; not orthogonal.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    device = torch.device(device)
    half = head_dim // 2
    if generator is not None:
        u = torch.rand(half, generator=generator, dtype=dtype, device=device)
        angles = torch.rand(half, generator=generator, dtype=dtype,
                            device=device) * (2.0 * torch.pi)
    else:
        u = torch.rand(half, dtype=dtype, device=device)
        angles = torch.rand(half, dtype=dtype, device=device) * (2.0 * torch.pi)
    scale = scale_low + u * (scale_high - scale_low)
    scale = torch.clamp(scale, min=math.sqrt(min_abs))
    a = scale * angles.cos()
    b = scale * angles.sin()
    M = torch.zeros(head_dim, head_dim, dtype=dtype, device=device)
    two_i = torch.arange(half, device=device) * 2
    M[two_i, two_i] = a
    M[two_i, two_i + 1] = -b
    M[two_i + 1, two_i] = b
    M[two_i + 1, two_i + 1] = a
    return M


def make_pairwise_complex_scaling_masks(
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    scale_low: float = 0.25,
    scale_high: float = 2.0,
) -> torch.Tensor:
    """Stack of ``num_heads`` complex-scaling masks ``[num_heads, D, D]``."""
    masks = [
        make_pairwise_complex_scaling_mask(
            head_dim, dtype, device, generator, scale_low, scale_high)
        for _ in range(num_heads)
    ]
    return torch.stack(masks, dim=0)


def is_pairwise_complex_scaling_mask(
    mask: torch.Tensor, atol: float = 1e-8,
) -> bool:
    """Validate ``mask`` ``[D, D]`` is block-diag ``[[a,-b],[b,a]]`` form.

    Requires square, even dim, each 2D block of the complex-scaling form,
    all off-block entries zero, and ``a^2 + b^2 > 0`` per block.
    """
    if mask.dim() != 2 or mask.shape[0] != mask.shape[1]:
        return False
    d = mask.shape[0]
    if d % 2 != 0:
        return False
    half = d // 2
    two_i = torch.arange(half, device=mask.device) * 2
    a = mask[two_i, two_i]
    b = mask[two_i + 1, two_i]
    ref = torch.zeros_like(mask)
    ref[two_i, two_i] = a
    ref[two_i, two_i + 1] = -b
    ref[two_i + 1, two_i] = b
    ref[two_i + 1, two_i + 1] = a
    if not torch.allclose(mask, ref, atol=atol, rtol=0.0):
        return False
    return bool(torch.all(a * a + b * b > atol * atol).item())


def pairwise_complex_scaling_inverse(mask: torch.Tensor) -> torch.Tensor:
    """Closed-form block-diagonal inverse of a complex-scaling mask.

    Handles ``[..., D, D]``. For each block ``[[a,-b],[b,a]]`` the inverse
    is ``1/(a^2+b^2) * [[a, b], [-b, a]]`` (no ``torch.linalg.inv``).
    """
    d = mask.shape[-1]
    half = d // 2
    two_i = torch.arange(half, device=mask.device) * 2
    a = mask[..., two_i, two_i]
    b = mask[..., two_i + 1, two_i]
    det = a * a + b * b
    inv_a = a / det
    inv_b = b / det
    inv = torch.zeros_like(mask)
    inv[..., two_i, two_i] = inv_a
    inv[..., two_i, two_i + 1] = inv_b
    inv[..., two_i + 1, two_i] = -inv_b
    inv[..., two_i + 1, two_i + 1] = inv_a
    return inv


def rope_commutation_error(
    x: torch.Tensor,
    mask: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> float:
    """Max abs error of ``RoPE(x @ M) - RoPE(x) @ M`` (should be ~0)."""
    lhs = apply_rope(x @ mask, cos, sin, position_ids)
    rhs = apply_rope(x, cos, sin, position_ids) @ mask
    return float((lhs - rhs).abs().max().item())
