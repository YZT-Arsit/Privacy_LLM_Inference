"""Amulet-style *lifted* nonlinear islands (CPU correctness prototype).

The existing :mod:`pllo.ops.nonlinear_islands` masks the activation input
with a permutation-only (``ZP``) view. This module adds an Amulet-style
variant that replaces the direct permutation view with a lifted
Kronecker / block view: each intermediate feature ``U[:, j]`` is expanded
into ``k`` lifted columns ``U[:, j] * R[j, ell]`` before the activation,
then squeezed back with a matrix ``S`` that is folded into the following
linear weight on the trusted side.

Two squeeze families are implemented:

* **ReLU (homogeneous).** Because ``ReLU(a x) = a ReLU(x)`` for ``a > 0``,
  a positive lift ``R`` can be inverted exactly by a least-squares squeeze
  ``S`` with ``sum_ell R[j, ell] Ucoef[j, ell] = 1``.
* **Selector (GELU / SiLU / SwiGLU).** Non-homogeneous activations cannot
  absorb a scale, so the lift uses ``R[j, valid_idx[j]] = 1`` with positive
  decoy scales elsewhere, and the squeeze ``S`` selects exactly the valid
  lifted column.

A LayerNorm "gadget" island uses a row-wise shift ``G = I + r 1^T`` so the
LayerNorm core is exactly invariant even with ``eps > 0`` (no scale is
applied -- ``LNCore(lambda x) != LNCore(x)`` when ``eps > 0``).

Design notes (kept consistent with the rest of the repo):

* Row-vector convention; hidden states ``[m, d]``; intermediate ``[m, h]``.
* CPU-only correctness prototype; not integrated into the GPT-2 wrapper.
* Pad stays OUTSIDE the nonlinear core and is compensated only at the
  preceding ``Linear`` boundary.
* Existing public APIs are untouched; these are additive functions.

Security caveat (selector mode): if the lifted projection
``W2_tilde_lift`` is visible to the accelerator and the decoy rows of the
squeeze are exactly zero, an observer can read off the valid selector
positions. This is a correctness prototype; a deployment would need a
trusted / fused squeeze or additional mitigation. See the per-island
``metadata`` warnings.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from pllo.ops.nonlinear_islands import (
    gelu_reference,
    layernorm_core,
    relu_reference,
    silu_reference,
)

__all__ = [
    "apply_feature_permutation",
    "block_lift",
    "inverse_permutation",
    "make_layernorm_shift_gadget",
    "make_relu_squeeze",
    "make_selector_lift_params",
    "run_layernorm_gadget_island",
    "run_relu_lifted_mlp_island",
    "run_selector_lifted_mlp_island",
    "run_swiglu_selector_lifted_mlp_island",
]


_SELECTOR_WARNING = (
    "Correctness prototype only: if W2_tilde_lift is visible and decoy "
    "rows are exactly zero, a GPU observer can infer valid selector "
    "positions. This must be treated as a correctness prototype or "
    "requires trusted/fused squeeze or additional mitigation."
)


# ---------------------------------------------------------------------------
# A. Block lift
# ---------------------------------------------------------------------------


def block_lift(U: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Lift ``U`` [m, h] by ``R`` [h, k] into [m, h*k].

    ``lift_R(U)[row, j*k + ell] = U[row, j] * R[j, ell]``.
    """
    if U.dim() != 2:
        raise ValueError(f"U must be 2D [m, h], got shape {tuple(U.shape)}")
    if R.dim() != 2:
        raise ValueError(f"R must be 2D [h, k], got shape {tuple(R.shape)}")
    m, h = U.shape
    h_r, k = R.shape
    if h_r != h:
        raise ValueError(
            f"R rows ({h_r}) must equal U columns ({h})"
        )
    # [m, h, 1] * [1, h, k] -> [m, h, k] -> [m, h*k]
    lifted = U.unsqueeze(-1) * R.unsqueeze(0)
    return lifted.reshape(m, h * k)


# ---------------------------------------------------------------------------
# B. ReLU squeeze (homogeneous activation)
# ---------------------------------------------------------------------------


def make_relu_squeeze(R: torch.Tensor) -> torch.Tensor:
    """Build ``S`` [h*k, h] with ``ReLU(block_lift(U, R)) @ S = ReLU(U)``.

    Requires ``R > 0``. Uses the stable least-squares coefficient
    ``Ucoef[j, ell] = R[j, ell] / sum_ell R[j, ell]^2`` so that
    ``sum_ell R[j, ell] Ucoef[j, ell] = 1``.
    """
    if R.dim() != 2:
        raise ValueError(f"R must be 2D [h, k], got {tuple(R.shape)}")
    if not bool((R > 0).all()):
        raise ValueError("ReLU lift requires strictly positive R")
    h, k = R.shape
    denom = (R * R).sum(dim=1, keepdim=True)  # [h, 1]
    ucoef = R / denom  # [h, k]
    S = torch.zeros(h * k, h, dtype=R.dtype, device=R.device)
    rows = torch.arange(h, device=R.device)
    for ell in range(k):
        S[rows * k + ell, rows] = ucoef[:, ell]
    return S


# ---------------------------------------------------------------------------
# C. Selector lift (non-homogeneous activations)
# ---------------------------------------------------------------------------


def make_selector_lift_params(
    h: int,
    k: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    *,
    decoy_low: float = 0.25,
    decoy_high: float = 2.0,
    avoid_unit_band: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selector lift parameters ``(R, valid_idx, S)``.

    * ``R[j, valid_idx[j]] = 1``; other entries are positive decoy scales
      sampled uniformly in ``[decoy_low, decoy_high]`` and nudged away from
      ``1`` by at least ``avoid_unit_band``.
    * ``S[j*k + valid_idx[j], j] = 1``; all other rows are zero.

    Then ``activation(block_lift(U, R)) @ S = activation(U)`` for any
    elementwise activation.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    device = torch.device(device)
    # Valid index per feature.
    if generator is not None:
        valid_idx = torch.randint(
            0, k, (h,), generator=generator, device=device,
        )
        decoys = (
            torch.rand(h, k, generator=generator, dtype=dtype, device=device)
            * (decoy_high - decoy_low) + decoy_low
        )
    else:
        valid_idx = torch.randint(0, k, (h,), device=device)
        decoys = (
            torch.rand(h, k, dtype=dtype, device=device)
            * (decoy_high - decoy_low) + decoy_low
        )
    # Push decoys out of the (1 - band, 1 + band) interval so they are not
    # accidentally close to the identity scale.
    near_unit = (decoys - 1.0).abs() < avoid_unit_band
    decoys = torch.where(
        near_unit,
        decoys + torch.sign(decoys - 1.0).clamp(min=0.0) * avoid_unit_band
        + (decoys <= 1.0).to(dtype) * (-avoid_unit_band),
        decoys,
    )
    # Guarantee strict positivity after the nudge.
    decoys = decoys.clamp(min=decoy_low * 0.5)

    R = decoys.clone()
    rows = torch.arange(h, device=device)
    R[rows, valid_idx] = 1.0

    S = torch.zeros(h * k, h, dtype=dtype, device=device)
    S[rows * k + valid_idx, rows] = 1.0
    return R, valid_idx, S


# ---------------------------------------------------------------------------
# D. Permutation helpers
# ---------------------------------------------------------------------------


def apply_feature_permutation(
    U: torch.Tensor, perm: torch.Tensor,
) -> torch.Tensor:
    """Permute the last (feature) dimension: ``U[:, perm]``."""
    return U.index_select(-1, perm)


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return ``inv`` with ``inv[perm] == arange`` (so ``U[:, perm][:, inv] == U``)."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


def _make_perm(h: int, generator: torch.Generator | None,
               device: torch.device) -> torch.Tensor:
    if generator is not None:
        return torch.randperm(h, generator=generator, device=device)
    return torch.randperm(h, device=device)


def _gen(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _positive_relu_lift(
    h: int, k: int, dtype: torch.dtype, device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Strictly positive lift matrix for the ReLU island."""
    if generator is not None:
        R = torch.rand(h, k, generator=generator, dtype=dtype, device=device)
    else:
        R = torch.rand(h, k, dtype=dtype, device=device)
    return R * 1.75 + 0.25  # in [0.25, 2.0], strictly positive


# ---------------------------------------------------------------------------
# E. ReLU lifted MLP island
# ---------------------------------------------------------------------------


def _masked_linear_perm_output(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
    n_in: torch.Tensor, n_in_inv: torch.Tensor, perm: torch.Tensor,
    pad_in: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``(x @ w + b)[:, perm]`` through the masked boundary.

    No pad:  ``(x n_in) (n_in_inv w[:, perm]) + b[perm]``.
    Pad:     ``((x - pad) n_in)(n_in_inv w[:, perm]) + b[perm] + pad w[:, perm]``.
    """
    w_perm = w.index_select(1, perm)
    w_tilde = n_in_inv @ w_perm
    b_tilde = b.index_select(0, perm)
    if pad_in is None:
        x_tilde = x @ n_in
        return x_tilde @ w_tilde + b_tilde
    x_tilde = (x - pad_in) @ n_in
    c1 = pad_in @ w_perm
    return x_tilde @ w_tilde + b_tilde + c1


def run_relu_lifted_mlp_island(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    n_in: torch.Tensor, n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    k: int = 4,
    pad_in: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """ReLU MLP island with a homogeneous (positive) block lift."""
    device = x.device
    dtype = x.dtype
    h = w1.shape[1]
    g = _gen(seed, device)
    perm = _make_perm(h, g, device)

    # Plain reference.
    z = x @ w1 + b1
    a = relu_reference(z)
    y_plain = a @ w2 + b2

    # Masked path -> Z[:, perm].
    z_perm_tilde = _masked_linear_perm_output(
        x, w1, b1, n_in, n_in_inv, perm, pad_in,
    )
    z_plain_permuted = z.index_select(1, perm)

    # Lift + activation.
    R = _positive_relu_lift(h, k, dtype, device, g)
    S_relu = make_relu_squeeze(R)
    z_lift = block_lift(z_perm_tilde, R)
    a_lift = relu_reference(z_lift)

    # Folded second linear (squeeze is absorbed on the trusted side).
    w2_perm_rows = w2.index_select(0, perm)
    w2_tilde_lift = S_relu @ w2_perm_rows @ n_out
    b2_tilde = b2 @ n_out
    y_tilde = a_lift @ w2_tilde_lift + b2_tilde

    expected_y_tilde = y_plain @ n_out
    squeeze_check = a_lift @ S_relu
    expected_squeeze = relu_reference(z.index_select(1, perm))

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "z_perm_tilde": z_perm_tilde,
        "z_plain_permuted": z_plain_permuted,
        "a_lift": a_lift,
        "squeeze_check": squeeze_check,
        "expected_squeeze": expected_squeeze,
        "metadata": {
            "k": int(k),
            "lift_dim": int(h * k),
            "activation_type": "relu",
            "lift_mode": "homogeneous_positive",
            "used_pad": pad_in is not None,
            "online_extra_matmul_count_delta": 0,
            "selector_rows_zero_for_decoys": False,
        },
    }


# ---------------------------------------------------------------------------
# F. GELU / SiLU selector lifted MLP island
# ---------------------------------------------------------------------------


def run_selector_lifted_mlp_island(
    activation_type: Literal["gelu", "silu"],
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    n_in: torch.Tensor, n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    k: int = 4,
    pad_in: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """GELU / SiLU MLP island with a selector block lift."""
    if activation_type == "gelu":
        act = gelu_reference
    elif activation_type == "silu":
        act = silu_reference
    else:
        raise ValueError(
            f"activation_type must be 'gelu' or 'silu', got {activation_type!r}"
        )

    device = x.device
    dtype = x.dtype
    h = w1.shape[1]
    g = _gen(seed, device)
    perm = _make_perm(h, g, device)

    z = x @ w1 + b1
    a = act(z)
    y_plain = a @ w2 + b2

    z_perm_tilde = _masked_linear_perm_output(
        x, w1, b1, n_in, n_in_inv, perm, pad_in,
    )
    z_plain_permuted = z.index_select(1, perm)

    R, valid_idx, S_selector = make_selector_lift_params(
        h, k, dtype, device, g,
    )
    z_lift = block_lift(z_perm_tilde, R)
    a_lift = act(z_lift)

    w2_perm_rows = w2.index_select(0, perm)
    w2_tilde_lift = S_selector @ w2_perm_rows @ n_out
    b2_tilde = b2 @ n_out
    y_tilde = a_lift @ w2_tilde_lift + b2_tilde

    expected_y_tilde = y_plain @ n_out
    squeeze_check = a_lift @ S_selector
    expected_squeeze = act(z.index_select(1, perm))

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "z_perm_tilde": z_perm_tilde,
        "z_plain_permuted": z_plain_permuted,
        "a_lift": a_lift,
        "squeeze_check": squeeze_check,
        "expected_squeeze": expected_squeeze,
        "metadata": {
            "k": int(k),
            "lift_dim": int(h * k),
            "activation_type": activation_type,
            "lift_mode": "selector",
            "used_pad": pad_in is not None,
            "online_extra_matmul_count_delta": 0,
            "selector_rows_zero_for_decoys": True,
            "selector_leakage_warning": _SELECTOR_WARNING,
        },
    }


# ---------------------------------------------------------------------------
# G. SwiGLU selector lifted island
# ---------------------------------------------------------------------------


def run_swiglu_selector_lifted_mlp_island(
    x: torch.Tensor,
    w_up: torch.Tensor, b_up: torch.Tensor,
    w_gate: torch.Tensor, b_gate: torch.Tensor,
    w_down: torch.Tensor, b_down: torch.Tensor,
    n_in: torch.Tensor, n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    k: int = 4,
    pad_in: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """SwiGLU island with a shared selector lift on up and gate paths."""
    device = x.device
    dtype = x.dtype
    h = w_up.shape[1]
    g = _gen(seed, device)
    perm = _make_perm(h, g, device)

    # Plain reference.
    a = x @ w_up + b_up
    b = x @ w_gate + b_gate
    g_act = a * silu_reference(b)
    y_plain = g_act @ w_down + b_down

    # Masked path -> A[:, perm] and B[:, perm] (shared perm).
    a_perm_tilde = _masked_linear_perm_output(
        x, w_up, b_up, n_in, n_in_inv, perm, pad_in,
    )
    b_perm_tilde = _masked_linear_perm_output(
        x, w_gate, b_gate, n_in, n_in_inv, perm, pad_in,
    )

    # Shared valid_idx / selector; independent decoy scales for up vs gate.
    R_up, valid_idx, S_selector = make_selector_lift_params(
        h, k, dtype, device, g,
    )
    # Re-derive a gate lift with the SAME valid_idx but fresh decoys.
    if g is not None:
        gate_decoys = (
            torch.rand(h, k, generator=g, dtype=dtype, device=device)
            * 1.75 + 0.25
        )
    else:
        gate_decoys = torch.rand(h, k, dtype=dtype, device=device) * 1.75 + 0.25
    R_gate = gate_decoys.clone()
    rows = torch.arange(h, device=device)
    R_gate[rows, valid_idx] = 1.0

    a_lift = block_lift(a_perm_tilde, R_up)
    b_lift = block_lift(b_perm_tilde, R_gate)
    g_lift = a_lift * silu_reference(b_lift)

    squeeze_check = g_lift @ S_selector
    expected_squeeze = g_act.index_select(1, perm)

    w_down_perm_rows = w_down.index_select(0, perm)
    w_down_tilde_lift = S_selector @ w_down_perm_rows @ n_out
    b_down_tilde = b_down @ n_out
    y_tilde = g_lift @ w_down_tilde_lift + b_down_tilde

    expected_y_tilde = y_plain @ n_out

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "a_perm_tilde": a_perm_tilde,
        "b_perm_tilde": b_perm_tilde,
        "g_lift": g_lift,
        "squeeze_check": squeeze_check,
        "expected_squeeze": expected_squeeze,
        "metadata": {
            "k": int(k),
            "lift_dim": int(h * k),
            "activation_type": "swiglu",
            "lift_mode": "selector",
            "used_pad": pad_in is not None,
            "online_extra_matmul_count_delta": 0,
            "selector_rows_zero_for_decoys": True,
            "selector_leakage_warning": _SELECTOR_WARNING,
        },
    }


# ---------------------------------------------------------------------------
# H. LayerNorm gadget island
# ---------------------------------------------------------------------------


def make_layernorm_shift_gadget(
    d: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Row-wise shift gadget ``G = I + r 1^T`` ([d, d]).

    For a row vector ``x``: ``x G = x + (x . r) 1^T`` -- a per-row scalar
    shift, under which the LayerNorm core is exactly invariant (even with
    ``eps > 0``).
    """
    device = torch.device(device)
    if generator is not None:
        r = torch.randn(d, 1, generator=generator, dtype=dtype, device=device)
    else:
        r = torch.randn(d, 1, dtype=dtype, device=device)
    ones_row = torch.ones(1, d, dtype=dtype, device=device)
    return torch.eye(d, dtype=dtype, device=device) + r @ ones_row


def run_layernorm_gadget_island(
    x: torch.Tensor,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    n_out: torch.Tensor,
    eps: float = 1e-5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Amulet-style LayerNorm gadget island (shift-only, exact with eps)."""
    device = x.device
    dtype = x.dtype
    d = x.shape[1]
    g = _gen(seed, device)
    perm = _make_perm(d, g, device)

    # Plain reference.
    core = layernorm_core(x, eps)
    scaled = core * norm_weight + norm_bias
    y_plain = scaled @ linear_weight + linear_bias

    # Gadget: shift + permutation, pushed through the input mask.
    gadget = make_layernorm_shift_gadget(d, dtype, device, g)
    p_mat = torch.eye(d, dtype=dtype, device=device).index_select(1, perm)
    g_tilde = n_in_inv @ gadget @ p_mat
    x_tilde = x @ n_in
    x_gadget = x_tilde @ g_tilde  # == (x G) P
    core_tilde = layernorm_core(x_gadget, eps)  # == core[:, perm]
    expected_core_tilde = core.index_select(1, perm)

    # Fold affine into the following linear, then permute rows + mask out.
    w_folded = torch.diag(norm_weight) @ linear_weight
    b_folded = norm_bias @ linear_weight + linear_bias
    w_tilde = w_folded.index_select(0, perm) @ n_out
    b_tilde = b_folded @ n_out
    y_tilde = core_tilde @ w_tilde + b_tilde

    expected_y_tilde = y_plain @ n_out

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "expected_y_tilde": expected_y_tilde,
        "core_tilde": core_tilde,
        "expected_core_tilde": expected_core_tilde,
        "metadata": {
            "activation_type": "layernorm_gadget",
            "gadget_mode": "shift_only",
            "eps": float(eps),
            "online_extra_matmul_count_delta": 0,
            "scale_invariant_with_eps": True,
            "note": "Shift gadget G = I + r 1^T keeps LayerNorm core exact "
                    "with eps>0; scale lifts are intentionally not used "
                    "because LNCore(lambda x) != LNCore(x) for eps>0.",
        },
    }
