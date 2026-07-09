"""Variant E — two-sided non-orthogonal LINEAR islands with nonlinear-boundary
mask switching.

Idea: keep the residual / RMSNorm / RoPE / KV boundary on a compatible mask
``N_res`` (signed permutation / scaled-orthogonal), but *inside a linear
sub-chain* switch to Variant D's dense two-sided non-orthogonal key matrices
(stronger anchor-attack resistance), then switch back before any nonlinear or
residual boundary. Switching is itself a plain linear weight and is EXACT:

    into island :  Switch_in  = N_res^{-1} P_in     (X N_res) Switch_in = X P_in
    out of island: Switch_out = P_out^{-1} N_res    (Y P_out) Switch_out = Y N_res

The nonlinear boundary is crossed with an EXISTING compatible primitive, never
with a dense keymat:
  * GELU (GPT-2): Amulet right-mask island with right mask ``P_ff`` (dense allowed).
  * SiLU + Hadamard (Qwen SwiGLU): a pure PERMUTATION mask ``S`` — permutations
    commute with elementwise ops, ``(a S) ⊙ (b S) = (a ⊙ b) S`` and
    ``SiLU(G S) = SiLU(G) S``; a dense S would break the Hadamard.

Hard invariants (asserted in the report fields):
    dense_keymat_crosses_rmsnorm = False
    dense_keymat_crosses_rope    = False
    dense_keymat_crosses_hadamard = False

Scope: EXACT for the linear islands + these compatible nonlinear boundaries. This
is a LOCAL island, not a full-model production path: the residual/RMSNorm/RoPE/KV
boundaries still ride ``N_res`` (so this does NOT change stable-state norm leakage).
No Gaussian noise, no RMSNorm expectation correction.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.ops.amulet_right_mask_islands import (
    amulet_right_mask_activation,
    make_right_mask_amulet_params,
)
from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat


# ---------------------------------------------------------------------------
# compatible residual-boundary masks (N_res) and elementwise masks (S)
# ---------------------------------------------------------------------------


def signed_perm_matrix(d: int, *, seed: int, dtype=torch.float64, device="cpu"):
    """RMSNorm/RoPE/KV-compatible residual mask: signed permutation (orthogonal).
    Returns ``(N, N_inv)`` with ``N_inv = N^T``."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(d, generator=g)
    signs = torch.where(torch.rand(d, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    n = torch.zeros(d, d, dtype=torch.float64)
    n[torch.arange(d), perm] = signs
    n = n.to(dtype=dtype, device=device)
    return n, n.transpose(0, 1).contiguous()


def perm_matrix(d: int, *, seed: int, dtype=torch.float64, device="cpu"):
    """Elementwise/Hadamard-compatible mask: PURE permutation (no signs, so it
    commutes with SiLU/GELU and the Hadamard product). Returns ``(S, S_inv=S^T)``."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    perm = torch.randperm(d, generator=g)
    s = torch.zeros(d, d, dtype=torch.float64)
    s[torch.arange(d), perm] = 1.0
    s = s.to(dtype=dtype, device=device)
    return s, s.transpose(0, 1).contiguous()


def switch_weight(from_mask_inv: torch.Tensor, to_mask: torch.Tensor) -> torch.Tensor:
    """Plain linear mask-switch weight: ``(X A) (A^{-1} B) = X B``."""
    return from_mask_inv @ to_mask


# ---------------------------------------------------------------------------
# GPT-2 GELU MLP island (Amulet-GELU at the nonlinear boundary)
# ---------------------------------------------------------------------------


def two_sided_gelu_mlp_island(
    x_res: torch.Tensor,
    w_up: torch.Tensor, b_up: torch.Tensor | None,
    w_down: torch.Tensor, b_down: torch.Tensor | None,
    *,
    n_res: torch.Tensor, n_res_inv: torch.Tensor,
    p_in: torch.Tensor, q_in: torch.Tensor,
    p_ff: torch.Tensor,
    p_out: torch.Tensor, p_out_inv: torch.Tensor,
    k: int = 3, seed: int = 0,
) -> dict[str, Any]:
    """Run ``GELU-MLP(X) N_res`` given the residual-masked input ``x_res = X N_res``.

    The GPU sees only: x_res, the switch weights, the folded weights, and the
    Amulet island params. Returns y_res (= Y N_res) and the exactness pieces.
    """
    m = x_res.shape[0]
    dff = w_up.shape[1]
    # 1. switch into the dense keymat space: X N_res -> X P_in
    x_dense = x_res @ switch_weight(n_res_inv, p_in)                 # = X P_in
    # 2. two-sided up projection: U P_ff
    w_up_t = q_in @ w_up @ p_ff
    u_dense = x_dense @ w_up_t
    if b_up is not None:
        u_dense = u_dense + b_up @ p_ff                             # = U P_ff
    # 3. Amulet-GELU with dense right mask P_ff: U P_ff -> GELU(U) P_ff.
    #    The Amulet R-factor factorisation is a precise algebraic construction that
    #    requires fp64 (its 1e-9 product tolerance is a pre-existing Amulet property,
    #    unmet in fp32/bf16), so the nonlinear boundary runs in fp64 and is cast back.
    in_dtype = u_dense.dtype
    params = make_right_mask_amulet_params(m, dff, k, p_ff.to(torch.float64),
                                           generator=torch.Generator().manual_seed(seed))
    v_dense = amulet_right_mask_activation(u_dense.to(torch.float64), params, "gelu").to(in_dtype)
    # 4. two-sided down projection: Y P_out
    p_ff_inv = torch.linalg.inv(p_ff)
    w_down_t = p_ff_inv @ w_down @ p_out
    y_dense = v_dense @ w_down_t
    if b_down is not None:
        y_dense = y_dense + b_down @ p_out                          # = Y P_out
    # 5. switch back to the residual mask: Y P_out -> Y N_res
    y_res = y_dense @ switch_weight(p_out_inv, n_res)               # = Y N_res

    # plaintext reference
    u = x_res @ n_res_inv @ w_up
    if b_up is not None:
        u = u + b_up
    y = torch.nn.functional.gelu(u) @ w_down
    if b_down is not None:
        y = y + b_down
    y_ref = y @ n_res
    return {
        "y_res": y_res, "y_ref": y_ref,
        "max_abs_error": float((y_res - y_ref).abs().max()),
        "u_dense": u_dense, "v_dense": v_dense, "y_dense": y_dense,
    }


# ---------------------------------------------------------------------------
# Qwen SwiGLU FFN island (permutation S at the Hadamard/SiLU boundary)
# ---------------------------------------------------------------------------


def two_sided_swiglu_island(
    x_res: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    *,
    n_res: torch.Tensor, n_res_inv: torch.Tensor,
    p_in: torch.Tensor, q_in: torch.Tensor,
    p_g: torch.Tensor, p_u: torch.Tensor,
    s_perm: torch.Tensor, s_perm_inv: torch.Tensor,
    p_mid: torch.Tensor, p_out: torch.Tensor, p_out_inv: torch.Tensor,
) -> dict[str, Any]:
    """Run ``SwiGLU-FFN(X) N_res``. Gate/up/down weights carry dense two-sided
    keymats; the SiLU+Hadamard boundary rides a pure permutation ``S``."""
    # switch in: X N_res -> X P_in
    x_dense = x_res @ switch_weight(n_res_inv, p_in)                 # X P_in
    # gate & up with dense two-sided keymat (output masks P_g, P_u)
    g_dense = x_dense @ (q_in @ w_gate @ p_g)                        # G P_g
    u_dense = x_dense @ (q_in @ w_up @ p_u)                          # U P_u
    # switch both branches to the elementwise-compatible permutation S
    g_elem = g_dense @ switch_weight(torch.linalg.inv(p_g), s_perm)  # G S
    u_elem = u_dense @ switch_weight(torch.linalg.inv(p_u), s_perm)  # U S
    # SiLU + Hadamard under permutation S (commutes exactly)
    v_elem = torch.nn.functional.silu(g_elem) * u_elem              # (SiLU(G)⊙U) S = V S
    # switch back to a dense keymat for the down projection
    v_dense = v_elem @ switch_weight(s_perm_inv, p_mid)             # V P_mid
    y_dense = v_dense @ (torch.linalg.inv(p_mid) @ w_down @ p_out)  # Y P_out
    y_res = y_dense @ switch_weight(p_out_inv, n_res)               # Y N_res

    # plaintext reference
    x = x_res @ n_res_inv
    g = x @ w_gate; u = x @ w_up
    y = (torch.nn.functional.silu(g) * u) @ w_down
    y_ref = y @ n_res
    return {
        "y_res": y_res, "y_ref": y_ref,
        "max_abs_error": float((y_res - y_ref).abs().max()),
        "g_elem": g_elem, "u_elem": u_elem, "v_elem": v_elem,
    }


# ---------------------------------------------------------------------------
# config / audit fields
# ---------------------------------------------------------------------------


def two_sided_island_report_fields(
    *, nonlinear_mode: str = "amulet_or_structured",
    max_abs_error: float | None = None,
    exact_lossless_tests_passed: bool = False,
    qwen_swiglu_verified: bool = False,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "variant": "two_sided_amulet_island",
        "aliases": ["exact_two_sided_nonlinear_switch", "variant_e"],
        "uses_two_sided_keymat": True,
        "uses_nonorthogonal_transform": True,
        "uses_noise": False,
        "uses_rmsnorm_expectation_correction": False,
        "uses_mask_switching_at_nonlinear_boundary": True,
        "nonlinear_mode": nonlinear_mode,        # amulet (GELU) | structured perm (SwiGLU)
        "residual_boundary_mask": "N_res_signed_permutation",
        "elementwise_boundary_mask": "pure_permutation",
        # hard invariants: dense keymat NEVER crosses these boundaries
        "dense_keymat_crosses_rmsnorm": False,
        "dense_keymat_crosses_rope": False,
        "dense_keymat_crosses_hadamard": False,
        "exact_scope": "linear_islands_plus_compatible_nonlinear_boundary",
        "exact_lossless_claim": bool(exact_lossless_tests_passed),
        "qwen_swiglu_island_verified": bool(qwen_swiglu_verified),
        "formal_security_claim": False,
        "is_aloepri_replication": False,
        "changes_stable_state_norm_leakage": False,   # residual still rides N_res
        "experiment_only": True,
        "production_qwen7b_integration": False,
    }
    if max_abs_error is not None:
        fields["max_abs_error"] = float(max_abs_error)
    return fields


__all__ = [
    "signed_perm_matrix", "perm_matrix", "switch_weight",
    "two_sided_gelu_mlp_island", "two_sided_swiglu_island",
    "two_sided_island_report_fields",
]
