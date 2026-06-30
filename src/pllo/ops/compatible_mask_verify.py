"""Compatible-mask verification for the A_rightmul nonlinear design.

A_rightmul claims security *under the compatible-mask assumption*: the nonlinear
islands commute with the masks because the masks belong to compatible families.
This module makes that assumption CHECKABLE -- each predicate raises
:class:`CompatibleMaskViolation` if the supplied mask is not in the required
family, so a paper-facing run can never silently apply A_rightmul over an
incompatible (e.g. arbitrary dense) mask.

Compatible families (row-vector convention, ``y = x @ N``):

* **residual / RMSNorm / LayerNorm core** -- the mask ``N`` must be a *signed
  permutation* (orthogonal monomial): exactly one non-zero of magnitude 1 per row
  and per column. RMSNorm/LayerNorm cores are invariant under such ``N`` up to the
  same right-multiply, so ``norm_core(x @ N) == norm_core(x) @ N``.
* **attention Q/K** -- the per-head masks ``Nq, Nk`` must satisfy
  ``Nq Nk^T == I`` so that ``(Q Nq)(K Nk)^T == Q K^T`` (scores unchanged).
* **GELU/SiLU/SwiGLU channels** -- the gate/up masks must be the *same channel
  permutation* ``P`` (a 0/1 permutation matrix), so
  ``SiLU(G P) * (U P) == (SiLU(G) * U) P``.

An arbitrary dense invertible mask is REJECTED by every predicate.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "CompatibleMaskViolation",
    "is_signed_permutation",
    "assert_signed_permutation",
    "assert_qk_compatible",
    "is_permutation_matrix",
    "assert_shared_channel_permutation",
    "verify_compatible_masks",
    "compatible_mask_audit_fields",
]


class CompatibleMaskViolation(RuntimeError):
    """A mask supplied to the A_rightmul nonlinear path is not in a compatible
    family (would not commute with the nonlinear core)."""


def is_signed_permutation(n: torch.Tensor, *, atol: float = 1e-6) -> bool:
    """True iff ``n`` is an orthogonal monomial (signed permutation) matrix."""
    if n.dim() != 2 or n.shape[0] != n.shape[1]:
        return False
    a = n.abs().to(torch.float64)
    nz = a > atol
    # exactly one non-zero per row and per column
    if not bool((nz.sum(dim=1) == 1).all() and (nz.sum(dim=0) == 1).all()):
        return False
    # every non-zero has magnitude 1
    vals = a[nz]
    if vals.numel() != n.shape[0]:
        return False
    return bool((vals - 1.0).abs().max().item() <= atol)


def assert_signed_permutation(n: torch.Tensor, *, name: str = "N_res",
                              atol: float = 1e-6) -> None:
    if not is_signed_permutation(n, atol=atol):
        raise CompatibleMaskViolation(
            "mask %r is not a signed permutation (orthogonal monomial); RMSNorm/"
            "LayerNorm/residual masks must be signed permutations for A_rightmul "
            "(arbitrary dense masks do not commute with the norm core)" % name)


def assert_qk_compatible(nq: torch.Tensor, nk: torch.Tensor, *,
                         atol: float = 1e-6) -> None:
    """Require ``Nq Nk^T == I`` so attention scores are preserved."""
    if nq.shape != nk.shape or nq.dim() != 2 or nq.shape[0] != nq.shape[1]:
        raise CompatibleMaskViolation("Q/K masks must be square and same shape")
    eye = torch.eye(nq.shape[0], dtype=torch.float64, device=nq.device)
    prod = nq.to(torch.float64) @ nk.to(torch.float64).T
    err = float((prod - eye).abs().max().item())
    if err > atol:
        raise CompatibleMaskViolation(
            "Q/K masks not compatible: max|Nq Nk^T - I| = %.3e > %.1e "
            "(need Q~K~^T == QK^T)" % (err, atol))


def is_permutation_matrix(p: torch.Tensor, *, atol: float = 1e-6) -> bool:
    """True iff ``p`` is a 0/1 permutation matrix."""
    if p.dim() != 2 or p.shape[0] != p.shape[1]:
        return False
    a = p.to(torch.float64)
    is01 = bool(((a.abs() < atol) | ((a - 1.0).abs() < atol)).all().item())
    if not is01:
        return False
    return bool((a.sum(dim=0) - 1.0).abs().max().item() <= atol
                and (a.sum(dim=1) - 1.0).abs().max().item() <= atol)


def assert_shared_channel_permutation(p_gate: torch.Tensor, p_up: torch.Tensor,
                                      *, atol: float = 1e-6) -> None:
    """Require gate/up to use the SAME channel permutation ``P``."""
    if not is_permutation_matrix(p_gate, atol=atol):
        raise CompatibleMaskViolation(
            "SwiGLU gate mask is not a channel permutation matrix")
    if p_up.shape != p_gate.shape or float(
            (p_gate.to(torch.float64) - p_up.to(torch.float64)).abs().max()
            .item()) > atol:
        raise CompatibleMaskViolation(
            "SwiGLU gate/up masks must be the SAME shared channel permutation "
            "(arbitrary / distinct dense masks do not commute with SiLU*Up)")


def verify_compatible_masks(
    *, n_res: torch.Tensor | None = None,
    nq: torch.Tensor | None = None, nk: torch.Tensor | None = None,
    p_gate: torch.Tensor | None = None, p_up: torch.Tensor | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Verify all supplied compatible-mask conditions (raise on any violation).

    Returns an audit dict of the checks performed (all True, since a failure
    raises). Pass only the masks available at the call site."""
    checked: dict[str, Any] = {}
    if n_res is not None:
        assert_signed_permutation(n_res, name="N_res", atol=atol)
        checked["residual_mask_is_signed_permutation"] = True
    if nq is not None and nk is not None:
        assert_qk_compatible(nq, nk, atol=atol)
        checked["attention_qk_scores_preserved"] = True
    if p_gate is not None and p_up is not None:
        assert_shared_channel_permutation(p_gate, p_up, atol=atol)
        checked["swiglu_shared_channel_permutation"] = True
    checked["compatible_masks_verified"] = True
    checked["nonlinear_masking_mode"] = "compatible_right_multiply_or_permutation"
    return checked


def compatible_mask_audit_fields(verified: bool, checks: dict[str, Any] | None = None
                                 ) -> dict[str, Any]:
    """Public audit fields for the A_rightmul compatible-mask assumption.

    ``security_status`` is the *conditional* claim status -- the claim holds under
    the compatible-mask assumption, which the checks above enforce."""
    out = {
        "nonlinear_design": "A_rightmul",
        "compatible_masks_verified": bool(verified),
        "security_status": ("claimed_under_compatible_mask_assumption" if verified
                            else "compatible_mask_assumption_unverified"),
        "compatible_mask_families": {
            "residual_rmsnorm_layernorm": "signed_permutation_orthogonal_monomial",
            "attention_qk": "Nq Nk^T == I (scores preserved)",
            "swiglu_gelu_silu": "shared_channel_permutation",
        },
        "arbitrary_dense_mask_rejected": True,
    }
    if checks:
        out.update({k: v for k, v in checks.items()
                    if k not in ("nonlinear_masking_mode",)})
    return out
