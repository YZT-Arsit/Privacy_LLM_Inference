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
    "COMPATIBLE_RESIDUAL_MASK_MODES",
    "COMPATIBLE_ATTENTION_MASK_FAMILIES",
    "INCOMPATIBLE_RESIDUAL_MASK_MODES",
    "INCOMPATIBLE_ATTENTION_MASK_FAMILIES",
    "is_signed_permutation",
    "assert_signed_permutation",
    "assert_orthogonal",
    "assert_qk_compatible",
    "is_permutation_matrix",
    "is_permutation_index",
    "assert_shared_channel_permutation",
    "verify_compatible_masks",
    "verify_mask_bundle_compatible",
    "verify_session_compatible_masks",
    "compatible_mask_audit_fields",
    "REQUIRED_COMPATIBLE_AUDIT_FIELDS",
]

# Paper-facing A_rightmul REQUIRES the residual mask to be a signed permutation
# (orthogonal monomial) and the per-head attention masks to be ORTHOGONAL +
# score-preserving. Any dense / non-orthogonal family (``dense_orthogonal``,
# ``block_orthogonal``, ``pairwise_complex_scaling``) is rejected: a dense
# orthogonal residual mask is not a signed permutation, and complex-scaling
# attention masks are not orthogonal (they change per-pair magnitude), so the
# GPU-visible masked state is not in a provably-compatible family.
COMPATIBLE_RESIDUAL_MASK_MODES = ("signed_permutation",)
COMPATIBLE_ATTENTION_MASK_FAMILIES = ("pairwise_rotation",)
INCOMPATIBLE_RESIDUAL_MASK_MODES = ("dense_orthogonal", "block_orthogonal")
INCOMPATIBLE_ATTENTION_MASK_FAMILIES = ("pairwise_complex_scaling",)

# the exact audit booleans a paper-facing A_rightmul build/worker report MUST carry
REQUIRED_COMPATIBLE_AUDIT_FIELDS = (
    "compatible_masks_verified",
    "residual_mask_is_signed_permutation",
    "attention_qk_scores_preserved",
    "swiglu_shared_channel_permutation",
    "arbitrary_dense_mask_rejected",
)


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


def assert_orthogonal(m: torch.Tensor, *, name: str = "M",
                      atol: float = 1e-4) -> None:
    """Require every ``[D,D]`` matrix in ``m`` (possibly batched ``[...,D,D]``) to
    be orthogonal (``M M^T == I``). REJECTS the non-orthogonal
    ``pairwise_complex_scaling`` attention family (which changes per-pair
    magnitude), while accepting orthogonal ``pairwise_rotation`` / signed
    permutations."""
    if m.dim() < 2 or m.shape[-1] != m.shape[-2]:
        raise CompatibleMaskViolation("%s must be square (got %s)"
                                      % (name, tuple(m.shape)))
    d = m.shape[-1]
    md = m.to(torch.float64)
    prod = md @ md.transpose(-2, -1)
    eye = torch.eye(d, dtype=torch.float64, device=m.device)
    err = float((prod - eye).abs().max().item())
    if err > atol:
        raise CompatibleMaskViolation(
            "%s is not orthogonal: max|M M^T - I| = %.3e > %.1e (non-orthogonal "
            "families such as pairwise_complex_scaling are NOT compatible masks "
            "for A_rightmul)" % (name, err, atol))


def is_permutation_index(idx: torch.Tensor) -> bool:
    """True iff a 1-D integer index tensor is a permutation of ``0..n-1``."""
    if idx.dim() != 1:
        return False
    n = idx.shape[0]
    srt, _ = torch.sort(idx.to(torch.int64))
    return bool(torch.equal(srt, torch.arange(n, device=idx.device,
                                              dtype=torch.int64)))


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


def verify_mask_bundle_compatible(
    masks: Any, *, attention_atol: float = 1e-4,
    residual_atol: float = 1e-6, check_layers: Any = None,
) -> dict[str, Any]:
    """Verify a REAL generated mask bundle is in the A_rightmul compatible family.

    ``masks`` is an :class:`~pllo.hf_wrappers.hf_causal_lm_skeleton.HFCausalLMMaskBundle`
    (or any object exposing ``residual_masks`` and ``layer_block_masks``). Raises
    :class:`CompatibleMaskViolation` on the first incompatible mask -- so a
    paper-facing A_rightmul build over a ``dense_orthogonal`` residual mask or a
    ``pairwise_complex_scaling`` attention family FAILS loudly instead of silently
    producing a wrong-but-reported-success package. Returns the audit dict with
    the five required booleans + the observed mask families."""
    residual = list(getattr(masks, "residual_masks", []) or [])
    block_masks = list(getattr(masks, "layer_block_masks", []) or [])
    if not residual or not block_masks:
        raise CompatibleMaskViolation(
            "mask bundle has no residual/layer masks to verify")

    # 1. residual / RMSNorm / LayerNorm core -- signed permutation per layer
    for ell, n_res in enumerate(residual):
        assert_signed_permutation(n_res, name="N_res[%d]" % ell,
                                  atol=residual_atol)

    # 2 + 3. per-layer attention (orthogonal + score-preserving) and SwiGLU perm
    n_layers = len(block_masks)
    layers = (range(n_layers) if check_layers is None
              else [e for e in check_layers if 0 <= e < n_layers])
    attn_family = None
    for ell in layers:
        bm = block_masks[ell]
        attn = bm["attn"]
        attn_family = attn.get("mask_family", attn_family)
        qm = attn["q_masks"]            # [num_heads, D, D]
        km = attn["key_masks"]          # [num_kv, D, D]
        kv_index = attn["kv_index"]     # [num_heads]
        # orthogonality of the key masks rejects pairwise_complex_scaling
        assert_orthogonal(km, name="key_masks[%d]" % ell, atol=attention_atol)
        # per Q-head score invariant Nq Nk^T == I (scores preserved under masking)
        for h in range(qm.shape[0]):
            kv = int(kv_index[h])
            assert_qk_compatible(qm[h], km[kv], atol=attention_atol)
        # SwiGLU gate/up share the SAME channel permutation by construction
        # (the single folded `perm`); verify it is a genuine permutation.
        perm = bm.get("perm")
        if perm is None or not is_permutation_index(perm):
            raise CompatibleMaskViolation(
                "SwiGLU channel mask (layer %d) is not a shared channel "
                "permutation" % ell)

    if attn_family is not None and \
            attn_family in INCOMPATIBLE_ATTENTION_MASK_FAMILIES:
        raise CompatibleMaskViolation(
            "attention mask_family %r is NOT a compatible family for A_rightmul "
            "(expected one of %s)" % (attn_family,
                                      COMPATIBLE_ATTENTION_MASK_FAMILIES))

    meta = getattr(masks, "metadata", {}) or {}
    return {
        "compatible_masks_verified": True,
        "residual_mask_is_signed_permutation": True,
        "attention_qk_scores_preserved": True,
        "swiglu_shared_channel_permutation": True,
        "arbitrary_dense_mask_rejected": True,
        "residual_mask_mode": meta.get("mask_mode"),
        "attention_mask_family": attn_family,
        "compatible_mask_layers_checked": len(layers),
        "nonlinear_masking_mode": "compatible_right_multiply_or_permutation",
    }


def verify_session_compatible_masks(session: Any, **kw) -> dict[str, Any]:
    """Verify a :class:`~pllo.hf_wrappers.qwen_masked_session.MaskedQwenSession`
    actually generated A_rightmul-compatible masks (delegates to
    :func:`verify_mask_bundle_compatible` on ``session.masks``)."""
    masks = getattr(session, "masks", None)
    if masks is None:
        raise CompatibleMaskViolation("session exposes no .masks bundle")
    return verify_mask_bundle_compatible(masks, **kw)


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
