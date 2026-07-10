"""Generation-aware re-audit of the Amulet nonlinear island.

Corrects a conflation in the earlier selector-visibility audit: it mixed the
*deployed* ``amulet_migrated`` implementation with the *paper-faithful* island.
This module separates three variants and classifies every finding into one of:

    implementation_bug | non_generation_compatible_left_mask
    selector_exposure   | scheme_level_leakage

Variants
--------
* **A. amulet_migrated** (deployed, ``nonlinear/amulet_backend.py``): a simplified
  per-feature lift ``U ⊗ R`` with ``R[i, valid]=1`` and a GPU-side ``gather(valid)``.
* **B. paper-faithful island** (``ops/amulet_right_mask_islands.py``): dense
  single-one ``R_bar`` (exactly one entry 1, the rest dense random decoys), secret
  permutations, ``M3 = P·π1ᵀ·E1·π3ᵀ`` — i.e. a LEFT operator over the row/token
  dimension. Faithful, but the left/token mask is incompatible with causal
  decoding + KV cache.
* **C. generation-aware right-mask-only island** (implemented here): ``P = I`` (no
  operator over the sequence dimension); only right hidden-dimension masks. A
  column-only lift/squeeze keeps the token axis untouched and preserves the
  right-mask invariant ``Y_tilde = phi(X)·N_out``.

Key correctness point (all variants): to return the true ``phi(X)`` the GPU must
apply ``phi`` to the true pre-activation values, so those values are present in
the GPU-visible lift by construction. That is an inherent property of any
nonlinear *offload* (scheme_level), NOT an implementation bug — see the
masked-Hadamard / CP-uniqueness analysis.

float64. No production path modified.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

_F64 = torch.float64

__all__ = [
    "right_only_activation_island", "audit_variant_a", "audit_variant_b",
    "audit_variant_c", "run_all",
]


# ===========================================================================
# Variant C: generation-aware, right-mask-only column island
# ===========================================================================


def right_only_activation_island(
    x: torch.Tensor, N_in: torch.Tensor, N_out: torch.Tensor,
    act: Callable[[torch.Tensor], torch.Tensor], *, k: int = 3, seed: int = 0,
) -> dict[str, Any]:
    """Right-mask-only nonlinear island: ``x·N_in -> phi(x)·N_out``.

    NO operator touches the row/token dimension ``m`` (generation-safe). The TEE
    holds ``N_in, N_out`` and a column-lift ``L``/squeeze ``S`` on the hidden dim;
    the GPU only computes ``act(z)`` on the lifted tensor ``z`` ([m, d·k]).
    """
    m, d = x.shape
    g = torch.Generator().manual_seed(seed)
    x_tilde = x @ N_in                                     # masked input (right)
    Ninv = torch.linalg.inv(N_in)
    b = int(torch.randint(0, k, (1,), generator=g))        # secret valid offset
    perm = torch.randperm(d * k, generator=g)              # secret col shuffle
    inv = torch.empty_like(perm); inv[perm] = torch.arange(d * k)

    # column-lift L [d, d·k]: real col of group j -> e_j (recovers x[:,j]); decoys dense
    L = torch.zeros(d, d * k, dtype=_F64)
    for j in range(d):
        for q in range(k):
            col = j * k + q
            if q == b:
                L[:, col] = Ninv[:, j]                     # z[:,col] = x[:,j]
            else:
                decoy = torch.randn(d, generator=g, dtype=_F64)
                L[:, col] = Ninv @ decoy                   # z[:,col] = x @ decoy (dense)
    L = L[:, perm]                                         # hide which columns are real
    z = x_tilde @ L                                        # [m, d·k]  GPU-VISIBLE
    s = act(z)                                             # GPU elementwise activation
    S = torch.zeros(d * k, d, dtype=_F64)                  # squeeze + right output mask
    for j in range(d):
        S[inv[j * k + b], :] = N_out[j, :]
    y = s @ S                                              # = phi(x) @ N_out

    # what the GPU can see / recover
    real_cols = torch.tensor([int(inv[j * k + b]) for j in range(d)])
    xvals = x.reshape(-1)
    zvals = z.reshape(-1)
    preact_values_in_z = int(sum(int(torch.isclose(zvals, v, atol=1e-9).any())
                                 for v in xvals))
    # equality-scan leak? is any GPU-visible per-column scale exactly 1?
    col_has_scale_one = bool(((z.abs() - 1.0).abs() < 1e-12).any())
    return {
        "output": y,
        "right_mask_invariant_ok": bool(torch.allclose(y, act(x) @ N_out, atol=1e-9)),
        "z_shape": list(z.shape),
        "touches_token_dim": False,                        # L,S are [d,dk],[dk,d]
        "kv_cache_causal_valid": True,
        "selector_index_gpu_visible": False,               # b, perm folded into L/S (TEE)
        "equality_scan_R_eq_1_leak": col_has_scale_one,    # no per-col scale=1
        "raw_preactivation_values_in_z": f"{preact_values_in_z}/{xvals.numel()}",
        "tee_computes": "z = x_tilde @ L  and  y = act(z) @ S  (2 linear ops on the "
                        "hidden dim, O(m·d·d·k)); GPU computes act(z).",
        "_real_cols": real_cols,
    }


# ===========================================================================
# Variant A: deployed amulet_migrated
# ===========================================================================


def audit_variant_a(*, h: int = 6, k: int = 3, seed: int = 0) -> dict[str, Any]:
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    be = AmuletMigratedNonlinearBackend(lift_k=k, seed=seed)
    valid, R = be._lift_factors(h, torch.device("cpu"), _F64)
    scan = (R.sub(1.0).abs() < 1e-12).float().argmax(dim=1)
    valid_by_scan = bool(torch.equal(scan, valid))
    x = torch.randn(4, h, dtype=_F64)
    res = be.silu(x)
    return {
        "variant": "A_amulet_migrated_deployed",
        "raw_preactivation_visible": True,   # lift built from unmasked U; valid col == U
        "selector_index_gpu_visible": True,  # gather(2, valid) runs on accelerator
        "valid_recoverable_from_R_scan": valid_by_scan,   # R[i,valid]==1
        "touches_token_dim": False,          # per-feature lift; rows independent
        "kv_cache_causal_valid": True,
        "right_mask_invariant_Y_eq_phiX_Nout": False,     # no mask at all: out == silu(x)
        "output_is_plain_activation": bool(torch.allclose(res.output, F.silu(x), atol=1e-9)),
        "classification": ["implementation_bug (R==1 equality-scan recovers valid)",
                           "selector_exposure (valid used on-device in gather)",
                           "scheme_level_leakage (no activation mask; raw x visible)"],
    }


# ===========================================================================
# Variant B: paper-faithful island
# ===========================================================================


def audit_variant_b(*, m: int = 4, d: int = 6, k: int = 3, seed: int = 1) -> dict[str, Any]:
    from pllo.ops.amulet_right_mask_islands import (
        sample_dense_single_one_rbar, make_right_mask_amulet_params,
        amulet_right_mask_activation)
    g = torch.Generator().manual_seed(seed)
    rbar, a, b = sample_dense_single_one_rbar(k, dtype=_F64, device="cpu", generator=g)
    decoys = rbar[rbar.sub(1).abs() > 1e-9]
    n = torch.randn(d, d, generator=g, dtype=_F64); n = n @ n.T + d * torch.eye(d, dtype=_F64)
    p = make_right_mask_amulet_params(m, d, k, n, torch.linalg.inv(n), seed=seed + 1)
    U = torch.randn(m, d, generator=g, dtype=_F64)
    V = amulet_right_mask_activation(U @ n, p, "silu")
    return {
        "variant": "B_paper_faithful_island",
        "rbar_exactly_one_unit": int((rbar.sub(1).abs() < 1e-9).sum()) == 1,
        "rbar_zero_decoys": bool((rbar.abs() < 1e-6).any()),
        "rbar_decoys_dense_random_range": [float(decoys.min()), float(decoys.max())],
        "equality_scan_R_eq_1_leak": False,   # single '1' is inside dense R_bar, never a
                                              # GPU-visible per-column scale (lift uses r2 dense)
        "touches_token_dim": not bool(torch.allclose(p.pi1, torch.eye(m, dtype=_F64))),
        "left_operator_over_sequence": "M3 = P·pi1^T·E1·pi3^T  (pi1 is [m,m] over tokens)",
        "kv_cache_causal_valid": False,       # per-length param regen + token-dim ops
        "right_mask_invariant_Y_eq_phiX_Nout": bool(torch.allclose(V, F.silu(U) @ n, atol=1e-9)),
        "classification": ["non_generation_compatible_left_mask (pi1/pi3/E1 over token dim)",
                           "NOT implementation_bug: no R==1 equality-scan (dense single-one R_bar)",
                           "scheme_level_leakage: pre-activation values present in the lift "
                           "(inherent to nonlinear offload), selector hiding rests on the "
                           "unproven secure-R assumption"],
    }


# ===========================================================================
# Variant C: generation-aware right-mask-only
# ===========================================================================


def audit_variant_c(*, m: int = 4, d: int = 6, k: int = 3, seed: int = 2) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, d, generator=g, dtype=_F64)
    N_in = torch.randn(d, d, generator=g, dtype=_F64); N_in = N_in @ N_in.T + d * torch.eye(d, dtype=_F64)
    N_out = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=_F64))[0]
    out = {}
    for name, act in (("silu", F.silu), ("gelu", F.gelu)):
        r = right_only_activation_island(x, N_in, N_out, act, k=k, seed=seed + 3)
        out[name] = {kk: vv for kk, vv in r.items() if not kk.startswith(("output", "_"))}
    return {
        "variant": "C_generation_aware_right_mask_only",
        "silu": out["silu"], "gelu": out["gelu"],
        "classification": ["generation_compatible (no token-dim operator; KV/causal safe)",
                           "right_mask_invariant holds: Y_tilde = phi(X)·N_out",
                           "no implementation_bug: no R==1 equality-scan, selector b/perm "
                           "folded into TEE-side L/S",
                           "scheme_level_leakage: pre-activation VALUES present in the "
                           "GPU-visible lift z (inherent to any nonlinear offload; CP theory). "
                           "Selector hiding rests on the secure-R-style assumption."],
    }


def run_all(*, seed: int = 0) -> dict[str, Any]:
    return {
        "stage": "amulet_generation_aware_audit",
        "variant_A": audit_variant_a(seed=seed),
        "variant_B": audit_variant_b(seed=seed + 1),
        "variant_C": audit_variant_c(seed=seed + 2),
        "dichotomy": "hidden selector => TEE computes the nonlinear (no offload); "
                     "offloaded nonlinear => selector and/or pre-activation exposed.",
    }
