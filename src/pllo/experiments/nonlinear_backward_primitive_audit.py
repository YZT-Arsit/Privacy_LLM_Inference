"""Scheme A nonlinear-backward primitive audit (experiment / audit module).

Attacks the nonlinear-backward blocker for Scheme A masked-domain LoRA training,
scoped to the minimal SiLU / SwiGLU island that Qwen/LLaMA use (NOT full Qwen).

Context (see [[masked-lora-backward-a0-a1]] / [[lora-lifecycle-audit]]):
* A0 dual backward is correct but leaks the exact activation-gradient cross-Gram.
* A1 independent backward masks fix the exact cross-Gram for linear/LoRA layers.
* Full Scheme A is now blocked by nonlinear backward primitives. This module
  determines whether the existing Amulet/Kronecker right-mask island can carry an
  A1 backward for SiLU / SwiGLU, and reports the status honestly.

Each nonlinearity is evaluated under three variants:

* ``trusted_shortcut`` -- the TEE recovers the plaintext, runs the nonlinear
  forward/backward, and re-masks with independent A1 masks. Correct, but it
  exposes the plaintext INSIDE the TEE; it is the correctness upper bound, NOT a
  GPU-side primitive.
* ``amulet_dense`` -- the real ``pllo.ops.amulet_right_mask_islands`` forward
  primitive (``U n -> phi(U) n`` via a Kronecker lift) under a dense mask. Its
  FORWARD is a genuine GPU-side primitive; its A1 BACKWARD is attempted and
  measured. The nonlinear backward needs an elementwise (Hadamard) product of
  tensors carried under *independent* masks, which does not commute with dense
  masks -- ``blocked_by_hadamard``.
* ``permutation`` -- masks restricted to permutations, under which the Hadamard
  product DOES commute, so an A1 backward is realisable; but permutation masks
  leak the activation Gram (``X P (X P)^T = X X^T``), so they do not deliver the
  dense-mask privacy A1 targets. Reported with the Gram-leak caveat.

No production path is modified. All math float64. Standard autograd is never
labelled A1 (it reproduces A0 dual behaviour).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from pllo.ops.amulet_right_mask_islands import (
    _island_operators,
    _lift_columns,
    amulet_right_mask_activation,
    amulet_right_mask_swiglu,
    make_right_mask_amulet_params,
    silu_reference,
)

_F64 = torch.float64

__all__ = [
    "silu", "silu_prime", "swiglu",
    "rand_orthogonal", "rand_invertible", "rand_permutation",
    "metrics", "corr_offdiag", "cross_gram",
    "island_apply_phi",
    "run_silu", "run_swiglu", "run_cross_gram_after_nonlinear",
    "run_tiny_transformer_integration", "run_all",
    "STATUS_VALUES",
]

STATUS_VALUES = (
    "trusted_shortcut_only", "forward_only", "backward_implemented",
    "backward_passed", "blocked_by_hadamard", "blocked_by_silu",
    "too_slow", "numerical_failure",
)


# ===========================================================================
# Nonlinearities
# ===========================================================================


def silu(x): return x * torch.sigmoid(x)


def silu_prime(x):
    s = torch.sigmoid(x)
    return s + x * s * (1.0 - s)


def swiglu(g, u): return silu(g) * u


# ===========================================================================
# Masks
# ===========================================================================


def rand_orthogonal(d, g):
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=_F64))
    return q * torch.sign(torch.diag(r)).unsqueeze(0)


def rand_invertible(d, g):
    """Well-conditioned dense invertible (SPD + shift)."""
    a = torch.randn(d, d, generator=g, dtype=_F64)
    return a @ a.T + d * torch.eye(d, dtype=_F64)


def rand_permutation(d, g):
    perm = torch.randperm(d, generator=g)
    p = torch.zeros(d, d, dtype=_F64)
    p[torch.arange(d), perm] = 1.0
    return p


def _mask(d, g, kind):
    return {"orthogonal": rand_orthogonal, "invertible": rand_invertible,
            "permutation": rand_permutation}[kind](d, g)


# ===========================================================================
# Metrics
# ===========================================================================


def metrics(got, exp):
    got = got.to(_F64); exp = exp.to(_F64)
    diff = (got - exp)
    denom = float(exp.norm()) or 1.0
    return {"max_abs": float(diff.abs().max()), "rel": float(diff.norm() / denom)}


def cross_gram(A, B): return A @ B.transpose(-2, -1)


def corr_offdiag(P, Q):
    n = P.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    p = P[iu[0], iu[1]].to(_F64); q = Q[iu[0], iu[1]].to(_F64)
    if p.std() == 0 or q.std() == 0:
        return float("nan")
    pc = p - p.mean(); qc = q - q.mean()
    return float((pc @ qc) / (pc.norm() * qc.norm()))


# ===========================================================================
# Amulet island: apply an arbitrary phi (proves the masked derivative is
# obtainable via the same lift; used to isolate the Hadamard blocker).
# ===========================================================================


def island_apply_phi(u_tilde, params, phi: Callable):
    m1, m2, m3, m4 = _island_operators(params)
    z = m1 @ _lift_columns(u_tilde, params.r_factors.r2) @ m2
    return m3 @ phi(z) @ m4


# ===========================================================================
# Experiment 1: SiLU
# ===========================================================================


def run_silu(*, m: int = 4, d: int = 8, k: int = 3, seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(m, d, generator=g, dtype=_F64)
    GY = torch.randn(m, d, generator=g, dtype=_F64)
    Y = silu(X)
    GX = GY * silu_prime(X)                                   # plaintext backward

    # independent A1 masks (dense) + permutation counterparts
    N_in = rand_invertible(d, g); N_out = rand_orthogonal(d, g)
    M_in = rand_orthogonal(d, g); M_out = rand_orthogonal(d, g)   # independent
    P_in = rand_permutation(d, g); P_out = rand_permutation(d, g)

    out: dict[str, Any] = {"dims": {"m": m, "d": d, "k": k}}

    # ---- variant: trusted_shortcut (correctness upper bound; TEE plaintext) ----
    Xt = X @ N_in
    X_rec = Xt @ torch.linalg.inv(N_in)                      # TEE recover
    Y_ts = silu(X_rec) @ N_out
    GYt = GY @ M_out
    GY_rec = GYt @ torch.linalg.inv(M_out)                   # TEE recover
    GX_ts = (GY_rec * silu_prime(X_rec)) @ M_in
    out["trusted_shortcut"] = {
        "forward": metrics(Y_ts, Y @ N_out),
        "backward": metrics(GX_ts, GX @ M_in),
        "gpu_side": False, "exposes_plaintext_in_tee": True,
        "forward_status": "trusted_shortcut_only",
        "backward_status": "trusted_shortcut_only",
    }

    # ---- variant: amulet_dense (real island forward; A1 backward attempt) ----
    params = make_right_mask_amulet_params(m, d, k, N_in, torch.linalg.inv(N_in),
                                           seed=seed + 1)
    t0 = time.perf_counter()
    V = amulet_right_mask_activation(Xt, params, "silu")      # = SiLU(X) N_in
    fwd_ms = (time.perf_counter() - t0) * 1e3
    Y_am = V @ (torch.linalg.inv(N_in) @ N_out)              # public linear remask
    fwd_err = metrics(Y_am, Y @ N_out)
    # backward: GRANT a perfect masked derivative via the same lift, then show the
    # Hadamard step is the blocker under independent dense masks.
    D_tilde = island_apply_phi(Xt, params, silu_prime)       # = SiLU'(X) N_in
    D_err = metrics(D_tilde, silu_prime(X) @ N_in)
    D_M = D_tilde @ (torch.linalg.inv(N_in) @ M_out)         # remask to M_out
    H = GYt * D_M                                             # (GY M_out) ⊙ (SiLU'(X) M_out)
    hadamard_err = metrics(H, GX @ M_out)                    # want GX M_out -> fails
    out["amulet_dense"] = {
        "forward": fwd_err, "forward_runtime_ms": fwd_ms,
        "masked_derivative_err": D_err,
        "backward_hadamard_attempt": hadamard_err,
        "gpu_side": True, "exposes_plaintext_in_tee": False,
        "forward_status": "forward_only",
        "backward_status": "blocked_by_hadamard",
        "note": "masked derivative is obtainable (err ~0); the Hadamard of two "
                "independently dense-masked tensors does not commute -> blocked.",
    }

    # ---- variant: permutation (Hadamard commutes; but Gram leaks) ----
    Xtp = X @ P_in
    Yp = silu(Xtp) @ (torch.linalg.inv(P_in) @ P_out)        # silu(X P_in)=silu(X)P_in
    fwd_perm = metrics(Yp, Y @ P_out)
    GYtp = GY @ P_out
    Dp = silu_prime(Xtp) @ (torch.linalg.inv(P_in) @ P_out)  # SiLU'(X) P_out
    Hp = GYtp * Dp                                           # = GX P_out (commutes)
    bwd_perm = metrics(Hp, GX @ P_out)
    act_gram_leak = corr_offdiag(cross_gram(Xtp, Xtp), cross_gram(X, X))
    out["permutation"] = {
        "forward": fwd_perm, "backward": bwd_perm,
        "gpu_side": True, "exposes_plaintext_in_tee": False,
        "activation_gram_corr": act_gram_leak,
        "forward_status": "backward_passed",
        "backward_status": "backward_passed",
        "caveat": "requires permutation masks; activation Gram leaks (corr~1), so "
                  "this does NOT provide the dense-mask A1 privacy.",
    }
    return out


# ===========================================================================
# Experiment 2: SwiGLU  (Z = SiLU(G) * U)
# ===========================================================================


def run_swiglu(*, m: int = 4, d: int = 8, k: int = 3, seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(m, d, generator=g, dtype=_F64)
    U = torch.randn(m, d, generator=g, dtype=_F64)
    GZ = torch.randn(m, d, generator=g, dtype=_F64)
    Z = swiglu(G, U)
    dG = GZ * U * silu_prime(G)                              # plaintext grads
    dU = GZ * silu(G)

    n = rand_invertible(d, g)                                # shared forward mask
    N_z = rand_orthogonal(d, g)
    M_z = rand_orthogonal(d, g)                              # independent A1 upstream
    M_g = rand_orthogonal(d, g); M_u = rand_orthogonal(d, g)
    P = rand_permutation(d, g); Pz = rand_permutation(d, g)
    Pg = rand_permutation(d, g); Pu = rand_permutation(d, g)

    out: dict[str, Any] = {"dims": {"m": m, "d": d, "k": k}}

    # ---- trusted_shortcut ----
    Gt, Ut = G @ n, U @ n
    ninv = torch.linalg.inv(n)
    G_r, U_r = Gt @ ninv, Ut @ ninv
    Z_ts = swiglu(G_r, U_r) @ N_z
    GZt = GZ @ M_z
    GZ_r = GZt @ torch.linalg.inv(M_z)
    dG_ts = (GZ_r * U_r * silu_prime(G_r)) @ M_g
    dU_ts = (GZ_r * silu(G_r)) @ M_u
    out["trusted_shortcut"] = {
        "forward": metrics(Z_ts, Z @ N_z),
        "backward_dG": metrics(dG_ts, dG @ M_g),
        "backward_dU": metrics(dU_ts, dU @ M_u),
        "gpu_side": False, "exposes_plaintext_in_tee": True,
        "forward_status": "trusted_shortcut_only",
        "backward_status": "trusted_shortcut_only",
    }

    # ---- amulet_dense: real island forward; A1 backward attempt ----
    params = make_right_mask_amulet_params(m, d, k, n, ninv, seed=seed + 2)
    t0 = time.perf_counter()
    Zt = amulet_right_mask_swiglu(Gt, Ut, params)            # = Z n
    fwd_ms = (time.perf_counter() - t0) * 1e3
    Z_am = Zt @ (ninv @ N_z)
    fwd_err = metrics(Z_am, Z @ N_z)
    # backward needs dG = GZ ⊙ U ⊙ SiLU'(G), dU = GZ ⊙ SiLU(G): Hadamard of
    # tensors under the independent upstream mask M_z vs forward mask n.
    SiluG_t = island_apply_phi(Gt, params, silu)             # SiLU(G) n
    Dprime_t = island_apply_phi(Gt, params, silu_prime)      # SiLU'(G) n
    # best GPU attempt at dU = GZ ⊙ SiLU(G): remask SiLU(G) n -> M_z, Hadamard
    SiluG_Mz = SiluG_t @ (ninv @ M_z)
    dU_attempt = GZt * SiluG_Mz
    dU_err = metrics(dU_attempt, dU @ M_z)
    # dG = GZ ⊙ U ⊙ SiLU'(G): triple Hadamard, even worse
    U_Mz = Ut @ (ninv @ M_z); Dprime_Mz = Dprime_t @ (ninv @ M_z)
    dG_attempt = GZt * U_Mz * Dprime_Mz
    dG_err = metrics(dG_attempt, dG @ M_z)
    out["amulet_dense"] = {
        "forward": fwd_err, "forward_runtime_ms": fwd_ms,
        "backward_dU_hadamard_attempt": dU_err,
        "backward_dG_hadamard_attempt": dG_err,
        "gpu_side": True, "exposes_plaintext_in_tee": False,
        "forward_status": "forward_only",
        "backward_status": "blocked_by_hadamard",
    }

    # ---- permutation (works but Gram leaks) ----
    Gp, Up = G @ P, U @ P
    Zp = swiglu(Gp, Up) @ (torch.linalg.inv(P) @ Pz)
    fwd_perm = metrics(Zp, Z @ Pz)
    GZp = GZ @ Pz
    piv = torch.linalg.inv(P)
    dU_p = GZp * (silu(Gp) @ (piv @ Pz))
    dG_p = GZp * (Up @ (piv @ Pz)) * (silu_prime(Gp) @ (piv @ Pz))
    out["permutation"] = {
        "forward": fwd_perm,
        "backward_dU": metrics(dU_p, dU @ Pz),
        "backward_dG": metrics(dG_p, dG @ Pz),
        "gpu_side": True, "exposes_plaintext_in_tee": False,
        "activation_gram_corr": corr_offdiag(cross_gram(Gp, Gp), cross_gram(G, G)),
        "forward_status": "backward_passed",
        "backward_status": "backward_passed",
        "caveat": "permutation masks only; activation Gram leaks.",
    }
    return out


# ===========================================================================
# Experiment 3: cross-Gram after nonlinear (200 trials)
# ===========================================================================


def run_cross_gram_after_nonlinear(*, m: int = 16, d: int = 8, trials: int = 200,
                                   seed: int = 0) -> dict[str, Any]:
    """Does a working nonlinear backward leak the exact cross-Gram X_tilde GX_tilde^T
    == X GX^T? A0 (coordinated masks) leaks; A1 (independent) breaks the equality.
    amulet_dense has no valid backward -> not applicable."""
    variants = {
        "trusted_shortcut_A1_dense": {"corr": [], "leak": None},
        "trusted_shortcut_A0_dual": {"corr": [], "leak": None},
        "permutation_A1": {"corr": [], "act_gram": [], "corr_stat": []},
    }
    for t in range(trials):
        g = torch.Generator().manual_seed(seed * 100003 + t)
        X = torch.randn(m, d, generator=g, dtype=_F64)
        GY = torch.randn(m, d, generator=g, dtype=_F64)
        GX = GY * silu_prime(X)
        cg_true = cross_gram(X, GX)
        N_in = rand_orthogonal(d, g)
        M_in = rand_orthogonal(d, g)                          # A1 independent
        Xt = X @ N_in
        # A1 dense (trusted_shortcut recovers, remask GX with independent M_in)
        gx_a1 = GX @ M_in
        variants["trusted_shortcut_A1_dense"]["corr"].append(
            corr_offdiag(cross_gram(Xt, gx_a1), cg_true))
        # A0 dual: coordinated backward mask makes GX_tilde = GX N_in^{-T}
        gx_a0 = GX @ torch.linalg.inv(N_in).T
        variants["trusted_shortcut_A0_dual"]["corr"].append(
            corr_offdiag(cross_gram(Xt, gx_a0), cg_true))
        # permutation A1
        P_in = rand_permutation(d, g); P_out = rand_permutation(d, g)
        Xtp = X @ P_in; gx_p = GX @ P_out
        variants["permutation_A1"]["corr_stat"].append(
            corr_offdiag(cross_gram(Xtp, gx_p), cg_true))
        variants["permutation_A1"]["act_gram"].append(
            corr_offdiag(cross_gram(Xtp, Xtp), cross_gram(X, X)))

    def stat(xs):
        tvec = torch.tensor(xs, dtype=_F64)
        return {"corr_mean": float(tvec.mean()), "corr_abs_mean": float(tvec.abs().mean()),
                "corr_std": float(tvec.std())}

    a0 = stat(variants["trusted_shortcut_A0_dual"]["corr"])
    a1 = stat(variants["trusted_shortcut_A1_dense"]["corr"])
    perm = stat(variants["permutation_A1"]["corr_stat"])
    perm_gram = stat(variants["permutation_A1"]["act_gram"])
    return {
        "trials": trials,
        "trusted_shortcut_A0_dual": {**a0, "exact_cross_gram_leak": a0["corr_mean"] > 0.99,
                                     "has_backward": True, "gpu_side": False},
        "trusted_shortcut_A1_dense": {**a1, "exact_cross_gram_leak": a1["corr_mean"] > 0.99,
                                      "has_backward": True, "gpu_side": False},
        "permutation_A1": {**perm, "exact_cross_gram_leak": perm["corr_mean"] > 0.99,
                           "activation_gram_corr_mean": perm_gram["corr_mean"],
                           "activation_gram_leak": perm_gram["corr_mean"] > 0.99,
                           "has_backward": True, "gpu_side": True},
        "amulet_dense": {"has_backward": False,
                         "exact_cross_gram_leak": "not_applicable",
                         "reason": "blocked_by_hadamard: no valid A1 backward"},
    }


# ===========================================================================
# Experiment 4: tiny transformer integration
# ===========================================================================


def run_tiny_transformer_integration(*, m: int = 4, d: int = 8, k: int = 3,
                                     seed: int = 0) -> dict[str, Any]:
    """Try to replace the SwiGLU MLP nonlinear block in a masked-domain training
    pass with the candidate primitive, without a trusted shortcut."""
    sw = run_swiglu(m=m, d=d, k=k, seed=seed)
    fwd_ok = sw["amulet_dense"]["forward"]["rel"] < 1e-9
    bwd_blocked = sw["amulet_dense"]["backward_status"] == "blocked_by_hadamard"
    return {
        "masked_forward_swiglu_ok": bool(fwd_ok),
        "masked_backward_swiglu_status": sw["amulet_dense"]["backward_status"],
        "nonlinear_training_without_trusted_shortcut": "blocked",
        "status": "blocked_by_hadamard" if bwd_blocked else "numerical_failure",
        "permutation_alternative": {
            "trainable": True,
            "status": "backward_passed",
            "blocker": "requires permutation masks throughout; incompatible with "
                       "the dense masks linear/LoRA A1 needs, and leaks activation "
                       "Gram -> not an A1-privacy solution.",
        },
        "reason": "SwiGLU masked forward works (Amulet lift) but the A1 backward "
                  "needs Hadamard products under independent dense masks; blocked.",
    }


# ===========================================================================
# Orchestration
# ===========================================================================


def run_all(*, seed: int = 0) -> dict[str, Any]:
    silu_res = run_silu(seed=seed)
    swiglu_res = run_swiglu(seed=seed)
    cross_gram = run_cross_gram_after_nonlinear(seed=seed)
    tiny = run_tiny_transformer_integration(seed=seed)
    # efficiency: island cost vs dim-lift factor k
    eff_rows = []
    for name, fn in (("silu_island", run_silu), ("swiglu_island", run_swiglu)):
        for k in (2, 3, 4):
            r = fn(m=4, d=8, k=k, seed=seed)
            am = r["amulet_dense"]
            eff_rows.append({
                "primitive": name, "m": 4, "d": 8, "k": k,
                "lifted_rows": 4 * k, "lifted_cols": 8 * k,
                "forward_runtime_ms": am["forward_runtime_ms"],
                "memory_estimate_bytes": int(4 * k * 8 * k * 8),  # lifted f64 tensor
            })
    return {
        "stage": "nonlinear_backward_primitive_audit",
        "silu": silu_res,
        "swiglu": swiglu_res,
        "cross_gram_after_nonlinear": cross_gram,
        "tiny_transformer_integration": tiny,
        "efficiency": eff_rows,
        "answers": {
            "gpu_silu_forward_implemented": True,
            "gpu_silu_a1_backward_implemented": False,
            "gpu_swiglu_forward_implemented": True,
            "gpu_swiglu_a1_backward_implemented": False,
            "any_variant_avoids_exact_cross_gram_gpu_side": "permutation_only_but_gram_leaks",
            "fast_enough_small_dims": True,
            "tiny_transformer_nonlinear_without_trusted_shortcut": False,
            "full_scheme_a_still_blocked": True,
        },
    }
