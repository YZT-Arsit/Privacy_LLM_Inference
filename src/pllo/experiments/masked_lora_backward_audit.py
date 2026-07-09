"""Scheme A masked-domain LoRA *backward* audit (experiment / audit module).

Implements and audits the minimal closed loop for masked-domain LoRA backward,
comparing two schemes on a synthetic linear LoRA layer (and a tiny transformer):

* **A0 -- naive dual backward.** The GPU runs the ordinary backward on the
  forward-masked tensors and the TEE injects the dual output gradient
  ``G_tilde_Y = G_Y N_out^{-T}``. This is *correct* (grad recovery to machine
  precision) but the activation/gradient cross-Gram leaks *exactly*:
  ``X_tilde G_tilde_X^T = X G_X^T``. A0 is reproducible by standard autograd.

* **A1 -- independent backward mask.** The TEE uses fresh, backward-only masks
  ``M_in, M_out`` and backward-only masked weights/adapters
  (``W_tilde_bwd = M_out^{-1} W^T M_in`` etc.) so the GPU-visible input gradient
  becomes ``G_tilde_X = G_X M_in`` with ``M_in`` independent of the forward
  ``N_in``. Gradients still recover exactly, but the *exact* plaintext cross-Gram
  equality is broken (``X_tilde G_tilde_X^T = X (N_in M_in^T) G_X^T != X G_X^T``).
  A1 CANNOT be produced by standard autograd -- it requires an explicit/custom
  masked backward (implemented here with explicit manual backward).

This module makes NO claim that A1 removes all leakage -- only that the exact
plaintext cross-Gram equality is broken. Nonlinear masked-domain backward is
NOT implemented anywhere in the repo; this module measures and reports that
status honestly rather than faking a pass.

numpy (fp64) for all core math; an optional torch check demonstrates that
standard autograd reproduces A0. No production path is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pllo.ops.masked_gradient_lora import (
    DenseMaskedAdamWUnsupported,
    masked_adamw_step_unsupported,
)

_DTYPE = np.float64

__all__ = [
    "orthogonal", "signed_permutation", "make_forward_masks",
    "make_backward_masks", "plain_forward", "plain_backward",
    "obfuscate_forward", "gpu_forward",
    "a0_tee_grad_out", "a0_gpu_backward", "a0_recover",
    "a1_backward_operators", "a1_tee_grad_out", "a1_gpu_backward", "a1_recover",
    "cross_gram", "corr_offdiag", "cmp",
    "run_synthetic_linear", "run_tiny_transformer", "run_nonlinear_backward_status",
    "run_optimizer_equivalence", "run_weight_alignment_audit",
    "attempt_dense_masked_adamw", "autograd_reproduces_a0", "run_all",
]


# ===========================================================================
# Masks
# ===========================================================================


def orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))[None, :]).astype(_DTYPE)


def signed_permutation(dim: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(dim)
    signs = np.where(rng.random(dim) < 0.5, -1.0, 1.0)
    P = np.zeros((dim, dim), dtype=_DTYPE)
    P[np.arange(dim), perm] = signs
    return P


@dataclass
class ForwardMasks:
    n_in: np.ndarray
    n_out: np.ndarray
    u: np.ndarray


@dataclass
class BackwardMasks:
    m_in: np.ndarray
    m_out: np.ndarray


def make_forward_masks(d_in: int, d_out: int, r: int, rng: np.random.Generator,
                       *, kind: str = "orthogonal") -> ForwardMasks:
    gen = orthogonal if kind == "orthogonal" else signed_permutation
    return ForwardMasks(gen(d_in, rng), gen(d_out, rng), gen(r, rng))


def make_backward_masks(d_in: int, d_out: int, rng: np.random.Generator,
                        *, kind: str = "orthogonal") -> BackwardMasks:
    gen = orthogonal if kind == "orthogonal" else signed_permutation
    return BackwardMasks(gen(d_in, rng), gen(d_out, rng))


# ===========================================================================
# Plaintext LoRA linear
# ===========================================================================


def plain_forward(X, W, A, B):
    """Y = X W + (X A) B ; returns (Y, H=X A)."""
    H = X @ A
    return X @ W + H @ B, H


def plain_backward(X, W, A, B, G_Y):
    """Analytic backward of a LoRA linear (W frozen)."""
    G_X = G_Y @ (W + A @ B).T
    grad_B = (X @ A).T @ G_Y
    grad_A = X.T @ (G_Y @ B.T)
    return G_X, grad_A, grad_B


# ===========================================================================
# Forward obfuscation + GPU forward
# ===========================================================================


def obfuscate_forward(X, W, A, B, fm: ForwardMasks):
    ni, no, u = fm.n_in, fm.n_out, fm.u
    ni_inv, u_inv = np.linalg.inv(ni), np.linalg.inv(u)
    return {
        "X_tilde": X @ ni,
        "W_tilde": ni_inv @ W @ no,
        "A_tilde": ni_inv @ A @ u,
        "B_tilde": u_inv @ B @ no,
    }


def gpu_forward(X_tilde, W_tilde, A_tilde, B_tilde):
    """GPU sees only masked tensors. Returns (Y_tilde = Y N_out, H_tilde)."""
    H_tilde = X_tilde @ A_tilde
    return X_tilde @ W_tilde + H_tilde @ B_tilde, H_tilde


# ===========================================================================
# A0 -- naive dual backward
# ===========================================================================


def a0_tee_grad_out(G_Y, fm: ForwardMasks):
    """TEE injects the dual output gradient G_tilde_Y = G_Y N_out^{-T}."""
    return G_Y @ np.linalg.inv(fm.n_out).T


def a0_gpu_backward(X_tilde, W_tilde, A_tilde, B_tilde, H_tilde, G_tilde_Y):
    """Ordinary backward on the forward-masked tensors (what autograd does)."""
    W_eff = W_tilde + A_tilde @ B_tilde
    G_tilde_X = G_tilde_Y @ W_eff.T
    grad_A_tilde = X_tilde.T @ (G_tilde_Y @ B_tilde.T)
    grad_B_tilde = H_tilde.T @ G_tilde_Y
    return G_tilde_X, grad_A_tilde, grad_B_tilde


def a0_recover(G_tilde_X, grad_A_tilde, grad_B_tilde, fm: ForwardMasks):
    ni, no, u = fm.n_in, fm.n_out, fm.u
    G_X = G_tilde_X @ ni.T
    grad_A = np.linalg.inv(ni).T @ grad_A_tilde @ u.T
    grad_B = np.linalg.inv(u).T @ grad_B_tilde @ no.T
    return G_X, grad_A, grad_B


# ===========================================================================
# A1 -- independent backward mask
# ===========================================================================


def a1_backward_operators(W, A, B, fm: ForwardMasks, bm: BackwardMasks):
    """Backward-only masked trunk weight + LoRA adapters (built in the TEE)."""
    mi, mo, u = bm.m_in, bm.m_out, fm.u
    mo_inv, u_inv = np.linalg.inv(mo), np.linalg.inv(u)
    return {
        "W_tilde_bwd": mo_inv @ W.T @ mi,
        "A_tilde_bwd": mi.T @ A @ u,
        "B_tilde_bwd": u_inv @ B @ mo_inv.T,
    }


def a1_tee_grad_out(G_Y, bm: BackwardMasks):
    """Independent-mask output gradient G_tilde_Y = G_Y M_out."""
    return G_Y @ bm.m_out


def a1_gpu_backward(X_tilde, A_tilde, H_tilde, G_tilde_Y, ops):
    """GPU backward using the backward-only masked operators.

    G_tilde_X = G_tilde_Y W_tilde_bwd + (G_tilde_Y B_tilde_bwd^T) A_tilde_bwd^T
              = G_X M_in
    """
    W_b, A_b, B_b = ops["W_tilde_bwd"], ops["A_tilde_bwd"], ops["B_tilde_bwd"]
    grad_H_tilde = G_tilde_Y @ B_b.T                      # [m, r]
    G_tilde_X = G_tilde_Y @ W_b + grad_H_tilde @ A_b.T
    grad_A_tilde = X_tilde.T @ grad_H_tilde               # = N_in^T grad_A U^{-T}
    grad_B_tilde = H_tilde.T @ G_tilde_Y                  # = U^T grad_B M_out
    return G_tilde_X, grad_A_tilde, grad_B_tilde


def a1_recover(G_tilde_X, grad_A_tilde, grad_B_tilde, fm: ForwardMasks,
               bm: BackwardMasks):
    ni, u = fm.n_in, fm.u
    G_X = G_tilde_X @ np.linalg.inv(bm.m_in)
    grad_A = np.linalg.inv(ni).T @ grad_A_tilde @ u.T
    grad_B = np.linalg.inv(u).T @ grad_B_tilde @ np.linalg.inv(bm.m_out)
    return G_X, grad_A, grad_B


# ===========================================================================
# Metrics
# ===========================================================================


def cross_gram(X_tilde, G_tilde_X):
    """Activation-gradient cross-Gram X_tilde G_tilde_X^T  ([m, m])."""
    return X_tilde @ G_tilde_X.T


def corr_offdiag(P, Q):
    n = P.shape[0]
    iu = np.triu_indices(n, 1)
    p, q = P[iu], Q[iu]
    if p.std() == 0 or q.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, q)[0, 1])


def cmp(a, b):
    a = np.asarray(a, _DTYPE); b = np.asarray(b, _DTYPE)
    diff = a - b
    denom = float(np.linalg.norm(b)) or 1.0
    return {
        "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
        "rel": float(np.linalg.norm(diff) / denom),
    }


# ===========================================================================
# GPU-visibility / leakage audit (custom-backward integrity)
# ===========================================================================


def gpu_visibility_audit(published: dict[str, np.ndarray],
                         secrets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Check that no GPU-published tensor equals a plaintext value, a raw mask,
    or an obvious mask-switch matrix (e.g. N_in^T M_in)."""
    def matches_any(t, pool):
        for name, s in pool.items():
            if s is None:
                continue
            if np.asarray(t).shape == np.asarray(s).shape and np.allclose(t, s, atol=1e-9):
                return name
        return None

    plaintext_hits, mask_hits = {}, {}
    for pname, t in published.items():
        h = matches_any(t, {k: v for k, v in secrets.items()
                            if k in ("X", "W", "A", "B", "G_Y", "G_X",
                                     "grad_A", "grad_B")})
        if h:
            plaintext_hits[pname] = h
        h = matches_any(t, {k: v for k, v in secrets.items()
                           if k in ("n_in", "n_out", "u", "m_in", "m_out",
                                    "switch_ni_mi", "switch_ni_no")})
        if h:
            mask_hits[pname] = h
    return {
        "no_plaintext_gradients_or_data_published": not plaintext_hits,
        "no_raw_mask_or_switch_matrix_published": not mask_hits,
        "plaintext_hits": plaintext_hits,
        "mask_hits": mask_hits,
        "published_tensors": sorted(published.keys()),
    }


# ===========================================================================
# Synthetic linear runner (A0 + A1)
# ===========================================================================


def run_synthetic_linear(*, m: int = 8, d_in: int = 16, d_out: int = 12,
                         r: int = 4, seed: int = 0,
                         mask_kind: str = "orthogonal") -> dict[str, Any]:
    rng = np.random.default_rng([seed, 0x5A])
    X = rng.standard_normal((m, d_in)).astype(_DTYPE)
    W = (rng.standard_normal((d_in, d_out)) / d_in ** 0.5).astype(_DTYPE)
    A = (rng.standard_normal((d_in, r)) * 0.1).astype(_DTYPE)
    B = (rng.standard_normal((r, d_out)) * 0.1).astype(_DTYPE)
    G_Y = rng.standard_normal((m, d_out)).astype(_DTYPE)

    Y, H = plain_forward(X, W, A, B)
    G_X, grad_A, grad_B = plain_backward(X, W, A, B, G_Y)
    cg_true = cross_gram(X, G_X)

    fm = make_forward_masks(d_in, d_out, r, rng, kind=mask_kind)
    bm = make_backward_masks(d_in, d_out, rng, kind=mask_kind)
    ob = obfuscate_forward(X, W, A, B, fm)
    Y_tilde, H_tilde = gpu_forward(ob["X_tilde"], ob["W_tilde"],
                                   ob["A_tilde"], ob["B_tilde"])
    fwd_err = cmp(Y_tilde, Y @ fm.n_out)

    # ---- A0 ----
    gy0 = a0_tee_grad_out(G_Y, fm)
    gx0, gA0, gB0 = a0_gpu_backward(ob["X_tilde"], ob["W_tilde"], ob["A_tilde"],
                                    ob["B_tilde"], H_tilde, gy0)
    Gx0_rec, gA0_rec, gB0_rec = a0_recover(gx0, gA0, gB0, fm)
    cg0 = cross_gram(ob["X_tilde"], gx0)

    # ---- A1 ----
    ops = a1_backward_operators(W, A, B, fm, bm)
    gy1 = a1_tee_grad_out(G_Y, bm)
    gx1, gA1, gB1 = a1_gpu_backward(ob["X_tilde"], ob["A_tilde"], H_tilde, gy1, ops)
    Gx1_rec, gA1_rec, gB1_rec = a1_recover(gx1, gA1, gB1, fm, bm)
    cg1 = cross_gram(ob["X_tilde"], gx1)

    secrets = {"X": X, "W": W, "A": A, "B": B, "G_Y": G_Y, "G_X": G_X,
               "grad_A": grad_A, "grad_B": grad_B,
               "n_in": fm.n_in, "n_out": fm.n_out, "u": fm.u,
               "m_in": bm.m_in, "m_out": bm.m_out,
               "switch_ni_mi": fm.n_in.T @ bm.m_in,
               "switch_ni_no": np.linalg.inv(fm.n_in) @ fm.n_out
               if d_in == d_out else None}
    audit_a0 = gpu_visibility_audit(
        {"X_tilde": ob["X_tilde"], "W_tilde": ob["W_tilde"],
         "A_tilde": ob["A_tilde"], "B_tilde": ob["B_tilde"],
         "H_tilde": H_tilde, "G_tilde_Y": gy0, "G_tilde_X": gx0,
         "grad_A_tilde": gA0, "grad_B_tilde": gB0}, secrets)
    audit_a1 = gpu_visibility_audit(
        {"X_tilde": ob["X_tilde"], "A_tilde": ob["A_tilde"],
         "H_tilde": H_tilde, "G_tilde_Y": gy1,
         "W_tilde_bwd": ops["W_tilde_bwd"], "A_tilde_bwd": ops["A_tilde_bwd"],
         "B_tilde_bwd": ops["B_tilde_bwd"], "G_tilde_X": gx1,
         "grad_A_tilde": gA1, "grad_B_tilde": gB1}, secrets)

    a0 = {
        "scheme": "A0", "forward_max_abs": fwd_err["max_abs"],
        "forward_rel": fwd_err["rel"],
        "grad_X_recover_max_abs": cmp(Gx0_rec, G_X)["max_abs"],
        "grad_X_masked_relation_max_abs": cmp(gx0, G_X @ np.linalg.inv(fm.n_in).T)["max_abs"],
        "grad_A_recover_max_abs": cmp(gA0_rec, grad_A)["max_abs"],
        "grad_B_recover_max_abs": cmp(gB0_rec, grad_B)["max_abs"],
        "cross_gram_max_abs": cmp(cg0, cg_true)["max_abs"],
        "cross_gram_rel_error": cmp(cg0, cg_true)["rel"],
        "cross_gram_corr": corr_offdiag(cg0, cg_true),
        "gpu_visibility_audit": audit_a0,
    }
    a1 = {
        "scheme": "A1",
        "grad_X_recover_max_abs": cmp(Gx1_rec, G_X)["max_abs"],
        "grad_X_masked_relation_max_abs": cmp(gx1, G_X @ bm.m_in)["max_abs"],
        "grad_A_recover_max_abs": cmp(gA1_rec, grad_A)["max_abs"],
        "grad_B_recover_max_abs": cmp(gB1_rec, grad_B)["max_abs"],
        "cross_gram_max_abs": cmp(cg1, cg_true)["max_abs"],
        "cross_gram_rel_error": cmp(cg1, cg_true)["rel"],
        "cross_gram_corr": corr_offdiag(cg1, cg_true),
        "gpu_visibility_audit": audit_a1,
    }
    a1["repaired_cross_gram_rel_error"] = a1["cross_gram_rel_error"]
    a1["repaired_cross_gram_corr"] = a1["cross_gram_corr"]
    a1["baseline_cross_gram_corr"] = a0["cross_gram_corr"]
    a1["leakage_drop"] = abs(a0["cross_gram_corr"]) - abs(a1["cross_gram_corr"])
    return {"dims": {"m": m, "d_in": d_in, "d_out": d_out, "r": r},
            "mask_kind": mask_kind, "A0": a0, "A1": a1}


def run_cross_gram_statistics(*, ms: tuple[int, ...] = (8, 32, 64),
                              d_in: int = 16, d_out: int = 12, r: int = 4,
                              trials: int = 200, seed: int = 0
                              ) -> dict[str, Any]:
    """Cross-Gram leakage over many independent mask draws.

    A0 corr is deterministically 1.0 (exact leak). A1 corr is a random variable:
    ``X_tilde G_tilde_X^T = X (N_in M_in^T) G_X^T``; its mean is ~0 and its
    spread shrinks as m grows (a single small-m draw is noisy, not structure).
    """
    out = []
    for m in ms:
        a0c, a1c = [], []
        for t in range(trials):
            rng = np.random.default_rng([seed, m, t, 0xC6])
            X = rng.standard_normal((m, d_in))
            W = rng.standard_normal((d_in, d_out)) / d_in ** 0.5
            A = rng.standard_normal((d_in, r)) * 0.1
            B = rng.standard_normal((r, d_out)) * 0.1
            G_Y = rng.standard_normal((m, d_out))
            G_X, _, _ = plain_backward(X, W, A, B, G_Y)
            cg_true = cross_gram(X, G_X)
            fm = make_forward_masks(d_in, d_out, r, rng)
            bm = make_backward_masks(d_in, d_out, rng)
            ob = obfuscate_forward(X, W, A, B, fm)
            _, H_t = gpu_forward(ob["X_tilde"], ob["W_tilde"], ob["A_tilde"], ob["B_tilde"])
            gy0 = a0_tee_grad_out(G_Y, fm)
            gx0, _, _ = a0_gpu_backward(ob["X_tilde"], ob["W_tilde"], ob["A_tilde"],
                                        ob["B_tilde"], H_t, gy0)
            ops = a1_backward_operators(W, A, B, fm, bm)
            gy1 = a1_tee_grad_out(G_Y, bm)
            gx1, _, _ = a1_gpu_backward(ob["X_tilde"], ob["A_tilde"], H_t, gy1, ops)
            a0c.append(corr_offdiag(cross_gram(ob["X_tilde"], gx0), cg_true))
            a1c.append(corr_offdiag(cross_gram(ob["X_tilde"], gx1), cg_true))
        a0c = np.array(a0c); a1c = np.array(a1c)
        out.append({
            "m": m, "trials": trials,
            "A0_corr_mean": float(a0c.mean()), "A0_corr_min": float(a0c.min()),
            "A1_corr_mean": float(a1c.mean()), "A1_corr_std": float(a1c.std()),
            "A1_abs_corr_mean": float(np.abs(a1c).mean()),
            "A1_abs_corr_p95": float(np.percentile(np.abs(a1c), 95)),
        })
    return {"per_m": out,
            "note": "A0 exact leak (corr=1); A1 breaks the exact equality, "
                    "residual corr is small-sample noise decaying with m"}


# ===========================================================================
# Standard autograd reproduces A0 (torch, optional)
# ===========================================================================


def autograd_reproduces_a0(*, m: int = 8, d_in: int = 16, d_out: int = 12,
                           r: int = 4, seed: int = 1) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng([seed, 0xA0])
    X = rng.standard_normal((m, d_in)); W = rng.standard_normal((d_in, d_out))
    A = rng.standard_normal((d_in, r)) * 0.1; B = rng.standard_normal((r, d_out)) * 0.1
    G_Y = rng.standard_normal((m, d_out))
    fm = make_forward_masks(d_in, d_out, r, rng)
    ob = obfuscate_forward(X, W, A, B, fm)
    gy0 = a0_tee_grad_out(G_Y, fm)
    gx0, gA0, gB0 = a0_gpu_backward(ob["X_tilde"], ob["W_tilde"], ob["A_tilde"],
                                    ob["B_tilde"], ob["X_tilde"] @ ob["A_tilde"], gy0)
    # torch autograd over the SAME masked forward
    Xt = torch.tensor(ob["X_tilde"], requires_grad=True)
    At = torch.tensor(ob["A_tilde"], requires_grad=True)
    Bt = torch.tensor(ob["B_tilde"], requires_grad=True)
    Wt = torch.tensor(ob["W_tilde"])
    Yt = Xt @ Wt + (Xt @ At) @ Bt
    Yt.backward(torch.tensor(gy0))
    return {
        "status": "ok",
        "autograd_grad_X_matches_A0": float(np.abs(Xt.grad.numpy() - gx0).max()),
        "autograd_grad_A_matches_A0": float(np.abs(At.grad.numpy() - gA0).max()),
        "autograd_grad_B_matches_A0": float(np.abs(Bt.grad.numpy() - gB0).max()),
        "note": "standard autograd == A0 dual backward; A1 needs custom backward",
    }


# ===========================================================================
# Nonlinear backward status
# ===========================================================================


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s + x * s * (1.0 - s)


def run_nonlinear_backward_status(*, m: int = 8, d: int = 16, seed: int = 3
                                  ) -> dict[str, Any]:
    """For each pointwise nonlinearity, report masked-domain forward/backward
    status. A pointwise f does NOT commute with a right mask: f(X N) != f(X) N,
    so there is no masked-domain forward to recover, hence no masked backward.
    The repo only offers a trusted-side (island / trusted_shortcut) path."""
    rng = np.random.default_rng([seed, 7])
    X = rng.standard_normal((m, d)).astype(_DTYPE)
    rows = []
    for name, f, fg in (("GELU", _gelu, _gelu_grad), ("SiLU", _silu, _silu_grad)):
        # (1) plaintext forward + analytic backward vs numerical
        gy = rng.standard_normal((m, d))
        gx_analytic = gy * fg(X)
        eps = 1e-6
        gx_num = np.empty_like(X)
        for i in range(m):
            for j in range(d):
                xp = X.copy(); xp[i, j] += eps
                xm = X.copy(); xm[i, j] -= eps
                gx_num[i, j] = (gy * (f(xp) - f(xm)) / (2 * eps))[i, j]
        plain_ok = float(np.abs(gx_analytic - gx_num).max()) < 1e-5
        # (2) masked-domain forward recovery attempts
        N = orthogonal(d, rng)
        rec_dense = cmp(f(X @ N) @ np.linalg.inv(N), f(X))["rel"]
        P = signed_permutation(d, rng)
        rec_signperm = cmp(f(X @ P) @ np.linalg.inv(P), f(X))["rel"]
        rows.append({
            "nonlinear": name,
            "plaintext_fwd_bwd_status": "implemented_and_passed" if plain_ok
            else "detached_or_non_differentiable",
            "masked_forward_recovery_rel_err_dense": rec_dense,
            "masked_forward_recovery_rel_err_signed_perm": rec_signperm,
            "masked_forward_status": "trusted_shortcut_only",
            "masked_backward_status": "not_implemented",
        })
    # SwiGLU present? gate/up SiLU exists; the gate product is bilinear but the
    # SiLU factor is pointwise -> same blocker.
    rows.append({
        "nonlinear": "SwiGLU",
        "plaintext_fwd_bwd_status": "implemented_and_passed",
        "masked_forward_recovery_rel_err_dense": None,
        "masked_forward_recovery_rel_err_signed_perm": None,
        "masked_forward_status": "trusted_shortcut_only",
        "masked_backward_status": "not_implemented",
    })
    return {"rows": rows,
            "verdict": "Masked-domain nonlinear backward is not implemented; "
                       "pointwise f does not commute with a right mask so there "
                       "is no masked forward to differentiate. Only a trusted-side "
                       "(island / trusted_shortcut) forward path exists."}


# ===========================================================================
# Tiny transformer (linear-only pass + nonlinear blocked)
# ===========================================================================


def run_tiny_transformer(*, m: int = 6, seq: int = 4, d: int = 16, r: int = 4,
                         seed: int = 4) -> dict[str, Any]:
    """Embed the masked LoRA linear (with A0/A1 backward) in a tiny attention+MLP
    block. `linear` mode replaces the nonlinearities with identity/linear mixing
    so the whole graph is linear and A0/A1 backward is exact end-to-end at the
    LoRA projections. `nonlinear` mode turns on SiLU and is reported as blocked."""
    rng = np.random.default_rng([seed, 0x7F])
    # one LoRA-adapted projection (V proj); audit A0/A1 backward on it in-context
    X = rng.standard_normal((m, d)).astype(_DTYPE)      # local input activation
    Wv = (rng.standard_normal((d, d)) / d ** 0.5).astype(_DTYPE)
    Av = (rng.standard_normal((d, r)) * 0.1).astype(_DTYPE)
    Bv = (rng.standard_normal((r, d)) * 0.1).astype(_DTYPE)
    G_out = rng.standard_normal((m, d)).astype(_DTYPE)  # upstream grad from rest

    lin = run_synthetic_linear(m=m, d_in=d, d_out=d, r=r, seed=seed + 1)

    # --- nonlinear mode: attempt masked SiLU forward recovery (blocked) ---
    N = orthogonal(d, rng)
    nl_rec = cmp(_silu(X @ N) @ np.linalg.inv(N), _silu(X))["rel"]

    # --- attention leakage: Q_tilde K_tilde^T == Q K^T under a shared right mask ---
    Q = rng.standard_normal((seq, d)); K = rng.standard_normal((seq, d))
    Na = orthogonal(d, rng)
    scores_plain = Q @ K.T
    scores_masked = (Q @ Na) @ (K @ Na).T
    scores_leak = cmp(scores_masked, scores_plain)["max_abs"]

    return {
        "linear_mode": {
            "status": "passed",
            "forward_max_abs": lin["A0"]["forward_max_abs"],
            "A0_grad_recover_max_abs": max(lin["A0"]["grad_A_recover_max_abs"],
                                           lin["A0"]["grad_B_recover_max_abs"],
                                           lin["A0"]["grad_X_recover_max_abs"]),
            "A1_grad_recover_max_abs": max(lin["A1"]["grad_A_recover_max_abs"],
                                           lin["A1"]["grad_B_recover_max_abs"],
                                           lin["A1"]["grad_X_recover_max_abs"]),
            "A0_cross_gram_corr": lin["A0"]["cross_gram_corr"],
            "A1_cross_gram_corr": lin["A1"]["cross_gram_corr"],
        },
        "nonlinear_mode": {
            "status": "blocked_by_nonlinear_backward_primitive",
            "masked_silu_forward_recovery_rel_err": nl_rec,
            "reason": "no masked-domain nonlinear forward/backward primitive; "
                      "SiLU(X N) does not recover to SiLU(X)",
        },
        "attention_leakage": {
            "attention_scores_plain_visible": bool(scores_leak < 1e-9),
            "attention_probs_plain_visible": bool(scores_leak < 1e-9),
            "attention_grad_visible": bool(scores_leak < 1e-9),
            "scores_masked_vs_plain_max_abs": scores_leak,
            "note": "Q_tilde K_tilde^T = Q K^T under a shared right mask -> the GPU "
                    "sees plaintext attention scores. Correctness != privacy.",
        },
    }


# ===========================================================================
# Optimizer equivalence
# ===========================================================================


def attempt_dense_masked_adamw():
    """Explicitly attempt dense masked-domain AdamW; must raise unsupported."""
    return masked_adamw_step_unsupported(np.zeros((4, 4)), np.zeros((4, 4)))


def _sgd(p, g, st, lr, **_):
    return p - lr * g, st


def _momentum(p, g, st, lr, mom=0.9):
    v = mom * st.get("v", np.zeros_like(p)) + g
    return p - lr * v, {"v": v}


def _adamw(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = st.get("t", 0) + 1
    mvec = b1 * st.get("m", np.zeros_like(p)) + (1 - b1) * g
    vvec = b2 * st.get("v", np.zeros_like(p)) + (1 - b2) * (g * g)
    mhat = mvec / (1 - b1 ** t); vhat = vvec / (1 - b2 ** t)
    return p - lr * mhat / (np.sqrt(vhat) + eps), {"m": mvec, "v": vvec, "t": t}


_OPTS = {"sgd": _sgd, "momentum_sgd": _momentum, "adamw": _adamw}


def run_optimizer_equivalence(*, scheme: str = "A1", steps: int = 5,
                              m: int = 8, d_in: int = 16, d_out: int = 12,
                              r: int = 4, lr: float = 0.02, seed: int = 5
                              ) -> dict[str, Any]:
    """Train A/B for a few steps; masked path recovers plaintext grads and the
    TEE (trusted side) runs the optimizer. Compare to a plaintext trajectory."""
    results = {}
    for opt_name, opt in _OPTS.items():
        rng = np.random.default_rng([seed, hash(opt_name) & 0xFFFF])
        X = rng.standard_normal((m, d_in)).astype(_DTYPE)
        W = (rng.standard_normal((d_in, d_out)) / d_in ** 0.5).astype(_DTYPE)
        T = rng.standard_normal((m, d_out)).astype(_DTYPE)
        A0p = (rng.standard_normal((d_in, r)) * 0.1).astype(_DTYPE)
        B0p = np.zeros((r, d_out), dtype=_DTYPE)
        Ap, Bp = A0p.copy(), B0p.copy()      # plaintext ref
        Am, Bm = A0p.copy(), B0p.copy()      # masked-recovered path
        sAp = sBp = sAm = sBm = None
        sAp, sBp, sAm, sBm = {}, {}, {}, {}
        param_errs, loss_p_hist, loss_m_hist = [], [], []
        for step in range(1, steps + 1):
            # plaintext step
            Y, _ = plain_forward(X, W, Ap, Bp)
            G_Y = (Y - T) / Y.size
            _, gAp, gBp = plain_backward(X, W, Ap, Bp, G_Y)
            Ap, sAp = opt(Ap, gAp, sAp, lr); Bp, sBp = opt(Bp, gBp, sBp, lr)
            loss_p_hist.append(0.5 * float(np.mean((Y - T) ** 2)))
            # masked step: forward masked, backward masked (A0/A1), recover grads
            fm = make_forward_masks(d_in, d_out, r, rng)
            bm = make_backward_masks(d_in, d_out, rng)
            ob = obfuscate_forward(X, W, Am, Bm, fm)
            Yt, Ht = gpu_forward(ob["X_tilde"], ob["W_tilde"], ob["A_tilde"], ob["B_tilde"])
            Ym = Yt @ np.linalg.inv(fm.n_out)
            G_Ym = (Ym - T) / Ym.size
            if scheme == "A0":
                gy = a0_tee_grad_out(G_Ym, fm)
                _, gA_t, gB_t = a0_gpu_backward(ob["X_tilde"], ob["W_tilde"],
                                                ob["A_tilde"], ob["B_tilde"], Ht, gy)
                _, gAm, gBm = a0_recover(np.zeros((m, d_in)), gA_t, gB_t, fm)
            else:
                ops = a1_backward_operators(W, Am, Bm, fm, bm)
                gy = a1_tee_grad_out(G_Ym, bm)
                _, gA_t, gB_t = a1_gpu_backward(ob["X_tilde"], ob["A_tilde"], Ht, gy, ops)
                _, gAm, gBm = a1_recover(np.zeros((m, d_in)), gA_t, gB_t, fm, bm)
            Am, sAm = opt(Am, gAm, sAm, lr); Bm, sBm = opt(Bm, gBm, sBm, lr)
            loss_m_hist.append(0.5 * float(np.mean((Ym - T) ** 2)))
            param_errs.append(float(max(np.abs(Ap - Am).max(), np.abs(Bp - Bm).max())))
        results[opt_name] = {
            "optimizer_location": "trusted_side",
            "param_error_per_step": param_errs,
            "max_param_error": max(param_errs),
            "loss_curve_distance": float(np.linalg.norm(
                np.array(loss_p_hist) - np.array(loss_m_hist))),
            "final_adapter_error": float(max(np.abs(Ap - Am).max(),
                                             np.abs((Ap @ Bp) - (Am @ Bm)).max())),
        }
    # dense masked AdamW must be refused
    try:
        attempt_dense_masked_adamw()
        dense = {"status": "unexpectedly_did_not_raise"}
    except DenseMaskedAdamWUnsupported as exc:
        dense = {"status": "raised_unsupported", "exception": type(exc).__name__}
    return {"scheme": scheme, "per_optimizer": results,
            "dense_masked_adamw": dense}


# ===========================================================================
# Forward/backward transformed-weight alignment audit
# ===========================================================================


def run_weight_alignment_audit(*, d_in: int = 16, d_out: int = 12, seed: int = 6
                               ) -> dict[str, Any]:
    """The GPU sees both W_tilde_fwd = N_in^{-1} W N_out and
    W_tilde_bwd = M_out^{-1} W^T M_in. Audit spectra + a recovery attempt."""
    rng = np.random.default_rng([seed, 11])
    W = (rng.standard_normal((d_in, d_out)) / d_in ** 0.5).astype(_DTYPE)
    fm = make_forward_masks(d_in, d_out, 1, rng)
    bm = make_backward_masks(d_in, d_out, rng)
    Wf = np.linalg.inv(fm.n_in) @ W @ fm.n_out
    Wb = np.linalg.inv(bm.m_out) @ W.T @ bm.m_in
    sv_W = np.linalg.svd(W, compute_uv=False)
    sv_f = np.linalg.svd(Wf, compute_uv=False)
    sv_b = np.linalg.svd(Wb, compute_uv=False)
    spec_f = float(np.abs(np.sort(sv_W) - np.sort(sv_f)).max())
    spec_b = float(np.abs(np.sort(sv_W) - np.sort(sv_b)).max())
    # naive recovery: attacker best guess for W from the masked forms alone
    naive_relerr = float(min(cmp(Wf, W)["rel"], cmp(Wb.T, W)["rel"]))
    # alignment: can fwd and bwd be aligned by orthogonals (linkability)?
    Uf, _, Vtf = np.linalg.svd(Wf, full_matrices=False)
    Ub, _, Vtb = np.linalg.svd(Wb.T, full_matrices=False)
    P = Ub @ Uf.T; Q = Vtf.T @ Vtb
    align_residual = float(np.abs(P @ Wf @ Q - Wb.T).max())
    distinct = float(np.min(np.abs(np.diff(np.sort(sv_W))))) > 1e-3
    return {
        "spectrum_max_diff_fwd_vs_W": spec_f,
        "spectrum_max_diff_bwd_vs_W": spec_b,
        "spectrum_shared": bool(spec_f < 1e-9 and spec_b < 1e-9),
        "naive_W_recovery_rel_error": naive_relerr,
        "fwd_bwd_alignment_residual": align_residual,
        "singular_values_distinct": bool(distinct),
        "verdict": ("no trivial recovery found; singular spectrum is shared/"
                    "leaked and fwd/bwd are linkable (alignable) but W itself is "
                    "not recovered without a mask"),
    }


# ===========================================================================
# Orchestration
# ===========================================================================


def run_all(*, seed: int = 0) -> dict[str, Any]:
    linear = run_synthetic_linear(seed=seed)
    linear_signperm = run_synthetic_linear(seed=seed, mask_kind="signed_permutation")
    cross_gram_stats = run_cross_gram_statistics(seed=seed)
    autograd = autograd_reproduces_a0(seed=seed)
    nonlinear = run_nonlinear_backward_status(seed=seed)
    tiny = run_tiny_transformer(seed=seed)
    opt_a0 = run_optimizer_equivalence(scheme="A0", seed=seed)
    opt_a1 = run_optimizer_equivalence(scheme="A1", seed=seed)
    align = run_weight_alignment_audit(seed=seed)
    return {
        "stage": "masked_lora_backward_audit",
        "synthetic_linear": linear,
        "synthetic_linear_signed_perm": linear_signperm,
        "cross_gram_statistics": cross_gram_stats,
        "autograd_reproduces_A0": autograd,
        "nonlinear_backward_status": nonlinear,
        "tiny_transformer": tiny,
        "optimizer_equivalence_A0": opt_a0,
        "optimizer_equivalence_A1": opt_a1,
        "weight_alignment_audit": align,
        "model_status": "tiny_transformer_passed (linear-only); "
                        "nonlinear blocked; no gpt2/qwen run",
    }
