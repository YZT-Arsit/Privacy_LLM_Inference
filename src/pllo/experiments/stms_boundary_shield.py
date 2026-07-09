"""Variant F — Selective Token-Mixing Shield (STMS), a BOUNDARY shield.

Idea
----
V4 showed that the observable residual boundary ``H_obs = H N`` (right hidden-dim
mask ``N``) leaks the token-token Gram ``H H^T`` because ``(HN)(HN)^T = H H^T``.
STMS additionally mixes the TOKEN dimension with a fresh matrix ``A`` sampled per
batch/window, only at the observable boundary::

    U = A (H N)                 # optionally with secret high-energy shield rows

``A`` is NEVER carried into attention / RMSNorm / RoPE / KV cache. Before any real
transformer computation or KV materialisation, the masked state is recovered::

    H N = A^{-1} U              # (shield rows, if any, are dropped here)

So STMS is a *boundary* shield, not a full-transformer left mixing.

Threat model (READ THIS — it is strictly weaker than V4's)
----------------------------------------------------------
STMS protects an observer of the shielded artifact ``U`` who (a) does not know
``A`` / the shield rows, and (b) does NOT observe the post-recovery state
``H N``. This models a passive observer of a transmitted/logged boundary snapshot
— NOT the compute-offload adversary of V4, who must be handed ``H N`` to run the
offloaded linear algebra. **If the adversary sees ``H N`` (the offload model),
STMS adds nothing** — the V4 / weight-recovery attacks apply to ``H N`` unchanged.
This limitation is surfaced everywhere and must not be hidden.

Security summary (measured in this module)
------------------------------------------
* orthogonal ``A``: ``U U^T = A (H H^T) A^T`` is an orthogonal *similarity*, so the
  eigenVALUES of ``H H^T`` (equivalently the singular values of ``H``) are
  INVARIANT. A spectrum-matching oracle still ranks candidates. -> partial only.
* non-orthogonal ``A``: breaks the spectrum, but a free-``A`` oracle overfits
  (any full-rank candidate fits), so security reduces to BSS/ICA hardness — a
  statistical, not information-theoretic, guarantee (report median AND p95 cosine).
* shield rows dilute BSS at a throughput cost (extra rows).
* block-local ``A`` (block size ``b``) is cheaper but attackable per block.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from pllo.experiments.public_weight_prompt_audit import (
    orthogonal_procrustes_residual,
    qr_orthogonal,
)

__all__ = [
    "token_mix_orthogonal", "token_mix_nonorthogonal", "token_mix_blocklocal",
    "sample_shield_rows", "stms_forward", "stms_recover",
    "oracle_spectrum_residual", "oracle_free_AQ_residual",
    "ica_bss_recovery", "stms_report_fields",
]


# ---------------------------------------------------------------------------
# token-dimension mixing matrices A (fresh per batch/window)
# ---------------------------------------------------------------------------


def token_mix_orthogonal(m: int, *, seed: int, dtype=torch.float64) -> torch.Tensor:
    """Fresh orthogonal token-mix ``A`` [m,m] (QR of a Gaussian)."""
    return qr_orthogonal(m, seed=seed, dtype=dtype)


def token_mix_nonorthogonal(
    m: int, *, seed: int, cond: float = 5.0, dtype=torch.float64
) -> torch.Tensor:
    """Fresh invertible token-mix ``A`` with a prescribed condition number ``cond``.

    ``A = U diag(s) V^T`` with ``s`` log-spaced in ``[1, cond]`` so ``cond(A)=cond``.
    """
    u = qr_orthogonal(m, seed=seed, dtype=dtype)
    v = qr_orthogonal(m, seed=seed + 1, dtype=dtype)
    s = torch.logspace(0, float(torch.log10(torch.tensor(cond))), m, dtype=dtype)
    return (u * s.unsqueeze(0)) @ v.transpose(0, 1)


def token_mix_blocklocal(
    m: int, *, block: int, seed: int, orthogonal: bool = True,
    cond: float = 5.0, dtype=torch.float64
) -> torch.Tensor:
    """Block-diagonal token-mix: independent [b,b] mixers on token blocks."""
    a = torch.zeros(m, m, dtype=dtype)
    for bi, start in enumerate(range(0, m, block)):
        end = min(start + block, m)
        bs = end - start
        if orthogonal:
            blk = qr_orthogonal(bs, seed=seed * 100 + bi, dtype=dtype)
        else:
            blk = token_mix_nonorthogonal(bs, seed=seed * 100 + bi, cond=cond, dtype=dtype)
        a[start:end, start:end] = blk
    return a


def sample_shield_rows(k: int, d: int, *, scale: float, seed: int, dtype=torch.float64) -> torch.Tensor:
    """Secret high-energy shield rows ``S`` [k,d] (never enter attention/KV)."""
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(k, d, dtype=dtype, generator=g)


# ---------------------------------------------------------------------------
# STMS forward + recovery (the boundary roundtrip)
# ---------------------------------------------------------------------------


def stms_forward(
    h: torch.Tensor, n: torch.Tensor, a: torch.Tensor | None,
    *, shield: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Shield the observable boundary. Returns ``U`` and bookkeeping.

    ``h`` [m,d] plaintext hidden; ``n`` [d,d] hidden-dim mask; ``a`` token-mix
    ([m,m] or [(m+k),(m+k)] when shielding); ``shield`` [k,d] optional.
    """
    hn = h @ n                                              # H N (existing right mask)
    m = h.shape[0]
    if shield is not None:
        full = torch.cat([hn, shield @ n], dim=0)          # [(m+k), d]; shield also masked by N
    else:
        full = hn
    if a is None:
        u = full                                           # Case A baseline: U = H N
    else:
        u = a @ full                                       # U = A (H N [; S N])
    return {"u": u, "hn": hn, "num_real_tokens": m,
            "has_shield": shield is not None,
            "num_shield_rows": 0 if shield is None else shield.shape[0]}


def stms_recover(u: torch.Tensor, a: torch.Tensor | None, *, num_real_tokens: int) -> torch.Tensor:
    """Recover ``H N`` before real compute: ``full = A^{-1} U`` then drop shield rows."""
    full = u if a is None else torch.linalg.solve(a, u)
    return full[:num_real_tokens]


# ---------------------------------------------------------------------------
# forward-matching oracles against the shielded boundary
# ---------------------------------------------------------------------------


def oracle_spectrum_residual(h_cand: torch.Tensor, u: torch.Tensor) -> float:
    """Two-sided orthogonal oracle: ``min_{A,Q orth} ||A H_cand Q - U||`` equals the
    L2 distance between the sorted singular-value spectra. This is the exact,
    principled attack against an orthogonal token-mix (it ignores A and N entirely).
    """
    su = torch.linalg.svdvals(u.double())
    sh = torch.linalg.svdvals(h_cand.double())
    k = min(su.numel(), sh.numel())
    # pad the shorter (shield rows add singular values) — compare the top-k
    su, sh = su[:k], sh[:k]
    denom = float(su.norm()) + 1e-30
    return float((su - sh).norm() / denom)


def _orthonormal_complement(b: torch.Tensor) -> torch.Tensor:
    """Orthonormal complement [d, d-m] of an orthonormal-column ``b`` [d, m]."""
    d, m = b.shape
    filler = torch.randn(d, d - m, dtype=b.dtype)
    q, _ = torch.linalg.qr(torch.cat([b, filler], dim=1))
    return q[:, m:]


def oracle_free_AQ_residual(h_cand: torch.Tensor, u: torch.Tensor) -> float:
    """Unconstrained oracle ``min_{A free, Q orth} ||A H_cand Q - U||``.

    With a free ``[m,m]`` ``A`` AND a free orthogonal ``Q``, the row space of
    ``H_cand`` can be rotated (by ``Q``) onto the row space of ``U`` and then
    remixed (by ``A``), so for ``m <= d`` full-rank ``H_cand`` the residual is ~0
    for ANY candidate. This demonstrates why the free-``A,Q`` oracle OVERFITS and
    cannot discriminate — the attacker must constrain ``A`` (spectrum / ICA / block),
    which is exactly what the other oracles do.
    """
    hc = h_cand.double()
    ud = u.double()
    if ud.shape[0] != hc.shape[0]:
        return float("inf")
    # orthonormal bases of the two row spaces (right singular vectors)
    _, _, vhc = torch.linalg.svd(hc, full_matrices=False)   # [m, d]
    _, _, vhu = torch.linalg.svd(ud, full_matrices=False)
    c = vhc.transpose(0, 1)                                   # [d, m] rowspace(H_cand)
    ub = vhu.transpose(0, 1)                                  # [d, m] rowspace(U)
    cperp = _orthonormal_complement(c)
    ubperp = _orthonormal_complement(ub)
    q = c @ ub.transpose(0, 1) + cperp @ ubperp.transpose(0, 1)   # orthogonal, Q Ub = C
    hcq = hc @ q                                              # rowspace(H_cand Q) == rowspace(U)
    a_hat = torch.linalg.lstsq(hcq.transpose(0, 1), ud.transpose(0, 1)).solution.transpose(0, 1)
    return float((a_hat @ hcq - ud).norm() / (ud.norm() + 1e-30))


def _rank_of(residuals: list[float], true_idx: int) -> int:
    order = sorted(range(len(residuals)), key=lambda i: residuals[i])
    return order.index(true_idx)


def candidate_attack(
    hidden_of, candidates: list[str], true_idx: int, *,
    case: str, n: torch.Tensor, seed: int = 0, cond: float = 5.0,
    block: int | None = None, shield_k: int = 0, shield_scale: float = 5.0,
) -> dict[str, Any]:
    """Run a forward-matching candidate attack for one STMS case.

    ``hidden_of(prompt) -> [m,d]``. Uses the baseline Procrustes oracle for
    Case A and the spectrum oracle otherwise (the strongest mask-agnostic attack).
    """
    h_true = hidden_of(candidates[true_idx])
    m, d = h_true.shape
    a = _build_A(case, m + shield_k, seed=seed, cond=cond, block=block)
    shield = None
    if shield_k > 0:
        shield = sample_shield_rows(shield_k, d, scale=shield_scale, seed=seed + 7, dtype=h_true.dtype)
    u = stms_forward(h_true, n, a, shield=shield)["u"]

    resids = []
    for c in candidates:
        hc = hidden_of(c)
        if case == "baseline":
            resids.append(orthogonal_procrustes_residual(hc, u))
        else:
            resids.append(oracle_spectrum_residual(hc, u))
    rank = _rank_of(resids, true_idx)
    order = sorted(range(len(resids)), key=lambda i: resids[i])
    margin = (sorted(resids)[1] - sorted(resids)[0]) if len(resids) > 1 else float("inf")
    return {
        "case": case, "num_candidates": len(candidates),
        "true_rank": rank, "top1": bool(rank == 0), "top5": bool(rank < 5),
        "true_residual": resids[true_idx], "residual_margin": margin,
        "recovered_idx": order[0],
    }


def _build_A(case: str, m: int, *, seed: int, cond: float, block: int | None):
    if case == "baseline":
        return None
    if case == "orthogonal":
        return token_mix_orthogonal(m, seed=seed)
    if case == "nonorthogonal":
        return token_mix_nonorthogonal(m, seed=seed, cond=cond)
    if case == "blocklocal":
        return token_mix_blocklocal(m, block=block or 32, seed=seed, orthogonal=True)
    raise ValueError(f"unknown case {case!r}")


# ---------------------------------------------------------------------------
# BSS / ICA recovery probe (statistical attack)
# ---------------------------------------------------------------------------


def ica_bss_recovery(u: torch.Tensor, hn_true: torch.Tensor, *, seed: int = 0) -> dict[str, Any]:
    """FastICA attempt to unmix ``U = A (H N)`` back to the rows of ``H N``.

    Reports median AND p95 of the best-match |cosine| between recovered sources
    and the true masked rows (higher = better recovery = worse for the defender).
    Returns NN-anchored recovery as well.
    """
    from sklearn.decomposition import FastICA

    m = hn_true.shape[0]
    x = u[:m].double().transpose(0, 1).numpy()             # [d, m]: d samples, m mixed signals
    try:
        ica = FastICA(n_components=m, random_state=seed, max_iter=1000, whiten="unit-variance")
        s = ica.fit_transform(x)                           # [d, m] recovered sources
    except Exception as exc:  # noqa: BLE001
        return {"ica_status": "failed", "error": str(exc)[:120]}
    rec = torch.tensor(s.T)                                # [m, d] recovered rows
    true = hn_true.double()
    # best |cosine| match per recovered source (greedy over true rows)
    rn = rec / (rec.norm(dim=1, keepdim=True) + 1e-30)
    tn = true / (true.norm(dim=1, keepdim=True) + 1e-30)
    cos = (rn @ tn.transpose(0, 1)).abs()                  # [m, m]
    best = cos.max(dim=1).values                           # best match per source
    best_sorted = best.sort().values
    p95 = float(best_sorted[max(0, int(0.95 * (m - 1)))])
    return {
        "ica_status": "ok", "num_sources": m,
        "median_cosine": float(best.median()),
        "p95_cosine": p95,
        "max_cosine": float(best.max()),
    }


# ---------------------------------------------------------------------------
# report fields
# ---------------------------------------------------------------------------


def stms_report_fields(
    *, roundtrip_max_abs: float | None = None,
    exact_lossless_tests_passed: bool = False,
) -> dict[str, Any]:
    return {
        "variant": "selective_token_mixing_shield",
        "aliases": ["stms_boundary_shield", "variant_f"],
        "uses_token_dim_mixing": True,
        "fresh_per_batch_mixing": True,
        "crosses_attention": False,
        "crosses_rmsnorm": False,
        "crosses_rope": False,
        "crosses_kv_cache": False,
        "shield_rows_optional": True,
        "is_boundary_shield_not_full_left_mixing": True,
        "protects_compute_visible_HN": False,     # only the shielded artifact U
        "threat_model": "observer of U without A; NOT the offload adversary who sees H N",
        "exact_lossless_claim": bool(exact_lossless_tests_passed) and "only_if_recovered_before_compute",
        "formal_security_claim": False,
        "production_qwen7b_integration": False,
        **({"roundtrip_max_abs": float(roundtrip_max_abs)} if roundtrip_max_abs is not None else {}),
    }


# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------


def _time(fn, *, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def efficiency(n: int, d: int, *, block: int = 128, cond: float = 5.0) -> dict[str, Any]:
    dt = torch.float64
    h = torch.randn(n, d, dtype=dt)
    nmask = qr_orthogonal(d, seed=0, dtype=dt)
    hn = h @ nmask
    a_o = token_mix_orthogonal(n, seed=1, dtype=dt)
    a_no = token_mix_nonorthogonal(n, seed=1, cond=cond, dtype=dt)
    a_bl = token_mix_blocklocal(n, block=block, seed=1, dtype=dt)
    return {
        "n": n, "d": d, "block": block,
        "gen_A_orthogonal_ms": _time(lambda: token_mix_orthogonal(n, seed=2, dtype=dt), iters=10),
        "gen_A_nonorth_ms": _time(lambda: token_mix_nonorthogonal(n, seed=2, cond=cond, dtype=dt), iters=10),
        "mix_ms": _time(lambda: a_o @ hn),
        "unmix_orth_ms": _time(lambda: a_o.transpose(0, 1) @ (a_o @ hn)),   # orth: A^-1=A^T
        "unmix_solve_ms": _time(lambda: torch.linalg.solve(a_no, a_no @ hn)),
        "mix_blocklocal_ms": _time(lambda: a_bl @ hn),
        "baseline_HN_ms": _time(lambda: h @ nmask),
    }
