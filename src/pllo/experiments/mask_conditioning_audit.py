"""CPU-only mask numerical-stability / condition-number audit (Section E).

Sweeps dense-mask condition number and reports correctness / recovered-gradient
error, solve residual, finite-ness, and norm amplification for masked linear and
masked LoRA forward+backward. float64 on CPU. Condition-number trends here are
CPU-float64 observations; they are NOT a promise about bf16/GPU ranges.
"""

from __future__ import annotations

import torch

from pllo.experiments.cpu_correctness_audit import DT, err, gen, make_mask, rel_fro

CONDS = (1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1e3, 1e4)


def _mask_for(family, dim, seed, cond):
    if family == "permutation":
        return make_mask("permutation", dim, seed=seed, cond=1.0)
    if family == "orthogonal":
        return make_mask("orthogonal", dim, seed=seed, cond=1.0)
    if family == "diagonal":
        return make_mask("positive_diagonal", dim, seed=seed, cond=cond)
    return make_mask("dense_gl", dim, seed=seed, cond=cond)   # well/ill-conditioned dense GL


def e_linear_conditioning(m=8, d_in=16, d_out=16, seed=0) -> list[dict]:
    rows = []
    g = gen(seed)
    X = torch.randn(m, d_in, generator=g, dtype=DT)
    W = torch.randn(d_in, d_out, generator=g, dtype=DT) / d_in ** 0.5
    GY = torch.randn(m, d_out, generator=g, dtype=DT)      # upstream grad
    for family in ("permutation", "orthogonal", "diagonal", "dense_gl"):
        conds = (1.0,) if family in ("permutation", "orthogonal") else CONDS
        for cond in conds:
            N_in, N_in_inv, c_in = _mask_for(family, d_in, seed + 1, cond)
            N_out, N_out_inv, c_out = _mask_for(family, d_out, seed + 2, cond)
            X_t = X @ N_in
            W_t = torch.linalg.solve(N_in, W) @ N_out
            Y_t = X_t @ W_t
            Y_ref = (X @ W) @ N_out
            # backward recovery: grad wrt W recovered from masked domain
            GY_t = GY @ N_out                                  # inject masked upstream grad
            gW_t = X_t.T @ GY_t                                # masked grad
            gW_rec = N_in_inv.T @ gW_t @ N_out_inv             # recover: N_in^{-T} gW_t N_out^{-1}
            gW_ref = X.T @ GY
            solve_res = (N_in @ torch.linalg.solve(N_in, W) - W).abs().max().item()
            fwd_ma, _ = err(Y_t, Y_ref)
            grad_ma, _ = err(gW_rec, gW_ref)
            finite = bool(torch.isfinite(Y_t).all() and torch.isfinite(gW_rec).all())
            rows.append({
                "section": "E_linear", "family": family, "cond_target": cond,
                "cond_in_measured": round(c_in, 3), "cond_out_measured": round(c_out, 3),
                "forward_max_abs": fwd_ma, "forward_rel_fro": rel_fro(Y_t, Y_ref),
                "recovered_grad_max_abs": grad_ma, "solve_residual": solve_res,
                "finite": finite,
                "norm_amplification": round((torch.linalg.norm(X_t) /
                                             (torch.linalg.norm(X) + 1e-30)).item(), 4),
                "max_intermediate_magnitude": round(X_t.abs().max().item(), 4),
                "pass": bool(fwd_ma < 1e-6 and grad_ma < 1e-6 and finite),
            })
    return rows


def e_lora_conditioning(m=8, d_in=16, d_out=16, r=4, seed=1) -> list[dict]:
    rows = []
    g = gen(seed)
    X = torch.randn(m, d_in, generator=g, dtype=DT)
    W = torch.randn(d_in, d_out, generator=g, dtype=DT) / d_in ** 0.5
    A = torch.randn(d_in, r, generator=g, dtype=DT) / d_in ** 0.5
    B = torch.randn(r, d_out, generator=g, dtype=DT) / r ** 0.5
    GY = torch.randn(m, d_out, generator=g, dtype=DT)
    for family in ("permutation", "orthogonal", "dense_gl"):
        conds = (1.0,) if family in ("permutation", "orthogonal") else CONDS
        for cond in conds:
            N_in, N_in_inv, c_in = _mask_for(family, d_in, seed + 1, cond)
            N_out, N_out_inv, _ = _mask_for(family, d_out, seed + 2, cond)
            R, R_inv, _ = _mask_for("dense_gl" if family == "dense_gl" else family, r, seed + 3, cond)
            X_t = X @ N_in
            A_t = N_in_inv @ A @ R
            B_t = R_inv @ B @ N_out
            W_t = N_in_inv @ W @ N_out
            Y_t = X_t @ (W_t + A_t @ B_t)
            Y_ref = (X @ (W + A @ B)) @ N_out
            # recover gradA from masked domain: masked output is Y_t = Y N_out, so the
            # injected upstream grad is G_Yt = GY N_out^{-T}; then gradA_t = N_in^T gradA R^{-T}.
            G_Yt = GY @ N_out_inv.T
            gA_t = X_t.T @ G_Yt @ B_t.T
            gA_rec = N_in_inv.T @ gA_t @ R.T
            gA_ref = X.T @ GY @ B.T
            fwd_ma, _ = err(Y_t, Y_ref)
            grad_ma, _ = err(gA_rec, gA_ref)
            finite = bool(torch.isfinite(Y_t).all() and torch.isfinite(gA_rec).all())
            rows.append({
                "section": "E_lora", "family": family, "cond_target": cond,
                "cond_in_measured": round(c_in, 3),
                "forward_max_abs": fwd_ma, "recovered_gradA_max_abs": grad_ma,
                "finite": finite,
                "pass": bool(fwd_ma < 1e-6 and grad_ma < 1e-6 and finite),
            })
    return rows


def recommended_condition_range(rows: list[dict]) -> dict:
    """Report the largest condition number that still holds fp64 correctness."""
    safe = [r["cond_target"] for r in rows
            if r["family"] == "dense_gl" and r["forward_max_abs"] < 1e-6
            and r.get("recovered_grad_max_abs", r.get("recovered_gradA_max_abs", 1)) < 1e-6]
    worst = [(r["cond_target"], r["forward_max_abs"]) for r in rows if r["family"] == "dense_gl"]
    return {
        "cpu_fp64_safe_max_condition": max(safe) if safe else None,
        "dense_gl_forward_error_by_cond": worst,
        "caveat": "CPU float64 observation only; do NOT extrapolate to bf16/fp16 or GPU.",
    }


def run_conditioning() -> dict:
    lin = e_linear_conditioning()
    lora = e_lora_conditioning()
    return {
        "E_linear": lin,
        "E_lora": lora,
        "recommendation": recommended_condition_range(lin + lora),
    }
