"""Orthogonal masked SGD / Momentum-SGD exactness audit (Gate 1.5-E, CPU fp64).

Frozen math (per LoRA layer), masks N_in, N_out, U all ORTHOGONAL and FIXED per
training session:
    A_t = N_in^-1 A U ,  B_t = U^-1 B N_out
    masked-domain grads (computed by the GPU from masked activations):
        gA_t = X_t^T (U_t B_t^T) = N_in^-1 gradA U     (orthogonal collapse of N_in^T . U^-T)
        gB_t = (X_t A_t)^T U_t   = U^-1 gradB N_out     (X_t=X N_in, U_t=U N_out since orthogonal)
    pure SGD (GPU only, no unmask):
        A_t <- A_t - lr gA_t   ==   N_in^-1 (A - lr gradA) U      (exact)
    momentum:
        mA_t <- beta mA_t + (1-damp) gA_t  ;  A_t <- A_t - lr (nesterov? gA_t+beta mA_t : mA_t)
        mA_t == N_in^-1 mA U   (exact)

Under ORTHOGONALITY the independent-backward form N_in^T gradA U^-T collapses to the
orthogonal form N_in^-1 gradA U (N_in^T=N_in^-1, U^-T=U). This module verifies that
collapse and the full multi-step equivalence, plus negative controls (mask refresh
without re-encoding; dense non-orthogonal GL). No trusted optimizer, no packed update.
"""

from __future__ import annotations

import torch

DT = torch.float64


def _gen(seed):
    g = torch.Generator(); g.manual_seed(int(seed) & 0x7FFFFFFF); return g


def orthogonal(n, g, dtype=DT):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=dtype))
    return q * torch.sign(torch.diagonal(r))            # unique orthogonal


def permutation(n, g, dtype=DT):
    return torch.eye(n, dtype=dtype)[:, torch.randperm(n, generator=g)]


def gl(n, g, dtype=DT, cond=8.0):
    q1 = orthogonal(n, g, dtype); q2 = orthogonal(n, g, dtype)
    s = torch.logspace(0, torch.log10(torch.tensor(cond)), n, dtype=dtype)
    return q1 @ torch.diag(s) @ q2.T


def _mask(kind, n, g, dtype=DT):
    if kind == "orthogonal":
        M = orthogonal(n, g, dtype); return M, M.T           # inv = transpose
    if kind == "permutation":
        M = permutation(n, g, dtype); return M, M.T
    if kind == "gl":
        M = gl(n, g, dtype); return M, torch.linalg.inv(M)
    raise ValueError(kind)


def orthogonal_collapse_check(m=6, d_in=12, d_out=10, r=4, seed=0) -> dict:
    """Verify the independent-backward masked grad N_in^T gradA U^-T equals the
    orthogonal form N_in^-1 gradA U when masks are orthogonal, and differs for GL."""
    g = _gen(seed)
    X = torch.randn(m, d_in, generator=g, dtype=DT)
    U_up = torch.randn(m, d_out, generator=g, dtype=DT)
    B = torch.randn(r, d_out, generator=g, dtype=DT)
    gradA = X.T @ (U_up @ B.T)
    out = {}
    for kind in ("orthogonal", "gl"):
        N_in, N_in_inv = _mask(kind, d_in, g)
        Umask, Umask_inv = _mask(kind, r, g)
        indep = N_in.T @ gradA @ Umask_inv.T          # general independent-backward form
        orth = N_in_inv @ gradA @ Umask               # orthogonal form
        out[kind] = float((indep - orth).abs().max())
    return {"orthogonal_forms_agree": bool(out["orthogonal"] < 1e-10),
            "gl_forms_differ": bool(out["gl"] > 1e-3),
            "orthogonal_diff": out["orthogonal"], "gl_diff": out["gl"]}


def run_alignment(*, optimizer="sgd", mask_kind="orthogonal", steps=50, m=6,
                  d_in=12, d_out=10, r=4, lr=0.02, momentum=0.0, dampening=0.0,
                  nesterov=False, weight_decay=0.0, wd_mode="coupled", accum=1,
                  refresh_masks=False, seed=0) -> dict:
    """Run plaintext vs GPU-masked SGD/momentum in lockstep; return per-step errors.

    The masked trainer NEVER unmasks: it holds A_t/B_t (and masked momentum) and
    updates them in the masked domain only. Equivalence is checked by unmasking for
    the test. refresh_masks=True (negative control) re-samples masks each step
    WITHOUT re-encoding the masked state -> must break."""
    g = _gen(seed)
    W = torch.randn(d_in, d_out, generator=g, dtype=DT) / d_in ** 0.5
    T = torch.randn(m, d_out, generator=g, dtype=DT)
    A0 = torch.randn(d_in, r, generator=g, dtype=DT) / d_in ** 0.5
    B0 = torch.zeros(r, d_out, dtype=DT)
    # shared synthetic input per step
    Xs = [torch.randn(m, d_in, generator=g, dtype=DT) for _ in range(steps * accum)]

    # plaintext state
    A, B = A0.clone(), B0.clone()
    mA = torch.zeros_like(A); mB = torch.zeros_like(B)
    # masked state (GPU domain)
    N_in, N_in_inv = _mask(mask_kind, d_in, g)
    Umask, Umask_inv = _mask(mask_kind, r, g)
    N_out, N_out_inv = _mask(mask_kind, d_out, g)
    A_t = N_in_inv @ A @ Umask
    B_t = Umask_inv @ B @ N_out
    mA_t = torch.zeros_like(A_t); mB_t = torch.zeros_like(B_t)

    rows = []
    idx = 0
    for step in range(steps):
        if refresh_masks:            # NEGATIVE: new masks but keep old masked state (no re-encode)
            N_in, N_in_inv = _mask(mask_kind, d_in, g)
            Umask, Umask_inv = _mask(mask_kind, r, g)
            N_out, N_out_inv = _mask(mask_kind, d_out, g)
        # accumulate grads
        gA = torch.zeros_like(A); gB = torch.zeros_like(B)
        gA_t = torch.zeros_like(A_t); gB_t = torch.zeros_like(B_t)
        for _ in range(accum):
            X = Xs[idx]; idx += 1
            G = X @ (W + A @ B) - T                     # plaintext upstream grad
            gA += X.T @ (G @ B.T); gB += (X @ A).T @ G
            # GPU masked-domain grad from masked activations
            X_t = X @ N_in
            U_t = G @ N_out_inv.T
            P_t = X_t @ A_t
            gA_t += X_t.T @ (U_t @ B_t.T); gB_t += (P_t.T @ U_t)
        if weight_decay and wd_mode == "coupled":
            gA = gA + weight_decay * A; gB = gB + weight_decay * B
            gA_t = gA_t + weight_decay * A_t; gB_t = gB_t + weight_decay * B_t
        # optimizer update
        if optimizer == "sgd" or momentum == 0.0:
            stepA, stepB = gA, gB
            stepA_t, stepB_t = gA_t, gB_t
        else:
            mA = momentum * mA + (1 - dampening) * gA
            mB = momentum * mB + (1 - dampening) * gB
            mA_t = momentum * mA_t + (1 - dampening) * gA_t
            mB_t = momentum * mB_t + (1 - dampening) * gB_t
            if nesterov:
                stepA, stepB = gA + momentum * mA, gB + momentum * mB
                stepA_t, stepB_t = gA_t + momentum * mA_t, gB_t + momentum * mB_t
            else:
                stepA, stepB = mA, mB
                stepA_t, stepB_t = mA_t, mB_t
        if weight_decay and wd_mode == "decoupled":     # multiplicative linear decay
            A = (1 - lr * weight_decay) * A - lr * stepA
            B = (1 - lr * weight_decay) * B - lr * stepB
            A_t = (1 - lr * weight_decay) * A_t - lr * stepA_t
            B_t = (1 - lr * weight_decay) * B_t - lr * stepB_t
        else:
            A = A - lr * stepA; B = B - lr * stepB
            A_t = A_t - lr * stepA_t; B_t = B_t - lr * stepB_t
        # unmask masked state for comparison (test only)
        A_un = N_in @ A_t @ Umask_inv
        B_un = Umask @ B_t @ N_out_inv
        mA_un = N_in @ mA_t @ Umask_inv
        # RELATIVE errors (the masking algebra is exact up to fp64 rounding, which is
        # relative; absolute error is meaningless if a config's magnitudes grow).
        def rel(a, b):
            return float((a - b).abs().max() / (b.abs().max() + 1e-12))
        probe = Xs[0]
        pred_p = probe @ (W + A @ B); pred_m = probe @ (W + A_un @ B_un)
        loss_p = 0.5 * float((pred_p - T).pow(2).mean())
        loss_m = 0.5 * float((pred_m - T).pow(2).mean())
        rows.append({
            "step": step + 1,
            "A_err": rel(A_un, A), "B_err": rel(B_un, B),
            "dW_err": rel(A_un @ B_un, A @ B), "momentum_err": rel(mA_un, mA),
            "loss_diff": abs(loss_p - loss_m) / (abs(loss_p) + 1e-12),
            "next_logit_err": rel(pred_m, pred_p),
        })
    worst = {k: max(rw[k] for rw in rows) for k in
             ("A_err", "B_err", "dW_err", "momentum_err", "loss_diff", "next_logit_err")}
    aligned = (worst["A_err"] < 1e-9 and worst["dW_err"] < 1e-9
               and worst["momentum_err"] < 1e-9 and worst["next_logit_err"] < 1e-9)
    return {"optimizer": optimizer, "mask_kind": mask_kind, "steps": steps,
            "momentum": momentum, "nesterov": nesterov, "dampening": dampening,
            "weight_decay": weight_decay, "wd_mode": wd_mode, "accum": accum,
            "refresh_masks": refresh_masks, "per_step": rows, "worst": worst,
            "aligned": bool(aligned)}


# ---- invocation-count model for the two profiles (protocol-level) ----
def invocation_counts(optimizer_mode: str, num_lora_layers: int) -> dict:
    if optimizer_mode in ("gpu_masked_sgd", "gpu_masked_momentum_sgd"):
        return {"optimizer_mode": optimizer_mode, "num_lora_layers": num_lora_layers,
                "input_invocation": 1, "loss_invocation": 1, "packed_update_invocation": 0,
                "optimizer_trusted_invocation": 0, "nonlinear_trusted_invocation": 0,
                "layerwise_trusted_invocation": 0, "total_trusted_invocations": 2,
                "worker_to_trusted_returns_after_input": 1}
    if optimizer_mode == "trusted_adamw":
        return {"optimizer_mode": optimizer_mode, "num_lora_layers": num_lora_layers,
                "input_invocation": 1, "loss_invocation": 1, "packed_update_invocation": 1,
                "optimizer_trusted_invocation": 1, "nonlinear_trusted_invocation": 0,
                "layerwise_trusted_invocation": 0, "total_trusted_invocations": 3,
                "worker_to_trusted_returns_after_input": 2}
    raise ValueError(optimizer_mode)


def run_audit() -> dict:
    return {
        "collapse": orthogonal_collapse_check(),
        "sgd_orthogonal": run_alignment(optimizer="sgd", mask_kind="orthogonal", steps=100),
        "sgd_permutation": run_alignment(optimizer="sgd", mask_kind="permutation", steps=50),
        "momentum_orthogonal": run_alignment(optimizer="momentum_sgd", momentum=0.9,
                                             lr=0.003, mask_kind="orthogonal", steps=100),
        "momentum_permutation": run_alignment(optimizer="momentum_sgd", momentum=0.9,
                                              lr=0.003, mask_kind="permutation", steps=50),
        "nesterov_orthogonal": run_alignment(optimizer="momentum_sgd", momentum=0.9,
                                             lr=0.003, nesterov=True, mask_kind="orthogonal", steps=50),
        "wd_coupled": run_alignment(optimizer="sgd", weight_decay=0.01, wd_mode="coupled",
                                    mask_kind="orthogonal", steps=50),
        "wd_decoupled": run_alignment(optimizer="sgd", weight_decay=0.01, wd_mode="decoupled",
                                      mask_kind="orthogonal", steps=50),
        "accum2": run_alignment(optimizer="sgd", accum=2, mask_kind="orthogonal", steps=30),
        "accum4": run_alignment(optimizer="momentum_sgd", momentum=0.9, lr=0.003, accum=4,
                                mask_kind="orthogonal", steps=30),
        # negative controls (must NOT align)
        "neg_refresh_masks": run_alignment(optimizer="sgd", mask_kind="orthogonal",
                                           refresh_masks=True, steps=20),
        "neg_non_orthogonal_gl": run_alignment(optimizer="sgd", mask_kind="gl", steps=20),
        "invocations": {"gpu_masked_sgd": invocation_counts("gpu_masked_sgd", 32),
                        "gpu_masked_momentum_sgd": invocation_counts("gpu_masked_momentum_sgd", 32),
                        "trusted_adamw": invocation_counts("trusted_adamw", 32)},
    }
