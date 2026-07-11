# O1 — optimizer equivalence formulas (verified)

Row-vector convention (`y = x W`, activations are rows). A masked LoRA leaf is a
two-sided transform of the plaintext leaf: `theta_t = P · theta · Q` with `P` from
the input/rank mask and `Q` from the output/rank mask. For the two adapters, under
the frozen convention `A_tilde = N_in^{-1} A U`, `B_tilde = U^{-1} B N_out`.

## Gradient chain (verified, err ≤ 2e-15)
    grad_theta_t = P^{-T} · grad_theta · Q^{-T}
i.e. for the adapters:
    grad_A_tilde = N_in^{T} · grad_A · U^{-T}
    grad_B_tilde = U^{-T} · grad_B · N_out^{T}

## Update that recovers plaintext SGD (verified, err ≤ 6e-15 across cond 1–32)
    theta_t_next = theta_t − lr · (P P^T) · grad_theta_t · (Q^T Q)
For the adapters this is exactly the spec:
    A_tilde_next = A_tilde − lr · (N_in^T N_in)^{-1} · grad_A_tilde · (U^T U)
    B_tilde_next = B_tilde − lr · (U^T U)^{-1} · grad_B_tilde · (N_out^T N_out)

## Key numerical facts (correctness.csv, trajectory.csv)
- **Orthogonal masks (cond=1):** `P P^T = I`, so the preconditioned update and the
  NAIVE update both equal plaintext SGD (err 6.7e-16). No Gram is used.
- **Non-orthogonal masks (cond>1):** the naive masked SGD is WRONG (err 0.11 → 0.20
  for cond 1.5 → 32); only the preconditioned update recovers plaintext SGD
  (err 8e-16 → 6e-15). The needed Gram `P P^T` is far from `I` (‖·−I‖ 0.9 → 660).
- **10-step trajectory:** O1-A 1.8e-15, O1-B 2.7e-15, O1-C 0.0 max abs loss diff vs
  plaintext — all three are correct; they differ only in leakage + invocation count.

## RMSNorm constraint (verified)
`rmsnorm_core(x N) == rmsnorm_core(x) N` iff `N N^T = I`. Measured invariance error:
4e-16 (orthogonal) vs 0.48 → 30 (cond 1.5 → 32). Under `paper_safe` (0 trusted
nonlinear) the residual/hidden mask the norm sees MUST be orthogonal; non-orthogonal
masking is confined to sub-bases that skip RMSNorm (attention `R`, V `S`, vocab `D`).
