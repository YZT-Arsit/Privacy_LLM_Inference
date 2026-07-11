# O1 — leakage implications

## The central tension (verified)
Non-orthogonal feature masks are the *candidate* defence against orthogonal-Gram
alignment. But two independent, verified constraints block them under `paper_safe`:

1. **RMSNorm forbids a non-orthogonal residual mask.** `rmsnorm_core(xN)=rmsnorm_core(x)N`
   iff `N N^T = I` (measured invariance error 0.48 → 30 for cond 1.5 → 32). The
   residual/hidden mask the norm sees must be orthogonal, so the residual-basis
   weight chain `W_tilde_l = N_l^{-1} W_l N_{l+1}` is **orthogonally** masked — the
   exact family the self-Gram attack recovers. Non-orthogonality can only live in
   sub-bases that skip the norm (attention `R`, V `S`, vocab `D`), which do NOT
   protect the residual weight chain.

2. **The O1-B optimizer re-exposes the Gram it was meant to hide.** Where a mask *is*
   non-orthogonal (`S`, `U`), the update `(P P^T) grad (Q^T Q)` puts the mask Gram on
   the GPU in closed form — strictly more than the self-Gram attack extracts.

## Consequence for profile selection
- **O1-A (orthogonal, direct SGD)** — correct, 0 extra crossings, no Gram exposed,
  BUT provides no protection against self-Gram weight-chain alignment. It is a
  correctness control, not a private-weight defence. Whether it is *secure enough*
  is the W1 go/no-go question (attacker recovering the orthogonal chain).
- **O1-B (non-orthogonal, preconditioned, GPU)** — correct, 2 crossings, but leaks
  the mask Gram on the GPU. **NOT paper-safe** as written. Only salvageable if a
  custom backward can apply the preconditioner without materialising the Gram AND an
  offline package attacker still cannot infer it — unresolved, presumed insufficient.
- **O1-C (non-orthogonal, trusted optimizer)** — correct, 3 crossings, no Gram on the
  GPU. This is the profile that can *actually* carry non-orthogonal feature masks
  into deployment, at the cost of one extra trusted crossing per step.

## What this does NOT settle
Even O1-C's non-orthogonal masking only helps in the sub-bases where it is allowed;
the residual weight chain stays orthogonal under `paper_safe` regardless of optimizer.
So the real security of the private-base *weight chain* is decided by **W1** (can the
attacker recover a functional model from the orthogonally-masked chain?) — not by the
optimizer choice. If W1 shows the orthogonal chain is recoverable, private-base under
`paper_safe` needs a stronger primitive than feature masking (e.g. more layers behind
the TEE), independent of O1.

## Verdict
- O1-A: paper-safe-eligible, **weak** private-weight defence (pending W1).
- O1-B: **not paper-safe** (Gram leak); do not mark secure.
- O1-C: paper-safe-eligible, carries non-orthogonal masks, +1 crossing; the deployable
  reference to compare against O1-A in S1.
Do not report O1-A correctness together with O1-B/C non-orthogonal security as if one
profile.
