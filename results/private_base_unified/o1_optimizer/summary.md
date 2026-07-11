# O1 — optimizer equivalence + leakage: summary

CPU/fp64 analysis (no H800/TDX). Verifies the three optimizer profiles for
masked-domain LoRA-SGD under the private-base convention and audits what each leaks.

## Correctness (all three recover plaintext SGD)
| profile | masks | update | 10-step max |Δloss| vs plaintext | trusted inv/step |
|---|---|---|---|---|
| O1-A | orthogonal | direct masked SGD | 1.8e-15 | 2 |
| O1-B | non-orthogonal | preconditioned `(PP^T)·g·(Q^TQ)` | 2.7e-15 | 2 |
| O1-C | non-orthogonal | trusted optimizer (packed grads) | 0.0 | 3 |
(correctness.csv: preconditioned recovers at cond 1–32; naive masked SGD fails for all
non-orthogonal, err 0.11 → 0.20. trajectory.csv, invocation_count.csv.)

## Two verified structural findings
1. **RMSNorm ⇒ orthogonal residual mask under paper_safe.** Invariance holds only for
   `N N^T = I`. Non-orthogonal masking is confined to attention `R`, V `S`, vocab `D`;
   the residual weight chain stays orthogonally masked.
2. **O1-B leaks the mask Gram** (`PP^T`, `Q^TQ`) onto the GPU — strictly more than the
   self-Gram attack extracts. Given (1), the left feature-Gram is trivial (`=I`) for
   q/k/v/gate/up (orthogonal residual input), but `S^TS` (o_proj) and `U^TU` (rank) are
   exposed. **O1-B is not paper-safe as written.**

## Recommendation (pre-W1)
- Use **O1-A** as the correctness control and the default direct-SGD profile.
- Use **O1-C** as the deployable non-orthogonal reference (+1 crossing, no Gram leak).
- Do **not** ship **O1-B** until a custom backward is shown to hide the Gram from an
  offline package attacker (presumed insufficient).
- The private-base *weight-chain* security is decided by **W1**, not by O1: under
  paper_safe the residual chain is orthogonal regardless of optimizer, so if W1 breaks
  the orthogonal chain, the fix is more layers behind the TEE, not a different optimizer.

## Not claimed
No security is claimed for any profile here. O1-B is explicitly not paper-safe.
Non-orthogonal-mask security is not claimed before W1/S1/A1.
