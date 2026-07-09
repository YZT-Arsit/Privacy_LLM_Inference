# Public-weight prompt-privacy audit — summary

## 1. Claim under test
Under a **public-weight** adversary (attacker holds the full plaintext model and
runs it as a forward oracle), the exact-lossless construction — protecting the
residual stream with a **hidden-dimension dense orthogonal mask `Q`** rather than
token-level permutations — protects the user prompt from **token-level recovery**,
at the cost of leaking some structural information (FFN neuron permutation / norm).

## 2. What IS (claimed to be) protected
- token-level prompt content recovery.

## 3. What is NOT protected (measured, not hidden)
- **norm leakage**: `||H Q|| = ||H||` for orthogonal `Q` — per-token norm profile
  is preserved. Measured rank-corr H vs H_obs = **1.000000**.
- **weight-alignment / FFN neuron permutation leakage**: recoverable structural info.
- **model-structure leakage**: architecture and per-layer Gram are exposed.
- **prompt-length leakage**: token count is visible (candidates of other lengths
  are trivially separable — see V4).

## 4. Correctness — SiLU/Amulet boundary (V1)
- fp64 max_abs **1.06e-13** (exact to rounding);
  fp32-IO max_abs 5.01e-06;
  native fp32 R-factor build **failed_as_expected**.
- negative control (dense P cannot cross SiLU): max_abs
  1.89e+00 (large ✓).
- **survives: True**

## 5. FFN permutation leakage (V2)
- Attacker GIVEN the ground-truth FFN neuron permutation.
- token recovery top1 = **0.00e+00** (random baseline
  1.99e-05); would be
  1.000 if `Q` were known.
- **FFN permutation does not unlock `Q`. survives: True**

## 6. ISA / internal-state inversion (V3)
- no-known-plaintext token top1 = **0.00e+00** (~0).
- **STATIC `Q` is broken by a known-plaintext attack**: token recovery → 1.0 at
  ~768 known rows (an approximate `Q`
  already argmaxes correctly), and `Q` is recovered **exactly** (~1e-14) once
  known rows ≥ d (1024 rows).
- fresh `Q` per session resists (top1 0.00e+00).
- **survives only without known-plaintext / with fresh Q: True**

## 7. Forward-matching oracle (V4) — the crux
- A small set: true ranks #1/6
  (residual 9.59e-15 vs margin 1.34e-03).
- B large set: top1 = True.
- C sensitive-token top1 = **1.00**.
- D multi-layer (fresh Q per layer): combined true rank #1.
- Fresh-Q ablation: residuals identical under any orthogonal Q → **freshness does not help**.
- **oracle_succeeds = True**

## 8. Efficiency (V5)
- dense orthogonal mask is a full [d,d] matmul (vs a signed-perm gather);
  Amulet-SiLU island is the dominant cost. fp64 exact; bf16 unsupported.
- classification: **stronger_audit_variant_not_main_path**.

## 9. Final decision: **NOT_SUPPORTED_UNDER_FORWARD_ORACLE**
Public plaintext weights enable candidate-verification attacks against observed obfuscated states. The construction does not provide token-level prompt privacy against a public-weight adversary without additional freshness or observation-limiting assumptions -- and freshness of Q does NOT help, because the token-token Gram H H^T is invariant to any orthogonal Q.

## 10. Exact wording allowed for the paper
> Public plaintext weights enable candidate-verification attacks against observed obfuscated states. The construction does not provide token-level prompt privacy against a public-weight adversary without additional freshness or observation-limiting assumptions -- and freshness of Q does NOT help, because the token-token Gram H H^T is invariant to any orthogonal Q.

### Separation of leakage axes (required)
- token-level content recovery: **BROKEN by forward oracle**
- norm / structure leakage: **present** (rank-corr 1.0000)
- weight-alignment leakage: **present** (FFN neuron permutation)
- model-structure leakage: **present**
- prompt semantic leakage: candidate-verification identifies the prompt / sensitive token.
