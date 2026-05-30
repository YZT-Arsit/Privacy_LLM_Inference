# Privacy LLM Obfuscation — Nonlinear Island Security Proxies (Stage 5.2)

## Experiment scope

Three lightweight proxies on the operator-compatible mask scheme used by the Stage 5.2 nonlinear islands. None of these proves formal security — they are naive-observer bounds, recorded so the paper can quote them directly under the security section.

## Permutation Recovery Proxy

fixed_permutation lets per-channel signatures align across sessions, so a naive attacker can match columns above chance. fresh_permutation_per_session breaks that alignment and drives top1 back toward 1/H. dense_sandwich_reference adds dense linear mixing on both sides and erases the column statistics entirely.

| strategy | top-1 recovery | top-5 recovery | mean signature error |
|---|---|---|---|
| fixed_permutation | 0.218750 | 0.625000 | 0.035143 |
| fresh_permutation_per_session | 0.123047 | 0.470703 | 0.071086 |
| permutation_pool | 0.113281 | 0.507812 | 0.054967 |
| dense_sandwich_reference | 0.015625 | 0.074219 | 0.315930 |

- Random-chance top-1 baseline at hidden_size=64: ``≈ 0.0156``.

## Island Linkability Proxy

fixed_perm_no_pad is the highest naive linkability because the GPU-visible tensor is a deterministic function of the plaintext. Adding pad at the Linear boundary collapses the value-level stability; sandwiching with a fresh dense mask removes the coordinate-multiset signal entirely.

| strategy | mean pair-cos | mean pair-L2 |
|---|---|---|
| fixed_perm_no_pad | 1.000e+00 | 0 |
| fixed_perm_with_linear_boundary_pad | 0.517313 | 3.216e+01 |
| fresh_perm_with_linear_boundary_pad | 0.008269 | 4.619e+01 |
| dense_to_perm_to_dense_sandwich | 0.003546 | 3.333e+01 |

- Linkability rank (high → low): fixed_perm_no_pad > fixed_perm_with_linear_boundary_pad > fresh_perm_with_linear_boundary_pad > dense_to_perm_to_dense_sandwich

## Mask Family Security Accounting

| mask family | where used | preserved invariants | leakage note |
|---|---|---|---|
| dense_invertible | Linear / Attention / KV cache boundaries | none beyond invertibility | Strong linear mixing; right multiply by a fresh dense N erases the per-channel signal. |
| orthogonal | RMSNorm core | row L2 norm (||X N||_2 = ||X||_2) | Norm-preserving by design — observer can recover row norms. |
| mean_preserving_orthogonal | LayerNorm core | row mean (X N · 1 = X · 1) AND row centered L2 norm | Mean + centered-norm preserved by design; an attacker observing many samples sees stable per-row mean and centered norm. |
| permutation | Activation island (GELU / ReLU / SiLU) | coordinate-value multiset (the sorted set of channel values is unchanged) | Permutation islands hide channel identity but do not hide coordinate-value multisets. Same multiset across sessions → permutation can be recovered by per-channel statistics if the permutation is reused. |
| paired_permutation | SwiGLU island (shared P for up and gate branches) | paired coordinate multiset for (up, gate) tuples | Same multiset leakage as permutation; additionally exposes that the up- and gate-branches use the *same* P (paired alignment). |

- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Permutation islands hide channel identity but do not hide coordinate-value multisets.
- Security depends on freshness, dense-mask sandwiching, and pad at Linear boundaries.
- Real TEE isolation is not implemented in this stage.

## Interpretation

- Fixed permutation top-1 = **0.219** vs fresh = **0.123** vs dense sandwich = **0.016**. Fresh permutation per session removes the cross-session signature alignment; the dense sandwich erases the column statistics entirely.
- Island linkability: `fixed_perm_no_pad` mean cosine = **1.0000** vs `fresh_perm_with_linear_boundary_pad` = **0.0083**. Linear-boundary pad collapses naive linkability.
- Mask family accounting (Proxy 3) catalogues the per-family leakage profile — *what is preserved by design*, not what an adversary can or cannot recover beyond that.

## Limitations

- These are proxy attacks, not adaptive learned attacks.
- Fresh permutation reduces stable statistical recovery but does not provide dense linear mixing.
- Coordinate-value multiset leakage exists inside activation islands.
- Security depends on limiting island lifetime and sandwiching with dense masks.
- No real TEE isolation is implemented.

## Next Stage Plan

- **Stage 5.3** — Adaptive / learned-inverter attacker that uses more than per-channel statistics (e.g. low-rank reconstruction, supervised inverter trained against a known model).
- **Stage 6.4** — Qwen / TinyLlama migration. The nonlinear-island proxies here motivate the freshness + dense-sandwich rules that the Qwen wrapper will be required to enforce.
