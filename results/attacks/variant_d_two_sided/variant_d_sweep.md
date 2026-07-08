# Variant D — two-sided non-orthogonal exact keymat (dim=256)

Exact for the LINEAR chain (no noise, no RMSNorm correction). Non-orthogonal P does not pass RMSNorm/nonlinearity exactly (scope: linear chain).

## Lambda sweep

| lambda | cond(P) | max_abs_err fp64 | fp32 | bf16 | surfaceA norm_corr | anchor gramL | anchor VMA top1 | mapping recovered | mask matmul ms |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 5.8e-13 | 7.6e-05 | 5.0e-01 | 1.000 | 0.652 | 0.191 | False | 0.0496 |
| 0.01 | 1.0 | 5.9e-13 | 6.9e-05 | 5.0e-01 | 1.000 | 0.654 | 0.190 | False | 0.0589 |
| 0.03 | 1.1 | 7.1e-13 | 6.5e-05 | 5.0e-01 | 0.999 | 0.658 | 0.190 | False | 0.0495 |
| 0.1 | 1.3 | 7.0e-13 | 6.9e-05 | 5.0e-01 | 0.990 | 0.703 | 0.190 | False | 0.0494 |
| 0.3 | 2.3 | 6.1e-13 | 8.0e-05 | 5.0e-01 | 0.928 | 0.997 | 0.189 | False | 0.0581 |
| 1 | 140.0 | 3.2e-12 | 6.7e-04 | 3.1e+00 | 0.743 | 124.915 | 0.189 | False | 0.0597 |

## Anchor-attack comparison (lambda=0.1, folded weight, public-weight anchor)

| variant | gram_right_rel_err | gram_left_rel_err | VMA col top1 | ArrowMatch | mapping recovered | stable-state unmask err |
|---|---|---|---|---|---|---|
| signed_perm | 0.000 | 0.000 | 1.000 | 0.543 | True | 0.0e+00 |
| dense_right_orthogonal | 0.030 | 0.000 | 0.263 | 0.000 | False | — |
| two_sided_nonorthogonal_exact | 0.008 | 0.703 | 0.190 | 0.000 | False | — |

**Reading.** `signed_perm` (A_rightmul production surface) is fully recovered (mapping acc 1.0 → exact stable-state un-mask). ObfuscaTune's right-orthogonal fold leaves the LEFT row-Gram invariant (gram_left≈0). Variant D scrambles BOTH Grams and no anchor attack recovers a mapping — while staying exact (fp64/fp32) for the linear chain. Surface-A norm_corr falls as lambda grows (non-orthogonal distorts per-token norm) but that same non-orthogonality breaks RMSNorm exactness — the tension formalised in docs/rmsnorm_exact_norm_impossibility.md. requires_plaintext_weight_anchor=True: none of these attacks applies under a private-weights deployment.
