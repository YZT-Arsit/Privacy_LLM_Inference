# SURROGATE_FIDELITY_PILOT

**SIMULATED_PROTOCOL_VIEW — not REAL_TDX_BACKED_VIEW.**

**Pilot verdict: FAIL.** source_identity_perfectly_separable_in_fixed_length_audit.

This bounded pilot measures agreement on a coarse generated-output length property. It does not reconstruct exact text and is not evidence of full model extraction.

## 1% query budget (executed first)

| View | Queries | Agreement | Macro-F1 | Cross-entropy |
|---|---:|---:|---:|---:|
| V0_output_only | 5 | 0.4667 | 0.4461 | 2.5700 |
| V2_protocol_faithful_simulated_gpu | 5 | 0.5556 | 0.5373 | 4.8078 |
| V0_plus_V2 | 5 | 0.5778 | 0.5639 | 4.6223 |
| plaintext_label_positive_control | 5 | 1.0000 | 1.0000 | 0.0014 |
| shape_only_negative_control | 5 | 0.5611 | 0.5247 | 2.0443 |
| random_gaussian_matched_control | 5 | 0.3426 | 0.3097 | 1.7438 |
| V2_shuffled_sample_feature_control | 5 | 0.3278 | 0.3139 | 9.0636 |

Every cell uses a random-init 1024→64→3 MLP with the same 300-step optimization budget. Train, validation, and evaluator groups are disjoint under exact text and word-bigram Jaccard ≥0.90 grouping; IDs are ordered/aligned with no duplicates; preprocessing is fit only on each queried training prefix. Label shuffle, sample-feature shuffle, Gaussian, shape-only, and source-identity controls are recorded. Because the fixed-length source classifier is perfect, this pilot fails and no V2 utility claim is permitted.
