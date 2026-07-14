# Old-Cohort Confound Attribution

This report attributes the 446 frozen disjoint-range features. No model inference was run.

| Tensor family | Disjoint features | Share |
|---|---:|---:|
| attention_scores | 213 | 47.8% |
| masked_v | 72 | 16.1% |
| masked_q | 66 | 14.8% |
| masked_k | 64 | 14.3% |
| transformed_hidden | 23 | 5.2% |
| masked_logits | 8 | 1.8% |

| Analysis category | Disjoint features | Share |
|---|---:|---:|
| hidden norm | 225 | 50.4% |
| attention extrema | 122 | 27.4% |
| attention entropy | 55 | 12.3% |
| attention moments | 36 | 8.1% |
| masked logits | 8 | 1.8% |

## Attribution conclusion

The signal is broad rather than isolated to one layer or one statistic. Attention-score features are the largest tensor family, while norm-preserving hidden/Q/K/V summaries also contribute heavily. All 446 are transform-invariant, so per-prompt coordinate refresh cannot remove them.

Both classes used the same capture code, run ID, package root, adapter hash, dtype, and tensor presence. Collection-time model state is therefore not supported as the explanation. The old design still cannot separate train/test semantic distribution from true membership because source split and membership are identical. References were not inputs to the prefill feature capture, so direct target/reference-field leakage is not supported.
