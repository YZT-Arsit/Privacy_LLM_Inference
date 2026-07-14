# G1/G2 Three-Seed Paired Statistics

All results use existing frozen, sample-aligned generations. No inference was rerun. Positive deltas mean G2 minus the seed-matched G1.

| Metric | Mean delta ± seed SD | Hierarchical 95% CI | s7 | s1234 | s2025 |
|---|---:|---:|---:|---:|---:|
| BLEU | -0.414 ± 1.011 | [-1.605, 0.634] | -0.113 | 0.413 | -1.541 |
| chrF | -1.245 ± 0.865 | [-2.148, -0.359] | -0.421 | -2.146 | -1.169 |
| ROUGE_L | -0.315 ± 0.793 | [-1.208, 0.600] | -0.498 | 0.553 | -1.001 |
| length | -1.455 ± 1.338 | [-2.867, -0.337] | -1.118 | -2.930 | -0.318 |
| repetition_pct | 0.016 ± 0.763 | [-0.664, 0.797] | -0.108 | -0.678 | 0.833 |
| invalid_pct | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 |

## Interpretation

These are exploratory three-seed intervals. With only three trained-model seeds, the seed-level uncertainty is necessarily imprecise; intervals are not equivalence tests.

The raw per-seed means, paired intervals, effect sizes, and sign counts are in `per_seed_paired_metrics.csv`.
