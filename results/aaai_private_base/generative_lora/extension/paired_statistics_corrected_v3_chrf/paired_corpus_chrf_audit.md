# Corrected paired corpus-chrF audit (v1)

The earlier hierarchical interval resampled per-example sentence-chrF differences. This version recomputes corpus chrF after every paired bootstrap draw.

| Seed | G1 corpus chrF | G2 corpus chrF | Corpus Δ | Mean sentence-chrF Δ | Paired corpus-Δ 95% CI |
|---:|---:|---:|---:|---:|---:|
| 7 | 72.41 | 72.16 | -0.26 | -0.421 | [-1.039, +0.535] |
| 1234 | 72.78 | 70.56 | -2.22 | -2.146 | [-3.011, -1.433] |
| 2025 | 72.43 | 71.26 | -1.17 | -1.169 | [-2.055, -0.249] |

## Three-seed hierarchical result

- Mean of the three seed-level corpus-chrF deltas: -1.217.
- Hierarchical paired corpus-chrF 95% CI: [-2.264, -0.216].
- Earlier sentence-chrF hierarchical CI: [-2.148, -0.359].

All pairs contain exactly 500 aligned examples with identical references and input hashes; no missing or duplicate IDs were found.
