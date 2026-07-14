# G2 Cost Delta versus Seed-Matched G1

| Seed | Train slowdown | Generation wall ratio | Total slowdown | Adapter Δ KiB |
|---:|---:|---:|---:|---:|
| 7 | 61.65× | 0.773× | 24.34× | 57.0 |
| 1234 | 61.59× | 0.741× | 24.39× | 56.4 |
| 2025 | 63.21× | 0.783× | 24.83× | 58.0 |

Generation wall ratios are descriptive, not a latency benchmark: frozen outputs have seed/method-dependent token counts. No timing run was launched.
