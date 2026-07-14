# Surrogate fidelity evaluation v2

**Under evaluated views and budgets, V2 does not provide consistent measurable additional surrogate utility.**

**SIMULATED_PROTOCOL_VIEW. This is not model extraction and does not establish black-box security.**

| Budget | Queries | V0 | V2 | V0+V2 | V2−V0 |
|---:|---:|---:|---:|---:|---:|
| 1% | 5 | 0.4667 | 0.5815 | 0.5537 | +0.1148 [-0.0833, +0.2222] |
| 5% | 25 | 0.6259 | 0.6167 | 0.6333 | -0.0093 [-0.0500, +0.0667] |
| 10% | 50 | 0.6815 | 0.5907 | 0.6630 | -0.0907 [-0.1111, -0.0722] |
| 20% | 100 | 0.7630 | 0.6481 | 0.7500 | -0.1148 [-0.1333, -0.1000] |

Target agreement, task accuracy, and output agreement all denote exact agreement on the preregistered coarse output-length class; exact generated-text reconstruction and KL are not applicable. All views use the same model, 500-sample pool, split, random-init architecture, optimizer, and steps. Preprocessing is fit on attack-training records only. Shuffled outputs, Gaussian features, random mapping, and the trusted plaintext label positive control are included in `metrics.csv`.
