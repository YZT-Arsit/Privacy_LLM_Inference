# STMS BSS/ICA recovery risk

- orthogonal-A baseline ICA: median cos 0.205, p95 cos 0.287

## Block-size sweep (orthogonal block-local)

| block | median_cos | p95_cos |
|---|---|---|
| 16 | 0.207 | 0.281 |
| 32 | 0.204 | 0.283 |
| 64 | 0.205 | 0.286 |

## Non-orthogonal condition-number sweep

| cond | unmix_max_abs | median_cos | p95_cos |
|---|---|---|---|
| 2.0 | 4.7e-13 | 0.205 | 0.286 |
| 5.0 | 4.8e-13 | 0.206 | 0.290 |
| 10.0 | 6.9e-13 | 0.203 | 0.273 |
| 50.0 | 9.1e-13 | 0.211 | 0.291 |
| 200.0 | 1.3e-12 | 0.201 | 0.291 |

## Shield sweep (p95 cosine range over scales)

| shield_fraction | best_p95 | worst_p95 |
|---|---|---|
| 0.0 | 0.287 | 0.287 |
| 0.01 | 0.281 | 0.309 |
| 0.03 | 0.278 | 0.284 |
| 0.05 | 0.277 | 0.290 |
| 0.1 | 0.262 | 0.284 |
