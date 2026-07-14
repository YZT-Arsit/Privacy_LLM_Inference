# Collection validation

All required pre-CPU gates passed.

| Shadow | Length-matched per class | Presence AUC | Missingness AUC | Shape-only AUC | Source AUC (p) | Shuffle mean | Pseudo mean | Disjoint ranges |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 433 | 0.500 | 0.500 | 0.476 | 0.500 (1.000) | 0.499 | 0.484 | 0 |
| 2 | 448 | 0.500 | 0.500 | 0.504 | 0.500 (1.000) | 0.488 | 0.483 | 0 |
| 3 | 444 | 0.500 | 0.500 | 0.474 | 0.500 (1.000) | 0.518 | 0.514 | 0 |

The shadow-ID diagnostic accuracy is 1.0. No shadow/run/session/checkpoint/path metadata is present in the matrix; this diagnostic therefore measures separability of independently trained model states, not a metadata collection confound.

Exact and semantic duplicate groups are all singletons. No global preprocessing was fitted before a split. Final membership estimation remains delegated to the CPU owner.
