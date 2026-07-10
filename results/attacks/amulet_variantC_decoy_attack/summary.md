# (a) Variant C — Statistical Real-vs-Decoy Column Distinguishing

**Question:** does variant C's selector hiding survive when the GPU-visible lift
`z` [m, d·k] is observed? The attacker sees `z` only and tries to identify the `d`
real columns (`z[:, valid] = x[:,j]`) among the `d·(k-1)` decoys.

**Method:** per-column statistics an attacker can compute from `z` alone — column
`norm`, `var`, `kurtosis`, `maxabs/norm` — scored for real-vs-decoy separation.
Metrics: **AUC** (best single statistic, direction-agnostic) and **top-1 selector
recovery** (calibrated/oracle-direction, within each group of `k`). Varied: decoy
mode, `k`, input distribution. `m=64, d=32`. float64. Script + CSV in this dir.

## Results

| input | decoy mode | k | best stat | AUC | top-1 recovery | chance |
|---|---|---|---|---|---|---|
| gaussian | naive | 2/4/8 | norm | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** | .50/.25/.125 |
| gaussian | unit_norm | 2/4/8 | kurtosis | 0.58 / 0.61 / 0.55 | 0.56 / 0.38 / 0.13 | — |
| gaussian | colnorm_matched | 2/4/8 | kurtosis | 0.53 / 0.53 / 0.55 | ~chance | — |
| gaussian | coord | 2/4/8 | var | 0.60 / 0.55 / 0.53 | ~chance | — |
| **structured** | naive | 2/4/8 | norm | **0.99 / 1.00 / 0.98** | **1.00 / 0.97 / 0.94** | — |
| **structured** | unit_norm | 2/4/8 | norm | **0.88 / 0.88 / 0.88** | **0.91 / 0.88 / 0.88** | — |
| structured | colnorm_matched | 2/4/8 | kurtosis | 0.59 / 0.58 / 0.57 | ~chance | — |
| structured | coord | 2/4/8 | kurtosis | 0.59 / 0.54 / 0.51 | ~chance | — |

("structured" = transformer-like activations: a fraction of **outlier channels**
with 6–20× scale + channel correlations, mimicking real pre-activation statistics.)

## Findings

1. **Naive decoys → selector fully recovered.** Decoy columns (`x @ randn`) have
   ~√d× the norm of a real column (a single coordinate), so a column-**norm** scan
   separates them perfectly: **AUC 1.0, top-1 1.0** for both inputs. The
   "secure-R" hiding collapses completely.
2. **Realistic activations leak even with unit-norm decoys.** For **structured**
   inputs, unit-norm decoys still give **AUC ≈ 0.88, top-1 ≈ 0.88** — the outlier
   channels make real columns (single channels) stand out against normalized
   decoy combinations. Real transformer activations are known to have large
   outlier features, so this is the realistic regime.
3. **Only distribution-matched decoys reach ~chance.** `colnorm_matched` and
   `coord` decoys bring AUC to ~0.51–0.60 and top-1 to chance — but:
   - a **residual higher-moment (kurtosis) signal** remains (AUC ~0.55–0.59):
     a real column is a *single coordinate*, a decoy is a *combination* (more
     Gaussian by CLT), so they differ in tails;
   - **`coord` decoys** (each decoy = another real coordinate of `x`) hide the
     selector best, but then **every column of `z` is a coordinate of `x`** — the
     full pre-activation is exposed; only the column-to-coordinate *grouping* is
     hidden.

## Verdict

**Variant C's selector hiding does NOT robustly survive when `z` is visible.** For
realistic (outlier-heavy) activations a trivial column-norm attack recovers the
selector (AUC ≈ 0.88–1.0) unless decoys are drawn to match the real activation
distribution — which requires the TEE to know that distribution and, at the
hiding-optimal limit (`coord` decoys), exposes all pre-activation values anyway. A
residual higher-moment signal (single-coordinate vs combination) persists even
under norm-matching.

This is the empirical confirmation of the earlier dichotomy and CP-uniqueness
theory: **an offloaded nonlinear leaks the pre-activation, and its selector hiding
rests on an assumption that is statistically fragile against realistic data.** Note
(stronger attacker, not run here): in the full masked-execution scheme the previous
Linear's output `x̃ = x·N_in` is also GPU-visible; correlating `z`'s pure-coordinate
columns against `x̃`'s mixed columns would strengthen the attack further.

**Scope reminder (from (b)):** under our paper's public-weights threat model this is
a **user-data / activation-privacy** finding, already reported by the paper as
proxy-only / `needs_more_evaluation`. It is **not** a violation of any `I(X;O)=0`
theorem — no such theorem exists in the repo.
