# Orientation-invariant audit provenance

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

This directory is append-only and derives deterministic symmetric-AUC point estimates from the previously frozen CSV metrics. Existing analysis files were not modified. No classifier fit, prediction regeneration, GPU operation, or data recollection occurred.

Inputs:

- `per_fold_metrics.csv`: `43aa40484641c68bfed74fd6f2e091535843e47b6008749347134a58082e3e1d`
- `group_disjoint_metrics.csv`: `b1bb4f4a61e241edd476c2fa33873ee528ca0a82b2905fda5230ce4795cd3091`
- `confidence_intervals.json`: `1dd685992d66f233883df1c7a28903d459a0bc260065221331b8fe9fdd3f7933`
