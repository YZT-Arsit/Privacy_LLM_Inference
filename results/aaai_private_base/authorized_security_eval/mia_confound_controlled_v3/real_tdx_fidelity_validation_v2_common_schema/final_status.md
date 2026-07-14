# REAL_TDX_COMMON_SCHEMA_FIDELITY_WITHHELD

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

The 610-column common schema has no semantic mismatch and no arbitrary post-hoc deletion. Source-fidelity gate: **FAIL** (raw AUC 0.805; normalized AUC 0.672, 95% CI [0.627, 0.718]; maximum univariate symmetric AUC 0.760).

Per the predeclared protocol, frozen classifier transfer was not run because source fidelity failed.

Blocker: failed normalized source-fidelity control. No real-TDX recollection, shadow training, full three-shadow MIA rerun, historical-output modification, registry update, or commit occurred.
