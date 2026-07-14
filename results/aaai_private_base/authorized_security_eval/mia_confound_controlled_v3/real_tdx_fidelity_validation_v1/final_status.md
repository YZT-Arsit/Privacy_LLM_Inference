# REAL_TDX_MIA_FIDELITY_VALIDATION_WITHHELD

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

Decision rule: **C**

Simulation-to-TDX fidelity validation is inconclusive because the collection paths are not semantically matched.

- Completed primary real-TDX collection: 250/250.
- Preserved corrective recapture: 250/250.
- Frozen classifier transfer: complete; real-TDX ROC-AUC 0.466.
- Failed gates: `hidden.final.last_l2` semantic collection mismatch, source-classifier AUC 1.000, and one disjoint simulated/real range.
- Remaining work: re-freeze a schema containing only terminal features actually materialized by both paths, regenerate the simulated classifier freeze, and then repeat this validation.
