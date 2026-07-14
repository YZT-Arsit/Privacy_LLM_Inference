# REAL_TDX_V2_FIDELITY_VALIDATION

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

| Evaluation | ROC-AUC | Balanced accuracy | Average precision | Max(TPR−FPR) | Score corr. vs sim |
|---|---:|---:|---:|---:|---:|
| Simulated same 250 | 0.468 | 0.505 | 0.540 | 0.035 | 1.000 |
| Real TDX primary | 0.466 | 0.474 | 0.538 | 0.033 | 0.888 |
| Real TDX pre-final sensitivity | 0.466 | 0.486 | 0.537 | 0.028 | 0.905 |

Fidelity audit: 1 invalid collection mismatch, 3 distribution-shifted features, 1 disjoint-range feature, raw source-classifier AUC 1.000. V0 agreement is 250/250.
