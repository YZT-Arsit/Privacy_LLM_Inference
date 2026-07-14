# Orientation-invariant MIA diagnostic

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

Raw AUC remains the primary cross-shadow generalization result. Symmetric AUC is a secondary conservative diagnostic that permits post-hoc score inversion.

| View | Raw AUC | Symmetric AUC | Group-disjoint raw AUC | Group-disjoint symmetric AUC |
|---|---:|---:|---:|---:|
| V0 | 0.675 | 0.675 | 0.605 | 0.605 |
| V2_combined | 0.459 | 0.541 | 0.487 | 0.518 |
| V0_plus_V2 | 0.473 | 0.527 | 0.504 | 0.510 |

## Paired symmetric-AUC differences

| Comparison | Fold effects | Equal-weight macro | 95% CI |
|---|---|---:|---|
| V2_combined_minus_V0 | -0.110, -0.157, -0.133 | -0.133 | withheld: prediction-level scores absent |
| V0_plus_V2_minus_V0 | -0.124, -0.174, -0.146 | -0.148 | withheld: prediction-level scores absent |

## Interpretation

- Raw AUC below 0.5 indicates unstable or reversed score orientation on a held-out shadow.
- Symmetric AUC is a conservative diagnostic allowing score inversion after observing orientation.
- Neither raw nor symmetric AUC permits interpreting V2 below 0.5 as active protection.
- The supported conclusion is only that V2 does not outperform the external V0 view.
- Paired bootstrap CIs are withheld because their exact calculation requires the frozen sample-level predictions, which were not persisted. No classifier was retrained.
