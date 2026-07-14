# Simulated versus real-TDX comparison

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

The real-TDX result is paper-facing; simulation is a larger controlled companion. Cross-domain differences are descriptive and not paired inference.

| View | Simulated LOSO AUC | Real-TDX LOSO AUC | Real − simulated |
|---|---:|---:|---:|
| V0 | 0.675 | 0.675 | -0.000 |
| V2_combined | 0.479 | 0.459 | -0.020 |
| V0_plus_V2 | 0.494 | 0.473 | -0.021 |

## Semantic-group-disjoint sensitivity

| View | Simulated AUC | Real-TDX AUC | Real − simulated |
|---|---:|---:|---:|
| V0 | 0.610 | 0.605 | -0.005 |
| V2_combined | 0.504 | 0.487 | -0.017 |
| V0_plus_V2 | 0.521 | 0.504 | -0.017 |
