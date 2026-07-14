# Paper-facing real-TDX MIA table

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

Primary classifier: L2 logistic regression; values are macro mean ± population SD over three held-out shadows.

| View | ROC-AUC | Balanced accuracy | Average precision | TPR@1% FPR | Advantage |
|---|---:|---:|---:|---:|---:|
| V0 | 0.675 ± 0.015 | 0.635 ± 0.003 | 0.637 ± 0.020 | 0.025 ± 0.008 | 0.289 ± 0.013 |
| V2_attention | 0.480 ± 0.004 | 0.492 ± 0.010 | 0.481 ± 0.004 | 0.006 ± 0.007 | 0.015 ± 0.006 |
| V2_kv | 0.464 ± 0.007 | 0.500 ± 0.002 | 0.474 ± 0.010 | 0.009 ± 0.002 | 0.007 ± 0.002 |
| V2_hidden | 0.494 ± 0.014 | 0.500 ± 0.007 | 0.497 ± 0.000 | 0.014 ± 0.009 | 0.032 ± 0.012 |
| V2_logit | 0.507 ± 0.009 | 0.498 ± 0.002 | 0.514 ± 0.004 | 0.019 ± 0.006 | 0.042 ± 0.016 |
| V2_combined | 0.459 ± 0.016 | 0.497 ± 0.002 | 0.474 ± 0.005 | 0.011 ± 0.003 | 0.008 ± 0.004 |
| V0_plus_V2 | 0.473 ± 0.015 | 0.501 ± 0.002 | 0.484 ± 0.004 | 0.011 ± 0.003 | 0.019 ± 0.008 |

Secondary attacker-capability summary: `max(V0, V2, V0+V2)` ROC-AUC = **0.675** (V0).
