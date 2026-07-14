# Paper-facing MIA table

Collection mode: SIMULATED_PROTOCOL_VIEW

Primary classifier: L2 logistic regression; values are macro mean ± fold SD.

| View | ROC-AUC | Balanced accuracy | Average precision | TPR@1% FPR | Advantage |
|---|---:|---:|---:|---:|---:|
| V0 | 0.675 ± 0.015 | 0.635 ± 0.003 | 0.637 ± 0.020 | 0.025 ± 0.008 | 0.289 ± 0.013 |
| V2_attention | 0.479 ± 0.004 | 0.492 ± 0.010 | 0.481 ± 0.004 | 0.006 ± 0.007 | 0.015 ± 0.006 |
| V2_kv | 0.482 ± 0.013 | 0.501 ± 0.001 | 0.489 ± 0.009 | 0.009 ± 0.007 | 0.006 ± 0.004 |
| V2_hidden | 0.499 ± 0.011 | 0.498 ± 0.002 | 0.498 ± 0.003 | 0.011 ± 0.008 | 0.041 ± 0.019 |
| V2_logit | 0.507 ± 0.009 | 0.498 ± 0.002 | 0.514 ± 0.004 | 0.019 ± 0.006 | 0.042 ± 0.016 |
| V2_combined | 0.479 ± 0.015 | 0.499 ± 0.001 | 0.486 ± 0.011 | 0.005 ± 0.006 | 0.005 ± 0.005 |
| V0_plus_V2 | 0.494 ± 0.010 | 0.499 ± 0.001 | 0.497 ± 0.010 | 0.005 ± 0.004 | 0.017 ± 0.016 |

Semantic-template-group-disjoint sensitivity (five subfolds pooled within each held-out shadow):

| View | ROC-AUC | Balanced accuracy | Average precision | TPR@1% FPR | Advantage |
|---|---:|---:|---:|---:|---:|
| V0 | 0.610 ± 0.008 | 0.581 ± 0.007 | 0.603 ± 0.018 | 0.029 ± 0.011 | 0.176 ± 0.010 |
| V2_combined | 0.504 ± 0.006 | 0.502 ± 0.014 | 0.508 ± 0.006 | 0.011 ± 0.008 | 0.042 ± 0.016 |
| V0_plus_V2 | 0.521 ± 0.006 | 0.512 ± 0.010 | 0.526 ± 0.015 | 0.021 ± 0.017 | 0.053 ± 0.009 |
