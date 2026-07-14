# Feature attribution

Collection mode: REAL_TDX_BACKED_VIEW  
Evaluation: three-shadow leave-one-shadow-out  
Feature schema: common-semantics v2, 610 columns

Coefficients use train-only standardization. Importance identifies association among validated transformed summaries; it does not imply plaintext recovery.

## Family coefficient magnitude

| Family | Mean absolute coefficient | Maximum absolute coefficient |
|---|---:|---:|
| attention | 0.020404 | 0.096804 |
| kv | 0.016449 | 0.085732 |
| hidden | 0.013690 | 0.049510 |
| logit | 0.018859 | 0.076038 |
| combined | 0.018869 | 0.096804 |

## Top standardized coefficients

| Feature | Family | Mean coefficient | Mean absolute | Fold directions |
|---|---|---:|---:|---|
| attention.layer_21.last_entropy_by_head.max | attention | 0.068968 | 0.068968 | +,+,+ |
| attention.layer_07.last_entropy_by_head.min | attention | 0.067727 | 0.067727 | +,+,+ |
| attention.layer_22.true_score_last_mean_by_head.min | attention | -0.059920 | 0.059920 | -,-,- |
| attention.layer_20.true_score_last_std_by_head.max | attention | -0.059850 | 0.059850 | -,-,- |
| attention.layer_23.true_score_last_mean_by_head.min | attention | -0.056811 | 0.056811 | -,-,- |
| kv.k.layer_18.last_l2_by_head.std | kv | -0.011061 | 0.056140 | -,+,- |
| attention.layer_08.true_score_last_std_by_head.min | attention | -0.032458 | 0.055974 | -,-,+ |
| kv.v.layer_06.last_l2_by_head.min | kv | 0.024204 | 0.055123 | -,+,+ |
| kv.k.layer_21.last_l2_by_head.std | kv | -0.052430 | 0.052430 | -,-,- |
| attention.layer_18.last_max_by_head.max | attention | 0.050180 | 0.050180 | +,+,+ |
| attention.layer_22.true_score_last_mean_by_head.mean | attention | -0.049525 | 0.049525 | -,-,- |
| attention.layer_02.true_score_last_std_by_head.min | attention | -0.004126 | 0.049132 | -,+,+ |
| kv.v.layer_19.last_l2_by_head.std | kv | 0.047256 | 0.047256 | +,+,+ |
| kv.k.layer_19.last_l2_by_head.max | kv | 0.046545 | 0.046545 | +,+,+ |
| attention.layer_19.true_score_last_std_by_head.min | attention | 0.046250 | 0.046250 | +,+,+ |
| kv.k.layer_22.last_l2_by_head.std | kv | 0.038074 | 0.045848 | +,+,- |
| kv.v.layer_12.last_l2_by_head.std | kv | -0.011790 | 0.045365 | +,+,- |
| kv.k.layer_14.last_l2_by_head.max | kv | -0.045287 | 0.045287 | -,-,- |
| attention.layer_05.last_entropy_by_head.min | attention | 0.044844 | 0.044844 | +,+,+ |
| attention.layer_21.true_score_last_std_by_head.max | attention | 0.044475 | 0.044475 | +,+,+ |
| attention.layer_22.true_score_last_std_by_head.std | attention | 0.019609 | 0.043970 | -,+,+ |
| attention.layer_22.true_score_last_std_by_head.min | attention | 0.043261 | 0.043261 | +,+,+ |
| attention.layer_15.last_entropy_by_head.min | attention | 0.026276 | 0.043161 | +,+,- |
| attention.layer_22.last_max_by_head.min | attention | 0.043070 | 0.043070 | +,+,+ |
| attention.layer_07.last_max_by_head.max | attention | -0.042437 | 0.042437 | -,-,- |

## Top univariate features

| Feature | Symmetric AUC | KS | Mutual information | Direction stable |
|---|---:|---:|---:|---|
| attention.layer_08.true_score_last_std_by_head.max | 0.5339 | 0.0920 | 0.007511 | False |
| attention.layer_11.true_score_last_mean_by_head.min | 0.5331 | 0.0733 | 0.003784 | False |
| kv.v.layer_16.last_l2_by_head.min | 0.5323 | 0.0800 | 0.015467 | False |
| kv.v.layer_12.last_l2_by_head.std | 0.5322 | 0.0740 | 0.011711 | False |
| attention.layer_11.true_score_last_mean_by_head.std | 0.5322 | 0.0700 | 0.014455 | False |
| attention.layer_00.last_max_by_head.std | 0.5312 | 0.0773 | 0.010067 | False |
| kv.v.layer_06.last_l2_by_head.mean | 0.5309 | 0.0707 | 0.017097 | False |
| attention.layer_00.last_max_by_head.min | 0.5289 | 0.0680 | 0.008825 | False |
| kv.v.layer_05.last_l2_by_head.min | 0.5285 | 0.0680 | 0.002588 | False |
| attention.layer_07.true_score_last_std_by_head.std | 0.5285 | 0.0693 | 0.002302 | False |
| attention.layer_11.last_entropy_by_head.min | 0.5284 | 0.0740 | 0.006737 | False |
| kv.v.layer_12.last_l2_by_head.max | 0.5281 | 0.0660 | 0.009018 | False |
| attention.layer_05.true_score_last_std_by_head.mean | 0.5277 | 0.0720 | 0.000004 | False |
| kv.v.layer_12.last_l2_by_head.mean | 0.5277 | 0.0573 | 0.002383 | False |
| attention.layer_08.true_score_last_std_by_head.std | 0.5270 | 0.0767 | 0.022119 | False |
| attention.layer_05.true_score_last_std_by_head.max | 0.5263 | 0.0667 | 0.010791 | False |
| kv.k.layer_16.last_l2_by_head.min | 0.5254 | 0.0627 | 0.002386 | False |
| kv.k.layer_18.last_l2_by_head.std | 0.5252 | 0.0647 | 0.005850 | False |
| kv.k.layer_16.last_l2_by_head.mean | 0.5251 | 0.0693 | 0.007305 | False |
| attention.layer_21.last_max_by_head.std | 0.5250 | 0.0627 | 0.001674 | False |
| attention.layer_07.true_score_last_std_by_head.max | 0.5248 | 0.0740 | 0.017364 | False |
| attention.layer_22.last_max_by_head.max | 0.5246 | 0.0653 | 0.009327 | True |
| kv.k.layer_22.last_l2_by_head.std | 0.5246 | 0.0727 | 0.001706 | False |
| kv.v.layer_06.last_l2_by_head.min | 0.5246 | 0.0580 | 0.006601 | False |
| attention.layer_15.last_entropy_by_head.min | 0.5245 | 0.0647 | 0.006276 | False |
