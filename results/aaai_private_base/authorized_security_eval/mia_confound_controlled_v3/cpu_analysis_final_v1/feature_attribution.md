# Feature attribution

Collection mode: SIMULATED_PROTOCOL_VIEW

Coefficients are from the primary L2 logistic model after train-only standardization. They identify associations among validated invariant summaries; they are not plaintext recovery.

## Coefficient family summaries

| Family | Mean absolute coefficient | Maximum absolute coefficient |
|---|---:|---:|
| attention | 0.480643 | 3.839877 |
| kv | 0.251062 | 2.609626 |
| hidden | 0.659665 | 4.204068 |
| logit | 0.655888 | 3.864591 |
| combined | 0.418693 | 4.204068 |

## Top standardized coefficients

| Feature | Family | Mean | Mean absolute | Directions across held-out shadows |
|---|---|---:|---:|---|
| hidden.layer_11.last_l2 | hidden | -1.970617 | 1.970617 | -,-,- |
| attention.layer_01.last_entropy_by_head.mean | attention | 1.831318 | 1.835508 | +,-,+ |
| attention.layer_12.true_score_last_mean_by_head.mean | attention | 1.766318 | 1.766318 | +,+,+ |
| attention.layer_10.last_max_by_head.mean | attention | -1.724393 | 1.724393 | -,-,- |
| logit.top_values.std | logit | 0.984872 | 1.655438 | +,+,- |
| hidden.layer_21.last_l2 | hidden | -0.852378 | 1.494812 | -,+,+ |
| attention.layer_13.last_max_by_head.max | attention | -1.470788 | 1.470788 | -,-,- |
| attention.layer_03.true_score_last_std_by_head.std | attention | -0.150698 | 1.464244 | -,-,+ |
| attention.layer_10.true_score_last_mean_by_head.min | attention | -1.433811 | 1.457088 | -,+,- |
| attention.layer_04.true_score_last_mean_by_head.mean | attention | 0.377830 | 1.437138 | +,+,- |
| attention.layer_10.last_entropy_by_head.mean | attention | -0.753436 | 1.425834 | +,+,- |
| attention.layer_13.true_score_last_mean_by_head.min | attention | -1.370978 | 1.419945 | -,+,- |
| attention.layer_07.true_score_last_mean_by_head.max | attention | -1.272922 | 1.344172 | +,-,- |
| hidden.layer_06.last_l2 | hidden | -0.310081 | 1.307784 | -,+,+ |
| attention.layer_17.true_score_last_mean_by_head.mean | attention | 1.273321 | 1.273321 | +,+,+ |
| attention.layer_12.true_score_last_std_by_head.max | attention | 1.247310 | 1.247310 | +,+,+ |
| attention.layer_13.true_score_last_mean_by_head.std | attention | -1.238288 | 1.238288 | -,-,- |
| attention.layer_23.true_score_last_std_by_head.max | attention | 1.206948 | 1.206948 | +,+,+ |
| hidden.layer_02.last_l2 | hidden | -0.201651 | 1.180707 | -,+,+ |
| attention.layer_19.last_entropy_by_head.mean | attention | -1.176213 | 1.176213 | -,-,- |
| attention.layer_19.last_max_by_head.mean | attention | -1.128239 | 1.174651 | -,+,- |
| attention.layer_21.last_entropy_by_head.mean | attention | 0.945032 | 1.156102 | -,+,+ |
| attention.layer_13.last_entropy_by_head.min | attention | -1.128201 | 1.152716 | -,+,- |
| attention.layer_09.last_max_by_head.std | attention | -1.140617 | 1.151230 | -,+,- |
| attention.layer_21.last_entropy_by_head.std | attention | 0.010176 | 1.148570 | +,+,- |

## Top univariate features

| Feature | Symmetric AUC mean±SD | KS mean | MI mean | Direction stable |
|---|---:|---:|---:|---|
| attention.layer_00.last_max_by_head.min | 0.5349±0.0182 | 0.0809 | 0.0109 | no |
| attention.layer_11.true_score_last_mean_by_head.min | 0.5346±0.0199 | 0.0728 | 0.0061 | no |
| kv.v.layer_16.last_l2_by_head.min | 0.5335±0.0058 | 0.0799 | 0.0202 | no |
| attention.layer_08.true_score_last_std_by_head.max | 0.5321±0.0156 | 0.0876 | 0.0104 | no |
| attention.layer_11.true_score_last_mean_by_head.std | 0.5319±0.0172 | 0.0704 | 0.0136 | no |
| attention.layer_05.true_score_last_std_by_head.max | 0.5311±0.0239 | 0.0825 | 0.0128 | no |
| attention.layer_23.true_score_last_mean_by_head.min | 0.5304±0.0213 | 0.0793 | 0.0072 | no |
| kv.v.layer_05.last_l2_by_head.min | 0.5286±0.0131 | 0.0665 | 0.0037 | no |
| attention.layer_05.true_score_last_std_by_head.mean | 0.5284±0.0239 | 0.0772 | 0.0000 | no |
| kv.k.layer_09.last_l2_by_head.std | 0.5280±0.0180 | 0.0726 | 0.0017 | no |
| kv.v.layer_03.last_l2_by_head.max | 0.5276±0.0159 | 0.0649 | 0.0000 | no |
| attention.layer_07.true_score_last_std_by_head.std | 0.5268±0.0096 | 0.0726 | 0.0042 | no |
| attention.layer_11.last_entropy_by_head.min | 0.5268±0.0208 | 0.0757 | 0.0061 | no |
| kv.v.layer_12.last_l2_by_head.std | 0.5266±0.0183 | 0.0666 | 0.0042 | no |
| kv.v.layer_20.last_l2_by_head.min | 0.5254±0.0056 | 0.0770 | 0.0191 | no |
| attention.layer_23.true_score_last_mean_by_head.mean | 0.5253±0.0195 | 0.0744 | 0.0019 | no |
| attention.layer_08.true_score_last_std_by_head.mean | 0.5247±0.0134 | 0.0758 | 0.0003 | no |
| kv.v.layer_12.last_l2_by_head.max | 0.5247±0.0200 | 0.0653 | 0.0108 | no |
| attention.layer_22.last_max_by_head.max | 0.5246±0.0079 | 0.0642 | 0.0090 | yes |
| kv.v.layer_13.last_l2_by_head.min | 0.5246±0.0148 | 0.0603 | 0.0016 | yes |
| kv.v.layer_06.last_l2_by_head.mean | 0.5245±0.0067 | 0.0612 | 0.0097 | no |
| kv.k.layer_21.last_l2_by_head.max | 0.5242±0.0211 | 0.0634 | 0.0049 | no |
| attention.layer_22.true_score_last_mean_by_head.min | 0.5242±0.0175 | 0.0614 | 0.0048 | yes |
| kv.k.layer_22.last_l2_by_head.std | 0.5240±0.0279 | 0.0706 | 0.0011 | no |
| hidden.layer_03.last_l2 | 0.5240±0.0130 | 0.0639 | 0.0023 | no |
