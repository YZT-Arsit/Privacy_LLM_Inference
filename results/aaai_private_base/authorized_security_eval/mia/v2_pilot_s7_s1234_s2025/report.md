# V2 Membership Privacy Pilot

Measured on 500 member and 500 non-member V2 records. The exact-length-matched cohort is the stronger control.

| Cohort | Features | n/class | ROC-AUC mean±sd | Advantage | TPR@1%FPR | TPR@0.1%FPR |
|---|---|---:|---:|---:|---:|---:|
| raw_balanced | shape_only | 500 | 0.7982±0.0062 | 0.4900 | 0.0300 | 0.0250 |
| raw_balanced | v2_no_attention | 500 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |
| raw_balanced | attention_only | 500 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |
| raw_balanced | v2_all | 500 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_length_matched | shape_only | 215 | 0.5039±0.0157 | 0.0543 | 0.0000 | 0.0000 |
| exact_length_matched | v2_no_attention | 215 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_length_matched | attention_only | 215 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_length_matched | v2_all | 215 | 1.0000±0.0000 | 1.0000 | 1.0000 | 1.0000 |

This is not the final MIA conclusion because matched V0 member outputs are still missing.

## Exact-length feature audit

**MIA audit verdict: FAIL.** The cohort contains 215 member and 215 nonmember records and 803 feature columns. 446 columns have disjoint member/nonmember ranges, so a threshold on any one of those columns identifies split/source-cohort identity perfectly on this cohort. This fails the strict leakage rule even though forbidden metadata fields are not directly selected.

All per-feature descriptive statistics, missing/zero rates, unique counts, univariate ROC-AUC, KS, mutual information, disjoint-range flags, and provenance are in `feature_audit.csv`. The disjoint subset is in `disjoint_range_features.csv`; the rule-by-rule decision is in `leakage_rule_audit.csv`.

### Leakage-rule checks

| Forbidden signal | Directly selected | Empirically encoded | Pass | Evidence |
|---|---:|---:|---:|---|
| source_file | False | True | False | 446 disjoint-range numerical features identify the aligned source cohort |
| run_or_session_id | False | False | True | not selected; same run_id sets across paths=True |
| checkpoint_id | False | False | True | not selected; identical package metadata across paths=True |
| training_or_eval_mode | False | False | True | no training/eval-mode feature or field is selected |
| step_or_epoch | False | False | True | no step/epoch feature or field is selected |
| path_dependent_tensor_presence_flag | False | False | True | all 803 columns are available in both paths=True |
| member_label_or_split_identity | False | True | False | 446 features have disjoint member/nonmember ranges |

### Feature provenance summary

| Tensor family | Features | Layers | Transform-invariant | Runtime metadata | Both paths | Disjoint |
|---|---:|---|---:|---:|---:|---:|
| transformed_hidden | 25 | 0-23, final | 25 | 0 | 25 | 23 |
| masked_logits | 9 | final | 9 | 0 | 9 | 8 |
| masked_q | 96 | 0-23 | 96 | 0 | 96 | 66 |
| masked_k | 96 | 0-23 | 96 | 0 | 96 | 64 |
| masked_v | 96 | 0-23 | 96 | 0 | 96 | 72 |
| attention_scores | 480 | 0-23 | 480 | 0 | 480 | 213 |
| tensor_shapes | 1 | global | 1 | 1 | 1 | 0 |

### Strongest univariate features

| Feature | Member-higher AUC | Direction-free AUC | KS | MI (nats) | Disjoint |
|---|---:|---:|---:|---:|---:|
| `transformed_hidden.layer_00.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_01.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_02.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_03.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_04.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_05.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_06.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_07.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_08.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_09.last_l2` | 1.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_11.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_12.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_13.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_14.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_15.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_16.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_17.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_18.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_19.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |
| `transformed_hidden.layer_20.last_l2` | 0.000000 | 1.000000 | 1.000000 | 0.694311 | True |

Statistics use sample standard deviation (`ddof=1`). Missing means non-finite. Zero rates are over finite observations. Mutual information is the sklearn continuous k-nearest-neighbor estimate in nats with fixed seed 20260714. Every feature is marked transform-invariant because it is a norm, an order-insensitive logit statistic under the permutation-only vocabulary mask, an attention quantity under Q/K mask cancellation, or sequence length. Only sequence length is runtime-metadata-derived.
