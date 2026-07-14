# Frozen-Cohort Negative Controls

Pipeline verdict for shuffled/pseudo controls: **PASS**.

| Control | Feature set | AUC mean ± SD | Materially above chance |
|---|---|---:|---:|
| N0_label_permutation | F0_shape_only | 0.488 ± 0.002 | False |
| N0_label_permutation | F2_invariant_hidden | 0.471 ± 0.042 | False |
| N0_label_permutation | F3_attention_no_shape | 0.482 ± 0.012 | False |
| N0_label_permutation | F4_masked_logits | 0.493 ± 0.027 | False |
| N0_label_permutation | F5_combined_metadata_free | 0.485 ± 0.022 | False |
| N1_pseudo_within_members | F0_shape_only | 0.519 ± 0.047 | False |
| N1_pseudo_within_members | F2_invariant_hidden | 0.498 ± 0.061 | False |
| N1_pseudo_within_members | F3_attention_no_shape | 0.489 ± 0.074 | False |
| N1_pseudo_within_members | F4_masked_logits | 0.507 ± 0.020 | False |
| N1_pseudo_within_members | F5_combined_metadata_free | 0.481 ± 0.111 | False |
| N2_pseudo_within_nonmembers | F0_shape_only | 0.550 ± 0.011 | False |
| N2_pseudo_within_nonmembers | F2_invariant_hidden | 0.559 ± 0.048 | False |
| N2_pseudo_within_nonmembers | F3_attention_no_shape | 0.594 ± 0.102 | False |
| N2_pseudo_within_nonmembers | F4_masked_logits | 0.549 ± 0.065 | False |
| N2_pseudo_within_nonmembers | F5_combined_metadata_free | 0.591 ± 0.080 | False |
| N3_source_train_vs_test | F0_shape_only | 0.454 ± 0.017 | False |
| N3_source_train_vs_test | F2_invariant_hidden | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F3_attention_no_shape | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F4_masked_logits | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F5_combined_metadata_free | 1.000 ± 0.000 | False |

N3 is deliberately a source-label classifier. High N3 performance confirms that the old membership task is source classification; it is not a pipeline-control failure.

Exact cross-source prompt overlap: 0; normalized decoded overlap: 0. Preprocessing is encapsulated in an attack-train-only sklearn Pipeline.
