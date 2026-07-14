# Frozen-Cohort Negative Controls

Pipeline verdict for shuffled/pseudo controls: **PASS**.

| Control | Feature set | AUC mean ± SD | Materially above chance |
|---|---|---:|---:|
| N0_label_permutation | F0_shape_only | 0.506 ± 0.026 | False |
| N0_label_permutation | F2_invariant_hidden | 0.496 ± 0.049 | False |
| N0_label_permutation | F3_attention_no_shape | 0.487 ± 0.044 | False |
| N0_label_permutation | F4_masked_logits | 0.518 ± 0.052 | False |
| N0_label_permutation | F5_combined_metadata_free | 0.487 ± 0.050 | False |
| N1_pseudo_within_members | F0_shape_only | 0.503 ± 0.047 | False |
| N1_pseudo_within_members | F2_invariant_hidden | 0.448 ± 0.077 | False |
| N1_pseudo_within_members | F3_attention_no_shape | 0.448 ± 0.071 | False |
| N1_pseudo_within_members | F4_masked_logits | 0.483 ± 0.059 | False |
| N1_pseudo_within_members | F5_combined_metadata_free | 0.438 ± 0.082 | False |
| N2_pseudo_within_nonmembers | F0_shape_only | 0.539 ± 0.053 | False |
| N2_pseudo_within_nonmembers | F2_invariant_hidden | 0.503 ± 0.059 | False |
| N2_pseudo_within_nonmembers | F3_attention_no_shape | 0.531 ± 0.073 | False |
| N2_pseudo_within_nonmembers | F4_masked_logits | 0.524 ± 0.057 | False |
| N2_pseudo_within_nonmembers | F5_combined_metadata_free | 0.535 ± 0.060 | False |
| N3_source_train_vs_test | F0_shape_only | 0.468 ± 0.019 | False |
| N3_source_train_vs_test | F2_invariant_hidden | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F3_attention_no_shape | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F4_masked_logits | 1.000 ± 0.000 | False |
| N3_source_train_vs_test | F5_combined_metadata_free | 1.000 ± 0.000 | False |

N3 is deliberately a source-label classifier. High N3 performance confirms that the old membership task is source classification; it is not a pipeline-control failure.

Exact cross-source prompt overlap: 0; normalized decoded overlap: 0. Preprocessing is encapsulated in an attack-train-only sklearn Pipeline.
