# Stage 5.7 — Permutation-Invariant Leakage Audit

## 1. Experiment Scope

We audit single-shot permutation-invariant leakage of the existing ZP-style compatible nonlinear islands under both selectable mitigation bundles (`fresh_perm_only` and `fresh_perm_plus_sandwich_plus_pad`). Trace pairs are collected with the Stage 5.5b real-token trace collector at the model-level wrapper boundaries. No protocol math, wrappers, defaults, or existing tests are modified.

## 2. Threat Model

Honest-but-curious accelerator that observes the visible (ZP-permuted) activations at the wrapper-side boundaries. No real TEE isolation or hardware side-channel resistance is evaluated. Raw tensors, masks, permutations, adapters, gradients, and private data are not exported.

## 3. Theoretical Claim Tested

Permutation-only nonlinear views provide channel-index hiding, not value hiding. Per-row L1 / L2 / Linf norms, extrema, sorted multisets, and quantiles are preserved by construction. The full mitigation bundle changes the freshness / temporal contract but does not remove single-shot permutation-invariant statistics inside the activation core.

## 4. Target Tensor Inventory

| tensor | prefill_present | decode_present |
|---|---|---|
| gate | True | True |
| up | True | True |
| swiglu_intermediate | True | True |
| post_island | True | True |
| q | True | True |
| k | True | True |
| v | True | True |
| boundary_input | True | True |
| final | True | True |

## 5. Norm Preservation

| bundle | tensor | scope | l1_corr | l2_corr | linf_corr | l2_max_abs_diff | l2_mean_abs_diff | label |
|---|---|---|---|---|---|---|---|---|
| fresh_perm_only | gate | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | gate | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | up | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | up | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | swiglu_intermediate | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | swiglu_intermediate | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | post_island | prefill | 0.9696 | 1.0 | 0.7436 | 0.0 | 0.0 | statistical_leakage_borderline |
| fresh_perm_only | post_island | decode | 0.9638 | 1.0 | 0.7479 | 0.0 | 0.0 | statistical_leakage_borderline |
| fresh_perm_only | q | prefill | 0.9087 | 1.0 | 0.7201 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | q | decode | 0.8812 | 1.0 | 0.727 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | k | prefill | 0.9211 | 1.0 | 0.7777 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | k | decode | 0.8946 | 1.0 | 0.7284 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | v | prefill | 0.9111 | 1.0 | 0.7505 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | v | decode | 0.9477 | 1.0 | 0.8384 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_only | boundary_input | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | boundary_input | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | final | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_only | final | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | gate | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | gate | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | up | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | up | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | post_island | prefill | 0.9696 | 1.0 | 0.7436 | 0.0 | 0.0 | statistical_leakage_borderline |
| fresh_perm_plus_sandwich_plus_pad | post_island | decode | 0.9638 | 1.0 | 0.7479 | 0.0 | 0.0 | statistical_leakage_borderline |
| fresh_perm_plus_sandwich_plus_pad | q | prefill | 0.9087 | 1.0 | 0.7201 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | q | decode | 0.8812 | 1.0 | 0.727 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | k | prefill | 0.9211 | 1.0 | 0.7777 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | k | decode | 0.8946 | 1.0 | 0.7284 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | v | prefill | 0.9111 | 1.0 | 0.7505 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | v | decode | 0.9477 | 1.0 | 0.8384 | 0.0 | 0.0 | statistical_leakage_low |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | final | prefill | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |
| fresh_perm_plus_sandwich_plus_pad | final | decode | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | statistical_leakage_detected_high |

## 6. Sorted Multiset Preservation

| bundle | tensor | scope | sorted_mse_mean | sorted_mse_max | sorted_l2_rel_mean |
|---|---|---|---|---|---|
| fresh_perm_only | gate | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | gate | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | up | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | up | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | swiglu_intermediate | prefill | 0.0 | 0.0 | 1e-06 |
| fresh_perm_only | swiglu_intermediate | decode | 0.0 | 0.0 | 1e-06 |
| fresh_perm_only | post_island | prefill | 0.00213317 | 0.01150422 | 0.320931 |
| fresh_perm_only | post_island | decode | 0.00199752 | 0.00751164 | 0.309181 |
| fresh_perm_only | q | prefill | 0.11439402 | 0.68020147 | 0.558965 |
| fresh_perm_only | q | decode | 0.13068706 | 1.14631987 | 0.597283 |
| fresh_perm_only | k | prefill | 0.13246053 | 1.14833903 | 0.609704 |
| fresh_perm_only | k | decode | 0.12616765 | 0.84086567 | 0.55858 |
| fresh_perm_only | v | prefill | 0.1543802 | 1.51804066 | 0.593934 |
| fresh_perm_only | v | decode | 0.16269389 | 1.0571506 | 0.577596 |
| fresh_perm_only | boundary_input | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | boundary_input | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | final | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_only | final | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | gate | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | gate | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | up | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | up | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | prefill | 0.0 | 0.0 | 1e-06 |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | decode | 0.0 | 0.0 | 1e-06 |
| fresh_perm_plus_sandwich_plus_pad | post_island | prefill | 0.00213317 | 0.01150422 | 0.320931 |
| fresh_perm_plus_sandwich_plus_pad | post_island | decode | 0.00199752 | 0.00751164 | 0.309181 |
| fresh_perm_plus_sandwich_plus_pad | q | prefill | 0.11439402 | 0.68020147 | 0.558965 |
| fresh_perm_plus_sandwich_plus_pad | q | decode | 0.13068706 | 1.14631987 | 0.597283 |
| fresh_perm_plus_sandwich_plus_pad | k | prefill | 0.13246053 | 1.14833903 | 0.609704 |
| fresh_perm_plus_sandwich_plus_pad | k | decode | 0.12616765 | 0.84086567 | 0.55858 |
| fresh_perm_plus_sandwich_plus_pad | v | prefill | 0.1543802 | 1.51804066 | 0.593934 |
| fresh_perm_plus_sandwich_plus_pad | v | decode | 0.16269389 | 1.0571506 | 0.577596 |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | decode | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | final | prefill | 0.0 | 0.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | final | decode | 0.0 | 0.0 | 0.0 |

## 7. Quantile / Extrema Preservation

| bundle | tensor | scope | max_corr | min_corr | quantile_mse_mean |
|---|---|---|---|---|---|
| fresh_perm_only | gate | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | gate | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | up | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | up | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | swiglu_intermediate | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | swiglu_intermediate | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | post_island | prefill | 0.6517 | 0.6514 | 0.00167916 |
| fresh_perm_only | post_island | decode | 0.6752 | 0.886 | 0.00158809 |
| fresh_perm_only | q | prefill | 0.3664 | 0.3967 | 0.09213679 |
| fresh_perm_only | q | decode | 0.3863 | 0.2812 | 0.11274756 |
| fresh_perm_only | k | prefill | 0.3364 | 0.4871 | 0.11347306 |
| fresh_perm_only | k | decode | 0.0781 | 0.6181 | 0.10241789 |
| fresh_perm_only | v | prefill | 0.2729 | 0.2985 | 0.12844291 |
| fresh_perm_only | v | decode | 0.3807 | 0.1899 | 0.13603656 |
| fresh_perm_only | boundary_input | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | boundary_input | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | final | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_only | final | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | gate | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | gate | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | up | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | up | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | post_island | prefill | 0.6517 | 0.6514 | 0.00167916 |
| fresh_perm_plus_sandwich_plus_pad | post_island | decode | 0.6752 | 0.886 | 0.00158809 |
| fresh_perm_plus_sandwich_plus_pad | q | prefill | 0.3664 | 0.3967 | 0.09213679 |
| fresh_perm_plus_sandwich_plus_pad | q | decode | 0.3863 | 0.2812 | 0.11274756 |
| fresh_perm_plus_sandwich_plus_pad | k | prefill | 0.3364 | 0.4871 | 0.11347306 |
| fresh_perm_plus_sandwich_plus_pad | k | decode | 0.0781 | 0.6181 | 0.10241789 |
| fresh_perm_plus_sandwich_plus_pad | v | prefill | 0.2729 | 0.2985 | 0.12844291 |
| fresh_perm_plus_sandwich_plus_pad | v | decode | 0.3807 | 0.1899 | 0.13603656 |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | decode | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | final | prefill | 1.0 | 1.0 | 0.0 |
| fresh_perm_plus_sandwich_plus_pad | final | decode | 1.0 | 1.0 | 0.0 |

## 8. Statistics-Only Classifier

| bundle | tensor | task | status | accuracy | chance_level | label |
|---|---|---|---|---|---|---|
| fresh_perm_only | gate | scope_classification | ok | 0.575 | 0.5 | proxy_attack_low |
| fresh_perm_only | gate | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | gate | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | up | scope_classification | ok | 0.575 | 0.5 | proxy_attack_low |
| fresh_perm_only | up | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | up | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | swiglu_intermediate | scope_classification | ok | 0.6 | 0.5 | proxy_attack_low |
| fresh_perm_only | swiglu_intermediate | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | swiglu_intermediate | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | post_island | scope_classification | ok | 0.5625 | 0.5 | proxy_attack_low |
| fresh_perm_only | post_island | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | post_island | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | q | scope_classification | ok | 0.5188 | 0.5 | proxy_attack_chance_level |
| fresh_perm_only | q | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | q | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | k | scope_classification | ok | 0.4625 | 0.5 | proxy_attack_chance_level |
| fresh_perm_only | k | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | k | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | v | scope_classification | ok | 0.5312 | 0.5 | proxy_attack_chance_level |
| fresh_perm_only | v | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | v | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | boundary_input | scope_classification | ok | 0.4125 | 0.5 | proxy_attack_chance_level |
| fresh_perm_only | boundary_input | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | boundary_input | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | final | scope_classification | ok | 0.5125 | 0.5 | proxy_attack_chance_level |
| fresh_perm_only | final | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_only | final | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | gate | scope_classification | ok | 0.575 | 0.5 | proxy_attack_low |
| fresh_perm_plus_sandwich_plus_pad | gate | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | gate | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | up | scope_classification | ok | 0.575 | 0.5 | proxy_attack_low |
| fresh_perm_plus_sandwich_plus_pad | up | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | up | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | scope_classification | ok | 0.6 | 0.5 | proxy_attack_low |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | swiglu_intermediate | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | post_island | scope_classification | ok | 0.5625 | 0.5 | proxy_attack_low |
| fresh_perm_plus_sandwich_plus_pad | post_island | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | post_island | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | q | scope_classification | ok | 0.5188 | 0.5 | proxy_attack_chance_level |
| fresh_perm_plus_sandwich_plus_pad | q | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | q | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | k | scope_classification | ok | 0.4625 | 0.5 | proxy_attack_chance_level |
| fresh_perm_plus_sandwich_plus_pad | k | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | k | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | v | scope_classification | ok | 0.5312 | 0.5 | proxy_attack_chance_level |
| fresh_perm_plus_sandwich_plus_pad | v | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | v | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | scope_classification | ok | 0.4125 | 0.5 | proxy_attack_chance_level |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | boundary_input | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | final | scope_classification | ok | 0.5125 | 0.5 | proxy_attack_chance_level |
| fresh_perm_plus_sandwich_plus_pad | final | prompt_id_linkability | skipped_no_per_row_prompt_label_in_stitched_view | None | None | proxy_attack_skipped |
| fresh_perm_plus_sandwich_plus_pad | final | position_bucket_classification | skipped_no_per_row_position_label_in_stitched_view | None | None | proxy_attack_skipped |

## 9. Freshness Ablation

Fixed-permutation vs fresh-permutation synthetic ablation:

- hidden_dim: `32`
- num_rows: `8`
- fixed_perm linkability accuracy: `1.0` (chance `0.125`)
- fresh_perm linkability accuracy: `1.0` (chance `0.125`)
- fixed_perm `sorted_l2_rel_mean = 0.0`, `l2_corr = 1.0`
- fresh_perm `sorted_l2_rel_mean = 0.0`, `l2_corr = 1.0`

Both fixed and fresh permutation views preserve per-row norms, sorted multisets, and quantiles by construction. The freshness contract changes the row-signature linkability (fixed perm preserves it exactly via deterministic channel re-mapping; fresh perm does not), but does not remove single-shot permutation-invariant statistics inside the activation core.

## 10. Interpretation

For raw permutation-island views (`gate`, `up`, `swiglu_intermediate`), the per-row norm, sorted-multiset, and quantile metrics are preserved at or near machine precision. The corresponding rows are labelled `statistical_leakage_detected_*`. This is the *channel-index hiding, not value hiding* property of ZP-style permutation masks. For dense post-island views (`post_island`, `boundary_input`, `final`), the same metrics generally do NOT match, because the dense right-mask plus boundary pad no longer share the multiset of the plain activations.

The two mitigation bundles (`fresh_perm_only` vs `fresh_perm_plus_sandwich_plus_pad`) are equivalent at the single-shot activation-core multiset level. The full bundle's added value is in temporal / boundary / freshness posture, not in single-shot permutation-invariant statistics.

## 11. Limitations

- Permutation-only nonlinear views provide channel-index hiding, not value hiding.
- Dense sandwiching and boundary pads mitigate temporal and boundary exposure but do not remove single-shot permutation-invariant statistics inside the activation core.
- This is a proxy leakage audit, not a formal security proof.
- No real TEE isolation or hardware side-channel resistance is evaluated.
- Raw tensors, masks, permutations, adapters, gradients, and private data are not exported.
- Per-row prompt-id and position-bucket labels are not available in the stitched [N, D] view; the corresponding classifier tasks are skipped rather than fabricated.

## 12. Next Stage Plan

Future work: add an enhanced nonlinear protection that breaks row-wise permutation invariance (for example, masked dense expansion + paired-permutation absorption that mixes channels across the islands), and a side-channel-aware proxy. Both are out of scope for Stage 5.7; they would lift the current permutation-invariant leakage from *channel-index hidden* to *value-multiset hidden* on these boundaries. As of this stage, no real TEE / GPU isolation or hardware side-channel resistance is claimed.

`formal_security_claim`: `False`

