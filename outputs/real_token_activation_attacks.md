# Real-Token-Prompted Real-Activation Attacker (Stage 5.5b)

## Experiment Scope

Stage 5.5b re-runs the Stage 5.5 adaptive proxy attackers (ridge linear inverter, small MLP inverter, signature / Sinkhorn permutation recovery, linkability proxy) but the (plain, visible) trace pairs now come from the Stage 6.4c modern decoder model-level wrapper driven by real (or deterministically synthetic) token IDs — embedding lookup → N blocks → final RMSNorm → optionally masked LM head — covering both PREFILL and DECODE_STEP paths. Default mode for the wider system remains `nonlinear_mode='trusted'`; default mitigation bundle remains `fresh_perm_only`. The numbers below are a real-token-prompted adaptive proxy evaluation of `fresh_perm_plus_sandwich_plus_pad` — they are NOT a formal security proof.

## Model and Tokenizer Loading Status

| field | value |
|---|---|
| model_loading.load_status | synthetic_only |
| model_loading.resolved_model_id | None |
| model_loading.model_family | None |
| model_loading.fallback_used | True |
| model_loading.load_error | attempt_real_model_load=False (default); pytest runs the synthetic fallback to avoid network downloads. |
| tokenizer_loading.tokenizer_status | not_requested |
| tokenizer_loading.tokenizer_id | None |
| tokenizer_loading.tokenizer_error | None |
| source | synthetic_block |
| num_layers_used | 2 |

## Prompt Set Summary

| field | value |
|---|---|
| token_source | synthetic_token_ids |
| tokenizer_status | not_requested |
| num_prompts | 8 |
| prompt_max_length | 8 |
| vocab_size_used | 64 |
| max_new_tokens | 3 |

## Trace Collection Summary

| field | value |
|---|---|
| block_spec.hidden_size | 32 |
| block_spec.intermediate_size | 64 |
| block_spec.num_attention_heads | 4 |
| block_spec.num_key_value_heads | 2 |
| block_spec.head_dim | 8 |
| block_spec.attention_variant | gqa |
| block_spec.rope_base | 10000.0 |
| all_prefill_allclose | True |
| all_decode_allclose | True |
| rope_position_increment | True |
| rope_positions_seen | [8, 9] |

## Target Tensor Inventory (Prefill)

| tensor_name | feature_dim | num_samples | plain abs_max | visible abs_max | plain_fingerprint | visible_fingerprint |
|---|---|---|---|---|---|---|
| boundary_input | 32 | 128 | 1.419 | 1.419 | 893826cad49bdc2a | 893826cad49bdc2a |
| q | 8 | 512 | 1.976 | 2.330 | 6a065677ff90d796 | 032ee0919d048952 |
| k | 8 | 256 | 1.974 | 1.937 | 027951f59ab101bc | 21c352b7f80ce921 |
| v | 8 | 256 | 2.088 | 2.091 | b8cc690205f7b798 | ff8dfe85c4c0cc3b |
| gate | 64 | 128 | 2.175 | 2.175 | c6f68599b79e005d | 2e0ded24dce3004d |
| up | 64 | 128 | 2.129 | 2.129 | 88a11f2163b3ab22 | 9b6c516565e79b0d |
| swiglu_intermediate | 64 | 128 | 2.065 | 2.065 | dd58c036535f73c7 | c7d0d0e8f4e5b470 |
| post_island | 32 | 128 | 0.833 | 0.805 | cf1822f7c60a3ea5 | 5b693f20a4e70a68 |
| final | 32 | 128 | 1.625 | 1.625 | 26e22bb664636878 | 26e22bb664636878 |

## Target Tensor Inventory (Decode)

| tensor_name | feature_dim | num_samples | plain abs_max | visible abs_max |
|---|---|---|---|---|
| boundary_input | 32 | 32 | 0.632 | 0.632 |
| q | 8 | 128 | 1.847 | 1.836 |
| k | 8 | 64 | 1.507 | 1.580 |
| v | 8 | 64 | 1.893 | 1.749 |
| gate | 64 | 32 | 2.160 | 2.160 |
| up | 64 | 32 | 1.730 | 1.730 |
| swiglu_intermediate | 64 | 32 | 1.442 | 1.442 |
| post_island | 32 | 32 | 0.415 | 0.553 |
| final | 32 | 32 | 1.235 | 1.235 |

## Prefill Real-Token Activation Attacks

Per-tensor attacker results on real-token-driven prefill activations. Tensor-by-tensor breakdowns follow in the Linear / MLP / Permutation sections.

## Decode-Step Real-Token Activation Attacks

Per-tensor attacker results on the single-token decode_step activations under the masked KV-cache append surface.

## Linear Inverter Results

| scope | tensor_name | bundle | relative_l2_error | mse | cosine_similarity |
|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | 0.0029 | 0.0000 | 1.0000 |
| prefill | q | fresh_perm_only | 1.0294 | 0.3195 | 0.0156 |
| prefill | k | fresh_perm_only | 1.0301 | 0.2729 | 0.0431 |
| prefill | v | fresh_perm_only | 1.0658 | 0.3578 | -0.0130 |
| prefill | gate | fresh_perm_only | 2.0847 | 1.3771 | 0.1046 |
| prefill | up | fresh_perm_only | 2.1867 | 1.6280 | 0.0071 |
| prefill | swiglu_intermediate | fresh_perm_only | 2.1561 | 0.1305 | -0.0005 |
| prefill | post_island | fresh_perm_only | 1.3642 | 0.0367 | 0.0881 |
| prefill | final | fresh_perm_only | 0.0011 | 0.0000 | 1.0000 |
| decode | boundary_input | fresh_perm_only | 0.5584 | 0.0067 | 0.8296 |
| decode | q | fresh_perm_only | 1.0679 | 0.3351 | 0.0310 |
| decode | k | fresh_perm_only | 1.0173 | 0.3060 | 0.2306 |
| decode | v | fresh_perm_only | 1.2403 | 0.4185 | 0.0617 |
| decode | gate | fresh_perm_only | 1.2521 | 0.4709 | -0.0545 |
| decode | up | fresh_perm_only | 1.2006 | 0.4951 | 0.0876 |
| decode | swiglu_intermediate | fresh_perm_only | 1.2758 | 0.0456 | 0.0245 |
| decode | post_island | fresh_perm_only | 1.8923 | 0.0638 | 0.0671 |
| decode | final | fresh_perm_only | 0.3834 | 0.0098 | 0.9236 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | 0.0029 | 0.0000 | 1.0000 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 1.0294 | 0.3195 | 0.0156 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | 1.0301 | 0.2729 | 0.0431 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 1.0658 | 0.3578 | -0.0130 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 2.0847 | 1.3771 | 0.1046 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 2.1867 | 1.6280 | 0.0071 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 2.1561 | 0.1305 | -0.0005 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | 1.3642 | 0.0367 | 0.0881 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | 0.0011 | 0.0000 | 1.0000 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | 0.5584 | 0.0067 | 0.8296 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | 1.0679 | 0.3351 | 0.0310 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | 1.0173 | 0.3060 | 0.2306 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 1.2403 | 0.4185 | 0.0617 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 1.2521 | 0.4709 | -0.0545 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 1.2006 | 0.4951 | 0.0876 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.2758 | 0.0456 | 0.0245 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | 1.8923 | 0.0638 | 0.0671 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | 0.3834 | 0.0098 | 0.9236 |

## Small MLP Inverter Results

| scope | tensor_name | bundle | relative_l2_error | mse | cosine_similarity | final_train_loss |
|---|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | 0.3974 | 0.0073 | 0.9188 | 0.0000 |
| prefill | q | fresh_perm_only | 1.1775 | 0.4180 | -0.0406 | 0.2075 |
| prefill | k | fresh_perm_only | 1.1760 | 0.3557 | 0.1653 | 0.0874 |
| prefill | v | fresh_perm_only | 1.2929 | 0.5265 | 0.0290 | 0.0829 |
| prefill | gate | fresh_perm_only | 1.1806 | 0.4417 | 0.1630 | 0.0004 |
| prefill | up | fresh_perm_only | 1.0767 | 0.3947 | 0.2437 | 0.0005 |
| prefill | swiglu_intermediate | fresh_perm_only | 1.2404 | 0.0432 | 0.0527 | 0.0000 |
| prefill | post_island | fresh_perm_only | 1.2181 | 0.0292 | 0.1351 | 0.0001 |
| prefill | final | fresh_perm_only | 0.3831 | 0.0155 | 0.9262 | 0.0001 |
| decode | boundary_input | fresh_perm_only | 0.7167 | 0.0110 | 0.6978 | 0.0000 |
| decode | q | fresh_perm_only | 1.3709 | 0.5522 | -0.0245 | 0.0163 |
| decode | k | fresh_perm_only | 1.2049 | 0.4293 | 0.3248 | 0.0009 |
| decode | v | fresh_perm_only | 1.3630 | 0.5054 | 0.1281 | 0.0004 |
| decode | gate | fresh_perm_only | 1.1481 | 0.3960 | 0.1439 | 0.0013 |
| decode | up | fresh_perm_only | 1.1353 | 0.4427 | 0.1195 | 0.0005 |
| decode | swiglu_intermediate | fresh_perm_only | 1.1818 | 0.0391 | 0.0705 | 0.0000 |
| decode | post_island | fresh_perm_only | 1.2078 | 0.0260 | 0.0651 | 0.0000 |
| decode | final | fresh_perm_only | 0.7314 | 0.0355 | 0.7153 | 0.0000 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | 0.4123 | 0.0079 | 0.9142 | 0.0000 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 1.1852 | 0.4235 | -0.0259 | 0.1728 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | 1.1810 | 0.3587 | 0.1505 | 0.1012 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 1.2969 | 0.5298 | -0.0418 | 0.0888 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 1.1266 | 0.4022 | 0.2145 | 0.0004 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 1.1616 | 0.4594 | 0.1120 | 0.0006 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.3250 | 0.0493 | 0.0066 | 0.0000 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | 1.2440 | 0.0305 | 0.1022 | 0.0001 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | 0.4148 | 0.0182 | 0.9118 | 0.0001 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | 0.7113 | 0.0109 | 0.7045 | 0.0000 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | 1.4096 | 0.5838 | -0.0573 | 0.0127 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | 1.2105 | 0.4333 | 0.3022 | 0.0008 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 1.3983 | 0.5319 | 0.1064 | 0.0005 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 1.1977 | 0.4309 | 0.1157 | 0.0010 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 1.1410 | 0.4472 | 0.1813 | 0.0007 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.2146 | 0.0413 | 0.0595 | 0.0000 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | 1.2191 | 0.0265 | -0.0091 | 0.0000 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | 0.7103 | 0.0335 | 0.7321 | 0.0000 |

## Permutation Recovery Results

Only the SwiGLU island tensors (gate / up / swiglu_intermediate) expose a column permutation; the other tensors are dense / orthogonal-masked or plain at the inter-block boundary.

| scope | tensor_name | bundle | random_chance | signature_top1 | soft_top1 | best_top1 |
|---|---|---|---|---|---|---|
| prefill | gate | fresh_perm_only | 0.0156 | 0.0156 | 0.0000 | 0.0156 |
| prefill | up | fresh_perm_only | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| prefill | swiglu_intermediate | fresh_perm_only | 0.0156 | 0.0156 | 0.0000 | 0.0156 |
| decode | gate | fresh_perm_only | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| decode | up | fresh_perm_only | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| decode | swiglu_intermediate | fresh_perm_only | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0000 | 0.0156 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0000 | 0.0156 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0156 | 0.0156 |

## Linkability Results

| scope | tensor_name | bundle | visible_vs_plain_cosine | mean_pairwise_cosine_visible | mean_linkability_rank |
|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | 1.0000 | 0.0432 | 0.3438 |
| prefill | q | fresh_perm_only | 0.0315 | 0.0118 | 310.7188 |
| prefill | k | fresh_perm_only | -0.0127 | 0.0066 | 153.3750 |
| prefill | v | fresh_perm_only | 0.0469 | 0.0194 | 132.6875 |
| prefill | gate | fresh_perm_only | 0.0201 | 0.0297 | 60.1719 |
| prefill | up | fresh_perm_only | 0.0118 | 0.0313 | 56.7188 |
| prefill | swiglu_intermediate | fresh_perm_only | 0.0139 | 0.0340 | 56.6562 |
| prefill | post_island | fresh_perm_only | 0.0156 | 0.0319 | 61.0000 |
| prefill | final | fresh_perm_only | 1.0000 | 0.1267 | 0.0000 |
| decode | boundary_input | fresh_perm_only | 1.0000 | 0.1203 | 0.2500 |
| decode | q | fresh_perm_only | 0.0641 | 0.0044 | 58.5156 |
| decode | k | fresh_perm_only | 0.0290 | 0.0262 | 29.5000 |
| decode | v | fresh_perm_only | 0.0502 | -0.0007 | 29.3906 |
| decode | gate | fresh_perm_only | 0.0327 | 0.0023 | 13.7188 |
| decode | up | fresh_perm_only | 0.0191 | 0.0018 | 15.5000 |
| decode | swiglu_intermediate | fresh_perm_only | 0.0301 | -0.0105 | 14.6875 |
| decode | post_island | fresh_perm_only | 0.0366 | 0.0165 | 13.4688 |
| decode | final | fresh_perm_only | 1.0000 | 0.1954 | 0.0000 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.0000 | 0.0432 | 0.3438 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 0.0315 | 0.0118 | 310.7188 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | -0.0127 | 0.0066 | 153.3750 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 0.0469 | 0.0194 | 132.6875 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 0.0201 | 0.0297 | 60.1719 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 0.0118 | 0.0313 | 56.7188 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0139 | 0.0340 | 56.6562 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0319 | 61.0000 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | 1.0000 | 0.1267 | 0.0000 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.0000 | 0.1203 | 0.2500 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | 0.0641 | 0.0044 | 58.5156 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | 0.0290 | 0.0262 | 29.5000 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 0.0502 | -0.0007 | 29.3906 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 0.0327 | 0.0023 | 13.7188 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 0.0191 | 0.0018 | 15.5000 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0301 | -0.0105 | 14.6875 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | 0.0366 | 0.0165 | 13.4688 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | 1.0000 | 0.1954 | 0.0000 |

## Bundle Comparison

Deltas are `full_bundle − fresh_only`: positive linear / MLP rel_l2 delta means the full bundle makes recovery harder (safer). The two Stage 5.3e bundles share the same per-call fresh-mask sampling under the Stage 6.4c wrapper, so deltas are 0.0 by construction — the bundle label distinguishes security posture, not numerical visibility.

| scope | tensor_name | inter_block_plain | linear_delta | mlp_delta | linkability_delta | perm_top1_delta | risk_fresh_only | risk_full_bundle |
|---|---|---|---|---|---|---|---|---|
| prefill | boundary_input | True | 0.0000 | 0.0150 | 0.0000 | n/a | high | high |
| prefill | q | False | 0.0000 | 0.0078 | 0.0000 | n/a | low | low |
| prefill | k | False | 0.0000 | 0.0050 | 0.0000 | n/a | low | low |
| prefill | v | False | 0.0000 | 0.0040 | 0.0000 | n/a | low | low |
| prefill | gate | False | 0.0000 | -0.0540 | 0.0000 | 0.0000 | low | low |
| prefill | up | False | 0.0000 | 0.0849 | 0.0000 | 0.0000 | low | low |
| prefill | swiglu_intermediate | False | 0.0000 | 0.0846 | 0.0000 | 0.0000 | low | low |
| prefill | post_island | False | 0.0000 | 0.0259 | 0.0000 | n/a | low | low |
| prefill | final | True | 0.0000 | 0.0317 | 0.0000 | n/a | high | high |
| decode | boundary_input | True | 0.0000 | -0.0055 | 0.0000 | n/a | high | high |
| decode | q | False | 0.0000 | 0.0388 | 0.0000 | n/a | low | low |
| decode | k | False | 0.0000 | 0.0056 | 0.0000 | n/a | low | low |
| decode | v | False | 0.0000 | 0.0352 | 0.0000 | n/a | low | low |
| decode | gate | False | 0.0000 | 0.0496 | 0.0000 | 0.0000 | low | low |
| decode | up | False | 0.0000 | 0.0057 | 0.0000 | 0.0000 | low | low |
| decode | swiglu_intermediate | False | 0.0000 | 0.0328 | 0.0000 | 0.0000 | low | low |
| decode | post_island | False | 0.0000 | 0.0113 | 0.0000 | n/a | low | low |
| decode | final | True | 0.0000 | -0.0211 | 0.0000 | n/a | high | high |

## Per-Bundle Headline

Headline grades are reported twice: `masked_only` excludes the structurally-plain inter-block tensors (`boundary_input`, `final`); `overall` includes them so the structural limitation stays visible.

| bundle | tensors | max_risk (masked_only) | max_risk (overall) | mean_lin_rel_l2 (masked) | mean_mlp_rel_l2 (masked) | mean_linkability_cos (masked) |
|---|---|---|---|---|---|---|
| fresh_perm_only | 18 | low | high | 1.419 | 1.212 | 0.028 |
| fresh_perm_plus_sandwich_plus_pad | 18 | low | high | 1.419 | 1.237 | 0.028 |

## Generation Token Match

| bundle | mean_token_match_rate | all_exact_match |
|---|---|---|
| fresh_perm_only | 1.000 | True |
| fresh_perm_plus_sandwich_plus_pad | 1.000 | True |

## Comparison with Stage 5.5 Random-Hidden Real-Activation Attacker

- Stage 5.5 artifact: `outputs/real_activation_attacks.json`
- Stage 5.5b artifact: `outputs/real_token_activation_attacks.json`

- Stage 5.5 feeds random hidden states directly to the Stage 6.4b block wrapper; Stage 5.5b feeds real (or deterministic synthetic) token IDs through the Stage 6.4c model wrapper (embedding + N blocks + final RMSNorm + LM head).
- Stage 5.5b therefore covers the prefill AND decode_step paths, including the masked KV-cache append surface, whereas Stage 5.5 covered only one block's prefill-style forward.
- Stage 5.5 traces use the Stage 6.4b N_res orthogonal residual mask around the block boundary; the Stage 6.4c model wrapper recovers between blocks, so boundary_input / final are PLAIN under Stage 5.5b. This is documented as a structural model-wrapper limitation, not a Stage 5.5b attacker finding.
- Masked-tensor recommendations (Q/K/V/gate/up/swiglu_intermediate/post_island) carry over: rel_l2 stays high, linkability stays low, permutation top1 stays near random chance for the SwiGLU island tensors.

## Recommendation

- `default_on_recommendation_full_bundle_masked_only = "acceptable_with_mitigation_under_real_token_proxy"`
- `default_on_recommendation_full_bundle_overall = "unsafe_default_on_under_real_token_proxy"`
- `default_on_recommendation_fresh_only_masked_only = "acceptable_with_mitigation_under_real_token_proxy"`
- `default_on_recommendation_fresh_only_overall = "unsafe_default_on_under_real_token_proxy"`
- `security_profile_detail_with_real_token_activation = "real-token-real-activation-adaptive-proxy-evaluated, not formal"`
- _Note_: Inter-block tensors (boundary_input, final) are plain at the model-wrapper boundary by construction; their high risk is STRUCTURAL, not a finding against the mitigation bundle. The masked-only recommendation grades the masked tensors only.

## Limitations

- These are real-token-prompted adaptive proxy attacks, not formal security proofs.
- If synthetic token fallback is used, results are not real tokenizer-driven traces.
- If synthetic model fallback is used, results are not real Qwen/TinyLlama traces.
- Prompt set is small and not representative of all user data.
- No black-box query attack is implemented.
- No side-channel attack is implemented.
- No real TEE isolation is evaluated.
- Dense sandwiching reduces tested recovery but does not imply semantic security.
- This stage only evaluates greedy generation traces, not sampling / beam search / top-k / top-p.
- Inter-block hidden states (boundary_input, final) are recovered to plain space between layers under the Stage 6.4c model-level wrapper; an attacker observing those boundaries sees plaintext. This is a known model-wrapper limitation, not a Stage 5.5b attacker finding.
- fresh_perm_only and fresh_perm_plus_sandwich_plus_pad share the same per-call mask sampling under the Stage 6.4c wrapper; the two bundles produce numerically identical traces for masked tensors.
- Not formal security; not a real TEE measurement; synthetic token fallback when tokenizer is unavailable; Dense sandwiching reduces tested recovery but does not imply semantic security.

## Next Stage Plan

- Stage 5.6 — stronger attacker variants under the real-token surface: black-box query attacker (no plaintext supervision), side-channel threat models (timing / cache), ML-based permutation recovery exploiting cross-attention information.
- Stage 7.0 (deferred) — LoRA private-training path under the same obfuscation envelope (mask scheduling under autograd, fresh-mask budget per step).
- Real Qwen / TinyLlama prompts via `--attempt-tokenizer-load --attempt-real-model-load --model-id <id>`.

