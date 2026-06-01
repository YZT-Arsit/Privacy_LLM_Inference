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
| boundary_input | 32 | 128 | 1.419 | 0.946 | 6875e19e32afa32c | 09f624fedc2ebd9e |
| q | 8 | 512 | 1.976 | 2.052 | 3b44fe0b5e1d4a9e | 0cf0c990c23ae3f0 |
| k | 8 | 256 | 1.974 | 1.602 | 33a9e5220726f324 | 3413cd1aa17c80b2 |
| v | 8 | 256 | 2.088 | 2.215 | a203d3b7c3318016 | be99f8e732961fa7 |
| gate | 64 | 128 | 2.175 | 2.175 | 464b84e1115fc172 | 7810286294a77bf3 |
| up | 64 | 128 | 2.129 | 2.129 | 27fbf88ca8e9b42d | 04fa045ae0acd3c2 |
| swiglu_intermediate | 64 | 128 | 2.065 | 2.065 | 918fc6c793567345 | a40a8e797ef476e6 |
| post_island | 32 | 128 | 0.833 | 0.848 | 9a85d85d6483e3ca | 09f08d81672de48a |
| final | 32 | 128 | 1.625 | 1.458 | c71ef525c9b0d2f0 | a3c3eede4cc9f35f |

## Target Tensor Inventory (Decode)

| tensor_name | feature_dim | num_samples | plain abs_max | visible abs_max |
|---|---|---|---|---|
| boundary_input | 32 | 32 | 0.632 | 0.726 |
| q | 8 | 128 | 1.847 | 1.918 |
| k | 8 | 64 | 1.507 | 2.058 |
| v | 8 | 64 | 1.893 | 2.115 |
| gate | 64 | 32 | 2.160 | 2.160 |
| up | 64 | 32 | 1.730 | 1.730 |
| swiglu_intermediate | 64 | 32 | 1.442 | 1.442 |
| post_island | 32 | 32 | 0.415 | 0.553 |
| final | 32 | 32 | 1.235 | 0.881 |

## Prefill Real-Token Activation Attacks

Per-tensor attacker results on real-token-driven prefill activations. Tensor-by-tensor breakdowns follow in the Linear / MLP / Permutation sections.

## Decode-Step Real-Token Activation Attacks

Per-tensor attacker results on the single-token decode_step activations under the masked KV-cache append surface.

## Linear Inverter Results

| scope | tensor_name | bundle | relative_l2_error | mse | cosine_similarity |
|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | 1.8045 | 0.1507 | -0.1596 |
| prefill | q | fresh_perm_only | 1.0398 | 0.3260 | 0.0133 |
| prefill | k | fresh_perm_only | 1.0254 | 0.2704 | 0.1061 |
| prefill | v | fresh_perm_only | 1.0659 | 0.3579 | 0.0278 |
| prefill | gate | fresh_perm_only | 2.0754 | 1.3649 | -0.0121 |
| prefill | up | fresh_perm_only | 1.8763 | 1.1986 | 0.1253 |
| prefill | swiglu_intermediate | fresh_perm_only | 2.1048 | 0.1244 | -0.0527 |
| prefill | post_island | fresh_perm_only | 1.4238 | 0.0400 | 0.0045 |
| prefill | final | fresh_perm_only | 1.6229 | 0.2789 | -0.0438 |
| decode | boundary_input | fresh_perm_only | 1.4481 | 0.0450 | 0.1184 |
| decode | q | fresh_perm_only | 1.0826 | 0.3443 | 0.0388 |
| decode | k | fresh_perm_only | 1.1091 | 0.3638 | 0.1619 |
| decode | v | fresh_perm_only | 1.2424 | 0.4199 | -0.0689 |
| decode | gate | fresh_perm_only | 1.0980 | 0.3622 | 0.2281 |
| decode | up | fresh_perm_only | 1.1337 | 0.4415 | 0.0657 |
| decode | swiglu_intermediate | fresh_perm_only | 1.3047 | 0.0476 | -0.0733 |
| decode | post_island | fresh_perm_only | 1.7099 | 0.0521 | -0.0619 |
| decode | final | fresh_perm_only | 1.3586 | 0.1226 | 0.1856 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.8045 | 0.1507 | -0.1596 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 1.0398 | 0.3260 | 0.0133 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | 1.0254 | 0.2704 | 0.1061 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 1.0659 | 0.3579 | 0.0278 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 2.0754 | 1.3649 | -0.0121 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 1.8763 | 1.1986 | 0.1253 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 2.1048 | 0.1244 | -0.0527 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | 1.4238 | 0.0400 | 0.0045 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | 1.6229 | 0.2789 | -0.0438 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.4481 | 0.0450 | 0.1184 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | 1.0826 | 0.3443 | 0.0388 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | 1.1091 | 0.3638 | 0.1619 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 1.2424 | 0.4199 | -0.0689 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 1.0980 | 0.3622 | 0.2281 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 1.1337 | 0.4415 | 0.0657 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.3047 | 0.0476 | -0.0733 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | 1.7099 | 0.0521 | -0.0619 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | 1.3586 | 0.1226 | 0.1856 |

## Small MLP Inverter Results

| scope | tensor_name | bundle | relative_l2_error | mse | cosine_similarity | final_train_loss |
|---|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | 1.2973 | 0.0779 | 0.0508 | 0.0005 |
| prefill | q | fresh_perm_only | 1.1405 | 0.3921 | 0.0105 | 0.1642 |
| prefill | k | fresh_perm_only | 1.1966 | 0.3682 | 0.0912 | 0.0910 |
| prefill | v | fresh_perm_only | 1.1586 | 0.4228 | 0.1814 | 0.0859 |
| prefill | gate | fresh_perm_only | 1.2089 | 0.4631 | 0.0834 | 0.0005 |
| prefill | up | fresh_perm_only | 1.1536 | 0.4531 | 0.1578 | 0.0005 |
| prefill | swiglu_intermediate | fresh_perm_only | 1.2262 | 0.0422 | 0.0359 | 0.0000 |
| prefill | post_island | fresh_perm_only | 1.2090 | 0.0288 | 0.1118 | 0.0001 |
| prefill | final | fresh_perm_only | 1.3377 | 0.1895 | 0.0471 | 0.0003 |
| decode | boundary_input | fresh_perm_only | 1.1784 | 0.0298 | 0.2261 | 0.0000 |
| decode | q | fresh_perm_only | 1.5024 | 0.6632 | -0.0316 | 0.0116 |
| decode | k | fresh_perm_only | 1.2939 | 0.4951 | 0.1542 | 0.0005 |
| decode | v | fresh_perm_only | 1.4185 | 0.5474 | 0.0084 | 0.0016 |
| decode | gate | fresh_perm_only | 1.1501 | 0.3974 | 0.1760 | 0.0011 |
| decode | up | fresh_perm_only | 1.0842 | 0.4038 | 0.1967 | 0.0011 |
| decode | swiglu_intermediate | fresh_perm_only | 1.2120 | 0.0411 | -0.0683 | 0.0003 |
| decode | post_island | fresh_perm_only | 1.1830 | 0.0249 | 0.0535 | 0.0000 |
| decode | final | fresh_perm_only | 1.1820 | 0.0928 | 0.2860 | 0.0000 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.3466 | 0.0839 | -0.0205 | 0.0004 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 1.2084 | 0.4402 | -0.0267 | 0.1685 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | 1.2182 | 0.3817 | 0.0746 | 0.0953 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 1.1597 | 0.4236 | 0.2001 | 0.0839 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 1.2720 | 0.5127 | 0.0293 | 0.0004 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 1.1540 | 0.4535 | 0.1438 | 0.0005 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.2693 | 0.0452 | 0.0092 | 0.0000 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | 1.2781 | 0.0322 | 0.0856 | 0.0001 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | 1.3592 | 0.1956 | 0.0166 | 0.0003 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.1018 | 0.0261 | 0.2620 | 0.0000 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | 1.4268 | 0.5981 | 0.0342 | 0.0140 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | 1.2879 | 0.4905 | 0.1625 | 0.0005 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 1.3523 | 0.4975 | 0.0408 | 0.0012 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 1.1350 | 0.3870 | 0.1599 | 0.0007 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 1.1000 | 0.4156 | 0.1656 | 0.0002 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.2401 | 0.0430 | -0.0932 | 0.0001 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | 1.1791 | 0.0248 | 0.0332 | 0.0000 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | 1.1050 | 0.0811 | 0.3045 | 0.0000 |

## Permutation Recovery Results

Only the SwiGLU island tensors (gate / up / swiglu_intermediate) expose a column permutation; the other tensors are dense / orthogonal-masked or plain at the inter-block boundary.

| scope | tensor_name | bundle | random_chance | signature_top1 | soft_top1 | best_top1 |
|---|---|---|---|---|---|---|
| prefill | gate | fresh_perm_only | 0.0156 | 0.0156 | 0.0312 | 0.0312 |
| prefill | up | fresh_perm_only | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| prefill | swiglu_intermediate | fresh_perm_only | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| decode | gate | fresh_perm_only | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| decode | up | fresh_perm_only | 0.0156 | 0.0312 | 0.0000 | 0.0312 |
| decode | swiglu_intermediate | fresh_perm_only | 0.0156 | 0.0469 | 0.0312 | 0.0469 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0312 | 0.0312 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0312 | 0.0156 | 0.0312 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0156 | 0.0156 | 0.0156 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0312 | 0.0000 | 0.0312 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0156 | 0.0469 | 0.0312 | 0.0469 |

## Linkability Results

| scope | tensor_name | bundle | visible_vs_plain_cosine | mean_pairwise_cosine_visible | mean_linkability_rank |
|---|---|---|---|---|---|
| prefill | boundary_input | fresh_perm_only | -0.0319 | 0.0263 | 64.3438 |
| prefill | q | fresh_perm_only | 0.0111 | 0.0135 | 229.0000 |
| prefill | k | fresh_perm_only | -0.0379 | 0.0082 | 141.0625 |
| prefill | v | fresh_perm_only | 0.0009 | 0.0127 | 123.0625 |
| prefill | gate | fresh_perm_only | 0.0408 | 0.0335 | 45.9062 |
| prefill | up | fresh_perm_only | 0.0312 | 0.0471 | 62.1406 |
| prefill | swiglu_intermediate | fresh_perm_only | 0.0333 | 0.0300 | 45.6406 |
| prefill | post_island | fresh_perm_only | -0.0384 | 0.0205 | 66.0781 |
| prefill | final | fresh_perm_only | -0.0357 | 0.1002 | 56.8281 |
| decode | boundary_input | fresh_perm_only | -0.0873 | 0.0106 | 19.4062 |
| decode | q | fresh_perm_only | -0.0363 | 0.0179 | 68.0156 |
| decode | k | fresh_perm_only | -0.0493 | 0.0626 | 33.5625 |
| decode | v | fresh_perm_only | 0.0099 | 0.0218 | 30.5469 |
| decode | gate | fresh_perm_only | 0.0518 | 0.0055 | 12.5000 |
| decode | up | fresh_perm_only | 0.0089 | -0.0030 | 15.6562 |
| decode | swiglu_intermediate | fresh_perm_only | 0.0595 | -0.0066 | 12.1562 |
| decode | post_island | fresh_perm_only | -0.0037 | -0.0026 | 15.4062 |
| decode | final | fresh_perm_only | -0.0874 | 0.0714 | 18.9062 |
| prefill | boundary_input | fresh_perm_plus_sandwich_plus_pad | -0.0319 | 0.0263 | 64.3438 |
| prefill | q | fresh_perm_plus_sandwich_plus_pad | 0.0111 | 0.0135 | 229.0000 |
| prefill | k | fresh_perm_plus_sandwich_plus_pad | -0.0379 | 0.0082 | 141.0625 |
| prefill | v | fresh_perm_plus_sandwich_plus_pad | 0.0009 | 0.0127 | 123.0625 |
| prefill | gate | fresh_perm_plus_sandwich_plus_pad | 0.0408 | 0.0335 | 45.9062 |
| prefill | up | fresh_perm_plus_sandwich_plus_pad | 0.0312 | 0.0471 | 62.1406 |
| prefill | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0333 | 0.0300 | 45.6406 |
| prefill | post_island | fresh_perm_plus_sandwich_plus_pad | -0.0384 | 0.0205 | 66.0781 |
| prefill | final | fresh_perm_plus_sandwich_plus_pad | -0.0357 | 0.1002 | 56.8281 |
| decode | boundary_input | fresh_perm_plus_sandwich_plus_pad | -0.0873 | 0.0106 | 19.4062 |
| decode | q | fresh_perm_plus_sandwich_plus_pad | -0.0363 | 0.0179 | 68.0156 |
| decode | k | fresh_perm_plus_sandwich_plus_pad | -0.0493 | 0.0626 | 33.5625 |
| decode | v | fresh_perm_plus_sandwich_plus_pad | 0.0099 | 0.0218 | 30.5469 |
| decode | gate | fresh_perm_plus_sandwich_plus_pad | 0.0518 | 0.0055 | 12.5000 |
| decode | up | fresh_perm_plus_sandwich_plus_pad | 0.0089 | -0.0030 | 15.6562 |
| decode | swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0595 | -0.0066 | 12.1562 |
| decode | post_island | fresh_perm_plus_sandwich_plus_pad | -0.0037 | -0.0026 | 15.4062 |
| decode | final | fresh_perm_plus_sandwich_plus_pad | -0.0874 | 0.0714 | 18.9062 |

## Bundle Comparison

Deltas are `full_bundle − fresh_only`: positive linear / MLP rel_l2 delta means the full bundle makes recovery harder (safer). The two Stage 5.3e bundles share the same per-call fresh-mask sampling under the Stage 6.4c wrapper, so deltas are 0.0 by construction — the bundle label distinguishes security posture, not numerical visibility.

| scope | tensor_name | inter_block_plain | linear_delta | mlp_delta | linkability_delta | perm_top1_delta | risk_fresh_only | risk_full_bundle |
|---|---|---|---|---|---|---|---|---|
| prefill | boundary_input | False | 0.0000 | 0.0494 | 0.0000 | n/a | low | low |
| prefill | q | False | 0.0000 | 0.0679 | 0.0000 | n/a | low | low |
| prefill | k | False | 0.0000 | 0.0217 | 0.0000 | n/a | low | low |
| prefill | v | False | 0.0000 | 0.0011 | 0.0000 | n/a | low | low |
| prefill | gate | False | 0.0000 | 0.0631 | 0.0000 | 0.0000 | low | low |
| prefill | up | False | 0.0000 | 0.0005 | 0.0000 | 0.0000 | low | low |
| prefill | swiglu_intermediate | False | 0.0000 | 0.0430 | 0.0000 | 0.0000 | low | low |
| prefill | post_island | False | 0.0000 | 0.0691 | 0.0000 | n/a | low | low |
| prefill | final | False | 0.0000 | 0.0215 | 0.0000 | n/a | low | low |
| decode | boundary_input | False | 0.0000 | -0.0766 | 0.0000 | n/a | low | low |
| decode | q | False | 0.0000 | -0.0756 | 0.0000 | n/a | low | low |
| decode | k | False | 0.0000 | -0.0060 | 0.0000 | n/a | low | low |
| decode | v | False | 0.0000 | -0.0662 | 0.0000 | n/a | low | low |
| decode | gate | False | 0.0000 | -0.0151 | 0.0000 | 0.0000 | low | low |
| decode | up | False | 0.0000 | 0.0158 | 0.0000 | 0.0000 | low | low |
| decode | swiglu_intermediate | False | 0.0000 | 0.0282 | 0.0000 | 0.0000 | low | low |
| decode | post_island | False | 0.0000 | -0.0039 | 0.0000 | n/a | low | low |
| decode | final | False | 0.0000 | -0.0770 | 0.0000 | n/a | low | low |

## Per-Bundle Headline

Headline grades are reported twice: `masked_only` excludes the structurally-plain inter-block tensors (`boundary_input`, `final`); `overall` includes them so the structural limitation stays visible.

| bundle | tensors | max_risk (masked_only) | max_risk (overall) | mean_lin_rel_l2 (masked) | mean_mlp_rel_l2 (masked) | mean_linkability_cos (masked) |
|---|---|---|---|---|---|---|
| fresh_perm_only | 18 | low | low | 1.418 | 1.230 | -0.009 |
| fresh_perm_plus_sandwich_plus_pad | 18 | low | low | 1.418 | 1.233 | -0.009 |

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
- `default_on_recommendation_full_bundle_overall = "acceptable_with_mitigation_under_real_token_proxy"`
- `default_on_recommendation_fresh_only_masked_only = "acceptable_with_mitigation_under_real_token_proxy"`
- `default_on_recommendation_fresh_only_overall = "acceptable_with_mitigation_under_real_token_proxy"`
- `security_profile_detail_with_real_token_activation = "real-token-real-activation-adaptive-proxy-evaluated, not formal"`
- _Note_: Inter-block tensors (boundary_input, final) are plain at the model-wrapper boundary by construction under plain_boundary mode; their high risk is STRUCTURAL, not a finding against the mitigation bundle. Under masked_boundary_experimental (Stage 5.6 extension) the inter-block residual is masked by a fresh orthogonal N_inter so boundary_input / final join the masked tensor set and the overall recommendation is bounded by the masked-only grade.

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

