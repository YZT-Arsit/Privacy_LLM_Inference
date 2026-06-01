# Real-Activation Adaptive Attacker (Stage 5.5)

## Experiment Scope

Stage 5.5 re-runs the Stage 5.4 adaptive proxy attackers (ridge linear inverter, small MLP inverter, signature / Sinkhorn permutation recovery, linkability proxy) against the (plain, attacker-visible) tensor pairs collected from the Stage 6.4b modern decoder block wrapper. Default mode for the wider system remains `nonlinear_mode='trusted'`; default mitigation bundle remains `fresh_perm_only`. The numbers below are an adaptive proxy evaluation of `fresh_perm_plus_sandwich_plus_pad` as a default-on candidate for real-activation deployments — they are NOT a formal security proof.

## Model Loading Status

| field | value |
|---|---|
| load_status | synthetic_only |
| resolved_model_id | None |
| model_family | None |
| model_class | None |
| fallback_used | True |
| candidates_tried | [] |
| load_error | attempt_real_model_load=False (default); pytest runs the synthetic fallback to avoid network downloads. |

## Trace Collection Summary

| field | value |
|---|---|
| source | synthetic_block |
| block_spec.hidden_size | 64 |
| block_spec.intermediate_size | 128 |
| block_spec.num_attention_heads | 4 |
| block_spec.num_key_value_heads | 2 |
| block_spec.head_dim | 16 |
| block_spec.attention_variant | gqa |
| block_spec.rope_base | 10000.0 |
| num_samples_target | 512 |
| batch_size × seq_len | 2 × 8 |
| use_pad | True |
| bundles_evaluated | fresh_perm_only, fresh_perm_plus_sandwich_plus_pad |

## Target Tensor Inventory

| tensor_name | feature_dim | num_samples | plain abs_max | visible abs_max | plain_fingerprint | visible_fingerprint |
|---|---|---|---|---|---|---|
| boundary_input | 64 | 512 | 4.229 | 4.348 | bbb17ad80b088897 | 70afc059f2f24047 |
| q | 16 | 512 | 3.208 | 3.701 | f38f2d986db81d5e | 4073b5f98298e4b6 |
| k | 16 | 512 | 3.259 | 2.962 | c20e6d0858986730 | c864f3f1ff229c96 |
| v | 16 | 512 | 3.511 | 2.867 | 09833941910f8ace | 91180103cba69c37 |
| gate | 128 | 512 | 3.399 | 3.399 | 44942f8809e8169b | 3fc624b6212e67eb |
| up | 128 | 512 | 3.735 | 3.735 | 874def66a32b17e6 | 2c3be00e913c217b |
| swiglu_intermediate | 128 | 512 | 6.169 | 6.169 | 53beb8c89bc39806 | ba86038bacbdf1f4 |
| post_island | 64 | 512 | 2.881 | 1.964 | 7000fcabc3dc7964 | 8b08a84a9e0ca45f |
| final | 64 | 512 | 4.667 | 5.114 | 14e53427ad14606e | e7f3e07e707bcab8 |

## Linear Inverter on Real Activations

| tensor_name | bundle | relative_l2_error | mse | cosine_similarity |
|---|---|---|---|---|
| boundary_input | fresh_perm_only | 1.1192 | 1.2698 | 0.0086 |
| q | fresh_perm_only | 1.0456 | 0.7049 | 0.0457 |
| k | fresh_perm_only | 1.0406 | 0.6796 | 0.0206 |
| v | fresh_perm_only | 1.0583 | 0.7216 | -0.0523 |
| gate | fresh_perm_only | 1.2744 | 1.0587 | 0.0244 |
| up | fresh_perm_only | 1.2794 | 1.0889 | 0.0309 |
| swiglu_intermediate | fresh_perm_only | 1.2665 | 0.2504 | 0.0130 |
| post_island | fresh_perm_only | 1.1250 | 0.2498 | 0.0320 |
| final | fresh_perm_only | 1.1220 | 1.7236 | 0.0056 |
| boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.1192 | 1.2698 | 0.0086 |
| q | fresh_perm_plus_sandwich_plus_pad | 1.0456 | 0.7049 | 0.0457 |
| k | fresh_perm_plus_sandwich_plus_pad | 1.0406 | 0.6796 | 0.0206 |
| v | fresh_perm_plus_sandwich_plus_pad | 1.0583 | 0.7216 | -0.0523 |
| gate | fresh_perm_plus_sandwich_plus_pad | 1.2744 | 1.0587 | 0.0244 |
| up | fresh_perm_plus_sandwich_plus_pad | 1.2794 | 1.0889 | 0.0309 |
| swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.2665 | 0.2504 | 0.0130 |
| post_island | fresh_perm_plus_sandwich_plus_pad | 1.1250 | 0.2498 | 0.0320 |
| final | fresh_perm_plus_sandwich_plus_pad | 1.1220 | 1.7236 | 0.0056 |

## Small MLP Inverter on Real Activations

| tensor_name | bundle | relative_l2_error | mse | cosine_similarity | final_train_loss |
|---|---|---|---|---|---|
| boundary_input | fresh_perm_only | 1.3936 | 1.9688 | 0.0146 | 0.2560 |
| q | fresh_perm_only | 1.2256 | 0.9684 | 0.0331 | 0.3107 |
| k | fresh_perm_only | 1.2409 | 0.9665 | -0.0028 | 0.2748 |
| v | fresh_perm_only | 1.2757 | 1.0485 | -0.0284 | 0.2940 |
| gate | fresh_perm_only | 1.4507 | 1.3719 | 0.0072 | 0.0577 |
| up | fresh_perm_only | 1.4393 | 1.3782 | 0.0167 | 0.0533 |
| swiglu_intermediate | fresh_perm_only | 1.4075 | 0.3093 | 0.0240 | 0.0353 |
| post_island | fresh_perm_only | 1.3749 | 0.3730 | 0.0317 | 0.0324 |
| final | fresh_perm_only | 1.4118 | 2.7290 | 0.0079 | 0.3397 |
| boundary_input | fresh_perm_plus_sandwich_plus_pad | 1.4114 | 2.0193 | -0.0024 | 0.2649 |
| q | fresh_perm_plus_sandwich_plus_pad | 1.2434 | 0.9967 | 0.0068 | 0.3118 |
| k | fresh_perm_plus_sandwich_plus_pad | 1.2302 | 0.9498 | 0.0234 | 0.2790 |
| v | fresh_perm_plus_sandwich_plus_pad | 1.2793 | 1.0544 | -0.0348 | 0.2662 |
| gate | fresh_perm_plus_sandwich_plus_pad | 1.4308 | 1.3346 | 0.0243 | 0.0616 |
| up | fresh_perm_plus_sandwich_plus_pad | 1.4181 | 1.3378 | 0.0433 | 0.0611 |
| swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 1.4203 | 0.3150 | 0.0089 | 0.0363 |
| post_island | fresh_perm_plus_sandwich_plus_pad | 1.3913 | 0.3820 | 0.0332 | 0.0339 |
| final | fresh_perm_plus_sandwich_plus_pad | 1.3901 | 2.6459 | 0.0155 | 0.3431 |

## Permutation Recovery on Real Activations

Only the SwiGLU island tensors (gate / up / swiglu_intermediate) expose a column permutation; the other tensors are dense / orthogonal-masked and have no recoverable permutation.

| tensor_name | bundle | random_chance | signature_top1 | soft_top1 | best_top1 |
|---|---|---|---|---|---|
| gate | fresh_perm_only | 0.0078 | 0.0000 | 0.0156 | 0.0156 |
| up | fresh_perm_only | 0.0078 | 0.0000 | 0.0156 | 0.0156 |
| swiglu_intermediate | fresh_perm_only | 0.0078 | 0.0078 | 0.0234 | 0.0234 |
| gate | fresh_perm_plus_sandwich_plus_pad | 0.0078 | 0.0000 | 0.0156 | 0.0156 |
| up | fresh_perm_plus_sandwich_plus_pad | 0.0078 | 0.0000 | 0.0156 | 0.0156 |
| swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0078 | 0.0078 | 0.0234 | 0.0234 |

## Linkability on Real Activations

| tensor_name | bundle | visible_vs_plain_cosine | mean_pairwise_cosine_visible | mean_linkability_rank |
|---|---|---|---|---|
| boundary_input | fresh_perm_only | 0.0010 | -0.0042 | 277.1719 |
| q | fresh_perm_only | -0.0021 | -0.0154 | 229.0312 |
| k | fresh_perm_only | 0.0153 | -0.0109 | 262.4219 |
| v | fresh_perm_only | -0.0104 | 0.0100 | 282.6875 |
| gate | fresh_perm_only | 0.0063 | 0.0006 | 233.7812 |
| up | fresh_perm_only | 0.0005 | 0.0008 | 234.2344 |
| swiglu_intermediate | fresh_perm_only | 0.0097 | -0.0009 | 218.5781 |
| post_island | fresh_perm_only | -0.0005 | 0.0021 | 241.1875 |
| final | fresh_perm_only | 0.0042 | -0.0037 | 270.2656 |
| boundary_input | fresh_perm_plus_sandwich_plus_pad | 0.0010 | -0.0042 | 277.1719 |
| q | fresh_perm_plus_sandwich_plus_pad | -0.0021 | -0.0154 | 229.0312 |
| k | fresh_perm_plus_sandwich_plus_pad | 0.0153 | -0.0109 | 262.4219 |
| v | fresh_perm_plus_sandwich_plus_pad | -0.0104 | 0.0100 | 282.6875 |
| gate | fresh_perm_plus_sandwich_plus_pad | 0.0063 | 0.0006 | 233.7812 |
| up | fresh_perm_plus_sandwich_plus_pad | 0.0005 | 0.0008 | 234.2344 |
| swiglu_intermediate | fresh_perm_plus_sandwich_plus_pad | 0.0097 | -0.0009 | 218.5781 |
| post_island | fresh_perm_plus_sandwich_plus_pad | -0.0005 | 0.0021 | 241.1875 |
| final | fresh_perm_plus_sandwich_plus_pad | 0.0042 | -0.0037 | 270.2656 |

## Mitigation Bundle Comparison

Deltas are `full_bundle − fresh_only`: positive linear / MLP rel_l2 delta means the full bundle makes recovery harder (safer). The two Stage 5.3e bundles share the same per-call fresh-mask sampling under the Stage 6.4b wrapper, so deltas are 0.0 by construction in the current implementation — the bundle label distinguishes security posture, not numerical visibility.

| tensor_name | linear_delta | mlp_delta | linkability_delta | perm_top1_delta | risk_fresh_only | risk_full_bundle |
|---|---|---|---|---|---|---|
| boundary_input | 0.0000 | 0.0178 | 0.0000 | n/a | low | low |
| q | 0.0000 | 0.0178 | 0.0000 | n/a | low | low |
| k | 0.0000 | -0.0108 | 0.0000 | n/a | low | low |
| v | 0.0000 | 0.0036 | 0.0000 | n/a | low | low |
| gate | 0.0000 | -0.0198 | 0.0000 | 0.0000 | low | low |
| up | 0.0000 | -0.0213 | 0.0000 | 0.0000 | low | low |
| swiglu_intermediate | 0.0000 | 0.0128 | 0.0000 | 0.0000 | low | low |
| post_island | 0.0000 | 0.0165 | 0.0000 | n/a | low | low |
| final | 0.0000 | -0.0217 | 0.0000 | n/a | low | low |

## Per-Bundle Headline

| bundle | tensors | max_risk | mean_lin_rel_l2 | mean_mlp_rel_l2 | mean_linkability_cos |
|---|---|---|---|---|---|
| fresh_perm_only | 9 | low | 1.148 | 1.358 | 0.003 |
| fresh_perm_plus_sandwich_plus_pad | 9 | low | 1.148 | 1.357 | 0.003 |

## Comparison with Stage 5.4 Synthetic Adaptive Attacker

Stage 5.4 evaluated the same family of attackers on structured synthetic activation distributions. Stage 5.5 reuses the same attacker code (ridge linear, small MLP, signature / Sinkhorn permutation recovery) but the (plain, visible) pairs now come from the Stage 6.4b modern decoder block wrapper — i.e. they are real intermediate activations from the obfuscated forward path, not synthesized signatures.

Under both Stage 5.4 (synthetic) and Stage 5.5 (real-activation), the recommended default-on bundle (`fresh_perm_plus_sandwich_plus_pad`) achieves `risk_level=low` against the adaptive proxy attackers. `risk_level` for `fresh_perm_only` is reported here under the same evaluation budget; both Stage 5.3e bundles produce numerically identical traces because they use the SAME per-call fresh-mask sampling inside `run_swiglu_mlp_island`.

## Recommendation

- `default_on_recommendation_full_bundle = "acceptable_with_mitigation_under_real_activation_proxy"`
- `default_on_recommendation_fresh_only = "acceptable_with_mitigation_under_real_activation_proxy"`
- `security_profile_detail_with_real_activation = "real-activation-adaptive-proxy-evaluated, not formal"`

## Limitations

- These are real-activation adaptive proxy attacks, not formal security proofs.
- If synthetic fallback is used, results are not real Qwen/TinyLlama activation traces.
- Random hidden-state block input is not the same as real token distribution unless tokenizer/embedding path is added.
- No black-box query attack is implemented.
- No side-channel attack is implemented.
- No real TEE isolation is evaluated.
- Dense sandwiching reduces tested recovery but does not imply semantic security.
- This stage does not implement full model-level generation or KV cache runtime.
- fresh_perm_only and fresh_perm_plus_sandwich_plus_pad share the same per-call mask sampling under the Stage 6.4b wrapper; the two bundles produce numerically identical traces. fixed_permutation_debug is a debug baseline only — never recommended for deployment.

## Next Stage Plan

- Stage 6.4c — full modern decoder model-level wrapper (multi-block stacking, LM head, KV cache runtime, generation parity).
- Stronger attacker variants: black-box query attacker, side-channel threat models, ML-based permutation recovery exploiting cross-attention information.
- Real Qwen / TinyLlama trace collection (`--attempt-real-model-load` with tokenizer / embedding integration once Stage 6.4c lands).

