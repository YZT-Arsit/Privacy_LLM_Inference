# Modern Decoder Block-Level Wrapper Smoke (Stage 6.4b)

## Experiment Scope

Stage 6.4b adds a block-level obfuscated forward for modern decoder-only architectures (LLaMA / TinyLlama / Qwen / Qwen2). The wrapper runs both a plain reference and a Stage 5.2a / 5.3e compatible-islands path on a single block, recovers the masked output, and reports allclose for each (use_pad × mitigation_bundle × nonlinear_mode) combination. Default mode for the wider system remains 'trusted' and the default mitigation bundle remains 'fresh_perm_only'.

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

## Modern Decoder Block Spec

| field | value |
|---|---|
| model_family | synthetic_modern_decoder |
| model_class | SyntheticLlamaBlock |
| block_path | synthetic.layers.0 |
| block_index | 0 |
| hidden_size | 64 |
| intermediate_size | 128 |
| num_attention_heads | 4 |
| num_key_value_heads | 2 |
| head_dim | 16 |
| norm_type | rmsnorm |
| activation_type | swiglu |
| position_encoding_type | rotary |
| attention_variant | gqa |
| rope_base | 10000.0 |
| rope_scaling_kind | None |

## Plain Reference vs HF Block Status

Plain reference is constructed from the **extracted weights** (no HF block forward call required); residual / RMSNorm / RoPE / GQA / SwiGLU paths are computed in row-vector convention. The obfuscated path is compared against this plain reference. For synthetic fallback, the plain reference is also synthetic.

## RMSNorm Handling

- Mode: **orthogonal_island_with_gamma_folded_into_qkv**.
- N_res is orthogonal so rmsnorm_core(X @ N_res) = rmsnorm_core(X) @ N_res.
- γ (input RMSNorm and post-attention RMSNorm) is folded into the adjacent q/k/v and gate/up projection weights.

## RoPE-Aware Attention Handling

- Mode: **rope_post_mask_only**.
- RoPE is applied to plain q/k first; per-head Q/K masks N_Q, N_K with N_Q N_K^T = I are applied AFTER RoPE.
- Pre-RoPE dense-mask commutation is not assumed.

## GQA / MQA Handling

- attention_variant: gqa
- num_attention_heads=4, num_key_value_heads=2, head_dim=16
- One N_K / N_V per kv-head; per-q-head N_Q is derived from the corresponding kv-head's N_K via N_Q = N_K^{-T}.
- repeat_kv is applied AFTER masking, matching HF semantics.

## SwiGLU Compatible Island Handling

- Mode: **compatible_island_paired_permutation**.
- run_swiglu_mlp_island with shared permutation P on the up- and gate-branches.
- pad_placement is linear_boundary_only; pad is never pushed through SwiGLU.
- online_extra_matmul_count = 0.

## Mitigation Bundle Results

| bundle | use_pad | nonlinear_mode | max_abs_error | rel_l2_error | allclose | dense_sandwich_enabled | boundary_pad_enabled | default_on_candidate |
|---|---|---|---|---|---|---|---|---|
| fresh_perm_only | false | compatible_islands | 1.907e-06 | 4.055e-07 | true | false | false | false |
| fresh_perm_only | true | compatible_islands | 2.146e-06 | 4.706e-07 | true | false | true | false |
| fresh_perm_plus_sandwich_plus_pad | false | compatible_islands | 1.691e-06 | 4.089e-07 | true | true | false | false |
| fresh_perm_plus_sandwich_plus_pad | true | compatible_islands | 2.146e-06 | 4.952e-07 | true | true | true | true |

- all_runs_allclose: **true**
- online_extra_matmul_count: 0
- implemented_block_level: true
- full_runtime_integrated: false
- mitigation_bundles_evaluated: ['fresh_perm_only', 'fresh_perm_plus_sandwich_plus_pad']

## Limitations

- Block-level integration; not a full Qwen / TinyLlama model-level wrapper.
- No generation / decode_step / KV cache runtime is implemented.
- RoPE is handled using post-RoPE per-head masking; mask-before-RoPE dense commutation is not assumed.
- If synthetic fallback is used, results are not from real Qwen / TinyLlama weights.
- RMSNorm γ is folded into adjacent projection weights; the norm core runs in an orthogonal residual mask space.
- Residual alignment uses the same orthogonal N_res on both branches.
- Inherits Stage 5.4 mitigation requirements (fresh permutation + dense sandwich + boundary pad).
- This is not a real TEE measurement.
- This is not formal security.

## Next Stage Plan

- Stage 5.5 — adaptive attacker on real modern-decoder activations once block-level extraction is stable.
- Stage 6.4c — full modern-decoder model-level wrapper (multi-block, LM head, generation).
- Stage 5.3d — full BERT / T5 wrapper (MLM head, encoder-decoder generation) remains scheduled but is engineering-heavy.

