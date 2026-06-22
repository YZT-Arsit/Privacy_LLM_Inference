# Stage 6.4.1 -- RoPE-Compatible Masked GQA/MHA Attention Probe

- Stage: **6.4.1_rope_gqa_complex_scaling**
- Status: **ok** | all_allclose: **True**
- Security status: **operator_compatible_leakage_reduction_not_semantic_security**
- no_intermediate_tee: **True**
- Synthetic tensor-level probe; CPU float64; no HF model loading.

> RoPE-compatible masks are correctness-preserving and reduce direct leakage, but they are weaker than dense masks. They are used because no intermediate TEE is allowed.

## Config

- batch_size: `2`
- seq_len: `8`
- decode_steps: `3`
- hidden_size: `32`
- num_heads: `4`
- num_key_value_heads: `2`
- rope_base: `10000.0`
- dtype: `float64`
- device: `cpu`
- seed: `2027`
- mask_family: `pairwise_complex_scaling`
- run_rotation_baseline: `True`
- run_complex_scaling: `True`
- run_leakage_proxy: `True`
- leakage_num_samples: `128`

## Correctness

## pairwise_rotation (correctness baseline)

### MHA case (num_heads=4, num_key_value_heads=4, head_dim=8)

| metric | max_abs_error |
|---|---|
| `rope_commutation_max_error` | 4.441e-16 |
| `score_max_abs_error` | 2.842e-14 |
| `prob_max_abs_error` | 1.665e-15 |
| `v_aggregation_max_abs_error` | 3.197e-14 |
| `output_max_abs_error` | 9.948e-14 |
| `prefill_cache_key_max_abs_error` | 3.553e-15 |
| `prefill_cache_value_max_abs_error` | 0.000e+00 |
| `rope_commutation_q_error` | 3.553e-15 |
| `cache_append_key_max_abs_error` | 1.776e-15 |
| `cache_append_value_max_abs_error` | 0.000e+00 |
| **allclose** | **True** |

Decode steps:

| step | position | output | key | value |
|---|---|---|---|---|
| 0 | 8 | 2.132e-14 | 1.776e-15 | 0.000e+00 |
| 1 | 9 | 3.091e-13 | 1.776e-15 | 0.000e+00 |
| 2 | 10 | 7.283e-14 | 1.776e-15 | 0.000e+00 |

### GQA case (num_heads=4, num_key_value_heads=2, head_dim=8)

| metric | max_abs_error |
|---|---|
| `rope_commutation_max_error` | 4.441e-16 |
| `score_max_abs_error` | 4.263e-14 |
| `prob_max_abs_error` | 2.026e-15 |
| `v_aggregation_max_abs_error` | 5.684e-14 |
| `output_max_abs_error` | 1.492e-13 |
| `prefill_cache_key_max_abs_error` | 1.776e-15 |
| `prefill_cache_value_max_abs_error` | 0.000e+00 |
| `rope_commutation_q_error` | 3.553e-15 |
| `cache_append_key_max_abs_error` | 1.943e-15 |
| `cache_append_value_max_abs_error` | 0.000e+00 |
| **allclose** | **True** |

Decode steps:

| step | position | output | key | value |
|---|---|---|---|---|
| 0 | 8 | 2.842e-14 | 1.776e-15 | 0.000e+00 |
| 1 | 9 | 7.105e-14 | 1.776e-15 | 0.000e+00 |
| 2 | 10 | 3.553e-14 | 1.943e-15 | 0.000e+00 |

## pairwise_complex_scaling (preferred RoPE-compatible mask)

### MHA case (num_heads=4, num_key_value_heads=4, head_dim=8)

| metric | max_abs_error |
|---|---|
| `rope_commutation_max_error` | 6.661e-16 |
| `score_max_abs_error` | 4.263e-14 |
| `prob_max_abs_error` | 2.442e-15 |
| `v_aggregation_max_abs_error` | 3.064e-14 |
| `output_max_abs_error` | 1.181e-13 |
| `prefill_cache_key_max_abs_error` | 7.105e-15 |
| `prefill_cache_value_max_abs_error` | 0.000e+00 |
| `rope_commutation_q_error` | 9.770e-15 |
| `cache_append_key_max_abs_error` | 3.553e-15 |
| `cache_append_value_max_abs_error` | 0.000e+00 |
| **allclose** | **True** |

Decode steps:

| step | position | output | key | value |
|---|---|---|---|---|
| 0 | 8 | 4.263e-14 | 3.553e-15 | 0.000e+00 |
| 1 | 9 | 2.345e-13 | 3.553e-15 | 0.000e+00 |
| 2 | 10 | 3.553e-14 | 3.553e-15 | 0.000e+00 |

### GQA case (num_heads=4, num_key_value_heads=2, head_dim=8)

| metric | max_abs_error |
|---|---|
| `rope_commutation_max_error` | 8.882e-16 |
| `score_max_abs_error` | 3.553e-14 |
| `prob_max_abs_error` | 1.305e-15 |
| `v_aggregation_max_abs_error` | 3.020e-14 |
| `output_max_abs_error` | 1.421e-13 |
| `prefill_cache_key_max_abs_error` | 3.553e-15 |
| `prefill_cache_value_max_abs_error` | 0.000e+00 |
| `rope_commutation_q_error` | 7.105e-15 |
| `cache_append_key_max_abs_error` | 3.553e-15 |
| `cache_append_value_max_abs_error` | 0.000e+00 |
| **allclose** | **True** |

Decode steps:

| step | position | output | key | value |
|---|---|---|---|---|
| 0 | 8 | 3.553e-14 | 3.553e-15 | 0.000e+00 |
| 1 | 9 | 5.684e-14 | 3.553e-15 | 0.000e+00 |
| 2 | 10 | 3.553e-14 | 3.553e-15 | 0.000e+00 |

## Leakage proxy (NOT a security proof)

| mode | same-session pair-norm corr | cross-session pair-norm corr | NN matching acc (pair-norm) |
|---|---|---|---|
| `no_mask` | 1.0000 | 1.0000 | 1.0000 |
| `pairwise_rotation` | 1.0000 | 1.0000 | 1.0000 |
| `pairwise_complex_scaling` | 0.7557 | 0.8896 | 0.0391 |

- num_samples: `128`, feature: `pair_norm`, leakage_proxy_is_not_security_proof: `True`
- preserved structure: RoPE pair partition; No cross-pair dense mixing inside RoPE-compatible region; Attention scores are preserved by construction; KV cache masks are reused within a generation session

## Mask structure

- `rope_pairwise_commuting`: True
- `default_mask_family`: pairwise_complex_scaling
- `q_mask`: per-query-head inverse-transpose of mapped KV mask
- `k_mask`: per-kv-head RoPE-compatible block mask
- `v_mask`: per-kv-head RoPE-compatible block mask
- `gqa_supported`: True
- `same_cache_mask_within_session`: True

## Leakage caveats

- RoPE pair partition is preserved
- No cross-pair dense mixing inside RoPE-compatible region
- rotation mode preserves per-pair norm
- complex-scaling mode changes per-pair norm but preserves pair structure
- KV cache requires same mask within a generation session
- attention scores remain visible to the GPU in this probe
- This is not a semantic-security proof

## Limitations

- Synthetic tensor-level probe, not a HF LLaMA/Qwen wrapper.
- q_proj/k_proj/v_proj/o_proj weight folding is not yet integrated (masks applied at the Q/K/V tensor level).
- RoPE scaling variants (NTK/YaRN) are not implemented.
- Real model embedding, LM head, sampling, and full generation are not covered.
- RoPE-compatible masks are weaker than dense masks; dense masks should be restored before/after RoPE-constrained regions in later block integration.
- CPU-only; no formal, cryptographic, or semantic security is claimed.

