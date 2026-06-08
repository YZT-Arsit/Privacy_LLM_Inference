# Sliding Window Attention / Rolling KV Cache

_Stage 7.8a: verify Stage 7.6g/h/i masked invariants under sliding window attention and rolling KV cache._

## Configuration

| Field | Value |
|---|---|
| batch_size | 2 |
| prompt_len | 6 |
| max_new_tokens | 4 |
| head_dim | 16 |
| num_q_heads | 2 |
| num_kv_heads | 2 |
| window_sizes | [2, 4, 999] |

## Standard Causal vs Sliding Window

Standard causal attention: query at position ``t`` attends to all keys ``[0, t]``. Sliding window with window ``w``: query at ``t`` only attends to keys in ``[max(0, t-w+1), t]``. When ``w >= s_total`` the two coincide (verified by the ``full_vs_sliding_match_when_window_ge_seqlen`` column below).

## Per-Window Results

| window | attention_privacy_mode | score_invariant_max | kv_window_invariant_max | attn_out_recover_max | window_eviction_correct | full_vs_sliding_max |
|---|---|---|---|---|---|---|
| 2 | `exact_visible_attention` | 4.441e-16 | 8.882e-16 | 1.776e-15 | True | n/a |
| 2 | `trusted_softmax_attention` | 6.106e-16 | 8.882e-16 | 1.332e-15 | True | n/a |
| 4 | `exact_visible_attention` | 8.882e-16 | 8.882e-16 | 1.429e-15 | True | n/a |
| 4 | `trusted_softmax_attention` | 6.661e-16 | 8.882e-16 | 8.882e-16 | True | n/a |
| full | `exact_visible_attention` | 8.882e-16 | 1.110e-15 | 9.992e-16 | True | 6.661e-16 |
| full | `trusted_softmax_attention` | 8.882e-16 | 8.882e-16 | 8.882e-16 | True | 4.441e-16 |

## Stage 7.6g Carry-Over Diagnostics

| Field | Value |
|---|---|
| use_pad | True |
| rope_mask_mode | pre_rope_block_diagonal_rotation |
| rope_transient_plain_qk_visible | False |
| qkv_projection_outputs_masked_directly | True |
| pad_enters_rmsnorm_core | False |
| pad_enters_rope_core | False |
| pad_enters_swiglu_core | False |
| pad_enters_softmax | False |
| greedy_token_match_rate | 1.0 |
| sequence_exact_match | True |

## Limitations

- CPU local emulation only.
- No real FlashAttention / sliding-window CUDA kernel.
- Window size policy is PUBLIC; cuts off attention beyond the window in a way that is observable by the accelerator.
- Timing / memory-access side channel from windowed KV access is NOT evaluated.
- Sliding window does not change the QK invariant; exact_visible_attention still exposes the windowed score matrix on the accelerator.
- Not formal cryptographic / semantic / differential-privacy security.
- No full Qwen / LLaMA deployment unless a real wrapper exists.

## Paper-Safe Wording

> Stage 7.6g/h/i masked invariants carry over to sliding window attention: within the active window the QK invariant holds, the rolling KV buffer obeys K_tilde = K @ N_K and V_tilde = V @ N_V per (layer, head), and the eviction policy is the public window size. Trusted-softmax mode hides the windowed attention map from the accelerator transcript at the cost of extra TEE round trips.

## Unsafe Wording to Avoid

- Real FlashAttention support.
- Real sliding-window CUDA kernel.
- Window policy is cryptographically hidden.
- Timing side channel evaluated.
- This is formal cryptographic security.

