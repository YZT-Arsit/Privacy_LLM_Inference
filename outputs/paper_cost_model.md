# Paper Complexity & Cost Model

_Stage 7.7f: symbolic and tiny / real-config estimates of FLOPs, transfers, and storage for every protocol mode. No real wall-clock measurement._

## Symbolic Formulas

### Linear Padded Transform

```
Linear ``Y = X W`` becomes ``X_pad = (X - T) M``, ``Y_tilde = X_pad W_tilde + C_linear`` with ``W_tilde = M^{-1} W N_out``, ``C_linear = T W N_out``. Trusted recovery: ``Y = Y_tilde N_out^{-1}``. Per-call FLOPs: O(B S d_in d_out) for the linear + O(B S d_in) for the (X-T) M transition. Compile FLOPs: O(d_in d_out d_out) for W_tilde and O(B S d_in d_out) for C_linear.
```

### Rmsnorm Sequence Chunk Token

```
RMSNorm core preserves orthogonal right-action: ``RMSNormCore(H Q) = RMSNormCore(H) Q``. sequence mode: 1 Q per layer per call (size [d, d]). chunk(k) mode: ceil(S/k) Q per layer per call. token mode: S Q per layer per call. Per-call extra trusted FLOPs in chunk/token modes: O(num_chunks d^3) sampling + O(B S d^2) per-row transitions.
```

### Rope Safe Block Rotation

```
Per-head right mask B_Q (B_K) is a block-diagonal 2D rotation in each RoPE pair (d_h/2 angles). Storage per head: d_h^2 (block-diagonal pattern, but stored as full d_h x d_h). Folded into q/k weights once per session: no per-call extra.
```

### Gqa Head Masks

```
N_K[h_kv], N_V[h_kv] orthogonal d_h x d_h per KV head; N_Q[h] = N_K[h // group_size]^{-T} per Q head (derived). Storage: 2 * h_kv * d_h^2 per layer.
```

### Kv Cache Append

```
Append per token per layer per KV head: K_tilde_new = K_new @ N_K, V_tilde_new = V_new @ N_V. FLOPs per token: 2 * L * h_kv * d_h^2.
```

### Lm Head Dense Permutation Block

```
Dense: z_tilde = z @ N_vocab, recovery z = z_tilde @ N_vocab^{-1}. Memory O(V^2), FLOPs O(B S V^2). Permutation: z_tilde[..., i] = z[..., perm[i]]. Memory O(V), FLOPs O(B S V). Block: per-block matmul, memory O(V b), FLOPs O(B S V b).
```

### Trusted Softmax Extra Boundary Cost

```
Per layer per call: ship Q_tilde, K_cache_tilde, V_cache_tilde to TEE (bytes ~ B (s_new + s_total) (h + 2 h_kv) d_h * bytes), TEE returns attn_out_tilde (bytes ~ B s_new h d_h * bytes). Round trips per decode step: 1 + L.
```

### Lora Rank R Overhead

```
Per LoRA-augmented linear: extra matmul X_pad A_tilde B_tilde with A_tilde [d_in, r_pad], B_tilde [r_pad, d_out]. Per-call FLOPs: O(B S d_in r_pad + B S r_pad d_out). Compile: O(d_in r_pad d_out).
```

## Tiny Config (synthetic decoder)

| Param | L | d | h | h_kv | d_h | s | V | r | block_size | lm_b |
|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 64 | 4 | 2 | 16 | 6 | 97 | 4 | 4 | 64 |

| Mode | round_trips | intermediate_tee_reentry | trusted_ops | accel_ops | mask_storage_bytes | lm_head_mask_overhead_bytes | asymptotic |
|---|---|---|---|---|---|---|---|
| `baseline_plain` | 0 | False | 0 | 368640 | 0 | 0 | O(L d^2) |
| `padded_correctness_trusted_fallback` | O(L) | True | 184320 | 368640 | 40960 | 75272 | O(L d^2 + L d^2 nonlinear-fallback) |
| `low_interaction_sequence_norm_exact_visible_attention` | 1 | False | 0 | 375872 | 40960 | 75272 | O(L d^2) per call |
| `low_interaction_token_norm_exact_visible_attention` | 1 | False | 1572864 | 1941504 | 204800 | 75272 | O(L s d^3) per call (per-row Q) |
| `trusted_softmax_attention` | 2 | True | 4896 | 368640 | 40960 | 75272 | O(L d^2 + L s^2 h d_h) |
| `rope_safe_pre_mask` | 1 | False | 0 | 368640 | 49152 | 75272 | O(L d^2) per call |
| `lora_enabled` | 1 | False | 0 | 390144 | 69632 | 75272 | O(L (d^2 + d r)) |
| `paged_kv` | 1 | False | 0 | 369664 | 40960 | 75272 | O(L d^2) |
| `scalable_lm_head_permutation` | 1 | False | 582 | 582 | 776 | 776 | O(V) lm-head, O(L d^2) rest |
| `scalable_lm_head_block` | 1 | False | 49152 | 49152 | 65536 | 65536 | O(V b) lm-head, O(L d^2) rest |

## Real Config Estimates (LLaMA-7B-ish)

| Param | L | d | h | h_kv | d_h | s | V | r | block_size | lm_b |
|---|---|---|---|---|---|---|---|---|---|---|
| value | 32 | 4096 | 32 | 32 | 128 | 1024 | 32000 | 16 | 16 | 1024 |

| Mode | round_trips | intermediate_tee_reentry | trusted_ops | accel_ops | mask_storage_bytes | lm_head_mask_overhead_bytes | asymptotic |
|---|---|---|---|---|---|---|---|
| `baseline_plain` | 0 | False | 0 | 8796093022208 | 0 | 0 | O(L d^2) |
| `padded_correctness_trusted_fallback` | O(L) | True | 4398046511104 | 8796093022208 | 4563402752 | 8192000000 | O(L d^2 + L d^2 nonlinear-fallback) |
| `low_interaction_sequence_norm_exact_visible_attention` | 1 | False | 0 | 8796257648640 | 4563402752 | 8192000000 | O(L d^2) per call |
| `low_interaction_token_norm_exact_visible_attention` | 1 | False | 2251799813685248 | 2260595906707456 | 4398314946560 | 8192000000 | O(L s d^3) per call (per-row Q) |
| `trusted_softmax_attention` | 33 | True | 277025390592 | 8796093022208 | 4563402752 | 8192000000 | O(L d^2 + L s^2 h d_h) |
| `rope_safe_pre_mask` | 1 | False | 0 | 8796093022208 | 4697620480 | 8192000000 | O(L d^2) per call |
| `lora_enabled` | 1 | False | 0 | 8826157793280 | 4798283776 | 8192000000 | O(L (d^2 + d r)) |
| `paged_kv` | 1 | False | 0 | 8796126576640 | 4563402752 | 8192000000 | O(L d^2) |
| `scalable_lm_head_permutation` | 1 | False | 32768000 | 32768000 | 256000 | 256000 | O(V) lm-head, O(L d^2) rest |
| `scalable_lm_head_block` | 1 | False | 34359738368 | 34359738368 | 268435456 | 268435456 | O(V b) lm-head, O(L d^2) rest |

## Notes Per Mode

- `baseline_plain` — No masks, no pads, no TEE.
- `padded_correctness_trusted_fallback` — Pad-only path with trusted fallback at every nonlinear core. Many TEE re-entries.
- `low_interaction_sequence_norm_exact_visible_attention` — Main protocol; one TEE-accelerator round trip per step.
- `low_interaction_token_norm_exact_visible_attention` — Token-wise Q: full Gram off-diagonal disrupted; per-row transition matmuls multiply by S.
- `trusted_softmax_attention` — Exact attention hidden from accelerator transcript at the cost of L extra TEE round trips per step.
- `rope_safe_pre_mask` — Block-diagonal B_Q/B_K folded into q/k weights.
- `lora_enabled` — Forward-only LoRA; rank-padded inner dimension r.
- `paged_kv` — Block-table indexing adds O(s/block_size) per (L, head).
- `scalable_lm_head_permutation` — Logit multiset is observable; index alignment hidden.
- `scalable_lm_head_block` — Block-membership of each vocab index observable.

## Limitations

- CPU local emulation only; no real wall-clock measurement.
- All numbers are FLOP / byte estimates, NOT real timings.
- Real GPU / TEE deployment cost is not modelled.
- Memory-bandwidth, kernel launch overhead, network round-trip latency are NOT modelled.
- LoRA training (backward pass) is NOT modelled.
- Not formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> We provide symbolic and tiny / real-config FLOP and storage estimates for every protocol mode. These are complexity-model evidence only; no real GPU / TEE wall-clock is measured.

## Unsafe Wording to Avoid

- Measured real GPU/TEE performance.
- Wall-clock latency benchmark.
- Throughput benchmark.
- This is formal cryptographic security.

