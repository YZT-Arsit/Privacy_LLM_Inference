# Appendix C — Complexity and Cost Model

This appendix consolidates the symbolic and tiny / real-config FLOP and storage estimates from `outputs/paper_cost_model.json`. **Numbers are not measured wall-clock.** No real GPU or TEE latency, throughput, memory-bandwidth, or kernel-launch numbers are reported in this work. Treat the tables below as *complexity-model evidence*.

## C.1. Symbol table

| Symbol | Meaning |
|---|---|
| `L` | number of decoder layers |
| `d` | hidden dimension (embedding / residual stream) |
| `h` | number of query heads |
| `h_kv` | number of KV heads (group_size = `h / h_kv`) |
| `d_h` | per-head dimension (typically `d = h * d_h`) |
| `s` | prompt / new-token sequence length |
| `s_total` | past_len + s (cache total length at the boundary of interest) |
| `V` | vocab size |
| `r` | LoRA true rank |
| `r_pad` | LoRA padded inner dimension (`r_pad >= r`) |
| `b` | block size (paged KV block size; LM-head block-mask size) |
| `w` | sliding-window attention window size |
| `bytes_dtype` | bytes per scalar at the precision under analysis |

## C.2. Symbolic formulas (excerpt)

* *Linear padded transform.* `Y_tilde = (X - T) M M^{-1} W N + T W N`. Per-call FLOPs: `O(B s d_in d_out)` for the linear + `O(B s d_in)` for the `(X - T) M` transition. Compile FLOPs: `O(d_in d_out d_out)` for `W_tilde`, `O(B s d_in d_out)` for `C_linear`.
* *RMSNorm under granularity.* Sequence mode: one `Q` per layer per call, `O(d^2)` storage. Chunk(`k`) mode: `ceil(s/k)` per layer per call. Token mode: `s` per layer per call; per-call extra trusted FLOPs `O(num_chunks d^3)` for sampling + `O(B s d^2)` for per-row transitions.
* *RoPE-safe block rotation.* Per head right mask `B_K` is a block-diagonal 2D rotation in each RoPE pair (`d_h / 2` angles). Folded into Q / K weights once per session; *no per-call extra*.
* *GQA head masks.* `N_K, N_V` orthogonal `d_h × d_h` per KV head; `N_Q[h] = N_K[h // group_size]^T` derived. Storage `2 h_kv d_h^2` per layer.
* *KV cache append.* Per token per layer per KV head: `K_tilde_new = K_new N_K`, `V_tilde_new = V_new N_V`. FLOPs per token: `2 L h_kv d_h^2`.
* *LM head.* Dense: `O(V^2)` memory + `O(B s V^2)` recovery. Permutation: `O(V)` memory + `O(B s V)` recovery. Block(`b`): `O(V b)` memory + `O(B s V b)` recovery.
* *Trusted softmax extra boundary cost.* Per layer per call: ship Q_tilde + K_cache_tilde + V_cache_tilde to TEE (`~ B (s_new + s_total) (h + 2 h_kv) d_h * bytes`); receive `attn_out_tilde` (`~ B s_new h d_h * bytes`). Round trips per decode step: `1 + L`.
* *LoRA rank-`r` overhead.* Per supported insertion site: extra matmul `(X - T) M A_tilde B_tilde` with `A_tilde [d_in, r_pad]`, `B_tilde [r_pad, d_out]`. Per-call FLOPs: `O(B s d_in r_pad + B s r_pad d_out)`. Compile: `O(d_in r_pad d_out)`.

## C.3. Mode-by-mode complexity table (asymptotic)

| Mode | TEE transfers per decode step | TEE compute | Accelerator compute | Storage overhead | Notes |
|---|---|---|---|---|---|
| `baseline_plain` | 0 | 0 | `O(L d^2)` | 0 | No masks, no pads, no TEE. |
| `padded_correctness_trusted_fallback` | `O(L)` | `O(L d^2)` (trusted fallback at every nonlinear) | `O(L d^2)` | `O(L d^2)` mask + `O(L s d)` pad | Pad-only path with trusted fallback at every nonlinear core. Many TEE re-entries. |
| `low_interaction_sequence_norm_exact_visible_attention` | **1** | 0 | `O(L d^2)` per call | `O(L d^2)` mask + `O(L s d)` pad + `O(V^2)` LM-head | Main protocol; one TEE-accelerator round trip per step. |
| `low_interaction_token_norm_exact_visible_attention` | **1** | `O(L s d^3)` (per-row Q sampling + transitions) | `O(L s d^3)` per call | `O(L s d^2)` mask | Token-wise Q: full Gram off-diagonal disrupted; per-row transitions multiply by `s`. |
| `trusted_softmax_attention` | **1 + L** | `O(L s_total s_new h d_h)` (softmax in TEE) | `O(L d^2)` | same as sequence + extra per-layer transfer bytes | Exact attention hidden from accelerator transcript at the cost of `L` extra TEE round trips per step. |
| `rope_safe_pre_mask` | **1** | 0 | `O(L d^2)` | `O(L d^2)` + `O(L h d_h^2)` per-head B masks | Block-diagonal `B_Q, B_K` folded into q/k weights. |
| `lora_enabled` | **1** | 0 | `O(L (d^2 + d r_pad))` | `O(L (d^2 + 7 d r_pad))` (7 supported sites) | Forward-only LoRA; rank-padded inner dimension `r_pad`. |
| `paged_kv` | **1** | 0 | `O(L d^2)` | `O(L d^2)` mask + block-table `O(L h_kv (s/b))` | Block-table indexing adds `O(s/b)` per (L, head). |
| `sliding_window_attention` | **1** (or `1 + L` under trusted softmax) | 0 | `O(L d^2)` | same as `paged_kv` with window cap `w` | Window size `w` is public; eviction by trimming. |
| `scalable_lm_head_permutation` | **1** | `O(s V)` (inverse-perm `index_select`) | `O(s V)` | `O(V)` (int64 indices) | Logit multiset observable; index alignment hidden. |
| `scalable_lm_head_block` | **1** | `O(s V b)` | `O(s V b)` | `O(V b)` | Block-membership of each vocab index observable. |

## C.4. Real-config FLOP estimates (LLaMA-7B-like, `L = 32, d = 4096, h = 32, d_h = 128, s = 1024, V = 32000, r = 16, block_size = 16, lm_b = 1024`)

| Mode | Round trips / decode step | TEE compute (ops) | Accelerator compute (ops) | Mask storage (bytes) |
|---|---|---|---|---|
| `baseline_plain` | 0 | 0 | 8.80e12 | 0 |
| `padded_correctness_trusted_fallback` | O(L) = O(32) | 4.40e12 | 8.80e12 | 1.07e8 |
| `low_interaction_sequence_norm_exact_visible_attention` | **1** | 0 | 8.80e12 | 4.30e9 |
| `low_interaction_token_norm_exact_visible_attention` | **1** | 2.25e15 | 2.26e15 | 4.39e12 |
| `trusted_softmax_attention` | **1 + 32 = 33** | 2.77e11 | 8.80e12 | 4.30e9 |
| `rope_safe_pre_mask` | **1** | 0 | 8.80e12 | 4.30e9 |
| `lora_enabled` | **1** | 0 | 8.83e12 | 4.32e9 |
| `paged_kv` | **1** | 0 | 8.80e12 | 4.30e9 |
| `scalable_lm_head_permutation` | **1** | 3.28e7 | 3.28e7 | 256 KiB (`V × 8` bytes) |
| `scalable_lm_head_block` | **1** | 3.44e10 | 3.44e10 | 256 MiB (`32k / 1k × 1k²` blocks) |

Key observations from the real-config table:

1. The exact `low_interaction_sequence_norm` mode adds negligible accelerator FLOPs over `baseline_plain` (8.80e12 vs 8.80e12, the difference is the boundary-transition overhead absorbed in compile).
2. Trusted-softmax raises round-trips by `L = 32` per decode step and adds `O(L s_total s_new h d_h) ≈ 2.77e11` trusted ops per step. The protocol does *not* claim "trusted-softmax preserves one round trip" — that wording is in the unsafe-wording list.
3. Token-norm granularity multiplies accelerator ops by `~ s = 1024` over sequence mode (8.80e12 → 2.26e15) because the per-row `Q_i` introduces an `s d^2`-per-row transition at every Q-output linear. This is a *security-efficiency knob*, not formal security.
4. Permutation LM-head is `~5` orders of magnitude cheaper than the dense `O(V^2)` baseline and `~3` orders cheaper than the block variant (256 KiB vs `> 134 GiB` for dense, vs 256 MiB for block).
5. LoRA adds `~0.4%` to accelerator FLOPs over the sequence baseline (`r_pad = 16` is tiny relative to `d = 4096`); the extra storage is `~ L × 7 × 2 × d × r_pad × 8 bytes ≈ 30 MiB`.

## C.5. Honest scope of the cost model

* Numbers are **symbolic / estimated FLOP counts**, not measured timings. Memory-bandwidth, kernel launch overhead, network round-trip latency, GPU-CPU transfer bandwidth, and TEE attestation cost are *not* modelled. Marked: `real_gpu_wall_clock_measured = false`, `real_tee_wall_clock_measured = false`.
* The real-config estimates assume float64 (`bytes_dtype = 8`); a real fp16 / bf16 deployment would halve the storage rows. Real int8 would quarter them. Real int4 would eighth them, but with the precision-stability caveats of `outputs/precision_quantization_stability.json`.
* The cost model does *not* include LoRA backward, MoE routing, MLA, M-RoPE, speculative decoding, or real serving-runtime overheads (paged-allocator behaviour, continuous-batching scheduling, prefix-cache deduplication). These are listed in the unsupported claims of Appendix B.
* The cost model does *not* account for output-length side-channel padding overhead. A real length-hiding policy (always emit `max_new_tokens`, then post-truncate inside the trusted side) would add up to `O(max_new_tokens - actual_emitted)` extra accelerator forwards.
* The cost model does *not* model integrity spot-check extra trusted compute beyond a per-mode estimate. The spot-check is `proxy_supported` only.

## C.6. Reading this appendix

The cost-model tables are regenerated by the aggregator (`outputs/paper_cost_model.json`) per stage release; any change to a per-mode formula reflects in the next run. The unsafe-wording list at the bottom of `outputs/paper_cost_model.json` enumerates the phrases the paper must avoid when discussing cost: "Measured real GPU / TEE performance", "Wall-clock latency benchmark", "Throughput benchmark", "This is formal cryptographic security".
