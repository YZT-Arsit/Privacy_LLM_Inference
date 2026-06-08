# Paper Claims Audit v2

_Stage 7.7g: table of every paper-relevant claim, its support status, safe wording, unsafe wording, and remaining blocker before real deployment._

## Summary

| Status | Count |
|---|---|
| supported | 10 |
| proxy_supported | 1 |
| cost_model_only | 0 |
| unsupported | 4 |

## Claims Table

| id | status | evidence | exists | remaining_blocker |
|---|---|---|---|---|
| `padded_full_generation_correctness` | **supported** | `outputs/modern_decoder_generation_correctness.json` | True | Real TEE/GPU deployment. |
| `one_round_low_interaction_exact_mode` | **supported** | `outputs/modern_decoder_low_interaction_correctness.json` | True | Real TEE/GPU deployment. |
| `rope_transient_plain_qk_eliminated` | **supported** | `outputs/modern_decoder_rope_safe_low_interaction.json` | True | Real GPU fused-kernel implementation. |
| `norm_full_gram_reduced_by_token_chunk_masks` | **supported** | `outputs/norm_granularity_low_interaction.json` | True | None for algebraic claim. |
| `attention_maps_hidden_only_in_trusted_softmax_mode` | **supported** | `outputs/attention_privacy_modes.json` | True | Real TEE with low boundary latency. |
| `attention_maps_visible_in_exact_low_interaction_mode` | **supported** | `outputs/attention_privacy_modes.json` | True | None for algebraic claim. |
| `scalable_lm_head_dense_mask_not_feasible` | **supported** | `outputs/lm_head_scalability.json` | True | Real serving runtime sample-loop integration. |
| `lora_integration_supported_for_specified_sites` | **supported** | `outputs/lora_protocol_integration.json` | True | LoRA backward / training not implemented. |
| `paged_kv_invariant_supported_in_synthetic_abstraction` | **supported** | `outputs/paged_kv_abstraction.json` | True | Real GPU paged-attention kernel. |
| `multi_session_mask_isolation_supported_in_cpu_simulation` | **supported** | `outputs/multi_session_batching.json` | True | Real serving scheduler integration. |
| `integrity_only_probabilistic_spot_check` | **proxy_supported** | `outputs/integrity_spotcheck.json` | True | Cryptographic verifiable computation / authenticated dataflow. |
| `no_real_gpu_or_tee_wall_clock` | **unsupported** | `outputs/paper_cost_model.json` | True | Actual hardware access (H100 CC / SGX). |
| `no_formal_cryptographic_security` | **unsupported** | `every-report-limitations-section` | True | Cryptographic protocol design or formal proof out of scope for this project. |
| `no_full_qwen_or_llama_deployment_unless_real_wrapper` | **unsupported** | `outputs/paper_claims_audit_v2.json` | False | Real model loader + real GPU. |
| `no_hardware_side_channel_evaluation` | **unsupported** | `every-report-limitations-section` | True | Real hardware + side-channel platform. |

## Safe Wording Per Claim

### `padded_full_generation_correctness`

- claim: Padded boundary linears integrate into full modern-decoder generation; pads never enter RMSNorm / RoPE / SwiGLU / softmax cores.
- safe: Padded full-generation correctness verified at float64 precision under CPU local emulation.
- unsafe: Padded generation gives cryptographic privacy.

### `one_round_low_interaction_exact_mode`

- claim: Low-interaction operator-compatible path achieves exact generation with one TEE-accelerator round trip per decode step under exact_visible_attention.
- safe: Exact low-interaction mode: greedy match 1.0, online_boundary_round_trips_per_decode_step = 1, intermediate_tee_reentry = false.
- unsafe: Exact low-interaction mode hides attention maps.

### `rope_transient_plain_qk_eliminated`

- claim: RoPE-safe pre-mask mode eliminates transient plain Q/K/V on the accelerator: qkv_projection_outputs_masked_directly = true.
- safe: Per-head block-diagonal RoPE-plane rotation masks commute with apply_rope and are folded into the qkv-projection, yielding masked Q_tilde / K_tilde / V_tilde directly.
- unsafe: Hides Q/K/V cryptographically.

### `norm_full_gram_reduced_by_token_chunk_masks`

- claim: Token/chunk norm-mask granularity disrupts the full-sequence Gram-matrix leakage that sequence-shared Q exhibits; row L2 norms remain preserved by RMSNorm correctness.
- safe: Token-wise masking preserves per-row L2 norms (required by RMSNorm) and disrupts off-diagonal Gram; chunk(k) preserves within-chunk Gram and disrupts cross-chunk Gram.
- unsafe: Token-wise masking hides row norms.

### `attention_maps_hidden_only_in_trusted_softmax_mode`

- claim: Attention maps are hidden from the accelerator transcript ONLY in trusted_softmax_attention mode; exact_visible_attention exposes them by construction.
- safe: Attention hiding requires trusted/secure softmax or approximate attention. Exact low-interaction with accelerator-side softmax exposes the QK invariant.
- unsafe: Row-wise score shifts provide attention privacy.

### `attention_maps_visible_in_exact_low_interaction_mode`

- claim: exact_visible_attention mode preserves the QK invariant by construction; attention scores and probabilities are visible on the accelerator.
- safe: Exact low-interaction baseline trades attention-map privacy for one-round-trip exactness.
- unsafe: Exact low-interaction mode hides attention.

### `scalable_lm_head_dense_mask_not_feasible`

- claim: Dense V x V LM-head orthogonal mask is not feasible for real LLM vocab sizes; permutation and block-diagonal masks scale but disclose multiset/block leakage.
- safe: Dense N_vocab not scalable to V >= 16k; permutation (O(V) storage) and block (O(V b) storage) are scalable alternatives with explicit leakage notes.
- unsafe: Dense vocab mask is scalable.

### `lora_integration_supported_for_specified_sites`

- claim: LoRA adapters integrate with the Stage 7.6h main protocol for q/k/v/o/up/gate/down_proj insertion sites; rank padding hides true rank but padded rank is observable.
- safe: Forward LoRA padded-boundary identity holds at every supported site at float64; A_tilde = M^{-1} A R, B_tilde = R^{-1} B N_out.
- unsafe: LoRA training is supported.

### `paged_kv_invariant_supported_in_synthetic_abstraction`

- claim: The per-(session, layer, head) masked KV invariant holds under a CPU synthetic paged cache with block-table remapping.
- safe: Block-table indexing preserves K_tilde = K @ N_K and V_tilde = V @ N_V per session; cross-session block sharing disabled by default.
- unsafe: Paged cache is cryptographically isolated.

### `multi_session_mask_isolation_supported_in_cpu_simulation`

- claim: Per-session masks (Q_l, N_K, N_V, N_vocab) are independent; same prompt under two sessions produces different masked boundary fingerprints; cross-session prefix sharing off by default.
- safe: Per-session orthogonal masks are sampled independently; boundary fingerprints differ across sessions for the same prompt.
- unsafe: Continuous batching is cryptographically isolated.

### `integrity_only_probabilistic_spot_check`

- claim: Active-adversary integrity is supported ONLY as a probabilistic spot-check prototype; no verifiable computation.
- safe: Detection rate scales with checked_fraction; no false alarms under correct execution; not a verifiable computation primitive.
- unsafe: Active malicious accelerator fully handled.

### `no_real_gpu_or_tee_wall_clock`

- claim: No real GPU or TEE wall-clock latency / throughput is measured; all numbers are FLOP / byte estimates or CPU emulated counts.
- safe: Complexity-model evidence only; no real wall-clock.
- unsafe: Measured real GPU/TEE performance.

### `no_formal_cryptographic_security`

- claim: No formal cryptographic / semantic / differential-privacy security is claimed.
- safe: Algebraic correctness + leakage / cost accounting only.
- unsafe: This is cryptographic security.

### `no_full_qwen_or_llama_deployment_unless_real_wrapper`

- claim: No full Qwen / LLaMA deployment unless a real wrapper exists; only synthetic tiny modern decoder is exercised.
- safe: Tiny modern decoder used as paper-ready surrogate; scaling to LLaMA / Qwen requires the corresponding tokenizer + model loader + GPU kernels.
- unsafe: Qwen / LLaMA deployed in TEE-GPU split.

### `no_hardware_side_channel_evaluation`

- claim: Hardware side channels (timing, memory, power, RDMA) are NOT evaluated.
- safe: Side-channel evaluation out of scope; would require real hardware platform and counter-measure design.
- unsafe: Side channels evaluated.

## Limitations

- Audit table is paper-ready guidance, not a formal proof.
- Evidence-artifact existence is checked; semantic validity is not re-evaluated here.
- Unsupported claims must be NEVER written as supported in the paper.

## Paper-Safe Wording

> All paper claims are classified into supported, proxy_supported, cost_model_only, or unsupported, with the corresponding safe and unsafe wording per claim.

## Unsafe Wording to Avoid

- Treating unsupported claims as supported.
- Citing missing-evidence artifacts as proof.
- Combining algebraic correctness with cryptographic security in the abstract.

