# Appendix B — Per-Claim Audit

This appendix consolidates the per-claim audit from `outputs/paper_claims_audit_v2.json` into a paper-ready table. Every claim carries a *status*, an *evidence artifact* path, a *safe wording* (which the paper may use verbatim), an *unsafe wording* (which the paper *must not* use, except inside an explicit "unsafe wording" enumeration), and a *remaining blocker*. **Unsupported claims must not be phrased as contributions.**

Status legend:

* **supported.** Backed by an artifact that exercises the claim algebraically or numerically under CPU local emulation.
* **proxy_supported.** Addressed by a proxy mechanism (e.g. probabilistic spot-check); not a full primitive. Paper must say "proxy" or "prototype".
* **partially_supported.** Addressed under a strictly weaker model (e.g. simulated, not measured). Paper must say "simulated", "audit-only", or equivalent.
* **cost_model_only.** Reflected in the symbolic / FLOP estimates of `outputs/paper_cost_model.json`; no end-to-end run.
* **unsupported.** Explicitly out of scope. The paper *must* phrase it as a non-claim or a future-work item.

Summary counts (from `outputs/paper_claims_audit_v2.json`): **15 supported, 1 proxy_supported, 0 cost_model_only, 10 unsupported** (total 26).

## B.1. Supported claims

| Claim id | Safe wording | Unsafe wording | Evidence | Remaining blocker |
|---|---|---|---|---|
| `padded_full_generation_correctness` | Padded full-generation correctness verified at float64 precision under CPU local emulation. | Padded generation gives cryptographic privacy. | `outputs/modern_decoder_generation_correctness.json` | Real TEE / GPU deployment. |
| `one_round_low_interaction_exact_mode` | Exact low-interaction mode: greedy match 1.0, `online_boundary_round_trips_per_decode_step = 1`, `intermediate_tee_reentry = false`. | Exact low-interaction mode hides attention maps. | `outputs/modern_decoder_low_interaction_correctness.json` | Real TEE / GPU deployment. |
| `rope_transient_plain_qk_eliminated` | Per-head block-diagonal RoPE-plane rotation masks commute with `apply_rope` and are folded into the qkv-projection, yielding masked `Q_tilde / K_tilde / V_tilde` directly. | Hides Q / K / V cryptographically. | `outputs/modern_decoder_rope_safe_low_interaction.json` | Real GPU fused-kernel implementation. |
| `norm_full_gram_reduced_by_token_chunk_masks` | Token-wise masking preserves per-row L2 norms (required by RMSNorm) and disrupts off-diagonal Gram; chunk(`k`) preserves within-chunk Gram and disrupts cross-chunk Gram. | Token-wise masking hides row norms. | `outputs/norm_granularity_low_interaction.json` | None for algebraic claim. |
| `attention_maps_hidden_only_in_trusted_softmax_mode` | Attention hiding requires trusted / secure softmax or approximate attention. Exact low-interaction with accelerator-side softmax exposes the QK invariant. | Row-wise score shifts provide attention privacy. | `outputs/attention_privacy_modes.json` | Real TEE with low boundary latency. |
| `attention_maps_visible_in_exact_low_interaction_mode` | Exact low-interaction baseline trades attention-map privacy for one-round-trip exactness. | Exact low-interaction mode hides attention. | `outputs/attention_privacy_modes.json` | None for algebraic claim. |
| `scalable_lm_head_dense_mask_not_feasible` | Dense `N_vocab` not scalable to `V >= 16k`; permutation (`O(V)` storage) and block (`O(V b)` storage) are scalable alternatives with explicit leakage notes. | Dense vocab mask is scalable. | `outputs/lm_head_scalability.json` | Real serving runtime sample-loop integration. |
| `lora_integration_supported_for_specified_sites` | Forward LoRA padded-boundary identity holds at every supported site at float64; `A_tilde = M^{-1} A R`, `B_tilde = R^{-1} B N_out`. | LoRA training is supported. | `outputs/lora_protocol_integration.json` | LoRA backward / training not implemented. |
| `paged_kv_invariant_supported_in_synthetic_abstraction` | Block-table indexing preserves `K_tilde = K N_K` and `V_tilde = V N_V` per session; cross-session block sharing disabled by default. | Paged cache is cryptographically isolated. | `outputs/paged_kv_abstraction.json` | Real GPU paged-attention kernel. |
| `multi_session_mask_isolation_supported_in_cpu_simulation` | Per-session orthogonal masks are sampled independently; boundary fingerprints differ across sessions for the same prompt. | Continuous batching is cryptographically isolated. | `outputs/multi_session_batching.json` | Real serving scheduler integration. |
| `sliding_window_attention_supported_in_cpu_synthetic_abstraction` | Within the active window, the QK invariant and rolling KV invariant hold at float64 precision; eviction policy is the public window size. | Real FlashAttention sliding-window kernel supported. | `outputs/sliding_window_attention.json` | Real GPU kernel. |
| `rolling_kv_window_invariant_supported` | Per-(layer, head) masked invariant preserved over the rolling window; old tokens evicted by the public window policy. | Rolling cache is cryptographically isolated. | `outputs/sliding_window_attention.json` | Real serving runtime. |
| `well_conditioned_masks_recommended_for_low_precision` | Condition-number sweep shows error scales with `cond(M)`; orthogonal masks remain at machine epsilon, dense ill-conditioned masks amplify error. | Any mask family is fine for fp16. | `outputs/precision_quantization_stability.json` | n/a for algebraic claim. |
| `generation_processors_safe_only_inside_trusted_side` | Main theorem: if recovered logits equal plain logits, every standard logit processor produces identical output under the same trusted randomness. | Bad word list cryptographically hidden from accelerator. | `outputs/generation_processor_coverage.json` | Output-length side-channel hiding. |
| `standard_1d_rope_scaling_covered_only_under_same_plane_rotation` | RoPE-plane block-diagonal mask commutes with `apply_rope` iff `B_K` is the same 2D rotation in each RoPE pair. | M-RoPE supported. | `outputs/modern_decoder_rope_safe_low_interaction.json` | Multi-axis RoPE-plane analysis for M-RoPE. |

## B.2. Proxy-supported claims

| Claim id | Safe wording | Unsafe wording | Evidence | Remaining blocker |
|---|---|---|---|---|
| `integrity_only_probabilistic_spot_check` | Detection rate scales with `checked_fraction`; no false alarms under correct execution; not a verifiable computation primitive. | Active malicious accelerator fully handled. | `outputs/integrity_spotcheck.json` | Cryptographic verifiable computation / authenticated dataflow. |

## B.3. Unsupported claims

These items must remain *unsupported* in the paper. The "safe wording" column gives the *non-claim* phrasing that may appear in the paper.

| Claim id | Safe wording (non-claim) | Unsafe wording (forbidden) | Evidence | Remaining blocker |
|---|---|---|---|---|
| `no_real_gpu_or_tee_wall_clock` | Complexity-model evidence only; no real wall-clock. | Measured real GPU / TEE performance. | `outputs/paper_cost_model.json` | Actual hardware access (H100 CC / SGX). |
| `no_formal_cryptographic_security` | Algebraic correctness + leakage / cost accounting only. | This is cryptographic security. | every-report-limitations-section | Cryptographic protocol design or formal proof out of scope. |
| `no_full_qwen_or_llama_deployment_unless_real_wrapper` | Tiny modern decoder used as paper-ready surrogate; scaling to LLaMA / Qwen requires the corresponding tokenizer + model loader + GPU kernels. | Qwen / LLaMA deployed in TEE-GPU split. | `outputs/paper_claims_audit_v2.json` | Real model loader + real GPU. |
| `no_hardware_side_channel_evaluation` | Side-channel evaluation out of scope; would require real hardware platform and counter-measure design. | Side channels evaluated. | every-report-limitations-section | Real hardware + side-channel platform. |
| `fp16_bf16_int8_int4_simulated_only_not_real_kernels` | CPU-simulated precision sweep; provides error bounds for the protocol's algebraic recovery, not real hardware performance. | Real GPU fp16 / bf16 / int8 / int4 performance. | `outputs/precision_quantization_stability.json` | Real GPU quantized kernels. |
| `output_length_side_channel_not_hidden_unless_separately_padded` | Output length is a side-channel that is NOT addressed by the current protocol; it must be hidden by additional padding or batching policy. | Output length hidden. | `outputs/generation_processor_coverage.json` | Explicit length-hiding policy with padded generation. |
| `m_rope_multimodal_unsupported` | Future work. | M-RoPE supported. | `outputs/decoder_component_coverage_audit.json` | Multi-axis RoPE-plane invariant derivation. |
| `moe_unsupported` | Future work. | MoE supported. | `outputs/decoder_component_coverage_audit.json` | Trusted routing or masked expert dispatch. |
| `speculative_decoding_unsupported` | Future work. | Speculative decoding supported. | `outputs/decoder_component_coverage_audit.json` | Speculative-decode threat model + masked draft model. |
| `quantized_real_model_deployment_unsupported_without_real_backend` | CPU-simulated quantization only; real deployment requires GPU int8 / int4 kernels and weight loaders. | Real quantized model deployment. | `outputs/precision_quantization_stability.json` | Real GPU quantization backend. |

## B.4. Audit invariants enforced by tests

The following invariants are enforced by `tests/test_paper_claims_audit_v2.py`, so that no paper draft can promote an unsupported claim by accident:

* The set of claim ids in `outputs/paper_claims_audit_v2.json` equals the 26-element union of the baseline 15 plus the Stage 7.8 addendum 11.
* Every claim whose id appears in the "must-be-unsupported" set (`no_real_gpu_or_tee_wall_clock`, `no_formal_cryptographic_security`, `no_full_qwen_or_llama_deployment_unless_real_wrapper`, `no_hardware_side_channel_evaluation`, `fp16_bf16_int8_int4_simulated_only_not_real_kernels`, `output_length_side_channel_not_hidden_unless_separately_padded`, `m_rope_multimodal_unsupported`, `moe_unsupported`, `speculative_decoding_unsupported`, `quantized_real_model_deployment_unsupported_without_real_backend`) has `status = unsupported`.
* Every claim whose id appears in the "must-be-supported" Stage 7.8 set has `status = supported`.
* No claim's safe-wording column contains any of the four forbidden phrases (`cryptographic security`, `hides attention maps in exact`, `side channels evaluated`, `wall-clock benchmark`).
* For every `supported` claim with an `outputs/` evidence path, the artifact file exists at audit time.
* `integrity_only_probabilistic_spot_check` has `status = proxy_supported`, never `supported`.
* The two attention-map claims (`attention_maps_hidden_only_in_trusted_softmax_mode`, `attention_maps_visible_in_exact_low_interaction_mode`) are both `supported`, never one without the other.

The audit is a paper-side firewall: it prevents the most common reviewer-rejecting overclaims (especially around cryptographic security, real-hardware performance, and unsupported model families) from entering the draft.
