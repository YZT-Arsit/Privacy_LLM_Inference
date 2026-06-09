# Tables

Paper-ready tables collected for direct insertion into the main text or appendix. Every cell is sourced from an artifact under `outputs/`; the citation column gives the JSON path. *Numbers are at float64 under CPU local emulation.*

## Table 1. Main mode comparison

| Mode | Exact | One round trip | Attention hidden | Norm leakage | RoPE plain Q / K / V | Use pad | Key limitation |
|---|---|---|---|---|---|---|---|
| `exact_visible_attention` (default) | yes | yes | no | full Gram (sequence), within-chunk (chunk), row norms (token) | none (`qkv_projection_outputs_masked_directly = true`) | yes | attention map visible by construction |
| `trusted_softmax_attention` | yes | no (1 + L per step) | yes | same as above | none | yes | extra `L` TEE round trips per decode step |
| `score_blinding_experimental` | yes | yes | no | same as above | none | yes | row-constant shift preserves softmax but not ranking; non-row-constant breaks softmax |
| `padded_correctness_trusted_fallback` (baseline) | yes | no (O(L)) | n/a | n/a | n/a | yes | trusted re-entry at every nonlinear; not low-interaction |

*Evidence:* `outputs/attention_privacy_modes.json`, `outputs/modern_decoder_low_interaction_correctness.json`, `outputs/modern_decoder_generation_correctness.json`.

## Table 2. Correctness summary

| Experiment | Max abs error | Greedy match | Sequence exact | Evidence |
|---|---|---|---|---|
| Padded full-generation correctness | `< 1e-14` (logits recovery) | 1.0 | yes | `outputs/modern_decoder_generation_correctness.json` |
| Low-interaction operator-compatible correctness | `< 1e-14` (logits recovery) | 1.0 | yes | `outputs/modern_decoder_low_interaction_correctness.json` |
| RoPE-safe low-interaction correctness | `4.0e-15` (QK score invariant), `1.8e-15` (RoPE commutation), `7.8e-16` (KV cache invariant) | 1.0 | yes | `outputs/modern_decoder_rope_safe_low_interaction.json` |
| Norm-mask granularity (sequence) | `2.1e-14` (full Gram error) | 1.0 | yes | `outputs/norm_granularity_low_interaction.json` |
| Norm-mask granularity (chunk, `k = 2`) | `56.0` (full Gram error) | 1.0 | yes | `outputs/norm_granularity_low_interaction.json` |
| Norm-mask granularity (token) | `44.2` (full Gram error) | 1.0 | yes | `outputs/norm_granularity_low_interaction.json` |
| Attention privacy modes (`exact_visible`) | `< 1e-14` (logits recovery) | 1.0 | yes | `outputs/attention_privacy_modes.json` |
| Attention privacy modes (`trusted_softmax`) | `< 1e-14` (logits recovery) | 1.0 | yes | `outputs/attention_privacy_modes.json` |
| Attention privacy modes (`score_blinding`) | `1.1e-16` row-constant; `5.6e-1` non-row-constant | 1.0 | yes | `outputs/attention_privacy_modes.json` |
| Sliding window attention | `4.4e-16` to `8.9e-16` (score invariant per window) | 1.0 | yes | `outputs/sliding_window_attention.json` |
| Paged KV abstraction | `< 1e-15` (per-block invariant) | n/a | n/a | `outputs/paged_kv_abstraction.json` |
| LoRA protocol integration (per site) | `< 1e-9` (padded-boundary identity) | 1.0 | yes | `outputs/lora_protocol_integration.json` |
| Generation processor coverage | `3.1e-15` (logit recovery) | 1.0 | yes | `outputs/generation_processor_coverage.json` |

## Table 3. Leakage summary

| Mechanism | Hidden from accelerator | Residual leakage | Mode knob | Evidence |
|---|---|---|---|---|
| Hidden states `H_l` | yes (under `Q_l` or per-row variant) | Gram block structure per granularity | `norm_mask_granularity` ∈ {sequence, chunk, token} | `outputs/norm_granularity_low_interaction.json` |
| RoPE plain Q / K | yes | per-RoPE-pair 2D norms preserved | `rope_mask_mode = pre_rope_block_diagonal_rotation` | `outputs/modern_decoder_rope_safe_low_interaction.json` |
| V plain | yes (under per-KV-head `N_V`) | none beyond protocol | (always on) | `outputs/modern_decoder_rope_safe_low_interaction.json` |
| Attention scores `S = Q K^T` | no in `exact_visible`; yes in `trusted_softmax` | attention map in exact mode | `attention_privacy_mode` | `outputs/attention_privacy_modes.json` |
| Post-softmax `P` | no in `exact_visible`; yes in `trusted_softmax` | attention map in exact mode | `attention_privacy_mode` | `outputs/attention_privacy_modes.json` |
| Logits `z` | yes (under `N_vocab`) | logit multiset (permutation), block partition (block) | LM-head mask choice | `outputs/lm_head_scalability.json` |
| KV cache contents | yes (under `N_K, N_V`) | cache length observable | (always on) | `outputs/paged_kv_abstraction.json`, `outputs/modern_decoder_low_interaction_correctness.json` |
| LoRA adapter `A, B` | yes | padded inner rank `r_pad` observable; true `r` hidden | rank padding | `outputs/lora_protocol_integration.json` |
| Output length / stop timing | no | observable on user-facing channel | (not implemented) | `outputs/generation_processor_coverage.json` |
| Window size (sliding window) | no (public policy) | window cut-off observable | `attention_window_size` | `outputs/sliding_window_attention.json` |
| Cross-session mask correlation | yes | per-session masks sampled independently | (always on) | `outputs/multi_session_batching.json` |
| Active adversary | partially (probabilistic detect) | corruption to UN-checked items undetected | `checked_fraction` | `outputs/integrity_spotcheck.json` |

## Table 4. Component coverage (short form)

| Category | Component | Status |
|---|---|---|
| Covered | RMSNorm | supported |
| Covered | SwiGLU | supported |
| Covered | standard 1D RoPE | supported |
| Covered | GQA / MQA causal attention | supported |
| Covered | KV cache (standard) | supported |
| Covered | paged KV abstraction | supported |
| Covered | LM head (dense, permutation, block) | supported |
| Covered | LoRA inference (7 sites) | supported |
| Covered | trusted-side generation processors | supported |
| Covered | sliding window attention | supported (CPU synthetic) |
| Partially | LayerNorm | audit-only |
| Partially | GELU MLP | audit-only |
| Partially | prefix cache cross-session sharing | audit-only |
| Partially | beam search | audit-only |
| Partially | grammar-constrained decoding | audit-only |
| Partially | quantisation (fp16 / bf16 / int8 / int4) | partially_supported (simulated) |
| Unsupported | M-RoPE / multimodal positional encoding | unsupported |
| Unsupported | MoE router / expert dispatch | unsupported |
| Unsupported | Multi-Head Latent Attention | unsupported |
| Unsupported | speculative decoding | unsupported |
| Unsupported | real vLLM / FlashAttention backend | unsupported |
| Unsupported | real GPU / TEE hardware side channels | unsupported |
| Unsupported | full active malicious security | unsupported |
| Unsupported | LoRA training (backward) | unsupported |
| Unsupported | full Qwen / LLaMA deployment | unsupported |

*Evidence:* `outputs/decoder_component_coverage_audit.json`. Full version in Appendix A.

## Table 5. LM-head scalability

| Mode | Exact | Storage | Online recovery cost | Leakage | Scalable to real `V`? |
|---|---|---|---|---|---|
| Dense orthogonal | yes | `O(V^2)` | `O(B s V^2)` | none beyond protocol | no |
| Permutation | yes | `O(V)` (int64 indices) | `O(B s V)` | logit *multiset* preserved (sorted-logits vector observable) | yes |
| Block-diagonal (block size `b`) | yes | `O(V b)` | `O(B s V b)` | block-membership of each vocab index observable | yes (with chosen `b`) |
| Top-k trusted recovery | top-1 exact (greedy); full distribution requires full recovery | `O(V)` (uses permutation underneath) | `O(B s V)` | same as permutation | greedy yes; sampling needs full recovery |

*Evidence:* `outputs/lm_head_scalability.json`. Dense baseline measured for `V ∈ {97, 1024, 4096}` only; `V ∈ {16384, 50000}` symbolic estimate.

## Table 6. LoRA supported sites

| Site | Target output mask `N_out` | Pad compensation | Supported? |
|---|---|---|---|
| `q_proj` | `B_Q` (block-diagonal RoPE-plane rotation) | `C_base = T W_q B_Q`, `C_lora = T A B B_Q` | yes |
| `k_proj` | `B_K` (block-diagonal RoPE-plane rotation) | analogous | yes |
| `v_proj` | `N_V` (orthogonal per KV head) | analogous | yes |
| `o_proj` | `Q_l` (residual-stream orthogonal) | analogous | yes |
| `up_proj` | `P` (paired permutation) | analogous | yes |
| `gate_proj` | `P` (same paired permutation) | analogous | yes |
| `down_proj` | `Q_l` (residual-stream orthogonal) | analogous | yes |

*Evidence:* `outputs/lora_protocol_integration.json`. All seven sites pass the padded-boundary identity at `< 1e-9` error.

## Table 7. Precision stability

| Precision / mask family | Error (orthogonal mask) | Greedy match | Recommendation |
|---|---|---|---|
| float64 | `2.8e-14` | 1.0 | reference (algebraic correctness) |
| float32 | `1.5e-6` | 1.0 | recommended for real fp32 GPU |
| bfloat16 (simulated) | `8.8e-2` | 1.0 | usable with well-conditioned masks |
| float16 (simulated) | `1.1e-2` | 1.0 | usable with well-conditioned masks |
| int8 weight-only (simulated) | `3.1e-1` | 1.0 | usable for greedy decoding with well-conditioned masks |
| int4 weight-only (symbolic) | `5.4` | 0.75 | not recommended without further QAT / calibration |
| Dense mask, `cond = 1000` (float32) | `>> 1e-3` (scales linearly with `cond`) | varies | not recommended for low precision |

Recommended mask families for low-precision deployment: orthogonal, permutation, RoPE-plane block rotation, block-diagonal well-conditioned. *Evidence:* `outputs/precision_quantization_stability.json`. fp16 / bf16 / int8 are *simulated*, int4 is *symbolic*; no real GPU kernel is measured.

## Table 8. Claim audit summary

| Status | Count | Key examples |
|---|---|---|
| supported | 15 | `padded_full_generation_correctness`, `one_round_low_interaction_exact_mode`, `rope_transient_plain_qk_eliminated`, `norm_full_gram_reduced_by_token_chunk_masks`, `attention_maps_hidden_only_in_trusted_softmax_mode`, `attention_maps_visible_in_exact_low_interaction_mode`, `scalable_lm_head_dense_mask_not_feasible`, `lora_integration_supported_for_specified_sites`, `paged_kv_invariant_supported_in_synthetic_abstraction`, `sliding_window_attention_supported_in_cpu_synthetic_abstraction`, ... |
| proxy_supported | 1 | `integrity_only_probabilistic_spot_check` |
| cost_model_only | 0 | — |
| unsupported (**must not be phrased as contributions**) | 10 | `no_real_gpu_or_tee_wall_clock`, `no_formal_cryptographic_security`, `no_full_qwen_or_llama_deployment_unless_real_wrapper`, `no_hardware_side_channel_evaluation`, `fp16_bf16_int8_int4_simulated_only_not_real_kernels`, `output_length_side_channel_not_hidden_unless_separately_padded`, `m_rope_multimodal_unsupported`, `moe_unsupported`, `speculative_decoding_unsupported`, `quantized_real_model_deployment_unsupported_without_real_backend` |

*Evidence:* `outputs/paper_claims_audit_v2.json`. Full per-claim wording in Appendix B.

## Table 9. Round-trip / cost summary (LLaMA-7B-like config)

| Mode | Round trips / decode step | Accelerator compute (ops) | Mask storage (bytes) |
|---|---|---|---|
| `baseline_plain` | 0 | 8.80e12 | 0 |
| `low_interaction_sequence_norm_exact_visible_attention` | 1 | 8.80e12 | 4.30e9 |
| `low_interaction_token_norm_exact_visible_attention` | 1 | 2.26e15 | 4.39e12 |
| `trusted_softmax_attention` | 1 + L = 33 | 8.80e12 | 4.30e9 |
| `rope_safe_pre_mask` | 1 | 8.80e12 | 4.30e9 |
| `lora_enabled` | 1 | 8.83e12 | 4.32e9 |
| `paged_kv` | 1 | 8.80e12 | 4.30e9 |
| `scalable_lm_head_permutation` | 1 | 3.28e7 (LM-head only) | 256 KiB |
| `scalable_lm_head_block (b = 1024)` | 1 | 3.44e10 (LM-head only) | 256 MiB |

*Evidence:* `outputs/paper_cost_model.json`. Numbers are symbolic / FLOP estimates, *not* measured wall-clock; full real-config table in Appendix C.

## Reading note

Every table above has one row per artifact path. The tables are intended as drop-in copies for the main text; numbers are at float64 and regenerated whenever the aggregator runs. If a row's evidence artifact is missing at audit time, the corresponding claim in `outputs/paper_claims_audit_v2.json` flips its `evidence_artifact_exists` flag to `false`, and the audit tests will fail. The tables therefore stay in sync with the underlying experiments by construction.
