# Experiments

We organise the empirical study by research questions, not by implementation milestones. Every question lists its setup, key metrics, the supported claim, and the limitation it does *not* address. All experiments run on CPU at float64 unless otherwise stated; no real GPU, TEE, or quantised kernel is invoked. The aggregator `outputs/paper_experiment_suite.json` collects 16 sub-experiments into one paper-safe report; the per-claim audit `outputs/paper_claims_audit_v2.json` classifies 26 claims (15 supported, 1 proxy-supported, 10 explicitly unsupported). The full pytest count at the time of writing is **1235 passed / 4 skipped / 0 failed**.

## Q1. Does padded masked generation preserve output equivalence?

**Setup.** A synthetic tiny modern decoder (LLaMA / Qwen-style: RMSNorm, rotate-half RoPE, GQA, SwiGLU, KV cache) is wrapped in the padded-boundary protocol with `use_pad = true`, fresh `T` per call, fresh `M` per call. Generation is greedy on a batch of two prompts of length six with three new tokens. Reference: full plaintext greedy generation.

**Key metrics.** Per-layer boundary invariant error, KV cache invariant error, recovered-logits error against plaintext, greedy-token match rate, sequence-exact-match flag, and the four `pad_enters_*` flags.

**Supported claim.** `padded_full_generation_correctness` (status: *supported*). Recovered logits match plaintext at float64 machine precision; greedy match `1.0`; sequence exact match `True`. The four `pad_enters_{rmsnorm_core, rope_core, swiglu_core, softmax}` flags are all `false`. Evidence: `outputs/modern_decoder_generation_correctness.json` and `outputs/modern_decoder_low_interaction_correctness.json`.

**Limitation.** The decoder is a synthetic tiny surrogate that mirrors the LLaMA / Qwen forward graph. We do *not* claim full Qwen / LLaMA deployment.

## Q2. Can the protocol run with one online boundary round trip per decode step?

**Setup.** The low-interaction operator-compatible wrapper is invoked with `attention_privacy_mode = exact_visible_attention`, `rope_mask_mode = pre_rope_block_diagonal_rotation`. Per-call boundary tables (`A`, `C_T`, `W_tilde`, `b_tilde`, `C_linear`) are compiled trusted-side, then the accelerator runs the full forward without re-entering the TEE between layers.

**Key metrics.** `online_boundary_round_trips_per_decode_step`, `intermediate_tee_reentry`, `trusted_fallback_used_in_main_path`, recovered-logits error.

**Supported claim.** `one_round_low_interaction_exact_mode` (status: *supported*). `online_boundary_round_trips_per_decode_step = 1`, `intermediate_tee_reentry = false`, `trusted_fallback_used_in_main_path = false`. Evidence: `outputs/modern_decoder_low_interaction_correctness.json` for the wrapper run; `outputs/paper_cost_model.json` for the per-mode round-trip count table.

**Limitation.** The protocol's `trusted_softmax_attention` mode trades the one-round-trip property for attention-map hiding (it adds `L` extra TEE round trips per decode step). We do *not* claim both at once.

## Q3. Does RoPE-safe masking remove plaintext Q / K / V exposure on the accelerator?

**Setup.** The wrapper is run with `rope_mask_mode = pre_rope_block_diagonal_rotation`. Per-head `B_K` is sampled from RoPE-plane block-diagonal rotations; per-head `B_Q = B_K[q_head // group_size]` is derived. Per-head right masks (`B_Q`, `B_K`, `N_V`) are folded directly into the qkv-projection weight; the accelerator-visible qkv output is therefore already masked per head.

**Key metrics.** `rope_transient_plain_qk_visible`, `rope_transient_plain_v_visible`, `qkv_projection_outputs_masked_directly`, `rope_commutation_max_abs_error`, `qk_score_invariant_max_abs_error`, `kv_cache_invariant_max_abs_error`.

**Supported claim.** `rope_transient_plain_qk_eliminated` (status: *supported*). `rope_transient_plain_qk_visible = false`, `rope_transient_plain_v_visible = false`, `qkv_projection_outputs_masked_directly = true`. `rope_commutation_max_abs_error ≈ 1.8e-15`, `qk_score_invariant_max_abs_error ≈ 4.0e-15`, `kv_cache_invariant_max_abs_error ≈ 7.8e-16`. Evidence: `outputs/modern_decoder_rope_safe_low_interaction.json`.

**Limitation.** Per-RoPE-pair 2D norms are preserved by construction (the mask is a 2D rotation in each plane). This is a *declared* leakage surface, not a bug.

## Q4. How does norm-mask granularity reduce Gram-matrix leakage at the layer-entry boundary?

**Setup.** Sequence, chunk(`k = 2`), and token granularity are exercised end-to-end on the same prompt; a second prompt provides a cross-prompt Gram-distance baseline; a second mask seed under the same prompt provides a same-prompt fresh-mask distance.

**Key metrics.** `row_norm_error`, `full_gram_error`, `off_diagonal_gram_error`, `within_chunk_gram_error`, `cross_chunk_gram_error`, `same_prompt_fresh_Q_gram_distance`, `different_prompt_gram_distance`.

**Representative values** (from `outputs/norm_granularity_low_interaction.json`):

| Granularity | row_norm_error | full_gram_error | within_chunk_gram | cross_chunk_gram |
|---|---|---|---|---|
| sequence | 1.78e-15 | 2.13e-14 | 2.13e-14 | 1.11e-14 |
| chunk(2) | 8.88e-16 | 56.0 | 2.84e-14 | 56.0 |
| token | 1.78e-15 | 44.2 | 24.4 | 44.2 |

`different_prompt_gram_distance ≈ 47.6` for context.

**Supported claim.** `norm_full_gram_reduced_by_token_chunk_masks` (status: *supported*). Sequence mode preserves the full Gram exactly; chunk mode preserves the within-chunk Gram block and disrupts cross-chunk Gram by the same order of magnitude as the cross-prompt baseline; token mode preserves only row L2 norms and disrupts off-diagonal Gram by the same order of magnitude. Greedy match `1.0` in all three modes.

**Limitation.** Row L2 norms are *not* hidden by any RMSNorm-compatible orthogonal mask; this is a mathematical requirement of RMSNorm correctness, not a design choice. The corresponding unsafe wording "token-wise masking hides row norms" is enumerated under the unsafe-wording list.

## Q5. What is the exactness vs hiding trade-off for the attention map?

**Setup.** The wrapper is invoked under three `attention_privacy_mode` values: `exact_visible_attention`, `trusted_softmax_attention`, `score_blinding_experimental`. Per mode, recovery error, greedy match, attention-visibility flags, and round-trip count are recorded.

**Key metrics.** `attention_scores_visible`, `attention_probs_visible`, `attention_map_hidden_from_accelerator_transcript`, `online_boundary_round_trips_per_decode_step`, `intermediate_tee_reentry`, `row_constant_blinding_softmax_max_abs_error`, `nonconstant_blinding_softmax_max_abs_error`, `greedy_token_match_rate`.

**Representative values** (from `outputs/attention_privacy_modes.json`):

| Mode | exact | one_round_trip | attention_hidden | round_trips | greedy match |
|---|---|---|---|---|---|
| `exact_visible_attention` | True | True | False | 1 | 1.0 |
| `trusted_softmax_attention` | True | False | True | 1 + L | 1.0 |
| `score_blinding_experimental` | True | True | False | 1 | 1.0 |

The `score_blinding_experimental` mode additionally reports `row_constant_blinding_softmax_max_abs_error ≈ 1.1e-16` (softmax is *exact* under row-constant shifts) and `nonconstant_blinding_softmax_max_abs_error ≈ 5.6e-1` (random non-row-constant blinding *breaks* softmax).

**Supported claims.** `attention_maps_hidden_only_in_trusted_softmax_mode` and `attention_maps_visible_in_exact_low_interaction_mode` (both *supported*). Evidence: `outputs/attention_privacy_modes.json`.

**Limitation.** Row-constant score shifts preserve softmax exactly but do *not* hide ranking or relative margins; the unsafe wording "row-wise score shifts provide attention privacy" is explicitly enumerated.

## Q6. Is LM-head masking scalable to real LLM vocab?

**Setup.** Four LM-head mask strategies (dense orthogonal, vocab permutation, block-diagonal orthogonal, top-k trusted recovery) are exercised at real vocab sizes `V in {97, 1024, 4096}` and symbolic vocab sizes `V in {16384, 50000}`. Dense is run only for `V <= 4096` to avoid allocating a `50000 × 50000` orthogonal mask; larger sizes are reported symbolically with an explicit infeasibility flag.

**Key metrics.** `exactness`, `max_abs_error`, `greedy_token_match_rate`, `memory_bytes_mask`, `online_recovery_ops_estimate`, `feasibility`, `logit_multiset_preserved_max_abs_error` (permutation only).

**Representative values** (from `outputs/lm_head_scalability.json`, `V = 4096`):

| Mode | error | memory_bytes | feasibility |
|---|---|---|---|
| dense | 2.4e-14 | 134.2 MiB | feasible_only_for_small_V |
| permutation | exact | 32.8 KiB | scalable |
| block_diagonal(b=64) | 3.8e-15 | 2.1 MiB | scalable_with_block_size_tunable |
| top-k | exact (top-1) | 32.8 KiB | scalable_top1_only |

**Supported claim.** `scalable_lm_head_dense_mask_not_feasible` (status: *supported*). Permutation and block-diagonal are exact; dense scales as `O(V^2)`. The unsafe wording "dense vocab mask is scalable" is explicitly enumerated.

**Limitation.** Permutation preserves the *multiset* of logits (the sorted-logits vector is observable). Block-diagonal makes the block partition observable unless the partition is itself permuted. Top-k trusted recovery is exact for greedy but *not* for full-distribution sampling.

## Q7. Does LoRA integrate with the latest main protocol at every supported insertion site?

**Setup.** LoRA adapters with `true_rank = 4`, `padded_rank = 8` are inserted at `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`. For each site, the padded-boundary identity from Lemma 10 is verified with site-appropriate `N_out` (`B_Q` for `q_proj`, `B_K` for `k_proj`, `N_V` for `v_proj`, residual-stream `Q_l` for `o_proj` and `down_proj`, paired permutation `P` for `up_proj` and `gate_proj`). End-to-end greedy generation is exercised with LoRA *merged* (`W_eff = W + A B`) under five `(norm_mask_granularity × attention_privacy_mode)` combinations.

**Key metrics.** Per-site `padded_boundary_identity_max_abs_error`, `trusted_recovery_max_abs_error`, `padded_AB_minus_true_AB_max_abs_error`; per-combination `greedy_token_match_rate`, `sequence_exact_match`, `lm_head_recovery_max_abs_error`, `intermediate_tee_reentry`, `online_boundary_round_trips_per_decode_step`.

**Supported claim.** `lora_integration_supported_for_specified_sites` (status: *supported*). All seven sites satisfy the padded-boundary identity at < 1e-9 error; all five end-to-end combinations greedy-match 1.0. Evidence: `outputs/lora_protocol_integration.json`.

**Limitation.** LoRA *training* (backward pass) is not implemented. The padded rank `r_pad` is observable on the accelerator side; only the *true* rank `r` is hidden by zero-padding.

## Q8. Do paged KV, sliding-window KV, and multi-session batching preserve the masked invariant?

**Setup.** A synthetic paged KV abstraction is exercised over 3 sessions × 2 layers × 2 KV heads, block size `4`, `max_tokens_per_session = 13` (not a multiple of the block size). The masked per-block invariant `K_tilde_block = K_plain_block N_K[s, l, kv_head]` is verified for every physical block and the full-cache invariant via `gather_full_tilde()`. A sliding-window attention experiment exercises `w in {2, 4, full}` under both attention privacy modes. A multi-session batching experiment runs 3 sessions with ragged prompt lengths (4 / 5 / 7) and ragged decode lengths (2 / 3 / 2); fingerprint isolation is verified by comparing two sessions running the *same* prompt under different mask seeds.

**Key metrics.** Per-block and full-cache invariant error; window-eviction-correct flag; full-vs-sliding equality when `w >= s_total`; per-session greedy match; fingerprint difference for same-prompt-different-session.

**Supported claims.** `paged_kv_invariant_supported_in_synthetic_abstraction`, `sliding_window_attention_supported_in_cpu_synthetic_abstraction`, `rolling_kv_window_invariant_supported`, `multi_session_mask_isolation_supported_in_cpu_simulation` (all *supported*). Paged invariants at < 1e-12; sliding-window invariants at < 1e-12; full-vs-sliding equality at < 1e-9; fingerprint isolation observed (different fingerprints for identical prompts under different sessions). Evidence: `outputs/paged_kv_abstraction.json`, `outputs/sliding_window_attention.json`, `outputs/multi_session_batching.json`.

**Limitation.** No real GPU paged-attention kernel; no real FlashAttention sliding-window kernel; no real serving scheduler. Timing / memory-access side channels and page-fault timing are *not* evaluated. The window size policy `w` is *public*.

## Q9. Are generation-time logit processors compatible with the recovered-logits boundary?

**Setup.** Trusted-side implementations of greedy, temperature, top-k, top-p, repetition penalty, stop-token, bad-words mask, forced-token mask, and reproducible temperature sampling under a trusted seed are applied to both plaintext logits and the masked-then-recovered logits. The two outputs are compared.

**Key metrics.** `logit_recovery_max_abs_error`, per-processor `max_abs_error_distribution`, `argmax_match_rate`, `discrete_equal`, `reproducible_under_same_trusted_seed`.

**Supported claim.** `generation_processors_safe_only_inside_trusted_side` (status: *supported*). Logit recovery error `3.1e-15`; all eight processors are exact under recovered logits; sampling is reproducible under the same trusted seed. Beam search and grammar-constrained decoding are marked `audit_only` (the main theorem of Lemma 9 applies but they are not implemented end-to-end). Evidence: `outputs/generation_processor_coverage.json`.

**Limitation.** Output length / stop timing remain observable unless separately padded. The unsafe wording "output length hidden" is explicitly enumerated under `output_length_side_channel_not_hidden_unless_separately_padded` (status: *unsupported*).

## Q10. What happens under low-precision / quantisation simulation?

**Setup.** The padded linear `Y_rec = (X - T) M M^{-1} W + T W` is run under six precision modes: `float64_reference`, `float32_simulated`, `bfloat16_simulated`, `float16_simulated`, `int8_weight_only_simulated`, `int4_weight_only_symbolic`. Mask families are: orthogonal, permutation, dense with controlled condition number `cond in {1, 2, 10, 100, 1000}`.

**Key metrics.** `logits_max_abs_error_vs_float64_plain`, `logits_relative_error`, `greedy_token_match_rate`, `overflow_detected`, `nan_detected`.

**Representative values** (from `outputs/precision_quantization_stability.json`, orthogonal mask):

| Precision | error | greedy |
|---|---|---|
| float64 | 2.8e-14 | 1.0 |
| float32 | 1.5e-6 | 1.0 |
| bfloat16 | 8.8e-2 | 1.0 |
| float16 | 1.1e-2 | 1.0 |
| int8_weight_only | 3.1e-1 | 1.0 |
| int4_weight_only_symbolic | 5.4 | 0.75 |

The condition-number sweep shows error scales roughly linearly with `cond(M)`; ill-conditioned dense masks (`cond = 1000`) amplify error by orders of magnitude over the orthogonal baseline.

**Supported claim.** `well_conditioned_masks_recommended_for_low_precision` (status: *supported*). Orthogonal / permutation / RoPE-plane block rotation / block-diagonal are recommended; ill-conditioned dense masks are *not* recommended.

**Limitation.** fp16 / bf16 / int8 are *simulated* via CPU round-trip casts; int4 is *symbolic only*. Real GPU tensor-core behaviour and accumulator types may differ. We mark `real_gpu_kernel_measured = false` and `real_quantized_model_loaded = false`.

## Q11. What integrity support exists?

**Setup.** Four detection modes (`no_check`, `spot_check_linear_projection`, `spot_check_lm_head_slice`, `spot_check_kv_cache_append`) are simulated at `checked_fractions = (0, 0.05, 0.1, 0.25, 0.5)` over `n_trials = 50` per setting; a single fixed-location corruption is injected and detected if the random spot-check sample covers it.

**Key metrics.** `empirical_detection_rate`, `expected_detection_probability_single_corruption`, `false_positive_rate`, `extra_trusted_compute_ops_estimate`.

**Proxy-supported claim.** `integrity_only_probabilistic_spot_check` (status: *proxy_supported*). Detection rate scales with `checked_fraction`; no false alarms under correct execution; `no_check` detects zero corruption by construction. Evidence: `outputs/integrity_spotcheck.json`.

**Limitation.** This is *not* verifiable computation, *not* an authenticated dataflow primitive, and *not* a cryptographic integrity proof. An adaptive adversary that observes which items are spot-checked can lower the effective detection rate. Privacy under a malicious accelerator (rather than integrity) is *not* addressed. The unsafe wording "active malicious accelerator fully handled" is explicitly enumerated.

## Q12. What is the component coverage across mainstream decoder-only architectures?

**Setup.** A coverage audit (`outputs/decoder_component_coverage_audit.json`) classifies each component into A (covered in main protocol), B (partially covered / extension), or C (unsupported / future work). The classification is dynamic: components whose evidence artifact exists are promoted from B to A (e.g. sliding window attention, which is `supported` because `outputs/sliding_window_attention.json` exists).

**Key metrics.** Category counts and per-component reason / required invariant / leakage surface / remaining blocker / artifact path.

**Supported claim.** Category A (`covered`) has 10 components: RMSNorm, SwiGLU, standard 1D RoPE, GQA / MQA, causal attention, KV cache, paged KV abstraction, LM head, LoRA inference, generation processors. Category B (`partially_covered`) has 6: sliding window attention (promoted to *supported*), LayerNorm (audit-only), GELU MLP (audit-only), prefix cache (audit-only), beam search (audit-only), quantisation (partially_supported via 7.8b artifact). Category C (`unsupported`) has 9: M-RoPE, MoE, MLA, speculative decoding, real vLLM / FlashAttention backend, real GPU / TEE hardware side channels, full active malicious security, LoRA training, full Qwen / LLaMA deployment.

**Limitation.** Every component in category C must remain category C in the paper. Treating M-RoPE, MoE, MLA, speculative decoding, real vLLM, or real Qwen / LLaMA as supported is in the explicit unsafe-wording list. The pytest enforces this via `test_no_unsupported_component_marked_supported` and `test_unsupported_marked_unsupported` in the claim audit.

## Summary metrics

The aggregator `outputs/paper_experiment_suite.json` reports 16 stage entries all `status = ok`. The claim audit reports 26 claims: 15 *supported*, 1 *proxy_supported*, 10 *unsupported*. The total test suite stands at **1235 passed / 4 skipped / 0 failed**. Every experiment runs CPU-only, deterministic, without network access, without any HuggingFace model download. No real GPU / TEE wall-clock is measured.
