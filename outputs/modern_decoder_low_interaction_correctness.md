# Low-Interaction Operator-Compatible Modern-Decoder Correctness

_CPU local emulation; main invariant `H_hat_l = H_l @ Q_l` with orthogonal `Q_l`; one boundary round-trip per decode step._

## No-Reentry Headline

| Field | Value |
|---|---|
| main_layer_invariant | `H_hat_l = H_l Q_l` |
| rmsnorm_mode | `operator_compatible_orthogonal` |
| trusted_fallback_used_in_main_path | False |
| intermediate_tee_reentry | False |
| online_boundary_round_trips_per_decode_step | 1 |
| use_pad | True |
| fresh_pad_used_at_linear_boundaries | True |

## Configuration

| Field | Value |
|---|---|
| vocab_size | 97 |
| hidden_size | 64 |
| intermediate_size | 176 |
| num_layers | 1 |
| num_query_heads | 4 |
| num_kv_heads | 2 |
| head_dim | 16 |
| max_position_embeddings | 128 |
| rope_base | 10000.0 |
| rms_norm_eps | 1e-06 |
| batch_size | 2 |
| prompt_len | 5 |
| max_new_tokens | 4 |
| dtype | float64 |
| device | cpu |

## Online Boundary Protocol (per decode step)

- **TEE -> accelerator**: h_hat_0 = h_0 @ Q_1 (masked current-token state)
- Accelerator runs every layer with no TEE re-entry.
- **Accelerator -> TEE**: z_tilde = z @ N_vocab (masked logits)
- TEE recovers `z = z_tilde @ N_vocab^{-1}` and samples.

| Property | Value |
|---|---|
| no_per_layer_trusted_recovery | True |
| no_trusted_rmsnorm_fallback | True |
| no_trusted_rope_recovery | True |
| rmsnorm_gamma_folded_into_following_linear | True |
| post_rope_per_head_masking_used | True |
| kv_cache_remasking_during_decode | False |

## Module Modes

| Module | Mode |
|---|---|
| rmsnorm_mode | `operator_compatible_orthogonal` |
| rope_mode | `post_rope_masking_with_transient_plain_qk_blocker` |
| swiglu_mode | `paired_permutation_with_boundary_pad` |
| attention_score_mode | `plaintext_scores_due_to_qk_invariant` |
| lm_head_mode | `padded_masked_logits_with_trusted_recovery` |
| qkv_projection_uses_transition_trick | `True` |
| o_proj_uses_transition_trick | `True` |
| mlp_in_uses_transition_trick | `True` |
| down_proj_uses_transition_trick | `True` |
| lm_head_uses_transition_trick | `True` |

## Pad Policy

| Check | Value |
|---|---|
| pad_at_linear_boundaries | True |
| pad_enters_rmsnorm_core | False |
| pad_enters_rope_core | False |
| pad_enters_swiglu_core | False |
| pad_enters_softmax | False |
| pad_compensated_before_nonlinear_core | True |

## Correctness Metrics

| Metric | Value |
|---|---|
| H_hat = H Q invariant max abs err | 2.665e-15 |
| RMSNormCore(H Q) - RMSNormCore(H) Q max | 1.776e-15 |
| Transition (Q^{-1} M, T M) max abs err | 1.665e-15 |
| SwiGLU paired-permutation max abs err | 1.998e-15 |
| o_proj recovery max abs err | 8.951e-16 |
| down_proj recovery max abs err | 1.917e-15 |
| N_Q N_K^T - I max abs err | 6.661e-16 |
| KV cache append invariant max abs err | 1.110e-15 |
| Prefill recovered logits max abs err | 9.315e-16 |
| Decode-step recovered logits max abs err | 9.315e-16 |
| LM head recovery max abs err | 9.315e-16 |
| greedy_token_match_rate | 1.0 |
| sequence_exact_match | True |

## RoPE Blocker (explicit)

Preferred path is post-RoPE per-head masking. The qkv-projection output is plain Q / K / V transiently on the accelerator -- RoPE is applied on plain Q / K and per-head masks are applied immediately afterwards. No TEE re-entry happens during this step (no data leaves the accelerator), but plain Q / K / V are transiently visible to the accelerator inside that block. This is the explicit blocker for the no-reentry path.

| Property | Value |
|---|---|
| blocker_present | True |
| transient_plain_qk_on_accelerator | True |
| tee_reentry_inside_rope_block | False |

## Norm Leakage Audit

**This is a leakage surface, not a security claim.** Orthogonal RMSNorm-compatible masking preserves row L2 norms and the full Gram matrix of the plain hidden states. Fresh `Q_l` per session does NOT randomise this structure.

| Metric | Value |
|---|---|
| row_norm_error (||H_hat_i||_2 - ||H_i||_2) | 1.776e-15 |
| gram_matrix_error (||H_hat H_hat^T - H H^T||_inf) | 2.132e-14 |
| same_prompt_fresh_Q_gram_linkability | 2.842e-14 |
| different_prompt_gram_distance | 31.3437 |
| nn_gram_match_rate_same_prompt | 1.0 |
| nn_gram_match_rate_different_prompt | 0.2000 |

Orthogonal RMSNorm-compatible masking preserves row L2 norms and the full Gram matrix of the hidden states. The accelerator boundary therefore exposes the token-pair similarity structure of the plain hidden states. Fresh Q per session does NOT randomise this structure (both runs yield the same Gram matrix).

## Repeated-Run Sanity Check

| Check | Value |
|---|---|
| same_input_two_runs_same_output | True |
| same_input_two_runs_different_masked_fingerprints | True |
| kv_cache_contains_plaintext | False |
| lm_head_logits_masked_before_recovery | True |
| sampling_on_trusted_recovered_logits | True |
| embedding_in_trusted_side | True |
| token_ids_exposed_to_accelerator | False |

## Limitations

- CPU local emulation only; no real TEE / GPU deployment.
- Synthetic tiny modern decoder (vocab=97, hidden=64, num_layers configurable).
- Default num_layers=1 demonstrates the invariant on a single layer; the wrapper supports N layers via accelerator-side inter-layer orthogonal change-of-basis matrices.
- RoPE remains the explicit blocker: post-RoPE per-head masking requires plain Q / K transiently on the accelerator. No TEE re-entry happens, but this is accelerator-side transient leakage of plain Q / K / V inside the qkv -> RoPE -> per-head-mask block.
- Attention scores / probabilities are plain by construction of the QK invariant (N_Q N_K^T = I); attention-map hiding is out of scope.
- Orthogonal RMSNorm-compatible masking preserves row L2 norms and Gram matrices; this is reported as an explicit boundary leakage surface.
- Low-interaction mode trades fewer TEE boundary crossings for norm-structure leakage; this is a deliberate trade-off, not a security claim.
- This is NOT formal cryptographic / semantic / differential privacy.
- No hardware side-channel evaluation.

## Paper-Safe Wording

> We verify that the operator-compatible orthogonal RMSNorm path, combined with fresh boundary pads and the trusted-precomputed Q^{-1} M transition trick, can run a full tiny modern-decoder generation step without any per-layer TEE re-entry. The trade-off, made explicit by the Gram-matrix leakage audit, is that the GPU-visible H_hat = H Q preserves row-norm and token-pair similarity structure.

## Unsafe Wording to Avoid

- The scheme is formally secure.
- Orthogonal masking hides hidden states cryptographically.
- Attention maps are hidden.
- We evaluate real TEE / GPU performance.
- Generic dense masks commute with RoPE.
- Pads can pass through nonlinear layers.
- We support full Qwen / LLaMA private generation on real TEE / GPU.
- No information leaks on the accelerator boundary.
- The Gram matrix is hidden.
- Row norms are randomised by fresh Q.

