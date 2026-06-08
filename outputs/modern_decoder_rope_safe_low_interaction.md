# RoPE-Safe Low-Interaction Modern-Decoder Correctness

_CPU local emulation; main invariant `H_hat_l = H_l @ Q_l`; pre-RoPE block-diagonal rotation masks `B_Q` / `B_K` eliminate plain-Q/K/V transient exposure on the accelerator._

## Stage 7.6g RoPE-Safe Headline

| Field | Value |
|---|---|
| main_layer_invariant | `H_hat_l = H_l Q_l` |
| rmsnorm_mode | `operator_compatible_orthogonal` |
| rope_mask_mode | `pre_rope_block_diagonal_rotation` |
| rope_transient_plain_qk_visible | False |
| rope_transient_plain_v_visible | False |
| qkv_projection_outputs_masked_directly | True |
| trusted_rope_recovery_used | False |
| generic_pre_rope_dense_commutation_used | False |
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
- Accelerator runs every layer with no TEE re-entry and no plain Q/K/V on the boundary.
- **Accelerator -> TEE**: z_tilde = z @ N_vocab (masked logits)
- TEE recovers `z = z_tilde @ N_vocab^{-1}` and samples.

| Property | Value |
|---|---|
| no_per_layer_trusted_recovery | True |
| no_trusted_rmsnorm_fallback | True |
| no_trusted_rope_recovery | True |
| no_plain_qkv_on_accelerator | True |
| rmsnorm_gamma_folded_into_following_linear | True |
| kv_cache_remasking_during_decode | False |

## Module Modes

| Module | Mode |
|---|---|
| rmsnorm_mode | `operator_compatible_orthogonal` |
| rope_mode | `pre_rope_block_diagonal_rotation` |
| rope_mask_mode | `pre_rope_block_diagonal_rotation` |
| swiglu_mode | `paired_permutation_with_boundary_pad` |
| attention_score_mode | `plaintext_scores_due_to_qk_invariant` |
| lm_head_mode | `padded_masked_logits_with_trusted_recovery` |
| qkv_projection_outputs_masked_directly | `True` |
| qk_invariant_via_B_Q_equal_B_K | `True` |

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
| RMSNormCore commutation max | 1.776e-15 |
| Transition trick max abs err | 1.554e-15 |
| SwiGLU paired-perm max abs err | 1.998e-15 |
| o_proj recovery max abs err | 8.283e-16 |
| down_proj recovery max abs err | 1.554e-15 |
| B_Q B_K^T - I max abs err | 2.220e-16 |
| RoPE commutation max abs err | 1.821e-15 |
| Q_rope_tilde K_rope_tilde^T = Q_rope K_rope^T max | 3.997e-15 |
| KV cache append invariant max abs err | 7.772e-16 |
| Prefill recovered logits max abs err | 8.812e-16 |
| Decode-step recovered logits max abs err | 8.882e-16 |
| LM head recovery max abs err | 8.882e-16 |
| greedy_token_match_rate | 1.0 |
| sequence_exact_match | True |

## RoPE-Pair Norm Leakage Audit

**This is the residual leakage surface of the RoPE-safe mask family**, not a security claim. ``B_Q[i]`` / ``B_K[k]`` are 2D rotations *within* each (channel j, channel j + head_dim/2) pair, so the per-pair 2D norm is preserved exactly.

| Metric | Value |
|---|---|
| rope_pair_norm_leakage | True |
| rope_pair_norm_max_abs_error | 2.220e-16 |
| rope_commutation_max_abs_error_audit | 2.220e-16 |

RoPE-plane block-rotation masking removes transient plain Q/K exposure but preserves per-RoPE-pair 2D norms. The masks act as 2D rotations within each (channel j, channel j+head_dim/2) pair, so |B_Q[i] x_pair| = |x_pair| for every pair. This is the residual leakage surface of the RoPE-safe mask family.

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
- Synthetic tiny modern decoder (vocab=97, hidden=64, default num_layers=1).
- RoPE-plane block-rotation masks B_Q / B_K preserve per-RoPE-pair 2D norms; this is the residual leakage surface (replaces the Stage 7.6f plain-Q/K/V transient exposure).
- Attention scores / probabilities are plain by construction (B_Q B_K^T = I); attention-map hiding is out of scope.
- Orthogonal RMSNorm-compatible masking preserves row L2 norms and full Gram matrices (carried over from Stage 7.6f); this remains a leakage surface.
- Stage 7.6g eliminates transient plain Q/K/V on the accelerator but does NOT eliminate norm-structure leakage.
- This is NOT formal cryptographic / semantic / differential-privacy security.
- No hardware side-channel evaluation.

## Paper-Safe Wording

> We eliminate the Stage 7.6f RoPE blocker via RoPE-plane block-rotation masks B_Q / B_K that commute with the repo's apply_rope, so the accelerator never holds a plain Q / K / V tensor. The residual leakage surface is the per-RoPE-pair 2D norm preservation, which we measure and report.

## Unsafe Wording to Avoid

- The scheme is formally secure.
- RoPE-plane masking hides Q / K cryptographically.
- Attention maps are hidden.
- We evaluate real TEE / GPU performance.
- Generic dense masks commute with RoPE.
- Pads can pass through nonlinear layers.
- Per-RoPE-pair norms are randomised by fresh B masks.
- RoPE-plane masking eliminates Gram-matrix leakage.

