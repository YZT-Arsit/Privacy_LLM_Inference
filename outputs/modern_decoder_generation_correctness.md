# Padded Modern Decoder Full-Generation Correctness

_CPU local emulation only -- no real TEE, no real GPU, no production framework. Main reported mode is `use_pad=True`._

## Configuration

| Field | Value |
|---|---|
| vocab_size | 97 |
| hidden_size | 64 |
| intermediate_size | 176 |
| num_layers | 2 |
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
| use_pad (main mode) | True |

## Coverage

| Component | Implemented |
|---|---|
| embedding | True |
| rmsnorm | True |
| rope | True |
| gqa_or_mqa | True |
| causal_attention | True |
| softmax | True |
| swiglu | True |
| residual | True |
| kv_cache | True |
| lm_head | True |
| greedy_generation | True |

## Pad Policy

| Check | Value |
|---|---|
| pad_at_linear_boundaries | True |
| pad_enters_rmsnorm_core | False |
| pad_enters_rope_core | False |
| pad_enters_swiglu_core | False |
| pad_enters_softmax | False |
| pad_compensated_before_nonlinear_core | True |

## Module Modes

| Module | Mode |
|---|---|
| rmsnorm_mode | `trusted_fallback_with_repad` |
| rope_mode | `post_rope_masking` |
| swiglu_mode | `paired_permutation_with_boundary_pad` |
| attention_score_mode | `plaintext_scores_due_to_qk_invariant` |
| lm_head_mode | `padded_masked_logits_with_trusted_recovery` |
| rmsnorm_gpu_compatible_claim | `False` |
| generic_pre_rope_mask_commutation_used | `False` |
| qk_projection_pad_compensated | `True` |
| swiglu_pad_compensated_before_core | `True` |
| swiglu_shared_permutation | `True` |

## Correctness Metrics

| Metric | Value |
|---|---|
| forward_logits_max_abs_error | 7.633e-16 |
| prefill_logits_max_abs_error | 9.992e-16 |
| decode_step_logits_max_abs_error_max | 9.992e-16 |
| kv_cache_invariant_max_abs_error | 8.882e-16 |
| qk_constraint_max_error | 6.661e-16 |
| swiglu_paired_permutation_max_error | 1.776e-15 |
| o_proj_recovery_max_error | 8.465e-16 |
| lm_head_recovery_max_error | 9.992e-16 |
| greedy_token_match_rate | 1.0 |
| sequence_exact_match | True |

## Repeated-Run Sanity Check

Two runs with the *same* input ids and *fresh* pads / masks must produce (a) identical recovered token sequences and (b) *different* GPU-visible masked-boundary fingerprints.

| Check | Value |
|---|---|
| same_input_two_runs_same_output | True |
| same_input_two_runs_different_masked_fingerprints | True |
| same_input_two_runs_recovered_logits_allclose | True |
| kv_cache_contains_plaintext | False |
| lm_head_logits_masked_before_recovery | True |
| sampling_on_trusted_recovered_logits | True |
| embedding_in_trusted_side | True |
| token_ids_exposed_to_accelerator | False |

This is a security-relevant *sanity check*, not a formal security proof.

## Ablation: use_pad=False

Provided only as an ablation row. `use_pad=False` is **not** the reported main mode.

| Metric | Value |
|---|---|
| use_pad | False |
| sequence_exact_match | True |

## Limitations

- CPU local emulation only; no real TEE/GPU.
- Synthetic tiny modern decoder; no full Qwen/LLaMA weights loaded.
- Attention scores/probabilities are not hidden in this correctness wrapper.
- Protecting attention maps requires an additional secure softmax or score obfuscation primitive.
- This validates padded masked algebraic correctness, not formal cryptographic security.
- No hardware side-channel evaluation.
- Per-call fresh pads and fresh per-Linear masks; per-KV-head masks fixed within a session so the KV cache append invariant holds.

## Paper-Safe Wording

> We verify that fresh boundary pads can be integrated into a full modern decoder-style generation path without breaking output equivalence. Pads are compensated before nonlinear cores, while KV cache and logits remain masked until trusted recovery.

## Unsafe Wording to Avoid

- The scheme is formally secure.
- The CPU wrapper proves cryptographic privacy.
- Attention maps are hidden.
- We evaluate real TEE/GPU performance.
- Generic dense masks commute with RoPE.
- Pads can pass through nonlinear layers.
- We support full Qwen/LLaMA private generation on real TEE/GPU.

