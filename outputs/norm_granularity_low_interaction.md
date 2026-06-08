# Norm-Mask Granularity Low-Interaction Correctness

_Stage 7.6h: tighten the RMSNorm-compatible orthogonal mask granularity (sequence / chunk / token) on top of Stage 7.6g's rope-safe no-reentry path._

## Inherited Stage 7.6g Guarantees

| Field | Value |
|---|---|
| use_pad | True |
| rope_mask_mode | pre_rope_block_diagonal_rotation |
| rope_transient_plain_qk_visible | False |
| qkv_projection_outputs_masked_directly | True |
| intermediate_tee_reentry | False |
| online_boundary_round_trips_per_decode_step | 1 |
| trusted_fallback_used_in_main_path | False |

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
| batch_size | 2 |
| prompt_len | 6 |
| max_new_tokens | 3 |
| chunk_size | 2 |
| dtype | float64 |
| device | cpu |

## Per-Mode Correctness

| Granularity | chunk_size | greedy_token_match_rate | sequence_exact_match | h_hat invariant max | qk_score invariant max | kv_cache invariant max | lm_head recovery max |
|---|---|---|---|---|---|---|---|
| `sequence` | 1 | 1.0 | True | 2.665e-15 | 3.997e-15 | 8.882e-16 | 1.235e-15 |
| `chunk` | 2 | 1.0 | True | 2.887e-15 | 3.109e-15 | 8.882e-16 | 9.437e-16 |
| `token` | 1 | 1.0 | True | 2.665e-15 | 3.220e-15 | 7.772e-16 | 8.188e-16 |

## Per-Mode Stage 7.6g Carry-Over Diagnostics

| Granularity | norm_q_is_per_row | use_pad | rope_transient_plain_qk_visible | qkv_projection_outputs_masked_directly | intermediate_tee_reentry | trusted_fallback_used_in_main_path | online_boundary_round_trips_per_decode_step |
|---|---|---|---|---|---|---|---|
| `sequence` | False | True | False | True | False | False | 1 |
| `chunk` | True | True | False | True | False | False | 1 |
| `token` | True | True | False | True | False | False | 1 |

## Norm + Gram Leakage Audit (Layer-Entry Boundary)

Boundary tensor: ``H_hat = H @ Q`` (embedded prompt, layer 0). All errors are ``max | metric(H_hat) - metric(H) |``. Row L2 norms are mathematically preserved in every mode (RMSNorm correctness requirement).

| Mode | row_norm_error | full_gram_error | off_diagonal_gram_error | within_chunk_gram_error | cross_chunk_gram_error | same_prompt_fresh_Q_gram_distance |
|---|---|---|---|---|---|---|
| `sequence` | 1.776e-15 | 2.132e-14 | 1.110e-14 | 2.132e-14 | 1.110e-14 | 2.842e-14 |
| `chunk` | 8.882e-16 | 55.985 | 55.985 | 2.842e-14 | 55.985 | 23.6356 |
| `token` | 1.776e-15 | 44.2493 | 44.2493 | 24.42 | 44.2493 | 26.2886 |

`different_prompt_gram_distance` (cross-prompt baseline): **47.6043**.

Sequence mode: full Gram preserved exactly (the leakage Stage 7.6g already reports). Chunk mode: within-chunk Gram preserved, cross-chunk Gram disrupted by independent Q_chunk. Token mode: only row L2 norms preserved (required by exact RMSNorm correctness); the full Gram off-diagonal is disrupted by independent Q_i.

## Limitations

- Token-wise orthogonal masking does not hide row norms, because exact RMSNorm requires row-norm preservation. Row L2 norms remain observable on every boundary.
- Token / chunk modes reduce full Gram leakage by avoiding a sequence-shared orthogonal basis; off-diagonal Gram terms become per-row Q_i Q_j^T mixtures rather than identity.
- Per-token / per-chunk Q sampling and transition tables introduce per-row matmuls (einsum over the token axis); this is a security-efficiency knob, not formal cryptographic security.
- Sequence mode (chunk size = full sequence) is the Stage 7.6g baseline; full Gram is preserved (carried-over leakage surface).
- CPU local emulation only; no real TEE / GPU deployment.
- Synthetic tiny modern decoder; num_layers default = 1.
- Attention scores / probs are plain by construction of the QK invariant (carried over from Stage 7.6g).
- RoPE-pair 2D norms are still preserved (Stage 7.6g RoPE-safe leakage surface, carried over).
- This is NOT formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> We add a granularity knob to the orthogonal RMSNorm-compatible mask. Token-wise and chunk-wise modes preserve per-row L2 norms (required by exact RMSNorm correctness) but disrupt the full-sequence Gram-matrix leakage that sequence-shared Q exhibits. Per-decode-step token-wise masking trades fresh per-row transition cost for reduced Gram leakage on the accelerator boundary; we present this as a security-efficiency knob, not a formal security guarantee.

## Unsafe Wording to Avoid

- Token-wise masking hides row norms.
- Token-wise masking cryptographically hides hidden states.
- Per-token Q eliminates all RMSNorm-compatible leakage.
- Gram leakage is zero.
- This is formal cryptographic security.

