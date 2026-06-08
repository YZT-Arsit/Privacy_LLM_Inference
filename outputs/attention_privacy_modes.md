# Attention-Map Protection / Secure-Attention Modes

_Stage 7.6i: add an ``attention_privacy_mode`` knob to the Stage 7.6h low-interaction wrapper. Compares an exact low-interaction baseline (visible attention) against an exact trusted-softmax baseline (hidden attention, extra TEE round trips) and a row-constant score-blinding demonstration._

## Inherited Stage 7.6h Guarantees

| Field | Value |
|---|---|
| use_pad | True |
| rope_mask_mode | pre_rope_block_diagonal_rotation |
| rope_transient_plain_qk_visible | False |
| qkv_projection_outputs_masked_directly | True |
| norm_mask_granularity | sequence |

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
| dtype | float64 |
| device | cpu |

## Summary Comparison

| Mode | exact | one_round_trip | attention_hidden | round_trips_per_decode_step | intermediate_tee_reentry | greedy_token_match_rate | sequence_exact_match |
|---|---|---|---|---|---|---|---|
| `exact_visible_attention` | True | True | False | 1 | False | 1.0 | True |
| `trusted_softmax_attention` | True | False | True | 2 | True | 1.0 | True |
| `score_blinding_experimental` | True | True | False | 1 | False | 1.0 | True |

## Attention Privacy: Exactness vs Hiding Tension

### Current Exact Score Invariant

```
Given Q_tilde = Q B_Q and K_tilde = K B_K with B_Q B_K^T = I:
  Q_tilde K_tilde^T = Q B_Q (K B_K)^T = Q B_Q B_K^T K^T = Q K^T.
So S_tilde = S and softmax(S_tilde) = softmax(S). Exact correctness,
attention scores / probabilities visible on the accelerator.
```

### Row Constant Shift Does Not Protect Attention

```
For any row-wise scalar c_i:
  softmax(S_i + c_i * 1) = softmax(S_i).
But (S_ij + c_i) - (S_ik + c_i) = S_ij - S_ik, so ranking,
relative margins, entropy, and attention topology are unchanged.
Row-constant additive blinding is NOT attention privacy.
```

### General Additive Score Pad Breaks Exact Softmax

```
For arbitrary R: softmax(S + R) != softmax(S) unless R is row-
constant. So additive score blinding with non-row-constant R
cannot be used with ordinary accelerator-side softmax while
preserving exactness.
```

### Exact Attention Hiding Requires One Of

```
Exact attention map hiding therefore requires one of:
  (i)   trusted / secure softmax,
  (ii)  cryptographic protocol,
  (iii) approximate attention,
  (iv)  a changed threat model (fused confidential kernel).
```

## Per-Mode Correctness

| Mode | greedy_match | seq_exact | lm_head_recovery_max | h_hat invariant max | qk_score invariant max | kv_cache invariant max |
|---|---|---|---|---|---|---|
| `exact_visible_attention` | 1.0 | True | 1.235e-15 | 2.665e-15 | 3.997e-15 | 8.882e-16 |
| `trusted_softmax_attention` | 1.0 | True | 1.318e-15 | 2.887e-15 | 3.997e-15 | 8.882e-16 |
| `score_blinding_experimental` | 1.0 | True | 1.082e-15 | 2.665e-15 | 3.997e-15 | 8.882e-16 |

## Boundary Round-Trip Metrics

| Mode | online_round_trips_per_decode_step | intermediate_tee_reentry | attention_extra_tee_round_trips_per_layer | trusted_softmax_used | trusted_fallback_used_in_main_path |
|---|---|---|---|---|---|
| `exact_visible_attention` | 1 | False | 0 | False | False |
| `trusted_softmax_attention` | 2 | True | 1 | True | False |
| `score_blinding_experimental` | 1 | False | 0 | False | False |

## Attention Leakage Audit

Each field is what the accelerator transcript exposes. ``attention_map_fingerprint_available`` answers: can the accelerator-visible tensors recover the attention map?

| Mode | attention_score_persistent_transcript_visible | attention_prob_persistent_transcript_visible | attention_entropy_visible | attention_top1_index_visible | attention_topk_indices_visible | attention_relative_margin_visible | attention_map_fingerprint_available |
|---|---|---|---|---|---|---|---|
| `exact_visible_attention` | True | True | True | True | True | True | True |
| `trusted_softmax_attention` | False | False | False | False | False | False | False |
| `score_blinding_experimental` | True | True | True | True | True | True | True |

## Score-Blinding Experimental Detail

| Field | Value |
|---|---|
| row_constant_shift_used | True |
| hides_relative_attention | False |
| attention_privacy_gain | none_against_relative_attention_observer |
| row_constant_blinding_softmax_max_abs_error | 1.110e-16 |
| nonconstant_blinding_softmax_max_abs_error | 0.562182 |

Row-constant shift: softmax exact, attention pattern fully preserved (privacy gain = none). Non-row-constant additive shift: softmax error is large (recorded as ``nonconstant_blinding_softmax_max_abs_error``) -- proves that arbitrary additive score blinding cannot be combined with ordinary accelerator-side softmax while preserving exactness.

## Trusted-Softmax Detail

| Field | Value |
|---|---|
| attention_privacy_mode | trusted_softmax_attention |
| attention_scores_visible | False |
| attention_probs_visible | False |
| attention_exact | True |
| attention_map_hidden_from_accelerator_transcript | True |
| trusted_softmax_used | True |
| intermediate_tee_reentry | True |
| online_boundary_round_trips_per_decode_step | 2 |
| attention_extra_tee_round_trips_per_layer | 1 |

## Topology-Private Attention (Experimental, Design-Only)

- implementation_status: `design_only_not_implemented`
- reason: Approximate or alternative attention forms (kernelized, top-k hidden by trusted selection, noisy/rank-obfuscated scores) require dropping the exactness guarantee. They are not merged into the main protocol because the wrapper currently insists on attention_exact = true; any approximate attention must be reported with attention_exact = false.

## Fused-Kernel Transcript-Hiding Threat Model (Design-Only)

| Field | Value |
|---|---|
| implementation_status | design_only_simulation_mode |
| requires_fused_kernel_assumption | True |
| not_cryptographic_security | True |
| attention_scores_persistent_visible | False |
| attention_scores_ephemeral_inside_kernel | True |

Assumption: Adversary observes persistent accelerator tensors / global memory transcript, but does NOT observe registers or ephemeral values inside a fused confidential attention kernel.

## Limitations

- exact_visible_attention exposes the attention map by construction: the QK invariant Q_tilde K_tilde^T = Q K^T intentionally preserves scores on the accelerator side.
- trusted_softmax_attention adds extra TEE round trips per decode step (one per layer) and re-enters the trusted primitive once per attention block.
- score_blinding_experimental shows that row-constant score shifts preserve softmax exactly but do NOT hide ranking, relative margins, entropy, or attention topology.
- Non-row-constant additive score blinding breaks softmax exactness; the wrapper records the numerical error of that alternative to make the trade-off explicit.
- All modes are CPU local emulation only; no real TEE / GPU deployment.
- Synthetic tiny modern decoder; num_layers default = 1.
- This is NOT formal cryptographic / semantic / differential-privacy security.

## Paper-Safe Wording

> Exact low-interaction attention with ordinary accelerator-side softmax exposes the attention map because the QK invariant intentionally preserves the score matrix. To hide attention maps exactly, the softmax computation must be moved to a trusted/secure primitive or replaced by an approximate/private attention mechanism. We therefore provide two modes: an exact low-interaction mode with visible attention maps, and an exact trusted-softmax mode that hides attention maps at the cost of additional trusted interaction.

## Unsafe Wording to Avoid

- The exact low-interaction mode hides attention maps.
- Row-wise score shifts provide attention privacy.
- This is cryptographic security.
- The trusted-softmax baseline preserves one round trip.
- Approximate attention is exact.

