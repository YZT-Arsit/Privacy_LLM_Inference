# LoRA Integration with Stage 7.6g/h/i Main Protocol

_Stage 7.7b: verify LoRA adapters at every supported insertion site under the padded low-interaction protocol._

## Supported LoRA Sites

| Site | n_out_kind | rope_safe | padded_boundary_identity_max | trusted_recovery_max | padded_AB_minus_true_AB_max |
|---|---|---|---|---|---|
| `q_proj` | B_Q_block_diagonal_rope_plane_rotation | True | 7.105e-14 | 7.105e-14 | 0.0 |
| `k_proj` | B_K_block_diagonal_rope_plane_rotation | True | 1.421e-13 | 1.226e-13 | 0.0 |
| `v_proj` | N_V_block_diagonal_orthogonal_per_head | True | 6.040e-14 | 5.684e-14 | 0.0 |
| `o_proj` | Q_l_residual_stream_orthogonal | True | 7.461e-14 | 7.816e-14 | 0.0 |
| `up_proj` | paired_permutation_P | True | 8.704e-14 | 8.704e-14 | 0.0 |
| `gate_proj` | paired_permutation_P | True | 9.237e-14 | 9.237e-14 | 0.0 |
| `down_proj` | Q_l_residual_stream_orthogonal | True | 1.492e-13 | 1.634e-13 | 0.0 |

## Merged-Weights Generation Across Modes

| norm_granularity | chunk | attention_privacy_mode | greedy_match | seq_exact | lm_head_recovery_max | h_hat_max | round_trips |
|---|---|---|---|---|---|---|---|
| `sequence` | 1 | `exact_visible_attention` | 1.0 | True | 1.499e-15 | 1.155e-14 | 1 |
| `token` | 1 | `exact_visible_attention` | 1.0 | True | 1.305e-15 | 6.273e-15 | 1 |
| `chunk` | 2 | `exact_visible_attention` | 1.0 | True | 1.388e-15 | 8.382e-15 | 1 |
| `sequence` | 1 | `trusted_softmax_attention` | 1.0 | True | 1.721e-15 | 1.510e-14 | 2 |
| `token` | 1 | `trusted_softmax_attention` | 1.0 | True | 1.686e-15 | 7.994e-15 | 2 |

## Rank Padding Policy

| Field | Value |
|---|---|
| rank_padding_policy | pad rank from true_rank to padded_rank with zeros in A and B; A_tilde / B_tilde share the inner dimension padded_rank on the accelerator side. |
| true_rank_hidden_from_shape | True |
| padded_rank_visible | True |
| lora_adapter_plaintext_visible | False |
| lora_training_backward_supported | False |

## Limitations

- CPU local emulation only; no real TEE / GPU.
- LoRA forward path validated only; backward / training is NOT implemented.
- Algebraic identity is verified per insertion site with fresh masks and pads; sample sizes are small.
- Rank padding hides the *true* inner rank but the *padded* rank (inner dimension of A_tilde and B_tilde) is observable on the accelerator side.
- The merged-weights generation path absorbs LoRA into the base weight; the resulting W_eff = W + A B is what the wrapper sees, equivalent to deploying a fine-tuned model.
- No formal cryptographic / semantic / differential-privacy security claim.

## Paper-Safe Wording

> LoRA adapters integrate with the low-interaction main protocol via the same padded-boundary algebra used for the base linear: A_tilde = M^{-1} A R, B_tilde = R^{-1} B N_out, with rank-space mask R. The forward path is exact at float64; rank padding hides the true rank but the padded rank is observable. We do not address LoRA training.

## Unsafe Wording to Avoid

- LoRA training is supported.
- Padded rank cryptographically hides true rank.
- LoRA fine-tuning is end-to-end private.
- This is formal cryptographic security.

