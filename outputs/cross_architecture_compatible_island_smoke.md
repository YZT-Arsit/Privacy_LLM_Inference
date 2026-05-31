# Cross-Architecture Compatible Island Smoke

- batch_size: 2
- seq_len: 8
- dtype: float32
- seed: 2026
- report_version: stage-5.3e-v1

## Integration headline per architecture

| architecture_type | model_id | status | activation / ffn_type |
|---|---|---|---|
| decoder_only | `sshleifer/tiny-gpt2` | implemented_model_level | model-level integrated |
| encoder_only | `hf-internal-testing/tiny-bert` | implemented_probe_level | probe-level FFN (gelu) |
| encoder_decoder | `hf-internal-testing/tiny-random-t5` | implemented_probe_level | probe-level FFN (t5_dense_relu_dense, relu) |

## GPT-2 (decoder_only) — model-level integrated

| use_pad | full_forward_allclose | full_forward_max_abs_error | generation_token_match_rate | blocks_with_compatible_islands | online_extra_matmul_count | pad_placement |
|---|---|---|---|---|---|---|
| False | True | 4.470e-08 | 1.0 | 2 | 0 | `n/a` |
| True | True | 6.706e-08 | 1.0 | 2 | 0 | `linear_boundary_only` |
| False | True | 6.706e-08 | 1.0 | 2 | 0 | `n/a` |
| True | True | 2.794e-08 | 1.0 | 2 | 0 | `linear_boundary_only` |

## BERT (encoder_only) — probe-level integrated

| use_pad | status | activation_type | permutation_dim | intermediate_size | max_abs_error | allclose | online_extra_matmul_count | pad_placement |
|---|---|---|---|---|---|---|---|---|
| False | loaded | gelu | 512 | 512 | 4.768e-06 | True | 0 | `n/a` |
| True | loaded | gelu | 512 | 512 | 6.199e-06 | True | 0 | `linear_boundary_only` |

## T5 / BART (encoder_decoder) — probe-level integrated

| use_pad | status | ffn_type | activation_type | permutation_dim | intermediate_size | max_abs_error | allclose | online_extra_matmul_count | pad_placement |
|---|---|---|---|---|---|---|---|---|---|
| False | loaded | t5_dense_relu_dense | relu | 37 | 37 | 3.725e-07 | True | 0 | `n/a` |
| True | loaded | t5_dense_relu_dense | relu | 37 | 37 | 5.066e-07 | True | 0 | `linear_boundary_only` |

## Integration scope

- GPT-2: model-level integrated.
- BERT: probe-level integrated.
- T5/BART: probe-level integrated.
- default mode remains trusted.
- LayerNorm remains trusted unless explicitly stated otherwise.
- no generation changes for BERT/T5.
- security follows Stage 5.2b caveats (fresh permutation per session, dense sandwich at Linear boundaries, pad at Linear boundaries only).
- not a real TEE measurement.
- not full BERT/T5 wrapper integration.

