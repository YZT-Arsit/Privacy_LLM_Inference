# Privacy LLM Obfuscation — Encoder-decoder Cross-attention Probe (Stage 6.2)

## Experiment scope

Encoder-decoder cross-attention probe: decoder hidden state supplies Q while encoder memory supplies K / V, so the input mask space for Q is independent of the input mask space for K / V. Validates the same mask + pad invariants as Stage 5.0 (decoder-only) and Stage 6.1 (encoder-only), plus a probe-level ``EncoderMemoryCache`` carrying obfuscated K / V.

## Model loading status
- **model_id**: `hf-internal-testing/tiny-random-t5`
- model class: `T5ForConditionalGeneration`
- family: `t5`
- hidden_size=32, num_heads=4, head_dim=8, inner_dim=32
- bias_present (q/k/v/o): {'q': False, 'k': False, 'v': False, 'o': False}
- cross-attention has_relative_attention_bias: False
- candidates tried: ['hf-internal-testing/tiny-random-t5']

## Encoder-decoder cross-attention invariants validated

1. `Q_dec_tilde = Q_dec N_Q_dec` (per head)
2. `K_enc_tilde = K_enc N_K_enc`
3. `V_enc_tilde = V_enc N_V_enc`
4. `N_Q_dec N_K_enc^T = I` (per head)
5. `Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T`
6. `softmax(Q_dec_tilde K_enc_tilde^T / sqrt(d) + M_enc) = softmax(Q_dec K_enc^T / sqrt(d) + M_enc)` for both `all_ones` and `padding` encoder masks
7. `AttnProb V_enc_tilde = (AttnProb V_enc) N_V_enc` (per head)
8. `W_O` projects from V-mask space → decoder residual mask space `N_dec_out`; use_pad pad compensation `C = T W N_out`.

## Encoder memory cache invariants

| batch_size | dec_seq_len | enc_seq_len | use_pad | K cache max err | V cache max err | cache allclose |
|---|---|---|---|---|---|---|
| 1 | 1 | 4 | true | 6.966e-07 | 5.960e-07 | true |
| 1 | 1 | 4 | false | 5.960e-07 | 5.960e-07 | true |
| 1 | 1 | 8 | true | 8.345e-07 | 9.537e-07 | true |
| 1 | 1 | 8 | false | 6.557e-07 | 5.737e-07 | true |
| 1 | 1 | 16 | true | 1.103e-06 | 8.345e-07 | true |
| 1 | 1 | 16 | false | 6.557e-07 | 8.568e-07 | true |
| 1 | 4 | 4 | true | 8.345e-07 | 8.941e-07 | true |
| 1 | 4 | 4 | false | 5.364e-07 | 5.960e-07 | true |
| 1 | 4 | 8 | true | 8.643e-07 | 9.537e-07 | true |
| 1 | 4 | 8 | false | 7.004e-07 | 5.234e-07 | true |
| 1 | 4 | 16 | true | 9.537e-07 | 1.192e-06 | true |
| 1 | 4 | 16 | false | 6.557e-07 | 6.557e-07 | true |
| 2 | 1 | 4 | true | 8.345e-07 | 7.749e-07 | true |
| 2 | 1 | 4 | false | 5.960e-07 | 5.960e-07 | true |
| 2 | 1 | 8 | true | 9.537e-07 | 1.013e-06 | true |
| 2 | 1 | 8 | false | 6.557e-07 | 7.153e-07 | true |
| 2 | 1 | 16 | true | 1.043e-06 | 1.162e-06 | true |
| 2 | 1 | 16 | false | 9.537e-07 | 8.345e-07 | true |
| 2 | 4 | 4 | true | 9.537e-07 | 1.103e-06 | true |
| 2 | 4 | 4 | false | 5.960e-07 | 4.768e-07 | true |
| 2 | 4 | 8 | true | 1.013e-06 | 1.073e-06 | true |
| 2 | 4 | 8 | false | 6.258e-07 | 8.345e-07 | true |
| 2 | 4 | 16 | true | 8.941e-07 | 1.073e-06 | true |
| 2 | 4 | 16 | false | 9.537e-07 | 8.345e-07 | true |

## Sweep results (per cell × encoder mask kind)
| batch_size | dec_seq_len | enc_seq_len | use_pad | encoder mask | score max err | prob max err | v_aggr max err | output max err | allclose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | true | all_ones | 3.874e-07 | 1.192e-07 | 3.874e-07 | 6.557e-07 | true |
| 1 | 1 | 4 | true | padding | 3.874e-07 | 8.941e-08 | 2.980e-07 | 4.582e-07 | true |
| 1 | 1 | 4 | false | all_ones | 2.980e-07 | 5.960e-08 | 2.086e-07 | 2.831e-07 | true |
| 1 | 1 | 4 | false | padding | 2.980e-07 | 8.941e-08 | 2.235e-07 | 1.751e-07 | true |
| 1 | 1 | 8 | true | all_ones | 5.215e-07 | 8.941e-08 | 2.533e-07 | 4.321e-07 | true |
| 1 | 1 | 8 | true | padding | 5.215e-07 | 1.192e-07 | 3.055e-07 | 3.502e-07 | true |
| 1 | 1 | 8 | false | all_ones | 3.017e-07 | 4.470e-08 | 2.980e-07 | 1.639e-07 | true |
| 1 | 1 | 8 | false | padding | 3.017e-07 | 4.470e-08 | 1.788e-07 | 1.341e-07 | true |
| 1 | 1 | 16 | true | all_ones | 4.917e-07 | 3.353e-08 | 2.086e-07 | 2.310e-07 | true |
| 1 | 1 | 16 | true | padding | 4.917e-07 | 4.470e-08 | 2.235e-07 | 4.098e-07 | true |
| 1 | 1 | 16 | false | all_ones | 5.364e-07 | 2.608e-08 | 1.565e-07 | 1.267e-07 | true |
| 1 | 1 | 16 | false | padding | 5.364e-07 | 5.960e-08 | 1.788e-07 | 1.043e-07 | true |
| 1 | 4 | 4 | true | all_ones | 4.619e-07 | 1.490e-07 | 4.247e-07 | 5.066e-07 | true |
| 1 | 4 | 4 | true | padding | 4.619e-07 | 1.192e-07 | 6.855e-07 | 7.153e-07 | true |
| 1 | 4 | 4 | false | all_ones | 4.061e-07 | 1.192e-07 | 2.384e-07 | 2.310e-07 | true |
| 1 | 4 | 4 | false | padding | 4.061e-07 | 1.490e-07 | 3.576e-07 | 3.725e-07 | true |
| 1 | 4 | 8 | true | all_ones | 8.345e-07 | 1.192e-07 | 4.172e-07 | 5.760e-07 | true |
| 1 | 4 | 8 | true | padding | 8.345e-07 | 1.788e-07 | 3.278e-07 | 5.495e-07 | true |
| 1 | 4 | 8 | false | all_ones | 8.345e-07 | 1.490e-07 | 2.682e-07 | 2.366e-07 | true |
| 1 | 4 | 8 | false | padding | 8.345e-07 | 1.788e-07 | 3.129e-07 | 2.353e-07 | true |
| 1 | 4 | 16 | true | all_ones | 9.835e-07 | 7.451e-08 | 2.086e-07 | 4.545e-07 | true |
| 1 | 4 | 16 | true | padding | 9.835e-07 | 1.043e-07 | 2.682e-07 | 4.917e-07 | true |
| 1 | 4 | 16 | false | all_ones | 5.960e-07 | 4.470e-08 | 2.086e-07 | 1.118e-07 | true |
| 1 | 4 | 16 | false | padding | 5.960e-07 | 5.960e-08 | 1.490e-07 | 1.416e-07 | true |
| 2 | 1 | 4 | true | all_ones | 5.960e-07 | 1.192e-07 | 2.831e-07 | 3.576e-07 | true |
| 2 | 1 | 4 | true | padding | 5.960e-07 | 1.192e-07 | 4.768e-07 | 7.004e-07 | true |
| 2 | 1 | 4 | false | all_ones | 8.345e-07 | 1.788e-07 | 3.278e-07 | 2.831e-07 | true |
| 2 | 1 | 4 | false | padding | 8.345e-07 | 2.086e-07 | 3.576e-07 | 3.278e-07 | true |
| 2 | 1 | 8 | true | all_ones | 7.451e-07 | 1.490e-07 | 3.055e-07 | 3.651e-07 | true |
| 2 | 1 | 8 | true | padding | 7.451e-07 | 1.788e-07 | 3.278e-07 | 4.694e-07 | true |
| 2 | 1 | 8 | false | all_ones | 6.258e-07 | 8.941e-08 | 2.515e-07 | 1.490e-07 | true |
| 2 | 1 | 8 | false | padding | 6.258e-07 | 1.192e-07 | 2.384e-07 | 1.341e-07 | true |
| 2 | 1 | 16 | true | all_ones | 8.903e-07 | 4.470e-08 | 2.086e-07 | 4.207e-07 | true |
| 2 | 1 | 16 | true | padding | 8.903e-07 | 6.706e-08 | 2.384e-07 | 4.396e-07 | true |
| 2 | 1 | 16 | false | all_ones | 6.557e-07 | 5.215e-08 | 1.788e-07 | 1.416e-07 | true |
| 2 | 1 | 16 | false | padding | 6.557e-07 | 8.941e-08 | 1.788e-07 | 1.639e-07 | true |
| 2 | 4 | 4 | true | all_ones | 5.960e-07 | 1.788e-07 | 4.265e-07 | 6.407e-07 | true |
| 2 | 4 | 4 | true | padding | 5.960e-07 | 1.788e-07 | 4.321e-07 | 4.834e-07 | true |
| 2 | 4 | 4 | false | all_ones | 4.768e-07 | 1.192e-07 | 3.576e-07 | 1.937e-07 | true |
| 2 | 4 | 4 | false | padding | 4.768e-07 | 1.341e-07 | 3.129e-07 | 1.788e-07 | true |
| 2 | 4 | 8 | true | all_ones | 7.376e-07 | 1.043e-07 | 3.874e-07 | 8.643e-07 | true |
| 2 | 4 | 8 | true | padding | 7.376e-07 | 1.192e-07 | 3.576e-07 | 6.109e-07 | true |
| 2 | 4 | 8 | false | all_ones | 5.811e-07 | 8.941e-08 | 3.576e-07 | 2.086e-07 | true |
| 2 | 4 | 8 | false | padding | 5.811e-07 | 8.941e-08 | 3.576e-07 | 2.086e-07 | true |
| 2 | 4 | 16 | true | all_ones | 8.643e-07 | 8.196e-08 | 2.757e-07 | 4.452e-07 | true |
| 2 | 4 | 16 | true | padding | 8.643e-07 | 1.043e-07 | 4.321e-07 | 5.215e-07 | true |
| 2 | 4 | 16 | false | all_ones | 6.557e-07 | 5.960e-08 | 1.788e-07 | 1.341e-07 | true |
| 2 | 4 | 16 | false | padding | 6.557e-07 | 7.451e-08 | 1.788e-07 | 1.490e-07 | true |

## Pad vs no-pad comparison
| use_pad | max output_err (any cell, any mask) | all cells allclose? | Q/K/V/O pad observed |
|---|---|---|---|
| true | 8.643e-07 | true | true |
| false | 3.725e-07 | true | true |

## All-ones vs padding encoder-mask comparison
| encoder mask | max output_err (any cell) | all cells allclose? |
|---|---|---|
| all_ones | 8.643e-07 | true |
| padding | 7.153e-07 | true |

## Limitations

- This stage validates only encoder-decoder cross-attention probe correctness.
- It does not implement full T5/BART obfuscated forward.
- It does not implement encoder-decoder generation.
- It does not implement decoder self-attention cache.
- It does not obfuscate LayerNorm.
- It does not obfuscate FFN / activation.
- It does not cover LM head.
- It does not claim real TEE security.
- Relative position bias is not handled unless explicitly added as a shared additive score bias in both plain and obfuscated paths.

## Next stage plan

- **Stage 6.3** — Cross-architecture workload + security experiments. Rerun the Stage 5.0.1 workload profiler over decoder-only / encoder-only / encoder-decoder to fill the 3×3 architecture × method matrix; document cost and leakage trade-offs per architecture using the probe data from Stages 5.0 / 6.1 / 6.2.

## Reproducibility

```bash
python scripts/run_cross_attention_experiments.py
```
