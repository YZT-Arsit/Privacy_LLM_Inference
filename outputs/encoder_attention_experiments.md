# Privacy LLM Obfuscation — Encoder-only Attention Probe (Stage 6.1)

## Experiment scope

Bidirectional self-attention probe for BERT-style encoder-only models. Tests the same mask + pad invariants as the Stage 5.0 GPT-2 attention probe, plus padding-mask coverage.

## Model loading status
- **model_id**: `hf-internal-testing/tiny-bert`
- model class: `BertForMaskedLM`
- hidden_size=128, num_heads=2, head_dim=64
- candidates tried: ['hf-internal-testing/tiny-bert', 'hf-internal-testing/tiny-random-BertModel', 'prajjwal1/bert-tiny']

## Encoder-only attention invariants validated

1. `Q_tilde = Q N_Q`, `K_tilde = K N_K`, `V_tilde = V N_V`
2. `N_Q N_K^T = I` (per head)
3. `Q_tilde K_tilde^T = Q K^T`
4. `softmax(Q_tilde K_tilde^T / sqrt(d) + M) = softmax(Q K^T / sqrt(d) + M)`
   for both `all_ones` and `padding` attention masks
5. `AttnProb V_tilde = (AttnProb V) N_V` (per head)
6. `W_O` projects from V-mask space to encoder residual mask space;
   `Y_tilde = Y N_out`; use_pad pad compensation `C = T W N_out`.

## Sweep results (per cell × mask kind)
| batch_size | seq_len | use_pad | mask kind | score max err | prob max err | attn_out max err | output max err | allclose |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | true | all_ones | 2.265e-06 | 2.682e-07 | 1.853e-06 | 4.411e-06 | true |
| 1 | 4 | true | padding | 2.265e-06 | 2.682e-07 | 1.853e-06 | 5.484e-06 | true |
| 1 | 4 | false | all_ones | 2.027e-06 | 2.384e-07 | 1.431e-06 | 3.099e-06 | true |
| 1 | 4 | false | padding | 2.027e-06 | 2.384e-07 | 1.431e-06 | 3.099e-06 | true |
| 1 | 8 | true | all_ones | 3.189e-06 | 3.576e-07 | 2.146e-06 | 3.934e-06 | true |
| 1 | 8 | true | padding | 3.189e-06 | 3.576e-07 | 2.801e-06 | 4.644e-06 | true |
| 1 | 8 | false | all_ones | 2.354e-06 | 2.980e-07 | 1.788e-06 | 2.310e-06 | true |
| 1 | 8 | false | padding | 2.354e-06 | 2.980e-07 | 2.682e-06 | 3.099e-06 | true |
| 1 | 16 | true | all_ones | 2.742e-06 | 4.768e-07 | 2.384e-06 | 3.293e-06 | true |
| 1 | 16 | true | padding | 2.742e-06 | 4.768e-07 | 2.146e-06 | 4.202e-06 | true |
| 1 | 16 | false | all_ones | 2.861e-06 | 2.384e-07 | 2.027e-06 | 2.623e-06 | true |
| 1 | 16 | false | padding | 2.861e-06 | 4.768e-07 | 1.788e-06 | 2.503e-06 | true |
| 2 | 4 | true | all_ones | 3.338e-06 | 2.384e-07 | 2.073e-06 | 5.901e-06 | true |
| 2 | 4 | true | padding | 3.338e-06 | 2.384e-07 | 2.073e-06 | 6.139e-06 | true |
| 2 | 4 | false | all_ones | 2.354e-06 | 2.384e-07 | 2.623e-06 | 3.338e-06 | true |
| 2 | 4 | false | padding | 2.354e-06 | 2.384e-07 | 2.623e-06 | 3.338e-06 | true |
| 2 | 8 | true | all_ones | 2.980e-06 | 5.662e-07 | 3.338e-06 | 4.292e-06 | true |
| 2 | 8 | true | padding | 2.980e-06 | 5.364e-07 | 3.159e-06 | 4.232e-06 | true |
| 2 | 8 | false | all_ones | 2.384e-06 | 2.384e-07 | 2.861e-06 | 3.338e-06 | true |
| 2 | 8 | false | padding | 2.384e-06 | 2.831e-07 | 2.623e-06 | 3.815e-06 | true |
| 2 | 16 | true | all_ones | 3.040e-06 | 5.364e-07 | 2.742e-06 | 5.245e-06 | true |
| 2 | 16 | true | padding | 3.040e-06 | 4.768e-07 | 2.503e-06 | 5.245e-06 | true |
| 2 | 16 | false | all_ones | 3.099e-06 | 7.749e-07 | 1.609e-06 | 3.815e-06 | true |
| 2 | 16 | false | padding | 3.099e-06 | 6.557e-07 | 2.295e-06 | 3.576e-06 | true |

## Pad vs no-pad comparison
| use_pad | max output_err (any cell, any mask) | all cells allclose? | Q/K/V/O pad observed |
|---|---|---|---|
| true | 6.139e-06 | true | true |
| false | 3.815e-06 | true | true |

## Padding mask coverage
| batch_size | seq_len | use_pad | score max err | prob max err | output max err | allclose |
|---|---|---|---|---|---|---|
| 1 | 4 | true | 2.265e-06 | 2.682e-07 | 5.484e-06 | true |
| 1 | 4 | false | 2.027e-06 | 2.384e-07 | 3.099e-06 | true |
| 1 | 8 | true | 3.189e-06 | 3.576e-07 | 4.644e-06 | true |
| 1 | 8 | false | 2.354e-06 | 2.980e-07 | 3.099e-06 | true |
| 1 | 16 | true | 2.742e-06 | 4.768e-07 | 4.202e-06 | true |
| 1 | 16 | false | 2.861e-06 | 4.768e-07 | 2.503e-06 | true |
| 2 | 4 | true | 3.338e-06 | 2.384e-07 | 6.139e-06 | true |
| 2 | 4 | false | 2.354e-06 | 2.384e-07 | 3.338e-06 | true |
| 2 | 8 | true | 2.980e-06 | 5.364e-07 | 4.232e-06 | true |
| 2 | 8 | false | 2.384e-06 | 2.831e-07 | 3.815e-06 | true |
| 2 | 16 | true | 3.040e-06 | 4.768e-07 | 5.245e-06 | true |
| 2 | 16 | false | 3.099e-06 | 6.557e-07 | 3.576e-06 | true |

## Limitations

- This stage validates only encoder self-attention probe correctness.
- It does not implement full BERT obfuscated forward.
- It does not obfuscate LayerNorm.
- It does not obfuscate GELU / FFN.
- It does not cover MLM head.
- It does not claim real TEE security.
- It does not cover encoder-decoder cross-attention.

## Next stage plan

- **Stage 6.2** — Encoder-decoder cross-attention probe (T5 / BART). Q from
  the decoder hidden state, K / V from cached encoder memory; new cache
  data structure for encoder memory invariants `K_enc_tilde = K_enc N_K`,
  `V_enc_tilde = V_enc N_V`.
- **Stage 6.3** — Cross-architecture workload + security experiments
  (rerun Stage 5.0.1 profiler over decoder-only / encoder-only / encoder-decoder).

## Reproducibility

```bash
python scripts/run_encoder_attention_experiments.py
```
