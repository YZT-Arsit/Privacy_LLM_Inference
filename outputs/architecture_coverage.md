# Privacy LLM Obfuscation — Architecture Coverage (Stage 6.0)

Scaffolding for paper-grade multi-architecture experiments. The coverage report identifies whether each Transformer family (decoder-only, encoder-only, encoder-decoder) can be loaded and classified by the inspector. Stage 6.0 does **not** implement obfuscated wrappers for non-GPT-2 architectures — those are deferred to Stages 6.1 / 6.2.

## Model coverage
| architecture | status | model_id | model_class | layers | heads | hidden | skip reason |
|---|---|---|---|---|---|---|---|
| decoder_only | loaded | sshleifer/tiny-gpt2 | GPT2LMHeadModel | 2 | 2 | 2 | — |
| encoder_only | loaded | hf-internal-testing/tiny-bert | BertForMaskedLM | 2 | 2 | 128 | — |
| encoder_decoder | loaded | hf-internal-testing/tiny-random-t5 | T5ForConditionalGeneration | 5 | 4 | 32 | — |

## Architecture type matrix
| architecture | type | encoder | decoder | cross-attn | causal self-attn | bidir self-attn | past_key_values |
|---|---|---|---|---|---|---|---|
| decoder_only | decoder_only | false | true | false | true | false | true |
| encoder_only | encoder_only | true | false | false | false | true | false |
| encoder_decoder | encoder_decoder | true | true | true | true | true | true |

## Output heads
| architecture | lm_head | mlm_head | classification_head | embedding path | self-attn path | cross-attn path | output head path |
|---|---|---|---|---|---|---|---|
| decoder_only | true | false | false | transformer.wte | transformer.h.0.attn | — | lm_head |
| encoder_only | false | true | false | bert.embeddings.word_embeddings | bert.encoder.layer.0.attention.self | — | cls.predictions |
| encoder_decoder | true | false | false | shared | encoder.block.0.layer.0.SelfAttention | decoder.block.0.layer.1.EncDecAttention | lm_head |

## Attention taxonomy
| kind | architecture | Q source | K source | V source | mask type | cache type |
|---|---|---|---|---|---|---|
| causal_self_attention | decoder_only | decoder_hidden_states | decoder_hidden_states | decoder_hidden_states | causal | autoregressive_kv_cache |
| bidirectional_self_attention | encoder_only | encoder_hidden_states | encoder_hidden_states | encoder_hidden_states | padding_bidirectional | none |
| cross_attention | encoder_decoder | decoder_hidden_states | encoder_memory | encoder_memory | encoder_padding_mask | encoder_memory_cache |

## Required invariants (per attention kind)
| kind | architecture | required invariant |
|---|---|---|
| causal_self_attention | decoder_only | Q_tilde K_tilde^T = Q K^T (per-head N_Q N_K^T = I); K_cache_tilde = K_cache N_K; V_cache_tilde = V_cache N_V. |
| bidirectional_self_attention | encoder_only | Q_tilde K_tilde^T = Q K^T (per-head N_Q N_K^T = I); no autoregressive cache (the whole sequence is seen at once). |
| cross_attention | encoder_decoder | Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T; K_enc_tilde = K_enc N_K; V_enc_tilde = V_enc N_V (encoder memory cached once per generation). |

## Attention kinds expected per loaded model
| architecture | expected attention kinds |
|---|---|
| decoder_only | causal_self_attention |
| encoder_only | bidirectional_self_attention |
| encoder_decoder | bidirectional_self_attention, causal_self_attention, cross_attention |

## Next-stage implementation plan

- **Stage 6.1** — Bidirectional self-attention probe for BERT-style encoder-only models (no autoregressive cache, padding masks instead of causal masks).
- **Stage 6.2** — Cross-attention probe for T5 / BART encoder-decoder models (K / V come from the encoder memory and stay constant across decoder steps, so the cache layout differs from Stage 4.8).
- **Stage 6.3** — Cross-architecture workload + security experiments (re-run the Stage 5.0.1 profiler + attention experiments over each architecture; verify the Q/K constraint and the cache invariant generalise).
- **Stage 6.4** — Qwen / ModelScope migration (delayed until the three baseline architectures are covered).

## Limitations

- Only the architecture inspector + attention taxonomy are populated in Stage 6.0. No obfuscated forward / cache / generation path exists for BERT / T5 / BART yet.
- Model coverage depends on HuggingFace Hub access at run time; missing models are skipped, not failed.
- `prajjwal1/bert-tiny` does not load via `AutoConfig` because the checkpoint config predates the modern `model_type` key. The registry falls back to `hf-internal-testing/tiny-bert` first.

## Reproducibility

```bash
python scripts/run_architecture_coverage.py
```
