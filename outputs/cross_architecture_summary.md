# Privacy LLM Obfuscation — Cross-Architecture Summary (Stage 6.3)

## Experiment scope

Cross-architecture summary aggregates Stage 5.0 (decoder-only), Stage 6.1 (encoder-only) and Stage 6.2 (encoder-decoder cross-attention) probe outputs plus the Stage 5.0.1 workload profile. It does not re-execute any probe.

## Cross-architecture coverage table

| architecture | status | model_id | model_class | attention_kind | cache_type | cells | rows |
|---|---|---|---|---|---|---|---|
| decoder_only | aggregated | sshleifer/tiny-gpt2 | GPT2LMHeadModel | causal_self_attention | autoregressive_kv_cache | 36 | 36 |
| encoder_only | aggregated | hf-internal-testing/tiny-bert | BertForMaskedLM | bidirectional_self_attention | none | 12 | 24 |
| encoder_decoder | aggregated | hf-internal-testing/tiny-random-t5 | T5ForConditionalGeneration | cross_attention | encoder_memory_cache | 24 | 48 |

## Attention invariant summary

| architecture | all allclose | max output err | max score err | max prob err | max cache err |
|---|---|---|---|---|---|
| decoder_only | true | 9.219e-09 | 2.765e-10 | 0 | 4.098e-08 |
| encoder_only | true | 6.139e-06 | 3.338e-06 | 7.749e-07 | — |
| encoder_decoder | true | 8.643e-07 | 9.835e-07 | 2.086e-07 | 1.192e-06 |

## Cache support summary

| architecture | cache_type | max cache err |
|---|---|---|
| decoder_only | autoregressive_kv_cache | 4.098e-08 |
| encoder_only | none | — |
| encoder_decoder | encoder_memory_cache | 1.192e-06 |

## Pad support summary

| architecture | use_pad values seen | padding mask supported | bias (q/k/v/o) | relative position bias |
|---|---|---|---|---|
| decoder_only | False/True | false | True/True/True/True | false |
| encoder_only | False/True | true | — | false |
| encoder_decoder | False/True | true | False/False/False/False | false |

## Workload summary (from Stage 5.0.1 profiler)

| method | implemented | boundary calls | boundary calls formula | trusted compute ops | gpu ops | measured wall-time (ms) | source |
|---|---|---|---|---|---|---|---|
| plain_hf_gpu | true | 0 | 0 (no boundary) | 0 | 4434424 | 3.826e+00 | measured |
| tslp_trusted_nonlinear_baseline | false | 32 | 3L + 2 = 8 per forward (LN_1 + LN_2 + GELU per layer + ln_f + LM head) | 1110230 | 4429848 | — | projected_from_op_counts |
| ours_current | true | 36 | 4L + 1 = 9 per forward (4 obfuscated linears per layer + LM head) | 1116310 | 4429848 | 6.169e+00 | measured |
| ours_ideal_gpu_nonlinear | false | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4434424 | — | projected_from_op_counts |
| amulet_style_reference | false | 4 | 1 per forward (single fused GPU pipeline round trip) | 1105654 | 4434424 | — | projected_from_op_counts |

## Trusted shortcuts still in place per architecture

- **decoder_only**:
  - `trusted_layernorm`
  - `trusted_gelu`
  - `lm_head_vocab_diag_mask_only`
- **encoder_only**:
  - `trusted_layernorm`
  - `trusted_gelu`
  - `no_mlm_head_obfuscation`
- **encoder_decoder**:
  - `trusted_layernorm`
  - `trusted_ffn_activation`
  - `no_decoder_self_attention_cache`
  - `no_relative_position_bias_obfuscation`

## Limitations

- **decoder_only**:
  - LayerNorm runs inside SimulatedTEE (trusted shortcut).
  - GELU runs inside SimulatedTEE (trusted shortcut).
  - LM head uses a diagonal vocab output mask only, no full pad.
  - Real TEE isolation is not implemented.
- **encoder_only**:
  - BERT obfuscated forward (LayerNorm / GELU / FFN / MLM head) is not implemented.
  - Only first-layer self-attention Q / K / V / O is validated.
  - Real TEE isolation is not implemented.
- **encoder_decoder**:
  - T5/BART obfuscated forward (LayerNorm / FFN / activation / LM head) is not implemented.
  - Decoder self-attention KV cache is not implemented.
  - Encoder-decoder generation is not implemented.
  - Relative position bias is not obfuscated.
  - Real TEE isolation is not implemented.
- This summary aggregates existing JSON; it does not re-run probes.
- It does not claim real TEE security; security claims are deferred to the security proxy report.

## Next stage plan

- **Stage 5.1** — GPU-side LayerNorm primitive (replaces the trusted LayerNorm shortcut shared by all three architectures).
- **Stage 5.2** — GELU / activation primitive feasibility (replaces the trusted activation shortcut).
- **Stage 6.4** — Qwen / ModelScope migration on top of Stage 6.0+'s architecture scaffold once a non-trusted nonlinear primitive is ready.
