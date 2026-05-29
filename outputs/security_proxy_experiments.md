# Privacy LLM Obfuscation — Security Proxy Experiments (Stage 6.3)

## Experiment scope

Four lightweight security proxies over the mask + pad scheme used in Stages 1–6.2. None of these constitutes a formal security proof; each is a naive-observer upper bound.

## Pad linkability proxy

fixed_mask_no_pad is the highest naive-linkability risk; fresh pad reduces the stability of the GPU-visible tensor across requests with identical plaintext input. This is a proxy under a naive observer model only, not a security proof.

| strategy | mean pair-cos | max pair-cos | min pair-cos | mean pair-L2 | interpretation |
|---|---|---|---|---|---|
| fixed_mask_no_pad | 1.000e+00 | 1.000e+00 | 1.000e+00 | 0 | Highest naive-linkability — same plaintext maps to identical GPU tensor. |
| fixed_mask_fresh_pad | 0.514901 | 0.574997 | 0.453671 | 4.530e+01 | Fresh pad alone scrambles values but reuses the same mask across requests. |
| fresh_mask_no_pad | 0.003563 | 0.092285 | -0.096952 | 4.679e+01 | Fresh mask alone removes the identity equivalence but preserves linear structure. |
| fresh_mask_fresh_pad | 5.332e-04 | 0.116745 | -0.098773 | 6.514e+01 | Fresh mask + fresh pad — lowest naive-linkability under this proxy. |

- Ranking (high → low linkability): fixed_mask_no_pad > fixed_mask_fresh_pad > fresh_mask_no_pad > fresh_mask_fresh_pad

## Mask freshness audit

Only sha256 fingerprints of mask tensors are kept. The condition number stats are aggregate-only and do not expose any single mask. Unexpected reuse counts duplicates outside the declared session-reuse policy.

| mask | expected policy | generated | unique fingerprints | unexpected reuse | cond mean | cond max | cond min |
|---|---|---|---|---|---|---|---|
| input_mask | fresh_across_trials | 32 | 32 | 0 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| output_mask | fresh_across_trials | 32 | 32 | 0 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| pad | fresh_across_trials | 32 | 32 | 0 | 2.634e+00 | 3.026e+00 | 2.282e+00 |
| kv_cache_mask | reused_within_session_fresh_across_sessions | 64 | 32 | 0 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| encoder_memory_mask | reused_within_encoder_memory_fresh_across_sessions | 64 | 32 | 0 | 1.000e+00 | 1.000e+00 | 1.000e+00 |

## Boundary leakage accounting

- compensation_terms are GPU-visible transcript objects.
- security proxy does not prove semantic security.
- real TEE isolation is not implemented in this stage.

### GPU-visible tensors

| name | contains plaintext | architecture scope | leakage note |
|---|---|---|---|
| obfuscated_input | false | all | X_tilde = (X - T) N_in or X N_in — never the plaintext input. |
| transformed_linear_weight | false | all | W_tilde = N_in_inv W N_out — depends on masks only. |
| transformed_lora_adapter | false | decoder_only | A_tilde, B_tilde — adapter content depends on rank mask. |
| compensation_terms | false | all | C = T W N_out is GPU-visible; depends on pad and masks, not on plaintext X. |
| obfuscated_q | false | all | Q_tilde = Q N_Q (per-head block-diag mask). |
| obfuscated_k | false | all | K_tilde = K N_K. |
| obfuscated_v | false | all | V_tilde = V N_V. |
| obfuscated_kv_cache | false | decoder_only | Stage 4.8 ObfuscatedGPT2KVCache: GPU keeps only masked K/V. |
| obfuscated_encoder_memory_cache | false | encoder_decoder | Stage 6.2 EncoderMemoryCache (probe-only) — GPU never sees K_plain/V_plain. |
| obfuscated_logits | false | decoder_only | logits_tilde = logits N_vocab; vocab mask is diagonal in Stage 4.7+. |

### Trusted-only tensors

| name | contains plaintext | architecture scope | leakage note |
|---|---|---|---|
| plaintext_input | true | all | Plaintext input never leaves SimulatedTEE. |
| plaintext_hidden_state | true | all | Intermediate plaintext hidden states stay on trusted side. |
| plaintext_logits | true | all | Recovered logits Y = Y_tilde N_out_inv stay trusted-only. |
| sampling_result | true | decoder_only | Argmax / token id produced by greedy decode is trusted-only. |
| masks | false | all | N_in, N_out, N_Q, N_K, N_V, N_vocab never sent to GPU. |
| mask_inverses | false | all | N_*_inv held by SimulatedTEE for input transform and output recovery. |
| pads | false | all | T held trusted-only; only T W N_out is sent across as compensation. |
| plaintext_lora_adapter | true | decoder_only | Plain LoRA A, B held trusted-only; only A_tilde, B_tilde cross to GPU. |
| optimizer_state | true | all | No optimizer state crosses the boundary in this inference-only stage. |

## Cache leakage proxy

This is a direct nearest-neighbour matching proxy under cosine similarity. It does not implement adaptive or learned inversion attacks. A low top1_match_rate for obfuscated→plain only bounds the naive observer; the absence of a stronger attack here is not a guarantee.

| cache kind | matching | top1 match rate | mean correct rank | cos correct pair | cos best wrong pair | queries |
|---|---|---|---|---|---|---|
| kv_cache | plain_to_plain_baseline | 1.000e+00 | 0 | 1.000e+00 | 0.369674 | 512 |
| kv_cache | obfuscated_to_plain | 0.001953 | 2.862e+02 | -0.025147 | 0.370044 | 512 |
| kv_cache | obfuscated_to_plain_v | 0.005859 | 2.243e+02 | 0.027118 | 0.369576 | 512 |
| encoder_memory_cache | plain_to_plain_baseline | 1.000e+00 | 0 | 1.000e+00 | 0.371327 | 512 |
| encoder_memory_cache | obfuscated_to_plain | 0.003906 | 2.384e+02 | 0.015097 | 0.371852 | 512 |

## Interpretation

- `fixed_mask_no_pad` mean pairwise cosine: **1.0000** vs `fresh_mask_fresh_pad`: **0.0005** — fresh mask + fresh pad collapses the naive linkability signal toward zero.
- KV cache: plain↔plain top1=1.0000; obf↔plain top1=0.0020. Naive cosine-matching cannot recover plaintext KV from K_tilde / V_tilde.
- Encoder memory cache: plain↔plain top1=1.0000; obf↔plain top1=0.0039. Same naive bound applies.

## Limitations

- These experiments are security proxies, not formal security proofs.
- They do not implement adaptive attacks.
- They do not implement learned inversion attacks.
- They do not evaluate real TEE isolation.
- They do not cover side channels.
- They do not prove LoRA adapter extraction resistance.

## Next stage plan

- **Stage 5.1** — GPU-side LayerNorm primitive (replaces a trusted-side leakage point counted under `trusted_only` above).
- **Stage 5.2** — GELU / activation primitive feasibility.
- **Stage 5.3** — stronger leakage experiments (adaptive observer, learned inverter); current experiments here are only naive-observer proxies.
