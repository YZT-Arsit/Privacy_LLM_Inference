# Modern Decoder Model-Level Wrapper Smoke (Stage 6.4c)

## Experiment Scope

Stage 6.4c stacks Stage 6.4b's per-block obfuscated forward into a multi-layer model-level wrapper with embedding lookup, final RMSNorm, an optionally-masked LM head, KV-cache-aware prefill / decode_step, and a hand-written greedy generation loop. Default mode for the wider system remains `nonlinear_mode='trusted'` and the default mitigation bundle remains `'fresh_perm_only'`.

## Model Loading Status

| field | value |
|---|---|
| load_status | synthetic_only |
| resolved_model_id | None |
| model_family | None |
| model_class | None |
| fallback_used | True |
| candidates_tried | [] |
| load_error | attempt_real_model_load=False (default); pytest runs the synthetic fallback to avoid network downloads. |

## Model-Level Wrapper Configuration

| field | value |
|---|---|
| source | synthetic_block |
| model_family | synthetic_modern_decoder |
| num_layers_used | 2 |
| hidden_size | 32 |
| intermediate_size | 64 |
| num_attention_heads | 4 |
| num_key_value_heads | 2 |
| head_dim | 8 |
| attention_variant | gqa |
| vocab_size | 64 |
| nonlinear_mode | compatible_islands |
| mitigation_bundles_evaluated | fresh_perm_only, fresh_perm_plus_sandwich_plus_pad |
| use_pad_values | [False, True] |
| max_new_tokens | 3 |

## Full Forward Correctness

| bundle | use_pad | allclose | max_abs_error | top1_match_rate | lm_head_status |
|---|---|---|---|---|---|
| fresh_perm_only | false | true | 7.451e-07 | 1.000 | single_dense_mask_pair_with_vocab_mask |
| fresh_perm_only | true | true | 8.941e-07 | 1.000 | single_dense_mask_pair_with_vocab_mask |
| fresh_perm_plus_sandwich_plus_pad | false | true | 7.451e-07 | 1.000 | single_dense_mask_pair_with_vocab_mask |
| fresh_perm_plus_sandwich_plus_pad | true | true | 8.941e-07 | 1.000 | single_dense_mask_pair_with_vocab_mask |

## Prefill / Decode-Step Correctness

| bundle | use_pad | allclose | max_abs_error | top1_match_rate | cache_seq_len | num_layers |
|---|---|---|---|---|---|---|
| fresh_perm_only | false | true | 8.643e-07 | 1.000 | 6 | 2 |
| fresh_perm_only | true | true | 8.643e-07 | 1.000 | 6 | 2 |
| fresh_perm_plus_sandwich_plus_pad | false | true | 8.643e-07 | 1.000 | 6 | 2 |
| fresh_perm_plus_sandwich_plus_pad | true | true | 8.643e-07 | 1.000 | 6 | 2 |

### Decode-Step (one token after prefill)

| bundle | use_pad | allclose | top1 | new_seq_len | position |
|---|---|---|---|---|---|
| fresh_perm_only | false | true | 1.000 | 7 | 6 |
| fresh_perm_only | true | true | 1.000 | 7 | 6 |
| fresh_perm_plus_sandwich_plus_pad | false | true | 1.000 | 7 | 6 |
| fresh_perm_plus_sandwich_plus_pad | true | true | 1.000 | 7 | 6 |

## Greedy Generation Correctness

| bundle | use_pad | sequence_exact_match | token_match_rate |
|---|---|---|---|
| fresh_perm_only | false | true | 1.000 |
| fresh_perm_only | true | true | 1.000 |
| fresh_perm_plus_sandwich_plus_pad | false | true | 1.000 |
| fresh_perm_plus_sandwich_plus_pad | true | true | 1.000 |

## KV Cache Invariants

- Per-layer cache holds `K_tilde = K @ N_K` and `V_tilde = V @ N_V` with one `N_K`/`N_V` per kv-head.
- Decode appends `k_new @ N_K` and `v_new @ N_V` along the seq axis using the cached mask material so the append invariant holds for the lifetime of one generation session.
- GQA: `repeat_kv` is applied on the masked cache *after* the append; per-q-head `N_Q = N_K[group]^{-T}` makes `q_tilde @ k_tilde_rep^T = q_rope @ k_rep^T`.

## RoPE / GQA Handling

- RoPE uses post-RoPE per-head masking; mask-before-RoPE commutation is not assumed (Stage 6.4 Probe A invariant).
- `decode_step` advances the RoPE absolute position via an explicit `position` argument; `_apply_rope_at` recomputes `cos`/`sin` at `[position, position+1)` for the new token.
- `rope_scaling` (linear / ntk / yarn) is recorded as a spec-level note; the default LLaMA-style base is used.

## Mitigation Bundle Results

| bundle | use_pad | dense_sandwich | boundary_pad | default_on_candidate |
|---|---|---|---|---|
| fresh_perm_only | false | false | false | false |
| fresh_perm_only | true | false | true | false |
| fresh_perm_plus_sandwich_plus_pad | false | true | false | false |
| fresh_perm_plus_sandwich_plus_pad | true | true | true | true |

## Trace Hook Status

- `collect_traces = false` (default off).
- The model wrapper re-exposes the Stage 5.5 block-level trace hook; setting `collect_traces=True` lets downstream stages (e.g. Stage 5.5b real-token-prompted attacker) capture per-layer intermediates without leaking raw tensors into JSON.

## Limitations

- This is model-level wrapper smoke, not a real TEE deployment.
- Real wall-time is not measured.
- Only greedy generation is implemented.
- Beam search / top-k / top-p are not implemented.
- RoPE scaling variants are not fully implemented unless explicitly supported.
- Qwen/TinyLlama real loading is opt-in and may be skipped.
- No LoRA training path is implemented.
- Security remains proxy-evaluated, not formal.
- Not formal security; not a real TEE measurement.
- Inter-layer hidden states are recovered to plain space between blocks.

## Next Stage Plan

- Stage 5.5b — Real-token-prompted real-activation attacker, now that tokenizer / embedding path is wired and decode-step traces are collectable.
- Stage 5.6 — Stronger attacker variants (black-box query, side-channel, ML-based permutation recovery).
- Stage 5.3d (deferred) — Full BERT and T5 obfuscated wrappers (not just probes).

