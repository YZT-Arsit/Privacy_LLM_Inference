# GPT-2 Model-Level Compatible Nonlinear Island — Stage 5.3b Smoke

- model_id: `sshleifer/tiny-gpt2`
- batch_size: 2
- seq_len: 8
- max_new_tokens: 4
- dtype: float32
- seed: 2026
- report_version: stage-5.3b-v1

## Full-model forward correctness

| use_pad | allclose | max_abs_error | relative_l2_error | cosine_similarity | top1_match_rate |
|---|---|---|---|---|---|
| False | True | 4.470e-08 | 1.582e-07 | 1.000054 | 1.0000 |
| True | True | 6.706e-08 | 2.275e-07 | 1.000054 | 1.0000 |

## Greedy generation correctness

| use_pad | max_new_tokens | sequence_exact_match | token_match_rate | top1_match_rate | max_logits_error |
|---|---|---|---|---|---|
| False | 4 | 1.0000 | 1.0000 | 1.0000 | 2.980e-08 |
| True | 4 | 1.0000 | 1.0000 | 1.0000 | 3.725e-08 |

## Island audit summary

| use_pad | num_blocks | blocks_with_compatible_islands | total_mlp_island_permutation_draws | online_extra_matmul_count | pad_placement | layernorm_remains_trusted |
|---|---|---|---|---|---|---|
| False | 2 | 2 | 2 | 0 | `n/a` | True |
| True | 2 | 2 | 2 | 0 | `linear_boundary_only` | True |

## Wrapper integration scope

- This is GPT-2 model-level wrapper integration (Stage 5.3b).
- LayerNorm remains trusted.
- LM head is not modified.
- KV cache and greedy generation control flow are not modified.
- BERT and T5 wrappers are not integrated.
- `compatible_islands` is not the default; default mode remains `trusted` for every wrapper.
- This is a measured GPT-2 model-level smoke, not a full cross-architecture measurement.

## Security caveats (Stage 5.2b)

- Security relies on Stage 5.2b mitigations: fresh permutation per session, dense sandwich at Linear boundaries, and pad at Linear boundaries only.
- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- Only GPT-2 model-level wrapper is integrated; BERT/T5 not integrated.
- This is not a real TEE measurement.
- This stage does not claim formal security; `compatible_islands` remains `proxy-evaluated, not formal`.

## Next stage

- Stage 5.3c — BERT and T5 wrapper / probe selective integration of the same `nonlinear_mode` feature flag.
