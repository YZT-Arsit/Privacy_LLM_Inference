# GPT-2 Compatible Nonlinear Island — Stage 5.3a Smoke

- model_id: `sshleifer/tiny-gpt2`
- block_index: 0
- batch_size: 2
- seq_len: 8
- dtype: float32
- seed: 2026
- report_version: stage-5.3a-v1

## Per-configuration results

| use_pad | allclose | max_abs_error | relative_l2_error | cosine_similarity | permutation_dim | pad_placement | online_extra_matmul_count |
|---|---|---|---|---|---|---|---|
| False | True | 4.768e-07 | 1.743e-07 | 1.000000 | 8 | `n/a` | 0 |
| True | True | 4.768e-07 | 1.494e-07 | 1.000000 | 8 | `linear_boundary_only` | 0 |

## Wrapper integration scope

- This is single-block wrapper integration.
- LayerNorm remains trusted.
- LM head is not modified.
- Generation path is not modified.
- GPT-2 model-level wrapper is not modified.
- BERT / T5 wrappers are not modified.
- `compatible_islands` is not enabled by default; default mode remains `trusted`.

## Security caveats (Stage 5.2b)

- Security relies on Stage 5.2b mitigations: fresh permutation per session, dense sandwich at Linear boundaries, and pad at Linear boundaries only.
- Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- This is not a real TEE measurement.
- This stage does not claim formal security; the `compatible_islands` mode is `proxy-evaluated, not formal`.

## Next stage

- Stage 5.3b — GPT-2 model-level wrapper integration of the same feature flag, followed by BERT / T5 wrappers.
