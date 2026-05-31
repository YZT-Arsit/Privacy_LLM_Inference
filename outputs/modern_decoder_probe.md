# Modern Decoder-Only (Qwen / TinyLlama) — Stage 6.4 Probe

## Experiment Scope

- batch_size: 2; seq_len: 8; hidden_size: 128; intermediate_size: 512
- num_query_heads: 4; num_kv_heads: 2; head_dim: 32
- dtype: float32; device: cpu; seed: 2026

## Model Loading Status

- status: `synthetic_only`
- model_id: `None`
- model_family: `None`
- reason: attempt_real_model_load=False (default); modern_decoder probe uses synthetic tensors to avoid network downloads.

## Modern Decoder Architecture Spec

- `architecture_type`: `decoder_only`
- `model_family`: `synthetic_modern_decoder`
- `norm_type`: `rmsnorm`
- `activation_type`: `swiglu`
- `position_encoding_type`: `rotary`
- `attention_variant`: `gqa`
- `hidden_size`: `128`
- `intermediate_size`: `512`
- `num_query_heads`: `4`
- `num_kv_heads`: `2`
- `head_dim`: `32`

## RMSNorm Orthogonal Island Probe

| use_pad | rms_core_max_abs_error | folded_output_max_abs_error | allclose | online_extra_matmul_count |
|---|---|---|---|---|
| False | 2.623e-06 | 2.861e-06 | True | 0 |
| True | 5.245e-06 | 2.384e-06 | True | 0 |

## SwiGLU Paired-Permutation Island Probe

| use_pad | max_abs_error | permutation_dim | shared_permutation_for_up_gate | online_extra_matmul_count | pad_placement |
|---|---|---|---|---|---|
| False | 8.106e-06 | 512 | True | 0 | `n/a` |
| True | 1.287e-05 | 512 | True | 0 | `linear_boundary_only` |

## RoPE-Aware Attention Probe

**Probe A — post-RoPE masking invariant (REQUIRED).**

- requirement: scores_tilde ≈ scores_plain after Q/K masking
- max_abs_error: 6.676e-06; relative_l2_error: 2.740e-07; allclose: True
- qk_constraint_error: 4.768e-07

**Probe B — pre-RoPE mask commutation (feasibility / negative result).**

| mask_family | expected_behavior | commutes | max_abs_error |
|---|---|---|---|
| `dense_invertible` | expected_failure | False | 5.052e+00 |
| `orthogonal` | expected_failure | False | 5.517e+00 |
| `block_diagonal_rotation` | expected_to_commute | True | 2.384e-07 |

Only masks that act as 2D rotations in the same planes as RoPE commute with RoPE. Generic dense and generic orthogonal masks DO NOT commute. The required path uses probe A (post-RoPE masking).

## GQA / MQA KV Shape Probe

- attention_variant: `gqa`
- group_size: 2
- mask_dimension: 32 (= head_dim)
- mask is per-head, NOT hidden_size, NOT num_heads.
- qk_constraint_max_error_per_q_head: 3.576e-07
- score_path: max_abs_error=8.345e-07, allclose=True
- value_path: max_abs_error=8.345e-07, allclose=True

## Workload / Integration Status

- architecture_type: `decoder_only`
- model_family: `synthetic_modern_decoder`
- norm_type: `rmsnorm`
- activation_type: `swiglu`
- position_encoding_type: `rotary`
- attention_variant: `gqa`
- integration_level: `probe_level` (Stage 6.4 — probe-level migration)
- all_required_probes_allclose: `True`
- online_extra_matmul_count: `0`
- default_nonlinear_mode: `trusted`
- workload_profiler integration: see `wrapper_integration_status.ours_compatible_nonlinear_islands.qwen_or_modern_decoder` and `cross_architecture_summary.compatible_island_integration_status`.

## Security Caveats from Stage 5.4

- security_profile: `inherits Stage 5.4 caveats`
- inherits Stage 5.4 mitigation table: fixed permutation is unsafe_default_on; fresh permutation alone needs_more_evaluation; dense sandwich + fresh permutation + pad at Linear boundaries is the recommended default-on candidate.

## Limitations

- Compatible mask families are weaker than unrestricted dense masks.
- Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.
- Probe-level migration only — not a full Qwen / TinyLlama wrapper integration.
- No Qwen / TinyLlama generation path is implemented.
- GQA / MQA is tensor-level only, not full runtime KV cache integration.
- Compatible islands inherit Stage 5.4 mitigation requirements: fresh permutation + dense sandwich + pad at Linear boundaries.
- This is not a real TEE measurement.
- This is not formal security.

## Next Stage Plan

- Stage 6.4b — Real Qwen / TinyLlama small-model loading and block-level wrapper integration behind the same `nonlinear_mode` feature flag (default `trusted`).
- Stage 5.3e — Dense-sandwich integration inside the existing wrapper / probe paths so the Stage 5.4 default-on mitigation bundle becomes selectable end-to-end.
- Stage 5.5 — Stronger adaptive attackers on real model activations.
