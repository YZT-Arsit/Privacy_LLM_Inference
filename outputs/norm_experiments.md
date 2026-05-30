# Privacy LLM Obfuscation — Norm Primitive (Stage 5.1)

## Experiment scope

Stage 5.1 validates a unified trusted norm primitive for both LayerNorm and RMSNorm under the project's right-multiply mask convention, plus a restricted feasibility probe for RMSNorm under orthogonal masks. The trusted primitive standardises the existing trusted-LayerNorm shortcut used in Stages 2–6.2 — it does not yet execute norm on the GPU side.

## Why general right-mask does not commute with LayerNorm

General right masks do not commute with LayerNorm. For ``X N`` with a non-orthogonal invertible ``N``, the column-wise mean and variance change in ways that depend on the off-diagonal mixing in ``N``, so ``LayerNorm(X N) ≠ LayerNorm(X) N``. The Stage 5.1 trusted primitive therefore continues to recover plaintext ``X`` on the trusted side, run the actual norm in cleartext, and re-mask the output.

## Trusted norm primitive correctness

| norm_type | batch_size | seq_len | hidden_size | use_pad | max output err | y_tilde invariant err | reference err | allclose |
|---|---|---|---|---|---|---|---|---|
| layernorm | 1 | 4 | 64 | true | 1.192e-06 | 0 | 4.768e-06 | true |
| layernorm | 1 | 4 | 64 | false | 5.364e-07 | 0 | 1.431e-06 | true |
| layernorm | 1 | 4 | 128 | true | 1.907e-06 | 0 | 5.722e-06 | true |
| layernorm | 1 | 4 | 128 | false | 1.431e-06 | 0 | 2.861e-06 | true |
| layernorm | 1 | 8 | 64 | true | 1.192e-06 | 0 | 2.384e-06 | true |
| layernorm | 1 | 8 | 64 | false | 1.431e-06 | 0 | 1.907e-06 | true |
| layernorm | 1 | 8 | 128 | true | 2.623e-06 | 0 | 4.292e-06 | true |
| layernorm | 1 | 8 | 128 | false | 1.669e-06 | 0 | 3.815e-06 | true |
| layernorm | 2 | 4 | 64 | true | 1.192e-06 | 0 | 2.384e-06 | true |
| layernorm | 2 | 4 | 64 | false | 1.431e-06 | 0 | 1.907e-06 | true |
| layernorm | 2 | 4 | 128 | true | 2.623e-06 | 0 | 4.292e-06 | true |
| layernorm | 2 | 4 | 128 | false | 1.669e-06 | 0 | 3.815e-06 | true |
| layernorm | 2 | 8 | 64 | true | 1.788e-06 | 0 | 2.861e-06 | true |
| layernorm | 2 | 8 | 64 | false | 1.431e-06 | 0 | 2.384e-06 | true |
| layernorm | 2 | 8 | 128 | true | 2.861e-06 | 0 | 4.530e-06 | true |
| layernorm | 2 | 8 | 128 | false | 1.907e-06 | 0 | 4.292e-06 | true |
| rmsnorm | 1 | 4 | 64 | true | 9.537e-07 | 0 | 5.245e-06 | true |
| rmsnorm | 1 | 4 | 64 | false | 4.768e-07 | 0 | 1.311e-06 | true |
| rmsnorm | 1 | 4 | 128 | true | 2.623e-06 | 0 | 5.245e-06 | true |
| rmsnorm | 1 | 4 | 128 | false | 2.384e-06 | 0 | 2.742e-06 | true |
| rmsnorm | 1 | 8 | 64 | true | 1.431e-06 | 0 | 2.861e-06 | true |
| rmsnorm | 1 | 8 | 64 | false | 1.192e-06 | 0 | 1.669e-06 | true |
| rmsnorm | 1 | 8 | 128 | true | 2.861e-06 | 0 | 4.411e-06 | true |
| rmsnorm | 1 | 8 | 128 | false | 1.431e-06 | 0 | 3.338e-06 | true |
| rmsnorm | 2 | 4 | 64 | true | 1.431e-06 | 0 | 2.861e-06 | true |
| rmsnorm | 2 | 4 | 64 | false | 1.192e-06 | 0 | 1.669e-06 | true |
| rmsnorm | 2 | 4 | 128 | true | 2.861e-06 | 0 | 4.411e-06 | true |
| rmsnorm | 2 | 4 | 128 | false | 1.431e-06 | 0 | 3.338e-06 | true |
| rmsnorm | 2 | 8 | 64 | true | 1.431e-06 | 0 | 2.950e-06 | true |
| rmsnorm | 2 | 8 | 64 | false | 1.431e-06 | 0 | 2.146e-06 | true |
| rmsnorm | 2 | 8 | 128 | true | 1.669e-06 | 0 | 4.292e-06 | true |
| rmsnorm | 2 | 8 | 128 | false | 1.907e-06 | 0 | 5.245e-06 | true |

## Restricted RMSNorm orthogonal-mask feasibility

If ``N`` is an orthogonal matrix, ``rms(X N) = rms(X)`` and ``normalize(X N) = normalize(X) N``. This is a restricted feasibility result — orthogonal masks form a strict subset of the general invertible-right-mask family used by the project, and they carry a different security profile that Stage 5.3 would have to evaluate before any GPU-side RMSNorm protocol could land.

| hidden_size | orthogonality err | rms preservation err | normalized state err | scalar gamma err | vector gamma err | allclose w/o gamma | allclose scalar gamma | allclose vector gamma |
|---|---|---|---|---|---|---|---|---|
| 64 | 3.576e-07 | 1.192e-07 | 1.669e-06 | 2.384e-06 | 9.679e+00 | true | true | false |
| 128 | 5.960e-07 | 1.192e-07 | 2.623e-06 | 3.815e-06 | 1.116e+01 | true | true | false |

## Gamma commutation analysis

Vector gamma breaks simple right-mask commutation. RMSNorm scales the normalised hidden state element-wise by ``gamma ∈ R^H``. For a scalar ``gamma`` (a single broadcast value) the right multiply by ``N`` commutes; for a vector ``gamma`` the mapping ``g ⊙ Z`` applied before ``Z N`` mixes channels differently than after, so ``gamma ⊙ (Z N) ≠ (gamma ⊙ Z) N`` in general. The probe's ``vector_gamma_max`` column quantifies this gap — the report calls vector-gamma RMSNorm out as the dominant blocker for any GPU-side RMSNorm protocol on production checkpoints (LLaMA / T5 / Qwen all ship per-channel gamma).

## Limitations

- TrustedNormPrimitive still runs norm in the trusted side.
- It standardizes the current trusted shortcut but does not eliminate trusted compute.
- General right masks do not commute with LayerNorm.
- General right masks do not commute with RMSNorm unless norm-preserving restrictions (orthogonal masks) are imposed.
- Vector gamma breaks simple right-mask commutation in RMSNorm.
- This stage does not implement GELU / activation obfuscation.
- This stage does not implement real TEE.
- This stage does not claim formal security.

## Next stage plan

- **Stage 5.2** — Activation primitive feasibility (GELU / SwiGLU / ReLU). Mirrors this stage: trusted primitive wrapper first, restricted-mask feasibility probe second.
- **Stage 5.3** — Security proxy experiments for the orthogonal-mask restriction (what does sampling N from O(H) leak vs sampling from GL(H)).
- **Stage 6.4** — Qwen / ModelScope migration once a GPU-side norm primitive exists for the per-channel-gamma case.
