# Stage 6.7 — Masked CausalLM Input/Output Boundaries

Validates the two trusted boundaries that wrap a masked decoder stack, so
the GPU never sees `input_ids`, plaintext embeddings, or plaintext logits.
This stage isolates the **boundaries**; it does not run a full decoder.
No tokenizer, no generation loop, no network. No formal/cryptographic/
semantic security is claimed.

## Boundary

```
TEE:  input_ids -> Embed -> X_plain
      X_tilde = (X_plain - T_in) @ N_res          (release only X_tilde)
GPU:  masked decoder -> masked hidden / masked logits
      (never sees input_ids, X_plain, or plaintext logits)
TEE:  recover plaintext logits, sample / stop-token / penalties,
      keep next token, look up + mask its embedding, release only the
      masked next embedding for the next step
```

## Input boundary

`X_tilde = (X_plain - T_in) @ N_res` (additive pad `T_in` optional). Only
`X_tilde` is released; `input_ids` and the embedding table stay trusted-side
(`input_ids_visible_to_gpu = False`,
`plaintext_embedding_visible_to_gpu = False`).

## Output boundary (vocab-logit mask)

With orthogonal `N_res`, `rmsnorm_core(H @ N_res) = rmsnorm_core(H) @ N_res`.
A **lightweight** vocab-logit mask `M_vocab = P_vocab @ D_vocab` (permutation
+ positive diagonal scaling — *not* a dense `vocab × vocab` matrix) is folded
with the final-norm affine and `N_res⁻¹` into the LM head:

```
W_lm_fold  = diag(gamma_final) @ W_lm
W_lm_unmask = N_res⁻¹ @ W_lm_fold
W_lm_tilde = W_lm_unmask[:, perm] * scale[None, :]      # = N_res⁻¹·W_lm_fold·M_vocab
```

The GPU computes `L_tilde = core_tilde @ W_lm_tilde = L @ M_vocab` (masked
logits only). The TEE recovers `L = L_tilde[:, inv_perm] * inv_scale` and
samples. `plaintext_logits_visible_to_gpu = False`,
`masked_logits_visible_to_gpu = True`, `logits_recovered_in_tee = True`.

> Permutation + scaling hides direct token-index/logit alignment but is
> **weaker than dense vocab masking** and gives no semantic security.

## Sampling boundary (trusted side)

`greedy_sample`, `apply_temperature`, `top_k_filter`, `top_p_filter`,
`sample_from_logits`, and `trusted_sample_from_masked_logits` (recover →
sample). Stochastic sampling is deterministic under a seeded generator.

## One-step decode boundary

`trusted_next_token_to_masked_embedding`: next token → embedding lookup →
`(emb − T_next) @ N_res` → release only the masked next embedding.

## Files

- `src/pllo/ops/causal_lm_boundaries.py`
- `src/pllo/experiments/causal_lm_boundary_probe.py`
- `scripts/run_causal_lm_boundary_probe.py` → `outputs/causal_lm_boundary_probe.{json,md}`
- `tests/test_causal_lm_boundaries.py` — 22 tests (no transformers needed).

## Result

All boundary metrics at float64 machine precision: embedding/next-embedding
mask error `0.0`, final-norm core `≤1.3e-15`, masked logits `≤4.4e-15`,
recovered logits `≤2.9e-15`; greedy + trusted-greedy match rate `1.0`;
seeded sampling deterministic.

## Limitations / next stage

Boundary probe only (no decoder stack); synthetic embeddings/LM head; GPU
sees *masked* logits (recovery + sampling in TEE); permutation+scaling vocab
mask weaker than dense; full-vocab LM-head cost not optimized; no generation
loop / tokenizer / chat template; output text semantics not protected once
returned to the user.

**Stage 6.8** — multi-layer mask handoff and a full masked CausalLM skeleton.
