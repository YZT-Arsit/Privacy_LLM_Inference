# 6. Correctness Analysis

This section gives algebraic proof sketches for the design's correctness identities and pairs each one with the artifact row that empirically verifies it. We do **not** claim formal cryptographic security here; "correctness" in this paper means: the GPU-visible masked computation, when un-masked by the trusted side, equals the plain reference output in our tested configurations (and, for the algebraic identities, on all inputs that satisfy the stated assumptions).

The artifact backing each theorem is summarized in **Table 3 (correctness summary)** and **Figure 4 (dense sandwich + pad compensation)**.

## Theorem 1 — Linear boundary with pad compensation

**Assumptions.** `N_in, N_out` are invertible matrices over `R`. `T` is an arbitrary trusted-side translation tensor of the shape of `X`.

**Statement.** For every `X`,
```
((X - T) N_in) (N_in^{-1} W N_out) + T W N_out + b N_out  =  (X W + b) N_out
```

**Proof sketch.** Expand the left-hand side:
```
(X - T) N_in N_in^{-1} W N_out  +  T W N_out  +  b N_out
= (X - T) W N_out + T W N_out + b N_out
= X W N_out - T W N_out + T W N_out + b N_out
= X W N_out + b N_out
= (X W + b) N_out.
```

**Artifact support.** Stage 1–4 GPT-2 model wrapper (`outputs/gpt2_model_correctness.json`, `outputs/gpt2_generation_correctness.json`, `outputs/workload_profile.json`) verifies the model-level identity end-to-end across prefill, decode_step, and greedy generation.

## Theorem 2 — Pointwise activation permutation island

**Assumptions.** `phi` is a pointwise function `R -> R`. `P` is a permutation matrix.

**Statement.** For every `Z`,
```
phi(Z P) = phi(Z) P.
```

**Proof sketch.** The `j`-th column of `Z P` is the `pi(j)`-th column of `Z` for some permutation `pi`. Applying `phi` column-by-column commutes with column reordering.

**Artifact support.** Stage 5.2 `outputs/nonlinear_island_experiments.json` and cross-architecture coverage `outputs/cross_architecture_summary.json` exercise this identity over GELU and ReLU islands in decoder-only, encoder-only, and encoder-decoder architectures.

## Theorem 3 — SwiGLU paired permutation

**Assumptions.** `P` is a permutation matrix. SiLU is pointwise.

**Statement.** For every `X`, `W_a`, `W_b`,
```
(X W_a P) * SiLU(X W_b P) = ((X W_a) * SiLU(X W_b)) P.
```

**Proof sketch.** By Theorem 2, `SiLU(X W_b P) = SiLU(X W_b) P`. The elementwise product `*` commutes with right multiplication by the *same* permutation:
```
(A P) * (C P) = (A * C) P.
```
Applying with `A = X W_a`, `C = SiLU(X W_b)` gives the claim. The paired permutation is required: using independent `P_a, P_b` breaks the alignment of the gate-value product.

**Artifact support.** Stage 6.4b / 6.4c modern-decoder wrappers (`outputs/modern_decoder_block_wrapper_smoke.json`, `outputs/modern_decoder_model_wrapper_smoke.json`) verify SwiGLU correctness in block- and model-level wrappers.

## Theorem 4 — RMSNorm orthogonal-mask invariance

**Assumptions.** `N` is an orthogonal matrix (`N N^T = I`). Separate the RMSNorm scale from the orthogonally-invariant core: `RMSNorm(X) = scale * RMSCore(X)` with `RMSCore(X) = X / RMS(X)` row-wise.

**Statement.** For every `X`,
```
RMSCore(X N) = RMSCore(X) N.
```

**Proof sketch.** For each row `x`, `RMS(x N) = sqrt(((x N)(x N)^T) / d) = sqrt(x x^T / d) = RMS(x)`. Therefore `(x N) / RMS(x N) = (x N) / RMS(x) = (x / RMS(x)) N`.

**Artifact support.** Stage 5.2 + 6.4 `nonlinear_island_experiments.json` + `modern_decoder_probe.json`.

## Theorem 5 — LayerNorm mean-preserving orthogonal-mask invariance

**Assumptions.** `N` is orthogonal and `N e = e`, where `e` is the all-ones column vector. Define `LNCore(X) = (X - rowMean(X) e^T) / RMS(centered)` row-wise.

**Statement.** For every `X`,
```
LNCore(X N) = LNCore(X) N.
```

**Proof sketch.** `rowMean(X N) = (X N e) / d = (X e) / d = rowMean(X)` because `N e = e`. The centered tensor satisfies `centered(X N) = centered(X) N`. Theorem 4 (RMS preservation under orthogonal `N`) applied to `centered(X)` then gives the claim.

**Artifact support.** Stage 5.2 `nonlinear_island_experiments.json` and cross-architecture LayerNorm coverage.

## Theorem 6 — Attention and KV cache invariant

**Assumptions.** `N_Q, N_K` are invertible per-head feature masks with `N_Q N_K^T = I`. The KV cache append concatenates along the token axis.

**Statement (attention identity).**
```
Q_tilde K_tilde^T = (Q N_Q)(K N_K)^T = Q (N_Q N_K^T) K^T = Q K^T.
```

**Statement (KV cache invariant).** At any decode step `t`,
```
[K_1 N_K; K_2 N_K; ...; K_t N_K]  =  [K_1; K_2; ...; K_t] N_K.
```

**Proof sketch.** Right-multiplication by `N_K` is a column-only operation; it distributes over the token-axis concatenation operator without re-indexing. The attention identity follows by substitution and the assumption `N_Q N_K^T = I`.

**Artifact support.** `outputs/kv_cache_correctness.json`, `outputs/gpt2_cache_correctness.json`, `outputs/modern_decoder_model_wrapper_smoke.json`.

## Theorem 7 — LoRA masked forward

**Assumptions.** `N_in, N_out, U` are invertible; `A in R^{d_in x r}`, `B in R^{r x d_out}`. Trusted side holds `T` and `(A, B)`.

**Statement.** With the masked definitions of Section 5.7,
```
Y_tilde + (alpha / r) T A B N_out  =  Y N_out
```
where `Y = X W + (alpha / r) X A B + b` is the plain LoRA output.

**Proof sketch.** `(N_in^{-1} A U) (U^{-1} B N_out) = N_in^{-1} A B N_out`. Multiplying by `X_tilde = (X - T) N_in` from the left,
```
X_tilde A_tilde B_tilde = (X - T) N_in N_in^{-1} A B N_out = (X - T) A B N_out.
```
Adding the trusted compensation `(alpha / r) T A B N_out` yields `(alpha / r) X A B N_out`. Combining with Theorem 1 gives the claim.

**Artifact support.** Stage 7.0 `outputs/lora_training_experiments.json` (single-step max_loss_diff), Stage 7.2 `outputs/lora_rank_padding_experiments.json` (`max_forward_err ≈ 2.2e-14`, allclose=True).

## Theorem 8 — LoRA masked backward (gradient identity)

**Assumptions.** Same as Theorem 7. The upstream gradient is `G = dL / dY` on the trusted side. The masked upstream gradient is `G_tilde = dL / dY_tilde`. Because `Y_tilde = Y N_out`,
```
G_tilde = (N_out^T)^{-1} G  =  G N_out^{-T}.
```

**Statement.** With masked backward as defined in Section 5.8,
```
dA_tilde  un-masks to  dA = (alpha / r) X^T G B^T,
dB_tilde  un-masks to  dB = (alpha / r) A^T X^T G.
```

**Proof sketch.** Substitute the boundary definitions into the masked-side update and collapse `N_out N_out^{-T}^T` via `(N_out)(N_out^T)^{-T} = I` and `U U^{-1} = I`. The trusted-side translation `T` contributes a publicly-computable correction that is absorbed before un-masking.

**Artifact support.** Stage 7.1 `outputs/lora_backward_experiments.json` (single-step `max_grad_a_err`, `max_grad_b_err`), Stage 7.2 `outputs/lora_rank_padding_experiments.json` (`max_grad_a_real_err ≈ 1.3e-15`, allclose=True), Stage 7.3 `outputs/multilayer_lora_training_experiments.json` (`max_grad_a_real_err ≈ 1.8e-15` across 14 modules in a 2-layer synthetic decoder, allclose=True).

## Theorem 9 — Rank padding (factor-product equality)

**Assumptions.** `A in R^{d_in x r}`, `B in R^{r x d_out}`. `A_pad = [A | A_dummy]`, `B_pad = [B; B_dummy]` with `A_dummy B_dummy = 0` exactly, or with a tracked trusted-side correction `Δ` such that `A_pad B_pad = A B + Δ` and `Δ` is held trusted-side.

**Statement.** For the four exact-cancellation dummy strategies (`zero_dummy`, `paired_cancellation_dummy`, `gaussian_matched_dummy`, `spectrum_matched_dummy`, `orthogonalized_cancellation_dummy`, `mixed_dummy_ensemble`):
```
A_pad B_pad = A B.
```
For `noise_injected_cancellation_dummy`, equality holds *after* the tracked trusted-side correction is folded back at recovery time.

**Proof sketch.** Block matrix multiplication: `[A | A_dummy] [B; B_dummy] = A B + A_dummy B_dummy`. Setting `A_dummy B_dummy = 0` (the cancellation property) recovers `A B`. For `noise_injected_cancellation_dummy`, `A_dummy B_dummy = Δ` with `Δ` held trusted-side and subtracted at un-masking.

**Artifact support.** Stage 7.4 `outputs/lora_stronger_dummy_experiments.json` (`max_forward_err ≈ 1.9e-14` to `3.0e-14` across all five strategies, allclose=True), with `rank_hidden_from_shape = True` recorded in Stage 7.2 `lora_rank_padding_experiments.json` (we read this as *true rank hidden from tensor shape only*; padded rank itself remains visible — see Limitations).

## What the correctness theorems do *not* claim

- They do not say "the masked transcript is indistinguishable from random". That is a *security* statement, evaluated separately in Section 7 under proxy attackers only.
- They do not say "the implementation is correct for all inputs". They say the algebraic identities hold under the stated assumptions, and that the *tested* configurations match the plain reference to float64 precision.
- They do not claim correctness for input regimes outside the tested architectures (GPT-2 model wrapper; modern decoder-only synthetic + tiny-HF wrapper; synthetic single- and multi-layer LoRA tiles).
