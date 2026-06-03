# 2. Background and Motivation

## 2.1 Decoder-only generation as the dominant LLM workload

Modern LLM serving is overwhelmingly *decoder-only* and *autoregressive*. A request consists of a prompt of length `T_p`, which is processed in a single *prefill* pass populating the KV cache for every layer; subsequent tokens are produced one at a time by *decode_step*, each step reading the existing cache, appending one new `K` and `V` row per layer, and emitting a single output token. Two consequences shape any obfuscation design:

- The cache append is performed on the **token axis** and must remain a no-op rewriter (i.e., the new row drops in at position `t` without re-indexing the existing rows).
- Each decode step contains *exactly the same* sequence of dense linear, attention, and nonlinear operations as a prefill row, so per-token work is dominated by Linear matmuls.

## 2.2 Modern architecture components and their symmetries

Beyond the classical GPT-2 block (LayerNorm + GELU + multi-head attention), the modern decoder-only stack adds three components, each with its own structural symmetry that an obfuscation scheme must respect:

- **RMSNorm.** Computes `RMSNorm(X) = scale * X / RMS(X)`. Separating the trainable scale from the orthogonally-invariant *core* `RMSCore(X) = X / RMS(X)` gives `RMSCore(X N) = RMSCore(X) N` for any orthogonal `N`. Arbitrary dense masks do *not* commute.
- **SwiGLU.** Implements `SwiGLU(X) = (XW_a) * SiLU(XW_b)`. SiLU is pointwise, so for any permutation `P`, `(AP) * SiLU(BP) = (A * SiLU(B)) P`. Permutation is the natural island mask family; dense masks break the elementwise gate-value product.
- **RoPE + GQA.** Rotary position embedding rotates pairs of feature dimensions inside each attention head; grouped-query attention shares `K` / `V` heads across multiple `Q` heads. Both are *block-structured* along the feature axis, so masks that act *within* a head block (or paired across the GQA group) preserve attention scores when the `Q`-`K` masks compose to identity.

## 2.3 LoRA personalization and its leakage profile

LoRA fine-tuning replaces a full weight update `dW` with a low-rank parameterization `dW = (alpha / r) A B`, with `A in R^{d_in x r}` and `B in R^{r x d_out}`. This is dramatically cheaper than full fine-tuning and has become the dominant form of *personal* model adaptation. Three leakage channels are now well-documented:

- **Shape leakage.** The published `A` and `B` factors reveal the rank `r` from their tensor shape, often a hyperparameter that correlates with the fine-tuning task.
- **Gradient leakage.** Per-step `dA` and `dB` reveal mini-batch identity and have been used for membership inference and partial training-data reconstruction.
- **Adapter extraction.** When `(A, B)` is published or queryable, downstream attackers can train classifiers that re-identify the fine-tuning corpus or recover prompt-style fingerprints from the merged effective weight.

Any obfuscation scheme that wants to be useful for LoRA personalization must close (or at least bound under proxy attackers) all three channels.

## 2.4 What boundary masking gives us, and what it does not

Amulet-style boundary masking (and the broader family of matrix-mask offload schemes) applies a trusted-side invertible transformation at the boundary of each Linear layer:

```
X_tilde = (X - T) N_in
W_tilde = N_in^{-1} W N_out
Y_tilde = X_tilde W_tilde + (T W) N_out + b N_out = (X W + b) N_out = Y N_out
```

The GPU only ever sees `(X_tilde, W_tilde, Y_tilde)`. With fresh `N_in`, `N_out`, and `T` per call, the transcript is randomized.

This idea is the algebraic skeleton of our work, but it is **not** sufficient by itself in three respects: it does not commute through nonlinear layers; it tends to assume a stable column structure that decoder-only generation perturbs; and it says nothing about LoRA factor masking, gradient masking, or rank hiding. Our design integrates the boundary skeleton with three new ingredients — operator-compatible nonlinear islands, generation-compatible right masking, and a private LoRA training path — and reports a proxy security evaluation rather than a formal one.

## 2.5 Trust assumptions we make and do not make

We assume the trusted-side controller (a TEE-like component in a real deployment; a local trusted runtime in this prototype) is honest, holds the user's data, samples masks, performs sampling and the loss / optimizer, and is never compromised. We assume the GPU is *honest-but-curious* in the canonical sense: it executes the kernels we send, observes everything it sees, can record full tensors, but does not insert malicious correctness errors that the trusted side could catch by re-computation. Hardware side-channels (cache, power, EM), microarchitectural transient-execution attacks, and a compromised TEE are explicitly **out of scope** in this paper. The Limitations section enumerates each such exclusion.
