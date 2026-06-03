# 5. Design

## 5.1 Overview

The trusted-side controller exposes an LLM forward / decode / generate / training API that is, at the source level, indistinguishable from a plain reference. Under the hood, every Linear `X W + b` is intercepted, masked, dispatched to the GPU, and recovered:

```
trusted: sample fresh masks (N_in, N_out, T)
trusted: X_tilde = (X - T) N_in
trusted: W_tilde = N_in^{-1} W N_out
GPU:     Z = X_tilde W_tilde
trusted: Y_tilde = Z + (T W) N_out + b N_out          # absorbed into trusted compensation
trusted: Y = (Y_tilde) N_out^{-1}                     # only if inter-block boundary requires plain space
```

Nonlinear layers are wrapped as **islands**: the *inside* of a nonlinear layer uses a mask family that the operator natively commutes with (Section 5.4), while the outside is dense Linear masking. Islands are *sandwiched* by dense Linear masks (Section 5.6) and protected by a boundary pad (Section 5.3).

Attention and KV cache are wrapped as feature-axis masks that compose to identity on the `Q^T K` dot product (Section 5.5). The KV cache is **stored in masked space**: the per-step append concatenates `K_t N_K` onto `K_{1:t-1} N_K` because the right mask distributes over the token-axis concatenation.

LoRA is wrapped with a paired inner mask `U` (Sections 5.7–5.8), and rank padding extends `(A, B)` to `(A_pad, B_pad)` with `A_pad B_pad = A B` (Section 5.9). The loss and optimizer remain on the trusted side throughout; the adapter is never merged into the public base weight.

A control-flow schedule (Section 5.10) describes when masks refresh, when recoveries to plain space happen, and which paths run in constant-time-emulated mode.

The system overview is illustrated in **Figure 1**, the right-masked generation in **Figure 2**, and the nonlinear-island construction in **Figure 3**.

## 5.2 Generation-compatible right masking

Let `X in R^{T x d_in}` be a feature-axis activation tensor. A *right mask* `N in R^{d_in x d_in}` is applied as `X N`. The key properties:

- **Token-axis concatenation commutes with right masking.**
  ```
  [X_{1:t-1}; X_t] N = [X_{1:t-1} N; X_t N]
  ```
  Therefore the masked KV cache can be appended row-by-row without re-masking the prefix.
- **Right masking is preserved by the trailing Linear.**
  ```
  (X N) W' = X (N W')
  ```
  so `W' = N^{-1} W` recovers `X W` on the GPU side.
- **Head-block structure is preserved.** If `N` is block-diagonal aligned with the per-head feature block, multi-head attention and grouped-query attention compose without cross-head mixing.

Left masks along the token axis (i.e., `M X` with `M in R^{T x T}`) do *not* commute with the token-axis cache append in general, so we use right masking throughout.

## 5.3 Linear boundary transformation and pad compensation

Plain Linear:
```
Y = X W + b
```

Masked Linear with boundary pad:
```
X_tilde = (X - T) N_in
W_tilde = N_in^{-1} W N_out
C_T    = T W N_out                          # trusted-side compensation
Y_tilde = X_tilde W_tilde + C_T + b N_out
        = (X - T) N_in N_in^{-1} W N_out + T W N_out + b N_out
        = X W N_out + b N_out
        = Y N_out
```

The boundary pad `T` is a trusted-side per-call sample. It serves two purposes: (i) it randomizes the *centered* activation seen on the GPU, so that an attacker who linearly inverts `X_tilde` sees a translation rather than the centered hidden state; (ii) the compensation `C_T = T W N_out` is *publicly computable from public `W` and trusted `T, N_out`*, so its accounting is trivial.

## 5.4 Operator-compatible nonlinear islands

The trick for nonlinear layers is to choose a mask family that the operator commutes with, restricted to the island.

**Pointwise activation islands (GELU / ReLU / SiLU).** A permutation matrix `P` is a special orthogonal mask. For any pointwise `phi`:
```
phi(Z P) = phi(Z) P
```
because permuting columns commutes with applying an elementwise function column-wise. The island is constructed as
```
W1_tilde = N_in^{-1} W1 P,    C1 = T W1 P
Z_tilde   = X_tilde W1_tilde + C1 + b1 P = (X W1 + b1) P
phi(Z_tilde) = phi((X W1 + b1)) P
W2_tilde   = P^{-1} W2 N_out
Y_tilde    = phi(Z_tilde) W2_tilde + ... = phi(X W1 + b1) W2 N_out
```

**SwiGLU island.** `SwiGLU(X) = (X W_a) * SiLU(X W_b)`. Apply the same permutation to both branches:
```
(X W_a P) * SiLU(X W_b P) = ((X W_a) * SiLU(X W_b)) P
```
The paired permutation across the two SwiGLU branches is required; using two independent permutations breaks the elementwise gate.

**RMSNorm island.** Separate the trainable scale from the orthogonally-invariant core:
```
RMSCore(X) = X / RMS(X),   RMSNorm(X) = scale * RMSCore(X)
```
For any orthogonal `N`:
```
RMS(X N) = RMS(X)        (orthogonal preserves the L2 row norm)
RMSCore(X N) = (X N) / RMS(X) = RMSCore(X) N
```
The mask family for an RMSNorm island is therefore the orthogonal group.

**LayerNorm island.** LayerNorm subtracts the row mean before normalizing. Let `N` be an orthogonal matrix with the additional property `N e = e` where `e = (1,...,1)^T` (i.e., the all-ones vector is a fixed point). Then `mean(X N) = mean(X) (N) = mean(X)` on each row, so the centered subspace is preserved and
```
LNCore(X N) = LNCore(X) N
```
The mask family is the *mean-preserving orthogonal group*, a strict subgroup of the orthogonal group.

The mask family for each operator is summarized in **Table 2** and exercised across decoder-only / encoder-only / encoder-decoder / modern-decoder architectures.

## 5.5 Attention, KV cache, RoPE, and GQA

Per-head attention computes `softmax((Q K^T) / sqrt(d_h)) V` per head. We mask each head's `Q, K, V` with its own right mask:
```
Q_tilde = Q N_Q
K_tilde = K N_K
V_tilde = V N_V
```
For attention scores to be preserved, we require `N_Q N_K^T = I`. The simplest construction is `N_K = (N_Q^{-1})^T`. Then:
```
Q_tilde K_tilde^T = Q N_Q (K N_K)^T = Q (N_Q N_K^T) K^T = Q K^T
```
The `V` mask is recovered by the trailing output projection: the masked attention output is `softmax(...) V N_V`, and the next Linear's mask `N_in` is chosen to absorb `N_V`.

**KV cache invariant.** During decode at step `t`, the trusted side appends `K_t N_K` and `V_t N_V` to the masked cache. Because right masking commutes with token-axis concatenation, the appended cache equals `K_{1:t} N_K` and `V_{1:t} N_V`. The attention scores at step `t` reduce to `Q_t (N_Q) (K_{1:t} N_K)^T = Q_t K_{1:t}^T`. The KV cache append is therefore a no-op rewriter in masked space.

**RoPE.** Rotary positional embedding rotates each two-dimensional feature block by a position-dependent angle. The rotation acts *within* a head's feature dimension; right masking with a head-block-respecting `N_Q, N_K` commutes through the rotation when the mask is restricted to commute with the 2-d rotation in each block (paired diagonal rotations).

**GQA.** Grouped-query attention shares `K, V` heads across `g` `Q` heads. Mask the shared `K, V` heads once per group; mask the `Q` heads independently per `Q` head with the inverse paired to the corresponding shared `K`. Attention scores reduce as in the multi-head case.

## 5.6 Dense sandwich and inter-block masking

A nonlinear island uses a restricted mask family — permutation, orthogonal, or mean-preserving orthogonal — which is *weaker* than a fully-dense Linear mask. To prevent a GPU-side attacker from inverting the island directly, every island is **sandwiched** between two dense-Linear-masked boundaries: the input is `X N_in` for a generic invertible `N_in`, then transformed to `X P` (where `P` is the island mask, which is special among invertible matrices), then transformed back to `X N_out`.

The boundary pad `T` is sampled fresh per call and translates the centered activation before the island starts. Its compensation `C_T = T W N_out` is publicly computable from `T, W, N_out`, all of which are trusted-side. The combined sandwich + pad construction is the *full mitigation bundle* used throughout the evaluation; it is **not** default-on for any individual operator wrapper (see Stage 7.5 contract).

Inter-block masked boundary (`masked_boundary_experimental` in our implementation) is an opt-in mode that *avoids* recovering to plain space between blocks, instead chaining the right-mask across the residual. We report it as an experimental ablation only; it is not default-on.

## 5.7 Private LoRA forward

Plain LoRA forward at a single linear layer:
```
Y = X W + (alpha / r) X A B + b
```

Masked LoRA forward with paired inner mask `U`:
```
A_tilde = N_in^{-1} A U
B_tilde = U^{-1} B N_out
X_tilde = (X - T) N_in
W_tilde = N_in^{-1} W N_out

Y_tilde = X_tilde W_tilde + (alpha / r) X_tilde A_tilde B_tilde + C_T + b N_out
        = X W N_out + (alpha / r) (X - T) A B N_out + T W N_out + b N_out
        = Y N_out  +  (alpha / r) ((X - T) A B - X A B) N_out
        = Y N_out  -  (alpha / r) T A B N_out
```

The last term `(alpha / r) T A B N_out` is fully trusted-side computable from `T, A, B, N_out` and is absorbed into the trusted compensation in the same place as `C_T`. The recovered output equals `Y N_out` exactly.

The adapter is **never** merged into `W`. Both `W_tilde` and `(A_tilde, B_tilde)` cross the boundary as separate tensors. The GPU sees `A_tilde, B_tilde` whose shape is `[d_in, r_pad]` and `[r_pad, d_out]` (i.e., the *padded* rank, see Section 5.9).

## 5.8 Masked LoRA backward

Plain backward of `Y = X W + (alpha / r) X A B` with respect to the LoRA factors, given upstream gradient `G = dL / dY`:
```
dA = (alpha / r) X^T G B^T
dB = (alpha / r) (X A)^T G = (alpha / r) A^T X^T G
```

Masked backward operates on `Y_tilde` and the masked upstream gradient `G_tilde`. Because `Y_tilde = Y N_out`, we have
```
G_tilde = G N_out^{-T}
```
(more precisely, `dL/dY_tilde = (dL/dY) (N_out^T)^{-1} = G (N_out^T)^{-1}`, which we write as `G N_out^{-T}` when `N_out` is invertible).

We then compute the masked gradients **on the GPU** as
```
dA_tilde = (alpha / r) X_tilde^T G_tilde B_tilde^T
dB_tilde = (alpha / r) A_tilde^T X_tilde^T G_tilde
```

By substituting the boundary definitions and using `U U^{-1} = I` and `N_out N_out^{-T}^T = I` (i.e., `N_out N_out^{-1} = I`), one obtains
```
dA_tilde = N_in^{-1} dA U     (up to absorbed trusted-side constants)
dB_tilde = U^{-1} dB N_out
```
which the trusted side un-masks into `dA, dB`. The optimizer step (SGD or AdamW) is then performed entirely on the trusted side; the optimizer state never leaves the trusted runtime. We report this contract explicitly: **loss and optimizer remain trusted-side**.

## 5.9 Rank padding and stronger dummy distributions

Let `r` be the true LoRA rank and `r_pad >= r` be the padded rank. Construct
```
A_pad = [A | A_dummy]   (d_in x r_pad)
B_pad = [B; B_dummy]    (r_pad x d_out)
```
with `A_dummy B_dummy = 0` (the *cancellation* property). Then
```
A_pad B_pad = A B + A_dummy B_dummy = A B
```

The GPU-visible shape is `[d_in, r_pad]` and `[r_pad, d_out]`; the *true* rank `r` is no longer determined by the tensor shape, but `r_pad` itself remains visible. Our Limitations section calls this out explicitly: **padded_rank is visible to the GPU.**

We evaluate five stronger dummy distributions:

- **zero_dummy.** `A_dummy = 0` (or `B_dummy = 0`). The cancellation is trivial; the spectral signature is highly distinguishable.
- **paired_cancellation_dummy.** `A_dummy = R`, `B_dummy = -R^+ A B` (or analogous paired construction) chosen so the product is exactly zero.
- **gaussian_matched_dummy.** Dummies sampled to match the per-row variance of the real factors.
- **spectrum_matched_dummy.** Dummies sampled with a spectral profile that matches the real factor's leading-singular-value distribution.
- **mixed_dummy_ensemble.** A per-call mixture over the previous four strategies.
- **noise_injected_cancellation_dummy.** A tracked trusted-side noise correction that is folded back at recovery time.
- **orthogonalized_cancellation_dummy.** An orthogonal basis is applied to the cancellation pair to reduce the spectral signature.

The *correctness* of all five is empirically verified to float64 precision (see Section 6). The *spectral inference proxy* — i.e., whether an attacker can recover `r` from the masked transcript despite the pad — remains `needs_more_evaluation` for the spectral-cliff strategy and `medium` for the dummy-strategy classifier; we report this faithfully in Section 7 rather than re-classifying it as `low`.

## 5.10 Runtime and control flow

The trusted controller exposes a top-level `generate(prompt, max_new_tokens)` API. The schedule per token is:

1. **Mask sampling.** Sample fresh `N_in, N_out, N_Q, N_K, N_V, P, U` from their respective families.
2. **Prefill (first call only).** For each layer, push `X_tilde, W_tilde, A_tilde, B_tilde`, plus per-head `N_Q, N_K, N_V`; receive `Y_tilde` and the masked KV entries. Append to the masked KV cache.
3. **Decode step.** For each layer: push `X_tilde_t` (a single row) and the existing masked KV cache; receive the masked logits.
4. **Recovery and sampling.** Trusted side un-masks the final logits and samples the next token (greedy / top-k / top-p).
5. **Optional constant-time emulation.** When the cost-model constant-time mode is engaged (`proxy_equalized`), the per-step trusted compute pads to the upper-bucket latency, flattening the cost-model timing classifier to near random chance. **This is a cost-model proxy, not a real wall-time gate.**

For LoRA training the schedule is analogous, with the additional backward / optimizer step after each forward.
