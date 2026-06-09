# Theory and Proofs

We collect the algebraic statements that the protocol relies on. Every step is shown explicitly. We avoid the words "obvious", "trivial", and "clearly". All matrix products are in row-vector convention (`x in R^{1 × d}` is right-multiplied by `W in R^{d_in × d_out}`); a tilde denotes the accelerator-visible masked tensor.

## Lemma 1 (Padded Linear Boundary)

**Statement.** Let `x in R^{B × s × d_in}` be an arbitrary input, `W in R^{d_in × d_out}` a weight, `bias in R^{d_out}` a bias, `T in R^{B × s × d_in}` a fresh additive pad, `M in R^{d_in × d_in}` an invertible input mask, and `N in R^{d_out × d_out}` an invertible output mask. Define

```
  W_tilde   := M^{-1} W N
  bias_tilde:= bias N
  C_linear  := T W N
  x_tilde   := (x - T) M           # the padded-boundary input
  y_tilde   := x_tilde W_tilde + bias_tilde + C_linear
```

Then `y_tilde = (x W + bias) N`.

**Proof.**

```
  y_tilde = (x - T) M  *  M^{-1} W N  +  bias N  +  T W N
          = (x - T) (M M^{-1}) W N  +  bias N  +  T W N        (associativity)
          = (x - T) W N            +  bias N  +  T W N         (M M^{-1} = I)
          = x W N - T W N          +  bias N  +  T W N         (distributivity)
          = x W N + bias N                                     (cancellation)
          = (x W + bias) N.                                    QED
```

**Remark.** Trusted recovery is `y = y_tilde N^{-1}`. Because `M, N` are invertible by construction, the protocol carries `M^{-1}` and `N^{-1}` only on the trusted side, never on the accelerator. The accelerator never holds `x` itself, only `x_tilde`. The pad `T` is cancelled by the `C_linear` term; therefore the next operator sees `(x W + bias) N`, not a pad-contaminated activation.

## Lemma 2 (RMSNorm Granularity)

**Statement.** Let `h_i in R^{d}` be a row, `Q_i in R^{d × d}` an orthogonal matrix (`Q_i Q_i^T = I`), and `eps > 0`. Define `rmsnorm_core(x) := x / sqrt(mean(x^2) + eps)`. Then

```
  rmsnorm_core(h_i Q_i) = rmsnorm_core(h_i) Q_i.
```

**Proof.** Let `r := mean((h_i Q_i)^2) = (1/d) sum_j (h_i Q_i)_j^2`. Then

```
  sum_j (h_i Q_i)_j^2 = (h_i Q_i)(h_i Q_i)^T
                     = h_i Q_i Q_i^T h_i^T
                     = h_i I h_i^T
                     = sum_j h_{i, j}^2.        (Q_i Q_i^T = I)
```

Hence `mean((h_i Q_i)^2) = mean(h_i^2)`, and

```
  rmsnorm_core(h_i Q_i) = (h_i Q_i) / sqrt(mean(h_i^2) + eps)
                       = h_i / sqrt(mean(h_i^2) + eps) * Q_i
                       = rmsnorm_core(h_i) Q_i.    QED
```

**Leakage corollaries.**

* *Sequence mode.* With `Q_i = Q` shared across all rows, the layer-entry boundary tensor is `H_hat = H Q`. The Gram matrix satisfies `H_hat H_hat^T = H Q Q^T H^T = H H^T` exactly: the token-pair Gram is preserved.
* *Chunk(k) mode.* With `Q_i = Q_{floor(i/k)}` shared within a chunk of size `k`, the inner product between rows `i, j` is `(h_i Q_{c(i)}) (h_j Q_{c(j)})^T = h_i Q_{c(i)} Q_{c(j)}^T h_j^T`. If `c(i) = c(j)` (same chunk) this reduces to `h_i I h_j^T = h_i h_j^T`. If `c(i) ≠ c(j)` (different chunks), it equals `h_i Q_{c(i)} Q_{c(j)}^T h_j^T`, a generic orthogonal mixture of `h_j^T`. The within-chunk Gram block is preserved; the cross-chunk Gram block is disrupted.
* *Token mode.* With `Q_i` independent per row, the diagonal `(H_hat H_hat^T)_{ii} = h_i Q_i Q_i^T h_i^T = ||h_i||^2` is preserved (this is the mathematical consequence of RMSNorm correctness). The off-diagonal `(H_hat H_hat^T)_{ij} = h_i Q_i Q_j^T h_j^T` is generically not `h_i h_j^T`; full Gram off-diagonal is disrupted.

Row L2 norms are *not* hidden by any RMSNorm-compatible orthogonal mask; this is a mathematical limit, not a design choice.

## Lemma 3 (RoPE-plane Commutation)

**Statement.** Let `X in R^{B × H × s × d_h}` and let `B in R^{d_h × d_h}` be block-diagonal with `d_h / 2` blocks where the `j`-th block is `[[cos phi_j, -sin phi_j], [sin phi_j, cos phi_j]]` for an arbitrary angle `phi_j`. Let `apply_rope(X, positions, base)` denote LLaMA / Qwen rotate-half RoPE. Then

```
  apply_rope(X B, positions, base) = apply_rope(X, positions, base) @ B.
```

**Proof.** RoPE acts as a 2D rotation in each `(j, j + d_h / 2)` channel pair: with `cos_t, sin_t` the rotation cosine / sine at position `t`, the RoPE update on `(x_j, x_{j + d_h/2})` is

```
  x_j_new            =  x_j cos_t - x_{j + d_h/2} sin_t
  x_{j + d_h/2}_new  =  x_{j + d_h/2} cos_t + x_j sin_t
```

i.e. left-multiplication by `R_t = [[cos_t, -sin_t], [sin_t, cos_t]]`. The mask `B`'s `j`-th block is `B_j = [[cos phi_j, -sin phi_j], [sin phi_j, cos phi_j]]`, a 2D rotation by angle `phi_j` in the *same* `(j, j + d_h/2)` plane. Two 2D rotations in the same plane commute: `R_t B_j = B_j R_t` because both equal the 2D rotation by `t + phi_j`. Therefore, for every position `t` and every channel pair `j`,

```
  apply_rope(x_t B_j, t) = R_t (x_t B_j) = (R_t B_j) x_t = (B_j R_t) x_t = B_j apply_rope(x_t, t)
```

which, transposed back into row-vector convention, gives `apply_rope(X B) = apply_rope(X) B`. Because `B` is block-diagonal in *all* RoPE planes, the per-plane commutation lifts to the full `d_h × d_h` matrix. QED

**Note.** The same lemma fails for arbitrary dense `B`: a generic orthogonal `B` does *not* preserve the RoPE plane structure, so `R_t B ≠ B R_t` in general. This is why the protocol uses *RoPE-plane block-diagonal* rotations, not generic orthogonal masks, for the Q / K masks.

## Lemma 4 (GQA / MQA Score Invariant)

**Statement.** Let `Q in R^{B × h × s × d_h}`, `K in R^{B × h_kv × s_total × d_h}`, `group_size = h / h_kv`. Let `B_K[kv_head]` be a per-KV-head orthogonal matrix (e.g. a RoPE-plane block-diagonal rotation). Define `B_Q[q_head] := B_K[q_head // group_size]`. Then with `Q_tilde[q_head] := Q[q_head] B_Q[q_head]` and `K_tilde[kv_head] := K[kv_head] B_K[kv_head]`,

```
  Q_tilde[q_head] @ K_tilde[group(q_head)]^T = Q[q_head] @ K[group(q_head)]^T
```

per Q head, where `group(q_head) = q_head // group_size`.

**Proof.**

```
  Q_tilde[q] K_tilde[group(q)]^T
    = (Q[q] B_Q[q]) (K[group(q)] B_K[group(q)])^T
    = Q[q] B_Q[q] B_K[group(q)]^T K[group(q)]^T            (transpose of a product)
    = Q[q] (B_Q[q] B_K[group(q)]^T) K[group(q)]^T
    = Q[q] (B_K[group(q)] B_K[group(q)]^T) K[group(q)]^T   (B_Q[q] := B_K[group(q)])
    = Q[q] I K[group(q)]^T                                 (B_K orthogonal)
    = Q[q] K[group(q)]^T.                                   QED
```

**Consequence.** `softmax(Q_tilde K_tilde^T / sqrt(d_h) + causal_mask) = softmax(Q K^T / sqrt(d_h) + causal_mask)`. The post-softmax probabilities and the value-gather output `probs @ V` therefore match the plaintext attention exactly, up to the V-side mask. The score matrix `S` is preserved by construction, so it is *visible* on the accelerator in `exact_visible_attention` mode.

## Lemma 5 (Attention Exactness-vs-Hiding)

**Statement.**

(a) Under `exact_visible_attention`, with `B_Q B_K^T = I`, the score matrix `S = Q K^T / sqrt(d_h) + causal_mask` and the post-softmax `P = softmax(S)` are *observable* on the accelerator.

(b) For any row-constant shift `c_i in R`, `softmax(S + c_i 1) = softmax(S)`, but `(S_ij + c_i) - (S_ik + c_i) = S_ij - S_ik` for all `k`, so ranking, relative margins, entropy, and attention topology are *unchanged*.

(c) For non-row-constant additive blinding `R`, `softmax(S + R) ≠ softmax(S)` in general; the softmax is *not* invariant under arbitrary additive shifts.

**Proof of (a).** Lemma 4 yields `Q_tilde K_tilde^T = Q K^T`, and the causal mask is fixed structural information. Hence `S_tilde = S` and `softmax(S_tilde) = softmax(S)`. Both are computed on the accelerator under the exact-visible mode.

**Proof of (b).** Let `c_i 1 in R^{s_total}` be the constant-`c_i` vector. By softmax homogeneity,

```
  softmax(S + c_i 1)_j = exp(S_ij + c_i) / sum_k exp(S_ik + c_i)
                       = (exp(c_i) exp(S_ij)) / (exp(c_i) sum_k exp(S_ik))
                       = exp(S_ij) / sum_k exp(S_ik)
                       = softmax(S)_j.
```

For the relative-margin observation, `(S_ij + c_i) - (S_ik + c_i) = S_ij - S_ik` by direct subtraction; the difference is independent of `c_i`, and equally so is the ranking it induces.

**Proof of (c).** Take any `R` such that `R_{i, j_1} - R_{i, j_2} ≠ 0` for some `(j_1, j_2)`. Then `softmax(S + R)_{j_1} / softmax(S + R)_{j_2} = exp(S_{i, j_1} + R_{i, j_1}) / exp(S_{i, j_2} + R_{i, j_2}) = exp((S_{i, j_1} - S_{i, j_2}) + (R_{i, j_1} - R_{i, j_2})) ≠ exp(S_{i, j_1} - S_{i, j_2}) = softmax(S)_{j_1} / softmax(S)_{j_2}`. Therefore `softmax(S + R) ≠ softmax(S)`. QED

**Consequence.** Exact attention-map hiding cannot be obtained with accelerator-side softmax plus additive blinding: row-constant shifts preserve softmax but not privacy; non-row-constant shifts hide ranking but break exact softmax. Exact attention hiding therefore requires (i) a trusted / secure softmax, (ii) a cryptographic protocol, (iii) an approximate / private attention proxy, or (iv) a fused confidential-kernel threat model.

## Lemma 6 (SwiGLU Paired Permutation)

**Statement.** Let `U, G in R^{B × s × d_int}` be the up and gate activations of an MLP block, and let `P in {0, 1}^{d_int × d_int}` be a permutation matrix (`P P^T = I`, exactly one `1` per row / column). Then `SwiGLU(U P, G P) = SwiGLU(U, G) P`, where `SwiGLU(U, G) := U ⊙ silu(G)`.

**Proof.** Let `pi : {0, ..., d_int - 1} -> {0, ..., d_int - 1}` be the permutation such that `(X P)_{..., i} = X_{..., pi(i)}`. Then `(U P)_{..., i} ⊙ silu((G P)_{..., i}) = U_{..., pi(i)} * silu(G_{..., pi(i)}) = SwiGLU(U, G)_{..., pi(i)} = (SwiGLU(U, G) P)_{..., i}`. The equality holds elementwise across the intermediate axis. QED

**Down absorption.** The down projection `Y = SwiGLU(U, G) @ W_down` becomes `Y = SwiGLU(U, G) P @ (P^{-1} W_down) = SwiGLU(U P, G P) @ (P^{-1} W_down)`. The trusted side therefore stores `W_down_compat = P^{-1} W_down` (a row permutation of `W_down`, implemented as `W_down.index_select(dim=0, index=perm)` because `P^T = P^{-1}` for a permutation). The accelerator never holds `P`; it only ever holds `W_down_compat`.

## Lemma 7 (KV Cache Append)

**Statement.** Let `K_{1:t-1} in R^{B × h_kv × (t - 1) × d_h}` be the cached past keys per session per (layer, KV head), let `N_K in R^{d_h × d_h}` be the per-(session, layer, head) orthogonal mask shared across all positions of the session, and let `K_t` be a new-token key. If

```
  K_tilde_{1:t-1} = K_{1:t-1} N_K        (per kv_head)
  K_tilde_t       = K_t N_K
```

then `concat(K_tilde_{1:t-1}, K_tilde_t) = K_{1:t} N_K`. The same statement holds for V with `N_V`.

**Proof.** Right-multiplication by `N_K` distributes over the sequence-axis concatenation: `concat(K_{1:t-1} N_K, K_t N_K) = concat(K_{1:t-1}, K_t) N_K = K_{1:t} N_K`. QED

**Extension to rolling window.** Under sliding-window attention with window size `w`, the cache is trimmed to `K_{t - w + 1 : t}`. Right-multiplication commutes with row-slicing: `K_{t - w + 1 : t} N_K = (K_{1:t} N_K)_{t - w + 1 : t} = K_tilde_{t - w + 1 : t}`. The masked rolling-window invariant `K_tilde_window = K_plain_window N_K` therefore holds across any monotonically advancing window cut-off. The same statement holds for V.

**Paged KV.** Under paged storage with block size `b`, each physical block of session `s` at layer `l` holds `K_tilde_block = K_plain_block N_K[s, l, kv_head]` for the same shared mask. Walking the block table yields the full masked cache. Cross-session block sharing is *off* by default; if enabled via an explicit `public_prefix` flag, the shared prefix's `K_tilde, V_tilde` rows are observable across sessions (a declared leakage surface).

## Lemma 8 (LM-head Recovery)

**Statement.** Let `z in R^{B × s × V}` be the plaintext logits and `N_vocab in R^{V × V}` an invertible matrix. Then `z = (z N_vocab) N_vocab^{-1}`.

**Proof.** By definition of inverse, `N_vocab N_vocab^{-1} = I`. Hence `(z N_vocab) N_vocab^{-1} = z (N_vocab N_vocab^{-1}) = z I = z`. QED

**Mask family choices.**

* *Dense orthogonal* `N_vocab in R^{V × V}` with `N_vocab^{-1} = N_vocab^T`. Storage `O(V^2)`, recovery `O(B s V^2)`. Exact. Not scalable to real LLM vocab.
* *Permutation* `P_vocab`. `N_vocab^{-1} = P_vocab^T = P_vocab.argsort()`. Storage `O(V)`, recovery `O(B s V)`. Exact. *Multiset of logits is preserved*: `sort(z N_vocab) = sort(z)` (the sorted-logits vector is identical, the index-to-value mapping is hidden).
* *Block-diagonal* with block size `b`. Storage `O(V b)`, recovery `O(B s V b)`. Exact. *Block membership of each vocab index is observable* unless the block partition is itself permuted.

The dense baseline is *not feasible* for `V >= 16k`; we report it symbolically for those sizes.

## Lemma 9 (Trusted Generation Processor)

**Statement.** Let `D` be a deterministic or stochastic logit-processor that depends only on (i) the input logits `z`, (ii) the generated history `H_hist`, (iii) the processor parameters `theta` (e.g. temperature, top-k value, top-p value, repetition-penalty value, bad-words list, forced-token id, stop-token id), and (iv) trusted-side randomness `rho`. Let `z_recovered = z_plain` at machine precision. Then `D(z_recovered, H_hist, theta, rho)` is identical to `D(z_plain, H_hist, theta, rho)` (deterministic case) or equal in distribution under the same `rho` (sampling case).

**Proof.** By assumption `D` is a function of `(z, H_hist, theta, rho)` only; it does not depend on any quantity that differs between `z_plain` and `z_recovered`. By substitution `D(z_recovered, ·) = D(z_plain, ·)`. In the sampling case, with the same `rho` the inverse-CDF (or Gumbel-max) sample is determined by the multinomial parameters, which agree because the logits agree. QED

**Coverage.** Greedy `argmax(z)`, temperature `z / T`, top-k mask, top-p mask, repetition penalty `(z[history] / penalty)` (or `* penalty` for negative entries per HF convention), stop-token / EOS pinning, bad-words mask (set bad-id entries to `-inf`), forced-token mask (set non-forced entries to `-inf`), and `torch.multinomial`-based sampling under a trusted seed all satisfy the hypothesis. Beam search and grammar-constrained decoding satisfy it too; we mark them `audit_only` because they are not implemented end-to-end in the artifact.

## Lemma 10 (LoRA Masked Boundary)

**Statement.** Let `X in R^{B × s × d_in}`, `W in R^{d_in × d_out}`, `A in R^{d_in × r}`, `B in R^{r × d_out}` (LoRA factors); let `M, N` be invertible boundary masks of appropriate shape, `R in R^{r × r}` an invertible rank-space mask, `T` a fresh pad. Define

```
  W_tilde = M^{-1} W N
  A_tilde = M^{-1} A R
  B_tilde = R^{-1} B N
  C_base  = T W N
  C_lora  = T A B N
  X_pad   = (X - T) M
```

Then

```
  X_pad W_tilde + C_base + X_pad A_tilde B_tilde + C_lora = (X W + X A B) N.
```

**Proof.** By Lemma 1, `X_pad W_tilde + C_base = X W N`. For the LoRA branch,

```
  X_pad A_tilde B_tilde + C_lora
    = (X - T) M  *  M^{-1} A R  *  R^{-1} B N  +  T A B N
    = (X - T) (M M^{-1}) A (R R^{-1}) B N        +  T A B N
    = (X - T) A B N                              +  T A B N      (M M^{-1} = I, R R^{-1} = I)
    = X A B N - T A B N                          +  T A B N
    = X A B N.
```

Summing,

```
  X_pad W_tilde + C_base + X_pad A_tilde B_tilde + C_lora
    = X W N + X A B N = (X W + X A B) N.        QED
```

**Rank-padding remark.** If `A`, `B` are zero-extended to `(d_in, r_pad)` and `(r_pad, d_out)` respectively, then `A_pad B_pad = A B` exactly, but the inner dimension `r_pad` (rather than the true rank `r`) is *observable* on the accelerator. The protocol thereby hides the true rank but does not hide the padded rank.

## Theorem 11 (Full Decoder-only Generation Correctness)

**Statement.** Let `M` be a decoder-only LLM whose forward graph is composed of: token embedding lookup, `L` decoder blocks each composed of (RMSNorm gamma_1, qkv projection, RoPE, GQA causal attention, o projection, residual add, RMSNorm gamma_2, up projection, gate projection, SwiGLU, down projection, residual add), final RMSNorm gamma_final, LM-head linear, and a trusted-side generation processor. Let the protocol of Sections 3-13 be instantiated with:

* per-layer orthogonal residual mask `Q_l` (or a per-row variant under chunk / token granularity);
* per-(layer, KV head) RoPE-plane block-diagonal rotation `B_K`, with `B_Q[q_head] := B_K[q_head // group_size]`;
* per-(layer, KV head) orthogonal V mask `N_V`;
* per-layer SwiGLU permutation `P`;
* per-call fresh boundary masks `M_qkv, M_o, M_mlp, M_down, M_lm`, and pads `T_*`;
* `N_vocab` (dense orthogonal, or permutation, or block-diagonal);
* trusted-side processor `D` with randomness `rho`.

Assume all gamma-folded weights, transition tables `(A_*, C_T_*)`, and per-row Q tables are computed correctly. Then for every greedy decode step:

```
  next_token_protocol(rho)  =  next_token_plain(rho)
```

at float64 machine precision; the sequence of generated tokens matches the plaintext reference exactly under greedy decoding, and matches in distribution under sampling with the same `rho`.

**Proof.** By induction over the `L` decoder blocks and over decode steps.

*Base case.* At layer entry of the first block, `H_hat_0 = H_0 Q_0` by Lemma 2's RMSNorm-compatibility extension (no RMSNorm has yet been applied, but the residual is multiplied by `Q_0` at the trusted side). The accelerator therefore sees `H_hat_0` consistent with the layer-1 invariant.

*Inductive step.* Assume `H_hat_l = H_l Q_l` (resp. its per-row variant) at the entry of layer `l`.

1. *Attention block.*
   * RMSNorm core: Lemma 2 gives `rmsnorm_core(H_hat_l) = rmsnorm_core(H_l) Q_l`, denoted `X_hat`.
   * QKV padded linear: Lemma 1 (with `M = M_qkv`, `T = T_qkv`, `W` = `gamma_1`-folded QKV weight, `N` = block-diagonal (`B_Q, B_K, N_V`)) yields the masked per-head Q, K, V outputs. Per-head Q output = `Q_plain B_Q`, K output = `K_plain B_K`, V output = `V_plain N_V`.
   * RoPE: Lemma 3 gives `apply_rope(Q_plain B_Q) = apply_rope(Q_plain) B_Q`, identically for K. No plaintext Q / K is materialised.
   * KV cache append: Lemma 7 preserves `K_cache_tilde = K_cache N_K`, `V_cache_tilde = V_cache N_V`.
   * Attention: Lemma 4 gives `Q_tilde K_tilde^T = Q K^T`, so the softmax output `probs` is exact; the value gather is `probs @ V_rep_tilde = (probs @ V_rep) @ N_V_block`, which equals `attn_out_plain @ N_V_block` per Q head.
   * o projection: Lemma 1 (with input mask `M_o`, output mask `Q_l`) gives `attn_o_tilde = attn_o_plain Q_l`.
   * Residual add: `H_hat_post_attn = H_hat_l + attn_o_tilde = (H_l + attn_o_plain) Q_l = H_post_attn Q_l`.

2. *MLP block.*
   * RMSNorm core: Lemma 2 again.
   * up / gate padded linears: Lemma 1 with output mask `P`; outputs are `U_plain P`, `G_plain P`.
   * SwiGLU: Lemma 6 gives `g_tilde = SwiGLU(U_plain P, G_plain P) = SwiGLU(U_plain, G_plain) P = g_plain P`.
   * down padded linear: Lemma 1 with input mask absorbing `P^{-1}` and output mask `Q_l`; output = `down_plain Q_l`.
   * Residual add: `H_hat_{l + 1, before_transition} = H_hat_post_attn + down_tilde = H_{l + 1} Q_l`.

3. *Layer transition.* If `l + 1 < L`, the accelerator applies the basis change `R = Q_l^{-1} Q_{l + 1}` so that `H_hat_{l + 1} = H_{l + 1} Q_{l + 1}`. Under chunk / token granularity, `R` is per-row; the same algebra applies row-wise.

4. *Final RMSNorm and LM head.* Lemma 2 followed by Lemma 1 with output mask `N_vocab` gives `logits_tilde = z_plain N_vocab`. Lemma 8 recovers `z = z_plain`. Lemma 9 applied to the trusted-side processor yields `D(z, H_hist, theta, rho) = D(z_plain, H_hist, theta, rho)`.

The same induction applies at every decode step. The KV cache invariant (Lemma 7) propagates across steps. Greedy `argmax(z) = argmax(z_plain)` produces the same next token. QED

## Theorem 12 (Leakage Characterisation)

**Statement.** Under the protocol of Sections 3-13, the accelerator transcript exposes the following information (and no other tensor that is not derivable from a public input):

* *Sequence-norm-granularity.* Full Gram `H_hat H_hat^T = H H^T` is preserved exactly at the layer-entry boundary (Lemma 2 corollary).
* *Chunk-norm-granularity.* Within-chunk Gram preserved; cross-chunk Gram is a random orthogonal mixture (Lemma 2 corollary).
* *Token-norm-granularity.* Row L2 norms preserved (mathematical consequence of RMSNorm correctness); off-diagonal Gram disrupted.
* *RoPE-plane masks.* Per-(`j`, `j + d_h / 2`) 2D norms are preserved (the mask is a 2D rotation in each plane).
* *Exact-visible attention.* `S = Q K^T / sqrt(d_h) + causal mask` and `P = softmax(S)` are visible.
* *Trusted softmax.* `S, P` are not visible on the accelerator; `1 + L` round trips per decode step.
* *Score blinding (row-constant).* Softmax is exact; ranking and relative margins are preserved (no privacy gain against a relative-attention observer).
* *Sliding window.* Cache rows outside `[t - w + 1, t]` are evicted; window size `w` is public.
* *Vocab permutation.* Sorted-logits vector is preserved; token-index-to-logit mapping is hidden.
* *Vocab block-diagonal.* Block membership of each vocab index is observable.
* *LoRA.* Inner padded rank `r_pad` is observable; true rank `r` is hidden by zero pad; `A, B` themselves not materialised.
* *KV cache length.* Observable for both standard and paged storage.
* *Output length / stop timing.* Observable unless separately padded; *not* hidden by the current protocol.
* *Boundary fingerprints across sessions.* Independent per-session masks make the masked layer-entry fingerprint different for identical prompts across sessions.

**Proof.** Each item follows directly from the corresponding lemma (Lemmas 2, 3, 4, 5, 7, 8) and from the definitions of the per-call masks. No new tensor enters the accelerator transcript outside of (transformed weight tables, boundary tables, masked activations, masked logits). QED

Every leakage item above is enumerated as a claim in `outputs/paper_claims_audit_v2.json` with a safe wording, an unsafe wording, and a remaining blocker.
