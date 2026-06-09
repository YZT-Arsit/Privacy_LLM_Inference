# Method

## Notation

We write `H_l in R^{B × s × d}` for the hidden state at the entry of decoder layer `l`. Row `i` is `h_{l, i} in R^{d}`. We use `Q_l` (orthogonal, `d × d`) for the residual-stream mask of layer `l`, `M` for per-call boundary masks at linear projections, `T` for fresh per-call additive pads, `B_Q, B_K` (block-diagonal `d_h × d_h` rotations in each RoPE plane) for the per-head Q / K masks, `N_V` (orthogonal `d_h × d_h`) for the per-head V mask, `P` (permutation `d_int × d_int`) for the SwiGLU paired permutation, `N_vocab` (orthogonal `V × V` or a scalable variant) for the LM-head mask, `R` (invertible `r_pad × r_pad`) for the LoRA rank-space mask, and `ρ` for trusted-side sampling randomness. A tilde over a tensor (`x_tilde`) denotes the accelerator-visible masked form.

## 1. The decoder-only computation graph

A modern decoder block is the composition

```
  H_l                                 (residual entry)
  RMSNorm gamma_1
  Q/K/V projections                    (linear; first padded boundary)
  RoPE on Q, K                         (structure-sensitive)
  GQA causal attention                 (softmax + value gather)
  o projection                         (linear; second padded boundary)
  residual add                         -> H_l_post_attn
  RMSNorm gamma_2
  gate, up projections                 (linear; third padded boundary)
  SwiGLU = up ⊙ silu(gate)             (element-wise)
  down projection                      (linear; fourth padded boundary)
  residual add                         -> H_{l+1}
```

Surrounding the `L` decoder blocks: token embedding lookup, a final RMSNorm `gamma_final`, the LM-head linear, and a trusted-side generation processor. The protocol's job is to preserve every linear's algebraic content while letting the *operator-compatible* mask flow through every nonlinear core unchanged.

## 2. Offline preprocessing and online generation

**Offline (per session, trusted side).** The trusted controller samples the per-layer orthogonal `Q_l`, per-(layer, head) `B_K` and `N_V`, derives `B_Q[h] = B_K[h // group_size]` so that `B_Q B_K^T = I`, samples the SwiGLU paired permutation `P` per layer, samples `N_vocab`, and folds the affine `gamma` of every RMSNorm into the *following* linear weight (so the accelerator never holds a fresh RMSNorm `gamma`). The trusted controller then ships transformed weight tables to the accelerator: `W_tilde = M^{-1} W N_out` for every linear, with the per-head right masks `B_Q`, `B_K`, `N_V` folded directly into the QKV projection so the qkv-projection output is masked *directly* on the accelerator.

**Per-call (trusted side).** For every forward call (prefill or one decode step), the trusted controller samples fresh per-call boundary masks `M_qkv, M_o, M_mlp, M_down, M_lm` and fresh pads `T_qkv, T_o, T_mlp, T_down, T_lm`, then compiles the boundary tables `A = Q^{-1} M`, `C_T = T M`, `W_tilde`, `b_tilde`, `C_linear = T W N_out`. Under chunk / token granularity, `Q_l` becomes a per-row tensor and `A_qkv`, `C_linear_o`, `A_mlp`, `C_linear_down` become per-row via the einsum `"ih,shk -> sik"`. The trusted controller does the embedding lookup, applies `Q_0` (or its per-row variant) to produce the layer-entry masked state `H_hat_0`, ships `H_hat_0` and the per-call tables to the accelerator.

**Online (accelerator).** The accelerator executes the full forward over masked tensors. Per layer it computes RMSNorm core (`gamma`-folded), the QKV padded-boundary transition + linear, RoPE on the *already-masked* Q / K (which is exact because `RoPE(Q B_Q) = RoPE(Q) B_Q` for RoPE-plane block rotations), the attention block (either exact-visible scores or trusted-softmax depending on mode), the o padded-boundary transition + linear, the residual add, the second RMSNorm core, the up / gate padded-boundary transition + linear, SwiGLU, the down padded-boundary transition + linear, the second residual add, and (if next layer exists) a basis-change `R = Q_l^{-1} Q_{l+1}` to transition the residual stream into the next layer's basis. After the final layer, final RMSNorm + LM-head boundary produces `logits_tilde`.

**Online (trusted recovery).** The trusted controller receives `logits_tilde`, applies `z = z_tilde N_vocab^{-1}`, runs the configured logit processors with trusted randomness `ρ`, emits the next token. In the exact mode this is the *single* TEE-accelerator round trip per decode step. In the trusted-softmax mode there are `1 + L` round trips.

## 3. Padded linear boundary

Every linear in the decoder is wrapped as a padded boundary. Given input state `x` (which may be `H_l Q_l` for a layer-entry RMSNorm-fed projection, or `g_tilde` for the down projection), input mask `M`, fresh pad `T`, weight `W` (with possibly folded `gamma`), output mask `N_out`, define

```
  W_tilde = M^{-1} W N_out
  C_linear = T W N_out
  x_pad_tilde = (x - T) M
  y_tilde = x_pad_tilde W_tilde + bias_tilde + C_linear
```

Then `y_tilde = x W N_out + bias N_out`. The accelerator never sees `x` directly: it only ever holds `x_pad_tilde`. The pad `T` is cancelled exactly by the `C_linear` term. The next operator (RMSNorm, RoPE, softmax, SwiGLU) therefore sees a clean transformed activation, not a pad-contaminated one. The four flags `pad_enters_rmsnorm_core, pad_enters_rope_core, pad_enters_swiglu_core, pad_enters_softmax` remain `false` per call.

## 4. The low-interaction invariant

The residual stream of layer `l` is masked by an orthogonal mask `Q_l`. Per row,

```
  h_hat_{l, i} = h_{l, i} Q_{l, c(i)}
```

where `c(i)` is a *chunk assignment* function. We support three granularities: `sequence` (`c(i) = 0` for all `i`; one `Q_l` shared across all token rows), `chunk(k)` (`c(i) = floor(i / k)`; one `Q_l` per chunk of size `k`), and `token` (`c(i) = i`; one `Q_l` per row). In sequence mode `Q_l` is `[d, d]`; in chunk / token mode `Q_l` is `[S, d, d]`. Every per-row matmul is dispatched through `_matmul_uniform_or_per_row(x, m)` which picks `x @ m` for 2-D `m` and `einsum("...si,sij->...sj", x, m)` for 3-D `m`. The chunk / token modes therefore impose a per-token transition cost (one extra `d^2` matmul per row at every transition / Q-output linear).

## 5. RMSNorm granularity

RMSNorm computes `rmsnorm_core(h_i) = h_i / sqrt(mean(h_i^2) + eps)`. For an orthogonal right-action `Q_i`, the row L2 norm is preserved, hence `rmsnorm_core(h_i Q_i) = rmsnorm_core(h_i) Q_i`. The protocol *folds* the affine `gamma` into the *next* linear at compile time (`W_first_proj = diag(gamma) W`), so the accelerator-visible RMSNorm is purely the core. Under sequence granularity, all rows share `Q_l`, and the boundary tensor `H_hat = H @ Q` exactly preserves the Gram matrix `H_hat H_hat^T = H Q Q^T H^T = H H^T`. Under chunk granularity with chunk size `k`, the within-chunk Gram block is preserved (rows in the same chunk share `Q_chunk`, so the inner product `h_i Q_chunk Q_chunk^T h_j^T = h_i h_j^T` for `i, j` in the same chunk); cross-chunk Gram is disrupted by `Q_chunk Q_chunk'^T` (a random orthogonal matrix between independently sampled chunks). Under token granularity, only per-row L2 norms are preserved (`||h_i Q_i|| = ||h_i||`); off-diagonal Gram is fully disrupted by `Q_i Q_j^T` between independently sampled per-row masks. Token-wise masking does *not* hide row norms; this is mathematically required by RMSNorm correctness.

## 6. RoPE-safe pre-mask

The LLaMA / Qwen `rotate_half` RoPE acts as a 2D rotation in each `(j, j + d_h/2)` channel pair. Any 2D rotation in the same plane *commutes* with RoPE's 2D rotation: for a block-diagonal mask `B` whose `j`-th block is `[[cos phi_j, -sin phi_j], [sin phi_j, cos phi_j]]`, we have `RoPE(X B) = RoPE(X) B`. The protocol therefore *folds* the per-head `B_Q`, `B_K` into the qkv-projection weight at compile time: the accelerator-visible qkv output is already `Q_plain B_Q`, `K_plain B_K`, `V_plain N_V` per head. RoPE is then applied directly to the masked Q / K. The diagnostics flag `rope_transient_plain_qk_visible` becomes `false`, and `qkv_projection_outputs_masked_directly` becomes `true`. The plaintext per-head `Q / K / V` tensor is *never* materialised on the accelerator.

## 7. GQA / MQA score invariant

For GQA / MQA with `h_kv` KV heads and `h` Q heads (`group_size = h / h_kv`), the protocol uses one `B_K[kv_head]` per KV head and derives `B_Q[q_head] = B_K[q_head // group_size]`. With `B_Q B_K^T = I` (a 2D rotation matrix is orthogonal in each plane, hence the product is the identity in each plane, hence overall the identity), the score matrix per Q head is

```
  Q_tilde K_tilde^T = (Q B_Q) (K B_K)^T = Q B_Q B_K^T K^T = Q K^T
```

so the post-softmax probability matrix and the post-`P V` value-gather output are *exact*. The QK invariant is what makes the attention block compatible with the operator-compatible mask family. Per Q head, the attention block output equals `(probs @ V_plain) @ N_V[kv_head]`, so the downstream o-projection boundary input is `attn_out_block_masked = attn_out_plain @ N_V_block` per the o-projection's expected input mask.

## 8. Attention privacy modes

The QK invariant intentionally preserves the score matrix. The accelerator therefore observes `S` and `softmax(S)` in `exact_visible_attention` mode. The protocol provides two alternative modes that change this exposure surface:

* `trusted_softmax_attention`. The accelerator ships masked Q / K / V to the trusted side per attention block. The trusted side recovers `Q_rope, K_rope, V` by applying `B_Q^T, B_K^T, N_V^{-1}` (cheap because they are orthogonal), runs softmax in trusted memory, computes `attn_out = probs @ V`, applies `N_V` per Q head, and returns the masked attention output. The accelerator never holds `S` or `P`. The protocol pays one extra TEE round trip per attention block, so per decode step the count is `1 + L`. We mark `intermediate_tee_reentry = true` and `attention_map_hidden_from_accelerator_transcript = true`.
* `score_blinding_experimental`. The trusted side ships a row-constant additive shift `c_i` per query row to the accelerator. Softmax is invariant under row-constant shifts (`softmax(S + c_i 1) = softmax(S)`), so the final attention output is exact. However ranking, relative margins, entropy, and attention topology are *not* hidden: `(S_ij + c_i) - (S_ik + c_i) = S_ij - S_ik`. The mode is a demonstration that trivial score blinding does *not* provide attention privacy; the wrapper additionally records the (non-zero) softmax error of a *non-row-constant* random shift `R` to make the trade-off explicit.

The exactness-vs-hiding tension is unavoidable on the accelerator-side softmax: arbitrary non-row-constant additive blinding breaks exact softmax; row-constant blinding preserves softmax but is not privacy. Exact attention-map hiding therefore requires (i) trusted / secure softmax (this mode), (ii) a cryptographic primitive, (iii) an approximate / private attention proxy, or (iv) a fused confidential-kernel threat model.

## 9. SwiGLU and MLP

The MLP intermediate is `g = a ⊙ silu(b)` where `a = up_proj(x)` and `b = gate_proj(x)`. Element-wise nonlinearities do not commute with generic right-multiplications, but they *do* commute with *shared* permutations along the intermediate axis: `(a P) ⊙ silu(b P) = (a ⊙ silu(b)) P`. The protocol samples a per-layer permutation `P` and applies it as the output mask of *both* the up and gate projections. The down projection then absorbs `P^{-1}` via row permutation: `W_down_compat = P^{-1} W_down`. Because `P^{-1}` is itself a permutation (the inverse-permutation index), the absorption is an `index_select` on the rows of `W_down`. The output of the SwiGLU block matches the plaintext SwiGLU exactly per row; the diagnostics flag `swiglu_paired_permutation_max_abs_error` is at float64 machine precision in every supported mode.

## 10. KV cache, paged KV, sliding-window KV

Per session per layer per head, the trusted side fixes `N_K, N_V` for the whole session. Every new-token K / V is masked at the qkv-projection output and appended to the cache:

```
  K_tilde_new = K_new N_K        # per kv_head per token
  V_tilde_new = V_new N_V
  K_tilde_cache <- concat(K_tilde_cache, K_tilde_new)   # per session
```

The masked append invariant `K_tilde_cache = K_cache N_K`, `V_tilde_cache = V_cache N_V` holds across all decode steps of the session.

**Paged KV.** Real serving stores K / V in fixed-size physical blocks indexed by a per-session block table. The masked invariant is *unchanged*: each physical block holds `K_tilde_block = K_plain_block N_K[layer, head]`. Walking the block table reconstructs the masked full cache `K_tilde_full = K_full N_K` consistent with the standard cache. Cross-session block sharing is *off* by default; prefix sharing requires an explicit `public_prefix` flag and is reported as a leakage surface when enabled.

**Sliding-window KV.** With window size `w`, query at absolute position `t` attends only to keys in `[max(0, t - w + 1), t]`. The rolling KV buffer keeps `w` physical entries per (layer, head). Per-rolling-window invariant: `K_tilde_window = K_plain_window @ N_K`. When `w >= s_total`, sliding window reduces to standard causal attention exactly. The window size policy `w` is *public*. Timing / memory-access side channels from windowed KV are *not* evaluated.

## 11. LM head

The LM head is a single linear `z = H @ W_lm + bias_lm` followed by an orthogonal vocab mask `N_vocab` of size `V × V`. The accelerator computes `z_tilde = z N_vocab`; the trusted side recovers `z = z_tilde N_vocab^{-1}` and applies the configured logit processor. For tiny vocab (`V = 97`) this is exact at float64 machine precision. For real LLM vocab (`V in {32k, 50k, 100k}`), the dense `[V, V]` orthogonal mask costs `O(V^2)` storage and `O(B S V^2)` per-call FLOPs, which is not feasible.

The protocol therefore provides three scalable LM-head alternatives:

* **Permutation mask.** `P_vocab` is a permutation. `z_tilde[..., i] = z[..., P_vocab[i]]`. Storage `O(V)`, recovery `O(V)`. The *multiset* of logits is preserved exactly (the sorted-logits vector is observable). The mapping from token-index to logit value is hidden.
* **Block-diagonal vocab mask.** Partition vocab into chunks of size `b`; each chunk has an independent orthogonal block mask. Storage `O(V b)`, recovery `O(V b)`. Within-block masking; block membership of each vocab index is observable unless the block partition is itself permuted.
* **Top-k trusted recovery.** The trusted side recovers full logits via inverse permutation and returns only the top-k values + indices. Greedy decoding (top-1) is *exact*; if a sampling rule depends on the full distribution, full recovery must be performed before truncation. We mark this mode `exact_for_greedy_top1__not_full_softmax_unless_full_recovery`.

The dense baseline is only feasible for small `V` (we run it for `V <= 4096`); larger sizes are reported symbolically with explicit infeasibility.

## 12. LoRA inference integration

LoRA adds a low-rank correction `Y = X W + X A B` at a supported insertion site. The protocol integrates LoRA into the padded-boundary algebra via three mask matrices: input mask `M`, output mask `N_out` (the same `N_out` used by the base linear's `W_tilde`), and a rank-space invertible mask `R` of size `r_pad × r_pad`. Define

```
  W_tilde = M^{-1} W N_out
  A_tilde = M^{-1} A_pad R
  B_tilde = R^{-1} B_pad N_out
  C_base = T W N_out
  C_lora = T A_pad B_pad N_out
```

where `A_pad`, `B_pad` are the zero-padded LoRA factors of inner dimension `r_pad >= r`. The accelerator computes

```
  Y_tilde = (X - T) M W_tilde + C_base + (X - T) M A_tilde B_tilde + C_lora
          = X W N_out + X A B N_out = (X W + X A B) N_out
```

at machine precision. The trusted side then recovers `Y = Y_tilde N_out^{-1}`. Per insertion site, the appropriate `N_out` is: `B_Q` for `q_proj`, `B_K` for `k_proj`, `N_V` for `v_proj`, `Q_l` (the residual-stream mask) for `o_proj` and `down_proj`, and the SwiGLU paired permutation `P` for `up_proj` and `gate_proj`. The padded rank `r_pad` is *observable* on the accelerator side (the inner dimension of `A_tilde`, `B_tilde`); the true rank `r` is hidden by the zero pad. The adapter values `A`, `B` themselves never reach the accelerator. LoRA *training* (backward pass) is *not* supported here.

## 13. Trusted-side generation processors

After logits recovery, the trusted side runs the configured processor: greedy `argmax`, temperature scaling, top-k mask, top-p (nucleus) mask, repetition penalty (using the generated history kept in the trusted side), stop-token / EOS handling, bad-words mask, forced-token mask. Sampling uses trusted randomness `ρ`. The processor never appears in the accelerator transcript; the policy parameters (top-k `k`, top-p `p`, bad-words list, forced-token id, stop-token id) are trusted-side inputs. Under the same `ρ`, the recovered logits produce identical samples to the plaintext reference. Beam search and grammar-constrained decoding satisfy the same theorem but are *audit-only* in the current artifact (not implemented end-to-end). The output length and stop timing remain observable unless a separate padding policy is applied; we do *not* claim output-length protection.

## 14. Integrity spot-check

For active-adversary resilience the protocol provides a probabilistic spot-check prototype. Per call, the trusted side samples a random fraction of accelerator-output items at a chosen boundary (qkv projection slice, LM-head slice, or KV cache append) and re-computes them in trusted memory; any mismatch is flagged. The detection rate per call equals `checked_fraction` for a single fixed-location corruption; false-positive rate is zero under correct execution. This is *not* verifiable computation, *not* an authenticated dataflow primitive, and *not* a cryptographic integrity proof: an adaptive adversary that observes which items are spot-checked can lower the effective detection rate. The corresponding paper claim is marked `proxy_supported` in the audit. Privacy under a malicious accelerator (rather than integrity) is *not* addressed by this mode.
