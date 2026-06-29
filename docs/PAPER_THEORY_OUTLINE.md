# Paper Theory Outline

This document consolidates the theoretical claims that the current
artifact supports, in the order they appear in the paper draft. Each
section states the claim, the operating assumptions, the proof
sketch, and a one-line link to the artifact that exercises it.

Throughout, `N`, `N_x`, `N_y`, `N_res`, `M`, `P` denote orthogonal
matrices (typically right-action masks); `Pi` denotes a (column or
head) permutation matrix; pad block sizes are denoted `p`. Honest
hedging is preserved:

- No formal, cryptographic, or semantic security is claimed.
- No real TEE or GPU wall-time is measured.
- Every empirical statement is either *algebraically proven*,
  *experimentally validated*, *proxy evaluated*, *cost-proxy only*,
  or *unsupported*.

The grading vocabulary is defined in `docs/PAPER_EVALUATION_MAP.md`
and applied verbatim in `paper_results/markdown/security_claims_table.md`.


## 1. Linear mask/pad correctness

**T1 (Right-action mask correctness).**
For an orthogonal `N in R^{d × d}` and any `X in R^{n × d}`, a linear
layer with weight `W` and bias `b` satisfies
```
  (X N) (N^T W) + b = X W + b.
```
**T2 (Boundary pad cancellation).**
Padding the input with `p` columns and the weight with `p` zero rows
preserves the linear output exactly: `[X | Z] [W; 0_{p × d_out}] = X W`
for any auxiliary `Z`. When pad columns are sampled fresh per call
they hide channel-aligned shape information from the accelerator
without changing the result.

**Assumptions.** `N` is held only on the trusted side; `N^T W` is
materialised on the trusted side and uploaded; the boundary pad sizes
are public.

**Proof sketch.** Both T1 and T2 are direct linear algebra; T1
uses `N N^T = I`, T2 uses block-multiplication with the zero block.

**Status.** *algebraically_proven*; cross-checked by the linear-with-
pad-compensation row of `ours_runtime_api_validation.json`.


## 2. Attention QK score invariance

**T3 (QKV right-mask invariance of scaled-dot-product attention).**
For fresh orthogonal head-space masks `N_h`, the masked queries and
keys `Q_tilde = Q N_h`, `K_tilde = K N_h` give
```
  Q_tilde K_tilde^T / sqrt(d_h)
    = (Q N_h)(K N_h)^T / sqrt(d_h)
    = Q (N_h N_h^T) K^T / sqrt(d_h)
    = Q K^T / sqrt(d_h),
```
so the softmax row-distribution `A = softmax(Q K^T / sqrt(d_h))` is
preserved exactly. Right-masking `V` with the *output* mask `N_out`
gives `A (V N_out) = (A V) N_out`, recovered on the trusted side by
`(A V N_out) N_out^T`.

**Assumptions.** Per-head orthogonal masks are sampled fresh on the
trusted side per call; the attention kernel runs the standard
softmax-times-V; numerical softmax stability is not affected because
softmax is invariant to additive constants only — here `Q K^T` is
preserved exactly, not just up to an additive constant.

**Proof sketch.** Two `N N^T = I` cancellations and bilinearity.

**Status.** *algebraically_proven*; per-head and per-block numerical
allclose checks recorded in `attention_experiments.{json,md}` and
`cross_attention_experiments.{json,md}` /
`encoder_attention_experiments.{json,md}`.


## 3. KV-cache append invariance

**T4 (KV-append commutativity with right-action masking).**
If at step `t` the cached `K_cache_tilde = K_cache N_h^{(t)}` and the
newly produced row `k_t_tilde = k_t N_h^{(t)}` share the *same*
per-head right-action mask `N_h^{(t)}` for the slice that participates
in the score, then
```
  concat([K_cache_tilde, k_t_tilde]) =
    concat([K_cache, k_t]) N_h^{(t)},
```
and the score `Q_t_tilde · concat([K_cache_tilde, k_t_tilde])^T`
equals the plaintext score by T3. The same holds for `V_cache_tilde`
under the *output*-side mask `N_out^{(t)}`.

**Assumption (essential).** Slices that take part in a single score
share the same right-action mask. If per-step masks are independently
freshly sampled across the cache, slice scores break. The Stage 5.3e
mitigation bundle controls *freshness across calls*, not freshness
across cached slices of the same call.

**Proof sketch.** Concatenation distributes over right multiplication.

**Status.** *algebraically_proven*; exercised by the
prefill / decode_step / greedy-generation rows of
`ours_runtime_api_validation.json` and by the model-level wrapper
output in `modern_decoder_model_wrapper_smoke.{json,md}`.


## 4. RMSNorm, RoPE, and GQA conditions

**T5 (RMSNorm commutes with orthogonal residual masks).**
For an orthogonal `N_res in R^{d × d}` and a residual hidden state
`H_tilde = H N_res`, `RMSNorm(H_tilde) = RMSNorm(H) N_res` *if and
only if* the RMS normaliser is computed in the same masked
coordinate system (the row-wise L2 norm is invariant under right
multiplication by orthogonal `N_res`). The `gamma` scale folds into
the adjacent linear, so the RMSNorm *core* runs in the orthogonal
`N_res` residual mask space.

**T6 (RoPE post-application masking).**
RoPE is a per-position block-diagonal rotation in pair-of-channels
coordinates. A general dense `N_h` does *not* commute with the RoPE
rotation. The artifact handles this by applying the per-head mask
*after* RoPE: `q_t_post_rope_tilde = q_t_post_rope N_h^{(t)}`,
preserving T3.

**T7 (Grouped-query attention).**
For GQA with `g` query heads sharing a `K, V` head, the per-K/V
right-action mask `N_kv` must be tied across the `g` query heads
that read it. Otherwise T3 fails. Artifact assumption: per-K/V mask
broadcast across the corresponding query heads.

**Proof sketch.** T5: `||H N||_2 = ||H||_2`. T6: explicit assertion
recorded as a structural condition. T7: same as T3 with the per-K/V
binding made explicit.

**Status.** *algebraically_proven* under stated assumptions; the
post-RoPE masking and GQA tying conditions are exercised in
`modern_decoder_model_wrapper_smoke.{json,md}` and in the modern-
decoder probes recorded by `attention_experiments.{json,md}`.


## 5. SwiGLU compatible-island correctness

**T8 (Compatible SwiGLU island).**
The SwiGLU island uses `silu(G) * U` with three fresh masks
sampled per call:
```
  G_tilde = G N_in,
  U_tilde = U Pi,
  silu_in_tilde = G_tilde Pi^{-1} N_in^T  // computed on trusted side
                                          //   for the silu domain
  A_tilde = (silu(G) Pi) * (U Pi) * N_out
          = (silu(G) * U) Pi N_out
        recovered to plain via (A_tilde N_out^T) Pi^{-1}.
```
The full Stage 5.3e bundle adds dense sandwich + boundary pad.

**Property.** `nonlinear_mode = "compatible_islands"` correctness
holds exactly per call, because `silu` is applied to a single masked
input and the multiplication distributes over the permutation.

**Known leakage (T8a).** The compatible island preserves
permutation-invariant activation statistics (row norms, sorted
multisets, quantiles) — see §6.

**Status.** *algebraically_proven* per call (correctness);
permutation-invariant leakage (T8a) is the subject of T9.


## 6. Permutation-invariant leakage theorem

**T9 (Permutation-invariant statistics theorem).**
For any column permutation `Pi`, the per-row L1/L2/Linf norms, the
per-row max/min, the per-row sorted multiset, and the per-row
quantiles of `X Pi` equal those of `X`. Therefore neither the
`fresh_perm_only` nor the `fresh_perm_plus_sandwich_plus_pad` bundle
removes single-shot permutation-invariant statistics inside the
activation core (the dense sandwich and boundary pad change
*freshness / temporal / boundary* posture, not value-level multiset
visibility).

**Corollary.** Any attacker that operates on permutation-invariant
features alone sees the same statistics as on plaintext. Per-row
linkability is the only freshness-controlled signal: with a fixed
permutation a fingerprint-style attacker succeeds; with a fresh
per-call permutation the same attacker collapses to chance.

**Proof sketch.** Direct from the definition of `Pi` as a column
permutation; norms / sorted / quantiles are coordinate-symmetric.

**Status.** *algebraically_proven* (the theorem itself);
*experimentally_validated* by `permutation_invariant_leakage.{json,csv,md}`
under both Stage 5.3e bundles, with conservative risk labels
(`statistical_leakage_detected_high` requires `l2_corr > 0.999`
AND `sorted_mse_mean < 1e-6` AND `sorted_l2_rel_mean < 1e-3`).


## 7. Lookup cost-proxy motivation

**T10 (Lookup table size scales as 2^(2b)).**
A two-input lookup proxy of SwiGLU with `b`-bit quantised inputs has
`2^(2b)` entries; preprocessing bandwidth is
`2^(2b) * entry_bytes * num_tables`; online lookup bandwidth is
`batch_size * seq_len * intermediate_size * entry_bytes`. Per-channel
tables would scale `2^(2b) * intermediate_size * entry_bytes` —
recorded as `impractical_proxy_only`.

**Interpretation.** Lookup-style nonlinear protection *may* improve
value hiding (it does not preserve the permutation-invariant
statistics theorem of T9 in the same way), but this artifact does
not implement any secure lookup primitive. The Stage 5.8 results
serve as a *cost baseline and future-work motivation*, not as a
security claim.

**Status.** *cost_proxy_only*. Headline numbers and microbenchmark
in `lookup_nonlinear_cost_proxy.{json,csv,md}`; the artifact records
`formal_security_claim = False`,
`cryptographic_lookup_implemented = False`,
`recommended_use = "cost-baseline-and-future-work-motivation"`.


## 8. Masked-gradient LoRA correctness

**T11 (Rank-space-mixed masked LoRA forward).**
For orthogonal `N_x in R^{d_in × d_in}`, `N_y in R^{d_out × d_out}`,
`M in R^{r_pad × r_pad}`, with masked factors
`A_tilde = N_x^T A M`, `B_tilde = M^T B N_y` and masked input
`X_tilde = X N_x`,
```
  X_tilde A_tilde B_tilde
    = X N_x (N_x^T A M)(M^T B N_y)
    = X A B N_y,
```
so `Y_tilde N_y^T = Y_plain` exactly.

**T12 (Gradient relations under masked forward).**
For MSE loss with `target_tilde = target N_y`,
```
  grad_A_tilde = N_x^T grad_A M,
  grad_B_tilde = M^T   grad_B N_y.
```
Right-multiplication by orthogonal `N_y` preserves the L2 loss exactly,
so loss equality also holds (`MSE(Y_tilde, target_tilde) = MSE(Y, target)`).

**T13 (Masked SGD / momentum SGD equivalence).**
Masked SGD `A_tilde <- A_tilde - lr * grad_A_tilde` and
heavy-ball momentum
`V_tilde <- mu V_tilde + grad_tilde; A_tilde <- A_tilde - lr V_tilde`
are linear in the gradient. Right multiplication by orthogonal masks
distributes over the linear combination, so both updates recover
exactly to plaintext SGD / momentum SGD after trusted-side recovery.

**T14 (Cancellation-padded rank).**
With `A_pad = [A_real, R, -R]` and `B_pad = vstack(B_real, S, S)`,
`A_pad B_pad = A_real B_real` exactly at initialisation; the visible
inner dimension is `r_pad` while the true rank `r_true` is hidden
from shape inspection. Dummy contribution norm is identically zero
at init.

**Proof sketch.** T11–T13 are sequences of orthogonal-mask
cancellations and linearity of the update; T14 is a direct algebraic
identity for the paired `[R, -R]` block.

**Status.** *algebraically_proven*; lockstep float64 verification at
machine precision in
`masked_gradient_lora_training.{json,csv,md}`
(`forward_max_abs_err <= 4.66e-15`,
 `loss_abs_err <= 1.11e-16`,
 grad relations `<= 2.52e-15`,
 SGD/momentum recovery `<= 1.78e-15`).


## 9. AdamW dense-mask limitation

**T15 (Dense masked AdamW exactness is unsupported).**
Coordinate-wise second moments `v <- beta_2 v + (1 - beta_2) g^2` are
*not* invariant under a dense orthogonal mixer: for a dense
orthogonal `Q`, `(g Q)_{i, j}^2 != (g^2 Q)_{i, j}` in general. So a
direct dense-mask masked AdamW step does not recover the plaintext
AdamW step.

**Consequence.** The artifact's
`masked_adamw_step_unsupported(...)` raises
`DenseMaskedAdamWUnsupported` rather than silently approximating. A
future stage could obtain AdamW exactness via (i) a trusted-assisted
update (recover, AdamW on plain, re-mask), (ii) signed-permutation
masks (the only orthogonal class that commutes with coordinate-wise
squaring), or (iii) a specialised masked optimiser.

**Proof sketch.** Counterexample: any dense `Q` with two non-zero
entries in a row makes `(g Q)^2 != g^2 Q` for a generic `g`.

**Status.** *algebraically_proven* (the limitation); the gate is
*experimentally_validated* by the AdamW-unsupported row of
`masked_gradient_lora_training.{json,csv,md}` and the unit test that
asserts `masked_adamw_step_unsupported(...)` raises.


## 10. End-to-end paper-safe wording

The paper-safe wording the artifact is consistent with:

> *Masked-gradient LoRA provides algebraic equivalence for SGD /
> Momentum under orthogonal masks and proxy-evaluated leakage
> mitigation; it does not provide formal, cryptographic, or
> semantic security.*

The same hedging applies to T1–T8 (correctness only), T9 (proven
theorem + experimentally validated; not a security guarantee), T10
(cost proxy only), and T15 (limitation explicitly raised). Every
related claim is graded in the next document.

`formal_security_claim`: `False` everywhere it is recorded.

## 11. Linear-boundary additive padding (production Qwen folded path)

**Claim.** Each Linear in the production folded path admits an additive input
pad whose compensation is precomputed and folded, so the GPU's matmul operand is
`X_tilde = (X − T) N_in` while the output stays in the existing compatible masked
basis `Y N_out`.

**Assumptions.** `N_in`, `N_out` orthogonal (signed-permutation / per-head / SwiGLU
permutation families); pad `T` is boundary-local (sampled in the masked basis as
`xpad = T N_in`).

**Proof sketch.** With `W_tilde = N_in^{-1} W N_out`, `b_tilde = b N_out`, and
`C_pad = T W N_out = xpad @ W_tilde`,
`(X−T)N_in · W_tilde + b_tilde + C_pad = (XW+b) N_out`. Recovery is unchanged, so
recovered logits and greedy tokens equal the mask-only path (and plaintext within
fp tolerance).

**Scope / limitation.** The pad obfuscates the Linear matmul operand view only; it
is compensated before any nonlinear core and is never persisted in the residual
stream. We do **not** claim that arbitrary dense affine masks commute with
RMSNorm / softmax / SwiGLU — those keep compatible permutation / Q·Kᵀ-invariant
masks. `formal_security_claim`: `False`.

**Artifact.** `src/pllo/deployment/linear_boundary_pad.py`,
`scripts/build_qwen7b_folded_package.py --linear-boundary-pad`,
`tests/test_qwen_linear_boundary_pad.py`,
[`linear_boundary_additive_padding.md`](linear_boundary_additive_padding.md).

## 12. Amulet-style right-mask nonlinear islands

**Claim.** For decoder-only generation with a right-mask stable invariant
`H_tilde = H N`, a nonlinear activation `phi` (ReLU/GELU/SiLU) and the two-input
SwiGLU operator can be evaluated on the untrusted GPU as an Amulet-style
lift/shuffle/squeeze island with `P = I`, `Q = N`, satisfying the external
contract `U N -> phi(U) N` without entering a TEE and without any intermediate
TEE boundary call.

**Construction.** Choose a dense target `R_bar ∈ R^{k×k}` with exactly one secret
entry `R_bar[a,b] = 1` and every other entry not equal to 1; factor
`R_bar = R1 R2 R3`. With permutation matrices `pi1..pi4` and selection matrices
`E1 = I_m ⊗ e_a^T`, `E2 = I_d ⊗ e_b`:

```
M1 = pi3 (pi1 ⊗ R1)        M2 = (N^{-1} pi2 ⊗ R3) pi4
M3 = pi1^T E1 pi3^T        M4 = pi4^T E2 pi2^T N
Z  = M1 (U_tilde ⊗ R2) M2 = pi3 ((pi1 U pi2) ⊗ R_bar) pi4
out_tilde = M3 phi(Z) M4
```

**Proof sketch.** Kronecker mixed-product gives
`(pi1 ⊗ R1)(UN ⊗ R2)(N^{-1}pi2 ⊗ R3) = (pi1 U pi2) ⊗ R_bar`. Permutations commute
with elementwise `phi`, so `phi(Z) = pi3 phi((pi1 U pi2) ⊗ R_bar) pi4`. Because
`R_bar[a,b] = 1`, the block at sub-position `(a,b)` is exactly `phi((pi1 U pi2))`,
which `E1 · E2` squeezes out; `pi1^T … pi2^T` and the trailing `N` restore
`phi(U) N`. SwiGLU shares the schedule across the gate/up branches so the same
unit-copy is selected after `SiLU(gate) ⊙ up`: `A_tilde = (SiLU(G) ⊙ U) N`.

**Stable-state / boundary properties.** `pi1` (token side) and `pi3` are
island-internal transient permutations undone by `M3`, so the stable decoder state
never carries a left/sequence mask (`uses_left_sequence_mask = false`); no
left mask is applied over the sequence dimension. `intermediate_tee_boundary_calls = 0`.
Any Linear-boundary additive pad is compensated *before* the island, which always
receives the clean masked activation `U N_ff` (`pad_enters_nonlinear_island = false`).

**Scope / limitation.** This construction assumes the adversary cannot reliably
identify the selected unit-copy channel inside the shuffled Kronecker-expanded
space. We do **not** claim arbitrary dense right masks commute with
GELU/SiLU/SwiGLU; correctness relies on the lift/shuffle/squeeze construction and
the unique-one secret coordinate `(a,b)`, which is never published.
`formal_security_claim`: `False`. This is a nonlinear-island experiment, not the
production Qwen7B path unless explicitly integrated.

**Artifact.** `src/pllo/ops/amulet_right_mask_islands.py`,
`scripts/run_amulet_right_mask_nonlinear_experiments.py`,
`tests/test_amulet_right_mask_nonlinear.py`,
`outputs/amulet_right_mask_nonlinear_experiments.{json,md}`.
