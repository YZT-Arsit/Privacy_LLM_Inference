# Exact-lossless norm protection across RMSNorm — a scoped impossibility note

**Status:** partial result. The theorem below is proven for **linear** obfuscations
with an **input-independent linear de-obfuscation** that is required to commute
(be *covariant*) with the RMSNorm normalization. It is **not** a full impossibility
result: it says nothing about token-dependent scaling, cross-token mixing, or
additive/nonlinear obfuscations (which escape the hypotheses — at the price of
exactness or KV semantics, as discussed at the end).

## Setup

RMSNorm on a hidden vector `x ∈ R^d` is
```
    RMSNorm(x) = g ⊙ x / sqrt(mean(x_i^2) + eps)
```
with a diagonal gain `g` and a small `eps`. Write the pure (gain-free, `eps→0`)
normalization as
```
    N(x) = x / rms(x),   rms(x) = ||x|| / sqrt(d).
```
The diagonal gain `g` is a fixed linear map and can be folded into the output-side
de-obfuscation, so we analyse `N`; `eps>0` only makes the equalities below hold up
to an `O(eps/||x||^2)` term, i.e. the exact statement is the `eps→0` limit.

We consider a **linear stable-state obfuscation** `x ↦ x T` with invertible `T`
(this is exactly the Variant-A/B right mask, ObfuscaTune's `R`, and Variant D's
`P_in`). "Exact-lossless with input-independent de-obfuscation across RMSNorm"
means: there exists a fixed (input-independent) invertible linear map `S` such that
the obfuscated normalization can be de-obfuscated without knowing `x`:
```
    (COV)     RMSNorm(x T) = RMSNorm(x) S      for all x ≠ 0.
```
Equivalently the obfuscated RMSNorm commutes with de-obfuscation: running RMSNorm
on the masked state `xT` and then applying `S` reproduces the masked normalized
state. This is the property any residual-stream mask must have to survive an
RMSNorm layer *without* entering a TEE or approximating.

## Theorem (scoped)

> If a linear obfuscation `T` (invertible) satisfies the covariance condition
> `(COV)` for some fixed invertible `S`, then
> ```
>     T Tᵀ = c I     for a scalar c > 0,
> ```
> i.e. `T` is a scalar multiple of an orthogonal matrix. Consequently
> `||x T|| = sqrt(c) · ||x||` for every `x`, so the per-token norm profile is
> preserved up to one global constant and its **ranking is exactly preserved**.

**Interpretation.** Within this (linear, input-independent, RMSNorm-covariant)
class, you *cannot* build an exact obfuscation that changes the per-token norm
ordering. Per-token norm leakage on the stable state is therefore unavoidable for
exact-lossless linear obfuscation across RMSNorm.

## Proof

Absorbing the gain, `(COV)` reads, for all `x ≠ 0`,
```
    (xT) / ||xT|| = (xS) / ||x||.                                   (1)
```
Take Euclidean norms of both sides. The left side is a unit vector, so
```
    1 = ||xS|| / ||x||   ⇒   ||xS|| = ||x||   for all x.
```
Hence `S` is **orthogonal**: `S Sᵀ = I`. Substituting `||xS|| = ||x||` back into
(1), the right side is `xS / ||xS||`, so both sides are unit vectors and
```
    xT / ||xT|| = xS / ||xS||.
```
Two equal unit vectors that are positive multiples of `xT` and `xS` give
`xT = λ(x) · xS` with `λ(x) = ||xT|| / ||x|| > 0`, i.e.
```
    x (T − λ(x) S) = 0   for all x.                                 (2)
```
Since `S` is invertible, set `y = xS` (so `x = y Sᵀ`). Then (2) becomes
`y (Sᵀ T) = λ(x) y` for all `y`: **every** vector `y` is a left-eigenvector of the
fixed matrix `M = Sᵀ T`. A matrix all of whose vectors are eigenvectors is a
scalar multiple of the identity (take a basis `e_i`: `M e_i = λ_i e_i`; then
`M(e_i+e_j) = λ_i e_i + λ_j e_j` must be a scalar multiple of `e_i+e_j`, forcing
`λ_i = λ_j`). Hence `Sᵀ T = c I`, i.e. `T = c S` with `c > 0`. Finally
```
    T Tᵀ = c² S Sᵀ = c² I,
```
so `T` is orthogonal up to the scalar `c` and `||xT|| = c ||x||` for all `x`. ∎

## What is NOT proven (scope)

The hypotheses are load-bearing; dropping any one escapes the conclusion, and each
escape costs something we are unwilling (or unable) to pay exactly:

1. **Token-dependent scaling** `x ↦ x T(x)` (the map depends on the token). This is
   not an input-independent linear `T`, so `(COV)` with a fixed `S` need not hold —
   but then de-obfuscation is no longer input-independent, i.e. the server must
   know per-token secrets, defeating offloading.
2. **Cross-token mixing** (mixing rows of the `[m, d]` block). This breaks the
   per-token factorisation entirely and destroys KV-cache / causal order, so it is
   incompatible with autoregressive decoding.
3. **Additive / nonlinear obfuscation** (e.g. adding noise, or an `x`-dependent
   correction). This leaves the linear class and is exactly what turns an *exact*
   scheme into an *approximate* one (next section).
4. **`eps > 0` and the gain `g`.** The gain is folded into `S`; `eps>0` makes `(1)`
   hold only up to `O(eps / ||x||²)`, so the statement is the `eps→0` (or
   `eps ≪ ||x||²`) exact limit, which is the regime real RMSNorm operates in.

So this is a **covariant-linear-obfuscation** impossibility, not a universal one.

## Relation to AloePri

AloePri uses invertible non-orthogonal key matrices with `P_hat Q_hat = I` (the
exact-invertible core we borrow in Variant D) **plus** Gaussian noise **plus** an
RMSNorm expectation correction
```
    kappa = E[ ||x P_hat|| / ||x|| ]
```
— an *input-independent scalar* meant to undo the norm change of a non-orthogonal
`P_hat`. Our theorem is exactly why this must be an **expectation** (approximate)
and not an identity: for a non-orthogonal `P_hat`, `P_hat P_hatᵀ ≠ c I`, so
`||x P_hat|| / ||x||` is **not** a single constant — it varies with the direction
of `x`. A fixed `kappa` can only match it on average, incurring a per-token
approximation error. AloePri accepts non-exactness precisely because it uses a
non-norm-preserving transform; that is the theorem's contrapositive in practice.

## Relation to our experiments (measured)

- **Norm-preserving obfuscations are exact and norm-leaking.** Right-mask /
  signed-perm / token-safe Kronecker lift all satisfy `T Tᵀ = c I` (permutation or
  orthogonal), so they are exact-lossless *and* leak the per-token norm profile:
  measured Surface-A `norm_corr ≈ 1.0` (e.g. `ours_fresh_signed_perm` on Qwen-7B:
  frequency/per-token-norm rank correlation `0.99999988`). This is the theorem's
  forward direction — exactness across RMSNorm ⇒ norm preservation ⇒ norm leakage.

- **Breaking norm leakage requires leaving the class — measured.** Variant D
  (`two_sided_nonorthogonal_exact`) uses a non-orthogonal `P` (`T Tᵀ ≠ c I` for
  `lambda > 0`). Its Surface-A `norm_corr` drops as `lambda` grows
  (`1.000 → 0.928 → 0.743` at `lambda = 0.0, 0.3, 1.0`, dim 256), i.e. it *does*
  distort the per-token norm ordering. But by the theorem this same non-orthogonality
  makes it **not** RMSNorm-covariant with an input-independent de-obfuscation: the
  test `test_scope_negative_nonlinearity_not_exact` confirms `gelu(xP)Q ≠ gelu(x)`
  (max-abs error `> 1e-2`), and no fixed `S` de-obfuscates `RMSNorm(xP)`. So Variant
  D buys norm-leakage reduction *exactly* by paying RMSNorm/nonlinearity exactness —
  it is exact only for the pure **linear chain**, and end-to-end would need a TEE
  crossing at every normalization/nonlinearity (ObfuscaTune-class).

- **Direct numerical check of `(COV)`.** Fitting the best input-independent linear
  `S` to `RMSNorm(x P) = RMSNorm(x) S` over 4000 random tokens (`d = 32`) gives
  relative residual:

  | lambda | cond(P) | best-fit `(COV)` residual | covariant? |
  |---|---|---|---|
  | 0.00 (orthogonal) | 1.00 | 5.5e-16 | yes |
  | 0.10 | 1.29 | 1.6e-2 | no |
  | 0.30 | 2.25 | 4.6e-2 | no |
  | 1.00 | 39.9 | 1.0e-1 | no |

  Exactly as the theorem predicts: a fixed de-obfuscation exists **iff** `T` is
  orthogonal-up-to-scalar (`lambda = 0`). Every non-orthogonal `P` that reduces
  norm leakage provably admits no input-independent RMSNorm de-obfuscation.

## Honest conclusion

Within exact-lossless **linear, input-independent, RMSNorm-covariant** obfuscation,
per-token norm leakage is unavoidable (theorem, proven). Escaping it provably
requires either (a) giving up exactness (AloePri's noise + expectation correction),
(b) token-dependent secrets (no offloading), or (c) cross-token mixing (no
KV-cache). We do **not** claim a universal impossibility — only this scoped one,
which already explains every norm measurement in the study.
