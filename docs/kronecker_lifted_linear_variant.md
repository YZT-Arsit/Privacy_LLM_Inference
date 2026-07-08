# Kronecker-Lifted-Linear Variant (variant C)

**Status:** experiment-only. `formal_security_claim = False`,
`production_qwen7b_integration = False`, attention / KV cache **not** lifted.

This document specifies variant **C** (`kronecker_lifted_linear`) and how it
relates to the two existing variants:

| variant | linear layer | nonlinear | stable state | public surface |
|---|---|---|---|---|
| A `right_pad_pure_right` | right-mask + pad | pure-right compatible | `H N` | `H N`, `N⁻¹WM` |
| B `right_pad_amulet_gelu` | right-mask + pad | right-mask Amulet Kronecker | `H N` | `H N`, `N⁻¹WM` |
| C `kronecker_lifted_linear` | Kronecker-lifted weight | squeeze / transient-lifted | `Π(H⊗R)Ω` | `Π(H⊗R)Ω`, `Ω⁻¹(W⊗S)Ω` |

## 1. Motivation

The Gram / norm / weight-alignment / token-recovery attacks in this repo act on
the **plain** right-masked stable state `H̃ = H N_res` and the **plain** folded
public weights `W̃ = N_in⁻¹ W N_out`. Amulet-GELU (variant B) only lifts a
*transient* nonlinear tensor `Z`; its `R_bar` never touches the squeezed stable
state, so **the stable-state Gram/norm/weight attacks are unaffected by
`R_bar`** (this is measured: variants A and B produce identical numbers on
surfaces A and B). To actually test whether a Kronecker expansion mitigates the
stable-state attacks, the *public surface itself* — stable state and/or folded
linear weights — must be lifted. That is variant C.

## 2. Invariant

Plain right-mask stable state:

```
H̃ = H N
```

Lifted stable state (`H ∈ ℝ^{m×d}`, `R ∈ ℝ^{k×k}` invertible):

```
Ĥ = Π (H ⊗ R) Ω        Ĥ ∈ ℝ^{mk × dk}
```

`Π` is the row-side operator, `Ω` the feature-side operator. **Stage 1 keeps `Π`
token-order-preserving** (`Π = I_m ⊗ P_k`, a per-token sub-row permutation that
never mixes token order) so causal order and KV-cache append survive. `Ω` is a
feature-side permutation. Both are orthogonal, so `Π⁻¹ = Πᵀ`, `Ω⁻¹ = Ωᵀ`.

## 3. Linear correctness theorem

For a plain linear `Y = X W + 1 bᵀ`, define the lifted weight and bias:

```
S = R_in⁻¹ R_out                       (so R_in S = R_out)
Ŵ = Ω_in⁻¹ (W ⊗ S) Ω_out
b̂ = Π_out ((1 bᵀ) ⊗ R_out) Ω_out
```

Then, with `Π_out = Π_in` (shared row side) and `X̂ = Π(X⊗R_in)Ω_in`, the
Kronecker mixed-product identity `(A⊗R)(B⊗S) = (AB)⊗(RS)` gives:

```
X̂ Ŵ + b̂
  = Π (X⊗R_in) Ω_in Ω_in⁻¹ (W⊗S) Ω_out + b̂
  = Π ((XW)⊗(R_in S)) Ω_out + b̂
  = Π ((XW + 1 bᵀ)⊗R_out) Ω_out
  = Ŷ.    ∎
```

Verified exact: fp64 `max_abs ≤ 1e-10` (measured ~1e-13 toy, ~1e-17 real GPT-2).

## 4. Squeeze (un-lift)

Recover `Y` from `Ŷ = Π(Y⊗R)Ω`: undo `Π`/`Ω` (transpose), then select sub-row
`a`, sub-col `b` of every `k×k` block and divide by `R[a,b]`. When `R` is a dense
single-one matrix (`R[a,b] = 1`, all others ≠ 1) the squeeze is a **pure
index-selection** (no divide) — the same construction proven exact for
Amulet-GELU. A generic invertible `R` also works (divide by `R[a,b] ≠ 1`); the
negative test confirms a *pure* selection on a generic `R` returns `Y·R[a,b]`,
not `Y`.

## 5. Relation to Amulet-GELU

* **Amulet-GELU (B)** lifts only the *transient* nonlinear tensor `Z = π₃((πUπ)⊗R_bar)π₄`;
  the squeeze returns to the plain `H N` stable state.
* **Lifted-linear (C)** lifts the *linear computation and (optionally) the stable
  residual stream*. The two stages:
  * **C1 `lifted_mlp_squeeze`** — up/gate/down folded weights are lifted, but the
    elementwise nonlinearity is computed at a plain squeeze point. Protects the
    *folded MLP weights only*; the stable residual state between blocks is plain
    `H N` → **does not defend a stable-residual attack.**
  * **C2 `lifted_mlp_residual`** — the residual stream stays lifted across the MLP
    block; only the elementwise nonlinearity transiently squeezes to the plain
    activation internally (unavoidable for an exact elementwise nonlinearity) and
    re-lifts. Numerically identical to C1; the difference is the *contract* — a
    stable-residual attacker sees `Π(H⊗R)Ω`, not `H N`. **Attention/KV are still
    not lifted**, so this is not a full-transformer lift.

## 6. Security boundary (the honest caveat)

A Kronecker lift does **not** information-theoretically remove norm/Gram:

```
‖X ⊗ R‖_F        = ‖X‖_F · ‖R‖_F
(X⊗R)(X⊗R)ᵀ      = (X Xᵀ) ⊗ (R Rᵀ)
```

Consequences, **measured** on surface A (stable state):

* A *token-order-preserving* lift (`Π = I_m ⊗ P_k`, required for KV) leaves the
  per-token norm exactly `‖H_i‖·‖R‖_F` and the token-block Frobenius Gram exactly
  `(H Hᵀ)·‖R‖_F²`. So `norm_corr = gram_corr = 1.0` — **the same leakage as a
  plain orthogonal right mask.** The lift alone buys nothing here.
* Only dense **cross-token** mixing reduces it (measured: `norm_corr` 1.0 → −0.36,
  `gram_corr` → 0.67 at k=3), and that mixing **breaks KV-cache / causal order**,
  so it is not usable in a decoder without a redesign (KV-UNSAFE, reference only).

Security therefore relies on hidden **block structure** + permutation/mixing +
the attacker's inability to align the lifted channels. **If the block structure
is recovered, the Gram/norm signal reappears.** No unconditional security is
claimed.

## 7. Limitations

* Increased compute/memory: lifted linear FLOPs and materialized-Kronecker peak
  memory scale ~`k²`; `Ŵ` is `dk × pk`.
* Attention / KV cache are **not** lifted in this stage (MLP island only).
* C1 squeezes to `H N` around the nonlinearity → does not protect the stable
  residual stream; only C2 keeps the residual lifted, and only across the MLP.
* `formal_security_claim = False`; experiment-only; not on the production Qwen7B
  path or in the attestation pipeline.

## 8. Reproduce

```
python scripts/run_variant_correctness.py --variant kronecker_lifted_linear --model tiny --lift-k 2
HF_HUB_OFFLINE=1 python scripts/run_variant_correctness.py --variant kronecker_lifted_linear --model tiny-gpt2 --layers 2 --lift-k 3
python scripts/run_variant_attack_eval.py --variant kronecker_lifted_linear --lift-k 3
python scripts/run_variant_comparison.py --model tiny --lift-k 2
```

Outputs: `outputs/variant_comparison/{right_pad_pure_right,right_pad_amulet_gelu,kronecker_lifted_linear,summary}/`.
Tests: `tests/test_kronecker_lifted_linear.py` (59 cases).
Code: `src/pllo/ops/kronecker_lifted_linear.py`,
`src/pllo/experiments/lifted_variants.py`,
`src/pllo/experiments/lifted_attack_surface.py`.
