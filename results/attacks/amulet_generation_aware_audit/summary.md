# Amulet Nonlinear — Generation-Aware Re-Audit (corrected)

**Correction to the prior `amulet_selector_visibility_audit`.** That audit
conflated two different things and over-attributed a leak to "Amulet":

- the `R==1` equality-scan recovery of the selector, and the "`M4` leaks `n` up to
  permutation" finding, are properties of a **specific implementation / a naive
  offload split**, *not* of the paper-faithful construction;
- I had **not** implemented a generation-aware right-mask-only variant.

This re-audit separates three variants and classifies every finding into exactly
one of four buckets, and does **not** claim the Amulet paper is broken. All facts
below are reproduced at float64 (`experiments/amulet_generation_aware_audit.py`).
No production path modified. Not committed.

## Classification buckets

1. **implementation_bug** — a leak from a coding choice, absent from the paper's
   construction, fixable without changing the scheme.
2. **non_generation_compatible_left_mask** — a left/row operator over the
   token/time dimension; correct in a full-sequence setting but incompatible with
   causal decoding + KV cache. A deployment incompatibility, *not* a security break.
3. **selector_exposure** — the valid index / selection matrices are directly
   GPU-visible.
4. **scheme_level_leakage** — a property of the construction itself under its own
   observable set (e.g. the pre-activation values being present in the lift). For
   the paper this is only a "break" if it violates the exact observable set of the
   paper's theorem — which this audit does **not** assert.

## Property matrix

| property | A · `amulet_migrated` (deployed) | B · paper-faithful island | C · gen-aware right-mask-only |
|---|---|---|---|
| raw pre-activation X visible | **yes** (no mask; valid col == X) | values present in lift (inherent) | values present in lift (**24/24**, inherent) |
| E1/E2 or `valid` index directly visible | **yes** (`gather(2,valid)` on device) | no (masks TEE-side) | no (`b`/perm folded into TEE `L`/`S`) |
| valid recoverable from GPU-visible R/scale | **yes** (`R==1` scan) | **no** (dense single-one `R_bar`, not a per-col scale) | **no** (no per-col scale = 1) |
| left mult touches token/time dim | no | **yes** (`M3 = P·π1ᵀ·E1·π3ᵀ`, `π1` is `[m,m]`) | no |
| KV cache / causal decoding valid | yes | **no** (token-dim ops + per-length param regen) | **yes** |
| right-mask invariant `Y_tilde = φ(X)·N_out` | **no** (no mask; `out == silu(x)`) | **yes** | **yes** |

Verified numbers: A — `valid` recovered by `R==1` scan (decoys ∈ [0.70,1.42]);
B — `R_bar` has exactly one unit, decoys dense random ∈ [-1.27,1.14] (no zero
decoys, no scannable per-column 1), `π1 ≠ I` over `m` tokens, invariant
`V == SiLU(U)·n` holds; C — invariant `Y == φ(X)·N_out` holds, `touches_token_dim
= False`, `selector_index_gpu_visible = False`, `equality_scan_R==1 = False`,
`raw_preactivation_values_in_z = 24/24`.

## Per-variant classification

**A · `amulet_migrated` (deployed, `nonlinear/amulet_backend.py`).**
- `implementation_bug`: `R[i, valid] = 1` ([:83](../../../src/pllo/nonlinear/amulet_backend.py#L83)) makes `valid` recoverable by an `R==1` scan.
- `selector_exposure`: `valid` used in an on-device `gather` ([:102](../../../src/pllo/nonlinear/amulet_backend.py#L102)).
- `scheme_level`: no activation mask at all — the raw pre-activation is the lift input.
- This is a simplified prototype, self-labeled `not_formally_claimed`; **it is not the paper's construction**, so its leaks say nothing about the paper.

**B · paper-faithful island (`ops/amulet_right_mask_islands.py`).**
- **NOT an `implementation_bug`**: `R_bar` is dense single-one (one entry = 1, the
  rest dense random incl. negatives); the lift uses the dense `r2`, so there is **no
  GPU-visible per-column scale of 1 to scan**. My earlier equality-scan / `R==1`
  claim does **not** apply here.
- `non_generation_compatible_left_mask`: `M3 = P·π1ᵀ·E1·π3ᵀ` applies a permutation
  `π1`/`π3` over the token/row dimension `m`, and the params are regenerated per
  sequence length — this breaks a stable KV cache and causal streaming. **This is
  the real reason it can't be dropped into a decoder**, and it is an incompatibility,
  not a security break.
- `scheme_level`: the pre-activation values are present in the GPU-visible lift by
  construction (the GPU must apply `φ` to true values). Under the paper's Z-only
  observable set the selector `(a,b)` is not equality-scannable; its hiding rests on
  the (unproven) secure-R assumption. **I do not have Theorem 5.2's exact observable
  set, so I do NOT declare a scheme-level break.** The prior "`M4` leaks `n` up to
  permutation" applies only if `M1..M4` are handed to the GPU (a naive offload
  split), not to the faithful split where only `Z`/`φ(Z)` cross.

**C · generation-aware right-mask-only island (implemented here).**
- Set `P = I` — no operator over the sequence dimension. A column-only lift `L`
  [d, d·k] / squeeze `S` [d·k, d] act only on the hidden dimension; the token axis
  passes through untouched.
- `right_mask_invariant` holds: `Y_tilde = φ(X)·N_out` (verified for SiLU and GELU).
- No `implementation_bug` (no `R==1` scale) and no `selector_exposure` (`b` + column
  permutation are folded into the TEE-side `L`/`S`); KV cache / causal decoding valid.
- `scheme_level` residual (unavoidable): the pre-activation **values** are present in
  the GPU-visible lift `z` (24/24 recovered as columns) — inherent to *any* nonlinear
  offload (a fixed nonlinear must be evaluated on the true values; see the
  masked-Hadamard / CP-uniqueness note). Selector-column hiding rests on the same
  secure-R-style assumption (dense decoys + secret column permutation).
- Per-op TEE cost: `z = x̃·L` and `y = φ(z)·S`, two `O(m·d²·k)` linear ops on the
  hidden dim; the GPU computes only `φ(z)`.

## The dichotomy (unchanged, and now cleanly attributed)

> **Hidden selector ⟹ the TEE computes the nonlinear (no offload). Offloaded
> nonlinear ⟹ the selector and/or the pre-activation is exposed.**

Variant C shows the *best achievable* offloaded design for a decoder: it removes the
implementation selector leak (A), removes the non-generation-compatible left mask
(B), and preserves the right-mask invariant — but the pre-activation values remain in
the offloaded lift (scheme-level, inherent), and selector-hiding remains an
assumption, not a proof.

## Allowed claims
- "The `R==1` equality-scan selector recovery is an **implementation bug in the
  deployed `amulet_migrated`**, not a property of the paper-faithful island."
- "The paper-faithful island is not droppable into a decoder because it applies a
  **left operator over the token dimension** (`π1`/`π3`/`E1`), which breaks KV cache /
  causal decoding — a **generation-incompatibility**, not a security break."
- "A **generation-aware right-mask-only** island exists (implemented, verified): no
  token-dim operator, KV-safe, and it preserves `Y_tilde = φ(X)·N_out` with no
  implementation selector leak."
- "In every offloaded variant the pre-activation **values** are present in the
  GPU-visible lift — inherent to nonlinear offload (scheme-level), distinct from an
  implementation selector leak."

## Disallowed claims
- ❌ "The Amulet paper leaks the selector by equality scan." (that is variant A's
  implementation bug; the faithful `R_bar` is dense single-one)
- ❌ "The paper-faithful Amulet island is broken." (not shown under Theorem 5.2's
  observable set; the decoder issue is the left/token mask, an incompatibility)
- ❌ "A generation-aware Amulet island hides the pre-activation." (values remain in
  the offloaded lift; hiding is combinatorial + assumption-based, not a linear mask)
- ❌ "Nonlinear can be offloaded with the pre-activation fully hidden." (contradicts
  the dichotomy + CP-uniqueness)
