# Amulet-style right-mask nonlinear islands

A GPU-side obfuscated nonlinear primitive for decoder-only generation. Each
island maps a right-masked activation to the right-masked activation output:

```
Input:   U_tilde = U N
Output:  V_tilde = phi(U) N        phi in {ReLU, GELU, SiLU}  (+ two-input SwiGLU)
```

This is a **nonlinear-island experiment**, not the production Qwen7B benchmark
unless and until explicitly integrated. No activation is moved into a TEE and no
intermediate TEE boundary call is added.

## Why right-mask only

Our main protocol is decoder-only and right-mask-only at the stable hidden-state
boundary `H_tilde = H N`. We do **not** use the original Amulet two-sided stable
state `H_tilde = P H Q`: the sequence/time dimension is never left-masked in the
stable state, which preserves KV-cache append semantics. Inside a nonlinear
island we may temporarily use an Amulet-style lift/shuffle/squeeze with Kronecker
products and permutation matrices, specialised to `P = I`, `Q = N`.

## Target Kronecker matrix `R_bar`

`R_bar ∈ R^{k×k}` is **dense and invertible** with exactly one secret entry equal
to 1:

```
R_bar[a, b] = 1
R_bar[i, j] != 1   for all (i, j) != (a, b)   (|R_bar[i,j] - 1| >= avoid_eps)
```

This is *not* a sparse one-hot — all other entries are random non-1 values. The
secret coordinate `(a, b)` is never published.

## Factorisation `R_bar = R1 R2 R3`

Sample invertible `R1, R2`, sample the target `R_bar`, then set
`R3 = (R1 R2)^{-1} R_bar` and verify `max|R1 R2 R3 - R_bar| <= tol` (fail loudly
otherwise). Independently sampling `R1, R2, R3` would not give the required
single-one product.

## Selection matrices

For `U ∈ R^{m×d}`, the lift `U ⊗ R_bar ∈ R^{mk×dk}`. With `E1 = I_m ⊗ e_a^T`
(`[m, mk]`) and `E2 = I_d ⊗ e_b` (`[dk, d]`):

```
E1 (U ⊗ R_bar) E2 = U           (since R_bar[a,b] = 1)
E1 phi(U ⊗ R_bar) E2 = phi(U)   (elementwise phi; selected coefficient is 1)
```

An index-based `squeeze_select` avoids materialising `E1`/`E2` for large tensors.

## Right-mask activation primitive

```
M1 = pi3 (pi1 ⊗ R1)        M2 = (N^{-1} pi2 ⊗ R3) pi4
M3 = pi1^T E1 pi3^T        M4 = pi4^T E2 pi2^T N

Z = M1 (U_tilde ⊗ R2) M2 = pi3 ((pi1 U pi2) ⊗ R_bar) pi4
S = phi(Z)
out_tilde = M3 S M4 = phi(U) N
```

Derivation: the Kronecker mixed-product rule collapses the three `R` factors into
`R_bar` and `U_tilde N^{-1} = U`; permutations commute with elementwise `phi`; the
unit copy is squeezed by `E1 · E2`; `pi*^T` and the trailing `N` restore
`phi(U) N`.

## Two-input SwiGLU

Qwen MLP uses `A = SiLU(G) ⊙ U`. Sharing `pi1..pi4` and the `R` factors (hence the
same selected `(a,b)`) across both branches:

```
Z_g = M1 (G_tilde ⊗ R2) M2     Z_u = M1 (U_tilde ⊗ R2) M2
S = SiLU(Z_g) ⊙ Z_u
A_tilde = M3 S M4 = [SiLU(G) ⊙ U] N
```

The Hadamard product of two identically-shuffled lifts equals the shuffled
Hadamard product, so the shared unit-copy is selected after the gate/up
combination.

## Qwen-style MLP integration

```
X_tilde = X N_in
W_gate_tilde = N_in^{-1} W_gate N_ff ;  W_up_tilde = N_in^{-1} W_up N_ff
G_tilde = X_tilde W_gate_tilde + b_gate N_ff = G N_ff
U_tilde = X_tilde W_up_tilde   + b_up   N_ff = U N_ff
A_tilde = AmuletSwiGLU(G_tilde, U_tilde) = A N_ff
W_down_tilde = N_ff^{-1} W_down N_out
Y_tilde = A_tilde W_down_tilde + b_down N_out = Y N_out
```

`recover(Y_tilde, N_out^{-1}) ≈ Y`.

## Interaction with the Linear-boundary additive pad

The additive pad stays boundary-local (`X_pad_tilde = (X - T) N_in`,
`C_pad = T W N_out`, `Y_tilde = X_pad_tilde W_tilde + b_tilde + C_pad = Y N_out`).
The nonlinear island receives the already-compensated `U N_ff`; the pad never
enters GELU/SiLU/SwiGLU. (No persistent residual additive pad is implemented
here.)

## Audit fields

```json
{
  "stage": "amulet_right_mask_nonlinear_island",
  "stable_state_invariant": "H_tilde = H N",
  "uses_left_sequence_mask": false,
  "intermediate_tee_boundary_calls": 0,
  "nonlinear_executed_on_gpu": true,
  "activation_supported": ["relu", "gelu", "silu", "swiglu"],
  "rbar_dense_single_one": true,
  "rbar_has_unique_one": true,
  "rbar_other_entries_not_one": true,
  "r_factor_product_verified": true,
  "selected_coordinate_public": false,
  "raw_rbar_visible_to_gpu": false,
  "raw_n_visible_to_gpu": false,
  "right_mask_output_verified": true,
  "swiglu_verified": true,
  "pad_enters_nonlinear_island": false,
  "nonlinear_island_input_form": "U N",
  "nonlinear_island_output_form": "phi(U) N"
}
```

The raw selected coordinate `(a, b)` is never included.

## Limitation (honest scope)

This construction assumes the adversary cannot reliably identify the selected
unit-copy channel inside the Kronecker-expanded shuffled space. We do **not**
claim arbitrary dense right masks commute with GELU/SiLU/SwiGLU; correctness
relies on the lift/shuffle/squeeze construction and the unique-one selected
coordinate. `formal_security_claim`: `False`.

## Artifacts

- `src/pllo/ops/amulet_right_mask_islands.py`
- `scripts/run_amulet_right_mask_nonlinear_experiments.py`
- `tests/test_amulet_right_mask_nonlinear.py`
- `outputs/amulet_right_mask_nonlinear_experiments.{json,md}`

## Validation

```
pytest tests/test_amulet_right_mask_nonlinear.py -q
python scripts/run_amulet_right_mask_nonlinear_experiments.py \
  --dtype float64 --hidden-size 16 --intermediate-size 32 \
  --batch-tokens 4 --kronecker-size 3 \
  --output-json outputs/amulet_right_mask_nonlinear_experiments.json \
  --output-md outputs/amulet_right_mask_nonlinear_experiments.md
```
