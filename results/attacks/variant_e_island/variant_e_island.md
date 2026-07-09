# Variant E — two-sided island (d_model=256, d_ff=1024, lambda=0.1)

## F. Anchor attacks (public-weight anchor) on ISLAND folded weights

| target | gram_right_rel_err | gram_left_rel_err | VMA col top1 | mapping recovered |
|---|---|---|---|---|
| island_W_up_tilde (two-sided Q_in W_up P_ff) | 0.020 | 0.056 | 0.214 | False |
| island_W_down_tilde (two-sided P_ff^-1 W_down P_out) | 0.013 | 0.030 | 0.095 | False |
| baseline_signed_perm_W_up | 0.000 | 0.000 | 1.000 | True |

Boundary switch weights (`N_res^{-1} P_in`, `P_out^{-1} N_res`): pure mask products, **no public-weight anchor** — Gram/VMA/ArrowMatch do not apply.

**Reading.** The island's two-sided folded weights inherit Variant D's anchor resistance (no mapping recovered), while the same public weights under a signed-perm fold are fully recovered (VMA top1 = 1.0). requires_plaintext_weight_anchor=True throughout.

## G. Performance (decomposed — Variant-E-specific vs inherited)

- **mask-switch only** (the 2 Variant-E matmuls): 0.0251 ms.
- linear island (switch + two-sided up/down, no nonlinear): 0.1174 ms; Variant-E-specific overhead vs plain masked MLP (0.1657 ms) = **+0.0000 ms**.
- **Amulet-GELU boundary (INHERITED from variant B): 277.194 ms** — dominant term (Kronecker lift at d_ff), NOT new to Variant E.
- full island 277.311 ms.
- cond(P_in/P_ff/P_out) = 1.3 / 1.3 / 1.3.
- island max_abs_error: fp64 1.3e-10, fp32 3.9e-02.

