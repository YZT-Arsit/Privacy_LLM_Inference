# Permutation Nonlinear-Island — Forward/Backward + Security-Boundary Audit

Experiment / audit only (float64). No production path modified; not committed.

## Stabilizer tests — does `phi(Z Q) = phi(Z) Q`?

| activation | permutation | signed_perm | pos_diagonal | dense_orth | dense_gl |
|---|---|---|---|---|---|
| gelu | True | False | False | False | False |
| silu | True | False | False | False | False |
| relu | True | False | True | False | False |

(pass = relation holds to fp64; each cell also has max_abs/rel_error in `stabilizer_tests.csv`.)

## MLP forward/backward (standard autograd nonlinear backward)

| activation | fwd Z | fwd U | fwd Y | bwd GU | bwd GZ | bwd GH |
|---|---|---|---|---|---|---|
| gelu | 7.69e-16 | 7.38e-16 | 5.73e-16 | 2.83e-15 | 2.86e-15 | 1.60e-15 |
| silu | 7.69e-16 | 7.58e-16 | 8.45e-16 | 2.83e-15 | 2.60e-15 | 1.71e-15 |
| relu | 7.69e-16 | 7.38e-16 | 7.76e-16 | 2.83e-15 | 2.68e-15 | 1.93e-15 |

All errors are relative; ~1e-14..1e-16 confirms the exact masked relations `Z_t=ZPi, U_t=UPi, Y_t=YN_out, GU_t=GUPi, GZ_t=GZPi, GH_t=GH M_in`.

**Experiment 3 (isolated autograd):** `GZ_tilde = GZ Pi` reproduced by standard `torch.autograd` for gelu/silu/relu — pass=True, custom nonlinear primitive used = **False**.

## Cross-Gram security boundary (200 trials)

| region | mean rel_error | mean corr | exact? |
|---|---|---|---|
| general linear (`H_t GH_t^T` vs `H GH^T`) | 1.82e+00 | 0.514 | False |
| nonlinear `Z` (`Z_t GZ_t^T` vs `Z GZ^T`) | 2.68e-15 | 1.000000 | True |
| nonlinear `U` (`U_t GU_t^T` vs `U GU^T`) | 2.87e-15 | 1.000000 | — |

Independent backward mask `M_in` protects the general linear region (`H_t GH_t^T = H N M_in^T GH^T != H GH^T`). The **nonlinear permutation region leaks the exact token×token cross-Gram** because `Pi Pi^T = I`.

## LoRA compatibility

| activation | up `Z` err | down `Y` err | rank⊥Pi |
|---|---|---|---|
| gelu | 1.33e-14 | 1.42e-14 | True |
| silu | 1.33e-14 | 1.50e-14 | True |

Rank-space masks `R_up`,`R_down` live in `r×r`; the activation permutation `Pi` lives in `d_ff×d_ff` — distinct spaces. Perturbing `R_up` leaves the `Pi`-invariant `Z_t = Z Pi` unchanged (rank⊥Pi), confirming independence.

## Tiny-transformer integration (experiment 6)

Ran on a real `PlainTransformerBlock` MLP (GELU, with biases); attention untouched (island applied only to MLP). forward_pass=True (recovered-output err 1.08e-15), autograd_backward_pass=True.

## Answers

**1. Does permutation mask close GELU/SiLU forward?**
Yes — `phi(ZPi)=phi(Z)Pi` holds to fp64 for GELU and SiLU (permutation pass = True/True).

**2. Do signed/diagonal/dense masks fail for GELU/SiLU?**
Yes — GELU/SiLU fail signed_perm (False/False), pos_diagonal (False/False), dense_orth (False/False) and dense_gl (False/False). ReLU additionally closes positive_diagonal (True) — positive-homogeneous — but not signed_perm (False).

**3. Does MLP forward remain exact?**
Yes — Z/U/Y masked relations hold at ~1e-14.

**4. Does MLP backward remain exact with standard autograd?**
Yes — GU/GZ/GH masked relations hold at ~1e-14 using torch autograd for the nonlinear.

**5. Is a nonlinear backward primitive needed?**
No — inside the permutation domain `GZ_t = GU_t (.) phi'(Z_t) = GZ Pi` is produced by standard autograd; no custom masked-Hadamard primitive is required.

**6. Does independent backward mask protect general regions?**
Yes — with independent `M_in`, `H_t GH_t^T` differs from `H GH^T` (mean rel_error 1.82e+00, exact=False).

**7. Does the nonlinear permutation region leak exact cross-Gram?**
Yes — `Z_t GZ_t^T = Z GZ^T` and `U_t GU_t^T = U GU^T` exactly (rel_error 2.68e-15, corr 1.000000). This is a real leak of token×token structure; permutation does NOT remove it.

**8. Are LoRA rank-space masks independent from activation permutation masks?**
Yes — `R` acts in rank space (`r×r`), `Pi` in activation space (`d_ff×d_ff`); changing `R` leaves the `Pi`-invariant unchanged.

**9. What claims are allowed/disallowed?**
See the two lists below.

### Allowed claims
- GELU/SiLU exact nonlinear islands require permutation-domain masking in this implementation.
- MLP forward/backward are exact under permutation islands.
- Standard autograd suffices for the nonlinear backward inside the permutation domain.
- Independent backward masks protect the linear/general regions.
- Nonlinear permutation regions leak the exact token×token cross-Gram.

### Disallowed claims (NOT supported by this experiment)
- Nonlinear cross-Gram is eliminated. (It is exactly preserved — see exp 4.)
- Permutation hides activation values. (It only permutes them; norm/Gram/value-multiset survive.)
- The experiment proves the full stabilizer theorem. (Finite tests only SUPPORT the algebraic claim.)
- The experiment proves input/token/adapter reconstruction (or its impossibility).
- The experiment proves end-to-end Qwen training.
- The nonlinear island provides formal privacy.
