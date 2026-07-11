# Claim support matrix (under the implemented gamma-fold convention)

| claim category | verdict |
|---|---|
| exact-arithmetic forward equivalence | **supported** (all targets; D4 top1 1.0, dry-run 2.7e-15) |
| exact-arithmetic backward relation (dL/dA_plain = dL/dA_tilde T_in^{-1}, dL/dB_plain = T_out dL/dB_tilde) | **supported** (grad relation err ~1e-16) |
| parameter-update equivalence, ordinary masked SGD (O1-A) | **supported ONLY for o_proj, down_proj**; NOT for q/k/v/gate/up (A-factor drifts up to 33% @10 steps; momentum up to 53%) |
| parameter-update equivalence, corrected (O1-B/O1-C) | **supported** for all targets (1e-16), but O1-B leaks gamma^2 on GPU (not paper_safe); O1-C is paper_safe |
| BF16 numerical behavior | **preregistered** for the matrix (parameter-equivalence is an exact-arithmetic property, decided at fp64; bf16 adds tolerance evidence only) |
| effective-weight trajectory (O1-A) | **empirical**: dW cosine vs plaintext 0.76-1.0 @10 steps -- functionally close, NOT exact |
| task-level utility | not claimed here (Gate 0.5 is optimizer equivalence, not utility) |
| trusted invocation count (main profile) | O1-C hybrid: 2 base crossings + 1 gradient-correction crossing = 3 logical/step for gamma-fed targets; o/down stay GPU-exact |

The broader theoretical direction is unchanged; only the optimizer/profile statement is
corrected: **ordinary masked SGD is NOT globally exact parameter-equivalent under the
gamma-fold; it is exact for o_proj/down_proj and effective-weight-aligned elsewhere.**
