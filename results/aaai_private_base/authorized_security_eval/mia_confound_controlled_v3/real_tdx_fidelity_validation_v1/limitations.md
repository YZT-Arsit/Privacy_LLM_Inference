# Limitations

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

- The subset has 250 samples (135 members, 115 nonmembers); TPR at 0.1% FPR is unsupported and TPR at 1% FPR is exploratory.
- Exactly one frozen feature lacks a semantically equivalent real-runtime tensor because terminal gamma is folded into `lm_head`.
- Raw simulated-versus-real source classification is perfect despite close per-feature numerics for most columns.
- The paired affine source calibration is label-independent but is sensitivity analysis, not the primary transfer result.
- The classifier is frozen from simulated shadows 1 and 3; no real-TDX membership labels were used for tuning.
- Results do not establish zero leakage, equivalence, formal privacy, universal resistance, or behavior outside the frozen subset.
