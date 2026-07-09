# V1_silu_amulet_boundary

- **threat_model**: n/a (correctness pre-requisite, not an attack)
- **attacker_observation**: n/a
- **attacker_knowledge**: n/a
- **attack_objective**: n/a; this establishes the exact-lossless foundation
- **metric**: max_abs_error, relative_l2_error of AmuletPhi(UQ,Q) vs phi(U)Q
- **result**: fp64 exact (max_abs 1.06e-13); fp32-IO stable (max_abs 5.01e-06); native fp32 R-factor build fails (fp64-only construction)
- **claim_survives**: True
- **note**: Boundary is exact-lossless at fp64 rounding; the Amulet R-factor construction is fp64-only, so fp32/bf16 inputs must be cast to fp64 for the nonlinear island. bf16 not supported (see V5).

## Boundary correctness

- fp64: max_abs 1.060e-13, rel_l2 3.297e-14 (n=54)
- fp32-IO: max_abs 5.007e-06, rel_l2 7.374e-07 (n=54)
- native fp32 build: failed_as_expected
- negative control (dense P across SiLU): 1.889e+00 (must be large)
