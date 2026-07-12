# Claim-support matrix — empirical closure

Tiers: **real** = real H800 + real TDX run this session; **fp64** = exact-arithmetic proof;
**MISSING** = registered but not executed (enumerated, not substituted).

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A | Executed code is content-bound (fix of the empty-hash defect) | **DONE** | `code_baseline_closure/` bundle `code_binding.json`; remote==local verified |
| C1 | BF16 O1-C runs on real H800+TDX, stable, correction in-enclave | **real** | `bf16_gate/` — gate_pass, finite, 48+120/step, counters 0 |
| C2 | BF16 does NOT underflow the gate/up correction | **real** | `corrected_grad_zeroed_fraction ≈ 0`; per-group numerics CSV |
| C3 | BF16 model-level equivalence vs fp64 plaintext | **real (bf16-limited)** | top1 ~0.976 / KL ~8e-3 (mantissa precision, not instability) |
| C4 | No γ / correction-matrix / plaintext-grad on untrusted GPU (BF16) | **real** | `untrusted_* = 0`, `silent_fallbacks = 0` |
| D2 | L11 momentum runs on real hardware; buffer accumulates corrected grad | **real (1/10-step)** | `momentum_real/`; buffer exactness fp64-proven |
| — | O1-C exact-parameter-equivalent (SGD/momentum/AdamW) | **fp64** | `full_lora_matrix/statistical_analysis.json` (worst 3.8e-13) |
| — | O1-A not parameter-equivalent; diverges on gate/up | **fp64** | same |
| D1 | Real SGD 50-step 3-seed trajectories (L1/L2/L10) | **MISSING** | ~23 h ferry; enumerated |
| D2′ | Real momentum 50-step 3-seed | **MISSING** | enumerated |
| D3 | **L12 AdamW on real hardware** | **MISSING** | needs TDX AdamW-state protocol; fp64-validated only |
| F/G | Converged GSM8K utility (L0/L5/L12, 3 seeds, exact-match) | **MISSING** | many GPU-hrs; L12 needs AdamW hardware |
| F/H | Converged SST-2 utility (L0/L5/L12, 3 seeds) | **MISSING** | enumerated |
| J | Real BF16 rank-mask confirmation | **MISSING** | needs worker L7/L8 support; fp64 ablation exact |
| 9 | Protected deployment provenance (package-native, 0 plaintext) | **real (DONE)** | `full_lora_matrix/deployment/` |
| K | Timing/communication decomposition; ferry ≠ deployment latency | **DONE** | `communication/`, `performance/` |

**Guardrails unchanged:** O1-A never "exact"; O1-B rejected (leaks γ²); O1-D infeasible; no
cryptographic-security / zero-leakage claim; L9 protects rank-only. **L12 is NOT claimed
hardware-validated** until the real TDX AdamW path runs.
