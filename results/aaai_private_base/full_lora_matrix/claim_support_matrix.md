# Claim-support matrix — full LoRA validation matrix (O1-C hybrid)

Evidence tiers: **fp64** = exact-arithmetic proof (decided at fp64, the correct place for a
parameter-equivalence property); **real** = real H800 + real Intel TDX run with fresh
attestation; **empirical** = measured, precision-limited.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Forward + backward-relation equivalence (masked ⇔ plaintext) for ALL 7 targets | **SUPPORTED (fp64 + real)** | matrix `gradA_relation_err`≈0; real gate top1 1.0, KL 7.9e-9 |
| 2 | **O1-C** exact-parameter-equivalent SGD, all targets | **SUPPORTED (fp64)** | `statistical_analysis.json`: O1-C A-update worst **3.8e-13** across all opt/groups/seeds |
| 3 | O1-C exact momentum incl. buffer state | **SUPPORTED (fp64)** | momentum O1-C A-update 6.1e-16; opt-state relerr ~1e-16 |
| 4 | O1-C exact AdamW semantics (state tensors) | **SUPPORTED (fp64)** | adamw O1-C A-update 3.8e-13; moment/state matched or TDX-basis exact |
| 5 | **O1-A** is NOT globally parameter-equivalent | **SUPPORTED (fp64)** | O1-A A-update exact only on o/down; qkv drift 0.68–1.10; **gate/up numerically diverges (NaN)** |
| 6 | O1-A is only functionally aligned (effective ΔW) on the drifting targets | **SUPPORTED (fp64)** | O1-A dW cosine high where finite; but gate/up instability makes even that fail at 50 steps |
| 7 | O1-C correction executes **inside the real TDX enclave** (not CPU sim) | **SUPPORTED (real)** | L10 gate: 120 corrected/step in-enclave, bundle TDX-only, attestation binds `optimizer_profile=o1c_hybrid` |
| 8 | No γ / correction-matrix / plaintext-grad on the untrusted GPU | **SUPPORTED (real)** | all `untrusted_*`=0; `silent_fallbacks`=0; O1-B rejected for exactly this leak |
| 9 | 3 logical trusted invocations/step for O1-C | **SUPPORTED (real)** | session + CE/dlogits + gradient-correction |
| 10 | AdamW: q/k **B-factor** ALSO needs TDX (2D RoPE rotation, non-monomial) | **SUPPORTED (fp64)** | monomial-criterion audit; A∈{q,k,v,gate,up} + B∈{q,k} in TDX; o/down fully GPU-exact |
| 11 | Rank mask (fixed / refresh w/ state transport) preserves exactness | **SUPPORTED (fp64)** | ablation: all exact ~1e-15; refresh without state transport BREAKS (0.79) |
| 12 | Protected adapter deploys package-native w/o plaintext reconstruction | **SUPPORTED (real)** | L10 deployment provenance (`deployment/`) |
| 13 | Protected utility EQUALS plaintext utility | **SUPPORTED-BY-CONSTRUCTION (fp64) + preregistered empirical** | exactness (claims 2–4) ⇒ identical model; held-out eval preregistered/budget |

## Wording guardrails
- O1-A must **never** be called exact-parameter-equivalent. On the γ-fed targets it drifts
  (q/k/v) or is numerically unstable (gate/up). It is retained ONLY as an efficiency ablation.
- O1-B is mathematically exact but **rejected as paper_safe** (GPU-visible correction reveals
  the private RMSNorm-gain spectrum). O1-D is infeasible.
- Parameter-equivalence is claimed at fp64; the real run is BF16-tolerance + systems validation.
- No cryptographic-security or zero-leakage claim (unchanged threat model).
