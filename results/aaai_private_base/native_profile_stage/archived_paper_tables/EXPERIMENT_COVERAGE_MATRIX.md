# Experiment coverage matrix (private-base AAAI)

Legend: ✅ done (real artifact) · ⏸ deferred (out of this round's scope) · ❌ not attempted.

## Core system + correctness
| Item | Status | Evidence |
|---|---|---|
| Private transformed package (fail-closed, plaintext-absent) | ✅ | `private_package/plaintext_absence_scan.json`, `build_validation.json` |
| Exact fold (8 Linear + RMSNorm + RoPE-commuting QK + monomial logits), fp64 < 1e-8 | ✅ | `private_package/build_validation.json` |
| Qwen2.5-0.5B private-base LoRA training on real A10 + real TDX | ✅ | `alicloud_a10_migration/**`, `utility_dataplane/**` |
| TDX protected loss + FP32 optimizer (SGD / momentum / AdamW exact) | ✅ | `full_lora_matrix/**`, `gate0_d4/**` |
| Real TDX attestation (quote verified) | ✅ | prior migration/attestation artifacts |

## Utility
| Item | Status | Evidence |
|---|---|---|
| SST-2 converged, 3 seeds, L0 / L5 / L12, best-dev + early stop | ✅ | `utility_dataplane/PHASE7_SST2_UTILITY_SUMMARY.json` |
| SST-2 equivalence L12 vs L5 (mean +0.0027, within 0.01 margin) | ✅ | same |
| GSM8K converged utility | ⏸ | compute-bound (measured L12 ≈ 74.6 h/seed) |

## Security (S1–S6, all positive controls PASS)
| ID | Attack | Status | Evidence |
|---|---|---|---|
| S1 | Representation inversion | ✅ | `security/S1_representation_inversion/s1_results.json` |
| S2 | LoRA adapter recovery | ✅ | `security/S2_lora_recovery/s2_results.json` |
| S3 | Logit leakage (perm-only vs monomial) | ✅ | `security/S3_logit_leakage/s3_results.json` |
| S4 | Gradient inversion (DLG/iDLG) | ✅ | `security/S4_gradient_inversion/s4_results.json` |
| S5 | Membership inference (Shokri shadow MIA) | ✅ | `security/S5_membership/s5_results.json` |
| S6 | KV-cache inversion | ✅ | `security/S6_kv_cache/s6_results.json` |

## System overhead
| Item | Status | Evidence |
|---|---|---|
| Per-step cost breakdown + batch sweep (real A10 + TDX) | ✅ | `utility_dataplane/PHASE_profiling_gate.json` |
| Cross-machine data-plane throughput (~170 MB/s) | ✅ | `alicloud_a10_migration/**` |

## Deliberately NOT done (scope guard)
| Item | Status | Reason |
|---|---|---|
| 7B model | ❌ | explicitly out of scope |
| External baselines (STIP / ObfuscaTune / Amulet) reproduction | ❌ | explicitly out of scope; internal plaintext/identity/random baselines used |
| GSM8K converged utility, per-token generation | ⏸ | compute-infeasible this round |
| Formal security proofs / theorems for masking | ❌ | not claimed; security is empirical-under-evaluated-attacks by design |

## Missing / future (honest)
- Deeper-layer KV inversion (S6 uses layer-0, attacker-favourable).
- Full attention/MLP LoRA MIA target (S5 uses a linear-probe proxy).
- White-box / adaptive attackers and side channels (out of the current threat model).
- Multi-task and larger-model utility; strict TOST equivalence needs more seeds.
