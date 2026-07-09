# V5_efficiency_dtype

- **threat_model**: n/a (cost/stability profiling)
- **attacker_observation**: n/a
- **attacker_knowledge**: n/a
- **attack_objective**: n/a
- **metric**: per-op latency (ms), boundary correctness by dtype
- **result**: dense orthogonal mask is a full [d,d] matmul (denser than a signed-perm gather); Amulet-SiLU island is the dominant cost (Kronecker lift). fp64 exact; fp32 tolerable for the mask matmul but the Amulet island is fp64-only; bf16 unsupported.
- **claim_survives**: None

## Latency (ms, d=768, m=32)

- plaintext_mlp_ms: 2.2080
- dense_orthogonal_mask_ms: 0.1158
- signed_perm_matmul_ms: 0.0823
- signed_perm_gather_ms: 0.0255
- amulet_silu_island_ms: 5737.6099

## dtype correctness (SiLU/Amulet boundary; island compute is fp64-only)

| io_dtype | io_roundtrip_max_abs | exact_lossless | native_build |
|---|---|---|---|
| fp64 | 6.22e-15 | True | fp64_only_construction |
| fp32 | 9.54e-07 | False | failed_as_expected |
| bf16 | 7.81e-03 | False | failed_as_expected |

- classification: **stronger_audit_variant_not_main_path**
