# G1/G2 effective configuration audit

Decision: **Statement A is true.** `G1_BF16_MATCHED` uses effective
`weight_decay=0.01` and matches the existing G2 optimizer recipe in the audited
non-protection dimensions. The existing G2 training cells can be retained; no G2
rerun is authorized or required by this audit.

## Shared model/data/LoRA facts

- Base: Qwen2.5-0.5B; weight hash
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Tokenizer hash:
  `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
- E2E train artifact hash:
  `7e46d5bca17ca3d31e843f2c4709854fac311b73a02fb09c7e1e619f7ee38d07`.
- Targets: q/k/v/o/gate/up/down projections; rank 8, alpha 16, dropout 0.
- Batch 16, gradient accumulation 1, 750 steps, LR 2e-4.

## Effective optimizer/precision facts

| Cell | WD (all groups) | Forward/backward/runtime | Authoritative LoRA; Adam m/v | AdamW | LR schedule; clip |
|---|---:|---|---|---|---|
| G1_STANDARD, seeds 1234/7/2025 | 0.0 (one trainable-LoRA group) | fp32 / fp32 | fp32; fp32 | PyTorch AdamW, betas .9/.999, eps 1e-8, fused not requested | 3% linear warmup then linear decay; global norm 1.0 |
| G1_BF16_MATCHED, seeds 1234/7/2025 | 0.01 (explicit update over every master parameter) | bf16 / bf16 runtime | fp32; fp32 | explicit unfused G2 formula, betas .9/.999, eps 1e-8 | constant 2e-4; no clipping |
| G2_PROTECTED, seeds 1234/7/2025 | 0.01 for GPU and TDX parameter partitions | bf16 / bf16 runtime | fp32 master; fp32 m/v (partitioned GPU/TDX) | explicit unfused formula, betas .9/.999, eps 1e-8 | constant 2e-4; no clipping |

G2 authoritative LoRA factors and AdamW state are partitioned according to the
protected protocol: trusted factors/moments in TDX and the exact monomial-safe
partition on GPU. Runtime copies are bf16. G1_BF16_MATCHED uses the same numerical
recipe without protection.

## Completion and order hashes

| Cell | Seed | Dataset/order hash | Steps | Generations | Primary terminal/artifact hash |
|---|---:|---|---:|---:|---|
| G1_STANDARD | 1234 | `e2e_schedule_s1234.json` (legacy meta; no terminal-manifest hash) | 750 | 500 | adapter `a099492ed5058d7b...`; generation `ec5a432031dcbfed...` |
| G1_STANDARD | 7 | `e2e_schedule_s7.json` (legacy meta; no terminal-manifest hash) | 750 | 500 | adapter `f313ea2900aa91e1...`; generation `0cfef06b7ace081b...` |
| G1_STANDARD | 2025 | `e2e_schedule_s2025.json` (legacy meta; no terminal-manifest hash) | 750 | 500 | adapter `3af4e992f8a978e7...`; generation `744d1758db59cff3...` |
| G1_BF16_MATCHED | 1234 | file `fb4f71f3a768c724...` | 750 | 500 | terminal config `755d453f...`; adapter `82ae93d32a929e8c...` |
| G1_BF16_MATCHED | 7 | file `f357013e1ccdddf7...` | 750 | 500 | terminal config `350a3b97...`; adapter `dcf868f5b8041c29...` |
| G1_BF16_MATCHED | 2025 | file `a42d9d8e54c06057...` | 750 | 500 | terminal config `8f3bb430...`; adapter `d5b27f3d8d5b6083...` |
| G2_PROTECTED | 1234 | effective schedule `818e2c072a939b9d...` | 750 | frozen profile says 500 | run `513c5729b6f40343...`; adapter `1e29821f0b4cf4eb...` |
| G2_PROTECTED | 7 | effective schedule `85073b38ecdff692...` | 750 | 500 | run `81f5905e1592212a...`; adapter `2684a8a59d67a55b...` |
| G2_PROTECTED | 2025 | effective schedule `7f4c2a930be8d17c...` | 750 | 500 | run `efe3638a0aea9f2c...`; adapter `d9ecfdfc74531842...` |

All audited training trajectories report finite completion. The G2 counters report
750 AdamW steps, FP32 authoritative state, BF16 runtime copies, verified
attestation, and zero silent fallbacks.

## Integrity exception requiring separate remediation

The frozen seed-1234 completion manifest records a 500-row generation artifact
with SHA-256 `d3b5ad9820c13e47...` and size 1,010,375 bytes. The current local file
has only 263 rows and SHA-256 `c3aa66cbbe899885...`. This does not change the
WD/configuration conclusion or require a G2 training rerun, but the local
seed-1234 generation artifact must not be presented as matching its frozen
terminal manifest. It should be restored from a verified frozen copy or audited
separately; this session did not modify it.

## Paper-facing roles

- `G1_STANDARD_WD0`: conventional plaintext baseline.
- `G1_MATCHED_WD001`: attribution control.
- `G2_PROTECTED_WD001`: protected system.

