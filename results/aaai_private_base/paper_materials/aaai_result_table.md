# AAAI final result tables (private-base, Qwen2.5-0.5B)

All numbers are copied from completed-experiment artifacts (paths in each caption). No
fabricated values. Security wording is scoped: *empirical resistance under the evaluated
threat model and published attacks* — never zero-leakage / impossible / information-theoretic.

## Table 1 — Utility (SST-2, dev accuracy)
Source: `utility_dataplane/PHASE7_SST2_UTILITY_SUMMARY.json` (real A10 GPU + real Intel TDX,
best-dev checkpoint, early stopping patience 3, 3 seeds).

| Arm | seed 1234 | seed 2025 | seed 7 | mean | vs L5 |
|---|---|---|---|---|---|
| L0 base floor (no LoRA) | — | — | — | **0.8853** | — |
| L5 plaintext-equivalent fp32 (reference) | 0.9163 | 0.9048 | 0.9255 | **0.9155** | — |
| L12 protected bf16 (deployment) | 0.9151 | 0.9232 | 0.9163 | **0.9182** | **+0.0027** |

Paired L12−L5 mean **+0.0027** (per-seed −0.0092…+0.0183), within the **0.01** equivalence
margin → *equivalent in mean; no systematic utility loss from the protected pipeline*.
95% CI (3 seeds, wide) [−0.033, +0.038] — reported honestly; not a strict TOST claim.

## Table 2 — Security (S1–S6, all positive controls PASS)
Sources: `security/S{1..6}_*/s*_results.json`, `security/security_run_manifest.json`.

| ID | Attack (paper) | Positive control | Ours (protected) | Empirical finding (scoped) |
|---|---|---|---|---|
| S1 | Representation inversion (Mahendran-Vedaldi'15; Fredrikson'15) | P0 cos 1.0, P1 cos 0.93 | strict cos **0.002 ≈ random**; known-pairs synthetic rel_err **5.7e-7** | resists strict inversion; orthogonal mask linearly invertible only if paired plaintext leaks; norm/Gram preserved (documented) |
| S2 | LoRA recovery (Hu'22) | plaintext ΔW rel_err **2.4e-15** | plaintext ΔW rel_err **1.41 ≈ √2** | masked product recoverable, plaintext ΔW not without side masks |
| S3 | Logit leakage (design_spec §F) | B0 multiset gap 0 | perm-only conf-corr **1.0** → monomial **0.009** (KL 3.72) | monomial removes the exactly-preserved confidence leak |
| S4 | Gradient inversion (Zhu'19 DLG; Zhao'20 iDLG) | plaintext token acc **0.625** ≫ rand 6.6e-6 | basis-invariance loss gap **9e-13**, tokens agree **100%** | mask = transparent basis change; defense = aggregation (sweep→0 at batch 8) + non-exposure |
| S5 | Membership inference (Shokri'17) | B0 AUC **0.614** > 0.5 | perm-only AUC **0.595**, monomial AUC **0.499** | membership leaks at trained weights; monomial removes readable signal |
| S6 | KV-cache inversion (S1 decoders) | plaintext KV top1 **0.541** ≫ chance 1.3e-3 | masked-transfer **0.003**, masked-adapted **0.536** | mask defeats plaintext-calibrated attacker; orthogonal-invertible given masked pairs (=TEE) |

## Table 3 — System overhead (SST-2 L12 protected training, real A10 + TDX)
Source: `utility_dataplane/PHASE_profiling_gate.json` (200-batch profile; selected config =
physical=effective batch 16, batched forward). Per-optimizer-step decomposition:

| Component | batch 16 | note |
|---|---|---|
| sec / batch (16 ex) | **0.459** | 34.9 examples/s |
| GPU peak memory | 2999 MB | on 23 GB A10 |
| bytes / example (each way) | 1.14 MB | CE logits + trusted-grad frames |
| trusted RPCs / example | 0.125 | 2 per opt step (ce_batch + adamw) ÷ 16 |
| **TDX AdamW compute** | **113.6 ms** | fixed per opt step (168-factor un-fold/AdamW/re-fold, CPU-only TDX) — **inherent** |
| CE network transfer | 44.5 ms | grows with batch × V |
| GPU forward / backward | 46 / 82 ms | after batched-forward fix (was 506 / 900 ms per-example) |

Batch sweep (throughput characterization only; 16 is the semantics-preserving max):
8→0.407 s, 16→0.459 s, 32→0.581 s, 64→1.059 s/batch. The fixed per-step TDX round trip
(~113 ms AdamW + ~44 ms CE net) dominates and is inherent to effective-batch-16 protected
training — not implementation overhead. (Cross-machine data-plane throughput ≈170 MB/s on
the A10↔TDX private VPC; see `alicloud_a10_migration/**`.)
