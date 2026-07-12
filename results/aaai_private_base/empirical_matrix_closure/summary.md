# Empirical matrix closure — summary

**Status: `INTERNAL_LORA_MATRIX_PARTIAL`** (not `EMPIRICALLY_COMPLETE`; the 10 FINAL-STOP-POINT
items are not all met — see `missing_runs_enumeration.json`). Nothing committed by the agent;
HEAD `9773d9c`. Real Qwen2.5-0.5B + package `bfd578b8…` + real H800 + real Intel TDX throughout.

## PHASE A — code provenance (DONE, corrects the prior defect)
The prior binding recorded `e3b0c442…` = SHA-256 of **empty content** (empty `git diff` while the
executed code was uncommitted) — it bound nothing. Fixed:
- `code_baseline_closure/`: HEAD, `git status --porcelain=v1`, `git diff --binary` + hash,
  untracked inventory, per-file hashes, **content-addressed `experiment_source_bundle.tar`** (26
  source files) + hash, `code_binding.json` (v2).
- **Commit mechanism (precise):** the agent wrote files via its tools into the worktree; an actor
  under the user's identity (author == committer == Zhaoting Yang, empty messages) ran `git add`/
  `git commit` outside the agent's tool calls — the agent issued **0** git commits. Chain
  `36b2d6c→d4f44dd→a95b59f→9773d9c` audited in `provenance_correction.md`.
- **Dependency on non-HEAD code:** at execution time YES (code was uncommitted); now NO — `9773d9c`
  captures the exact files, and **remote (on-hardware) hashes match local byte-for-byte**
  (`code_binding.json → remote_executed_code_sha256_16`, verified this run).

## PHASE C — real BF16 O1-C gate (RUNNING → `bf16_gate/`)
Real H800 + real TDX, fresh attestation per run bound to `optimizer_profile=o1c_hybrid`, 3 seeds
{1234,2025,7} × {1,10}-step, **BF16 forward**, in-enclave correction.
- Confirmed stable: seeds 1234/2025 1-step **gate_pass, finite, top1 0.976, KL ~8e-3**,
  `corrected_grad_zeroed_fraction ≈ 0`, 48 GPU-exact + 120 in-enclave corrections/step, 0 missing.
- Security counters `untrusted_gamma/correction_matrix/plaintext_grad_materializations = 0`,
  `silent_fallbacks = 0`; 3 logical invocations/step.
- Per-group BF16 numerics (q/k/v · gate/up · o/down separately, not hidden in a mean) →
  `bf16_gate/per_group_bf16_numerics.csv`; 3-seed stats → `bf16_gate/statistics.json`.
- **BF16 is stable — the gate is NOT replaced by FP32 or O1-A.**

## PHASE D2 — real L11 momentum (RUNNING → `momentum_real/`)
Same harness, `--momentum 0.9`, buffer persisted in a worker sidecar; the corrected-A buffer
accumulates the **TDX-corrected** gradient (buffer lives in the folded metric). 3 seeds × {1,10}-step.
Buffer *exactness* is fp64-proven (main matrix); this validates the buffer-corrected update on real
hardware. (Real 50-step momentum trajectories: missing — ferry cost.)

## PHASE K — timing / communication (DONE)
Per step: 3 logical invocations; ~25.5 MB logits + 3.48 MB correction each way (correction = 12% of
GPU→TDX); in-enclave compute ~0.05 s vs ~184 s step wall (cross-cloud ferry). Prototype wall time
reported transparently, **not** a deployment-latency claim.

## Missing registered runs (enumerated, no analytic substitution)
SGD/momentum **50-step 3-seed**; **L12 AdamW on real hardware** (needs a TDX AdamW-state protocol —
fp64-validated only, NOT hardware-claimed); converged **GSM8K/SST-2 utility** (L0/L5/L12, 3 seeds);
**real** rank-mask confirmation (Phase J; fp64 ablation already exact). Reasons + measured ferry
costs in `missing_runs_enumeration.json` and `limitations.md`.

## Do-not-rerun honored
D2, D3, package construction, Gate 0, Gate 0.5, prior generation-correctness, archived rank-only
50-step — none rerun.
