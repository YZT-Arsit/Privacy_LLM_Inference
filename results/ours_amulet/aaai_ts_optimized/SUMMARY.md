# Public benchmarks — `trusted_shortcut` (Amulet) folded-remote, fp32, optimized

Real run on **Qwen2.5-7B-Instruct**, H800, 2026-07-04. Backend = the "current
scheme" this stage optimized: **`trusted_shortcut` (op_backend `amulet_migrated`)
folded-remote**, fp32 fold, with the precision-neutral latency features on
(amulet lift-factor cache + keep-alive + precompute-embed) — all **verified
bit-identical** to the un-optimized path, precision unchanged. Worker + boundary
co-located on the H800 (127.0.0.1). Runner: `run_aaai_generation_benchmark.py`
(`--backend folded_remote --nonlinear-backend trusted_shortcut`), crash-safe with
`--resume` + a self-restarting watchdog.

## Headline results

| benchmark | metric | OURS (trusted_shortcut, fp32) |
|---|---|---|
| **IFEval** (541) | strict / loose **prompt** | **0.686 / 0.704** |
| | strict / loose **instruction** | 0.769 / 0.785 |
| **GSM8K** (1319) | exact-match | **0.905** (1194/1319) |
| **HumanEval** (164) | pass@1 (sandbox) | **0.823** (135/164) |
| **MT-Bench** (80×2) | two-turn generations | 160 produced (judge deferred) |

- IFEval: 834 instructions, **100% coverage, 0 failed records**.
- HumanEval pass@1 **0.823 exactly matches** the prior A_rightmul fp32 result
  (0.8232) — an independent cross-check that the two nonlinear designs give
  identical code-gen accuracy at fp32. (Prior real plaintext = 0.780, so OURS is
  even marginally ahead — within noise.)
- MT-Bench: answers generated for all 80 questions (both turns); MT-Bench scoring
  needs an external LLM judge (FastChat GPT-4-style), deferred.

## Latency (greedy, H800)

| | tokens/sec |
|---|---|
| **OURS** (trusted_shortcut, fp32 folded, optimized) | **~25.6** (25.35–25.80 across the 4 benchmarks) |
| **plaintext** (`plaintext_local`, bf16, HF) | **62.79** (clean, uncontended) |
| ratio | **~2.45×** |

The OURS throughput is stable at ~25.6 tok/s across all four benchmarks. The
~2.45× gap vs plaintext is **not** apples-to-apples on precision: OURS runs
**fp32** (required for parity), plaintext runs its native **bf16** — fp32 alone
doubles the per-token weight-memory traffic, and at batch-1 decode that memory
bandwidth is the floor. (No prior plaintext latency record existed, so this was
measured fresh on the same H800.)

## Method / integrity

- **Precision unchanged**: fp32 fold throughout; the three latency features are
  precision-neutral and were verified bit-identical (token/text/exact-token match
  1.0). No bf16/TF32 anywhere in OURS.
- **Crash-resume**: every benchmark ran with `--resume` (skips completed ids) under
  a watchdog that restarts the worker on a health-check failure and re-runs on a
  nonzero exit. **Result: 0 restarts, 0 failed records across all 4 benchmarks.**
- **Volume**: 2,184 examples, 604,617 generated tokens.
- Malformed-output scanner fixed to a script-agnostic Unicode letter test
  (`[^\W_]`) so correct non-Latin output (Kannada/Devanagari/Arabic/…) is no
  longer falsely flagged (commit `bed9c7a`).

Raw: `outputs/aaai_ts_optimized/{ifeval,gsm8k,humaneval,mt_bench}/` on the H800
(responses.jsonl + report.json + status/heartbeat); consolidated numbers in
`results.json`.
