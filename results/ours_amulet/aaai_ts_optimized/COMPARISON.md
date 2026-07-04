# OURS vs plaintext — throughput · latency · accuracy (Qwen2.5-7B, fp32)

Complete side-by-side of the **current scheme** (`trusted_shortcut` folded-remote,
fp32, optimized) against the **plaintext** baseline, on the same H800 / model /
greedy decoding. This consolidates the fresh trusted_shortcut run
(`results/ours_amulet/aaai_ts_optimized/`) with the plaintext responses + MT-Bench judge from
`results/ours_a_rightmul/aaai_fp32_final/`.

## Design equivalence (why this is one comparison, not two)

The fresh `trusted_shortcut` generations are **bit-identical to `A_rightmul`** on
every benchmark checked: MT-Bench **160/160 token-identical**, GSM8K and HumanEval
accuracies match to the digit (0.9052, 0.8232). Both designs are exact fp32
reconstructions of the same computation, so they produce the same greedy output.
→ every `aaai_fp32_final` (A_rightmul) plaintext-comparison result — **including the
Claude-Opus MT-Bench judge** — transfers directly to `trusted_shortcut`, with no
re-judging.

## 1. Accuracy (OURS vs plaintext) — PARITY

| benchmark | metric | OURS | plaintext | Δ (ours−plain) | scorer |
|---|---|---|---|---|---|
| IFEval-541 | strict-prompt acc | **0.686** | 0.641 | +0.045 | approx reimpl (this repo, same both sides) |
| IFEval-541 | strict-prompt acc | 0.688 | 0.662 | +0.026 | official google `instruction_following_eval` (aaai_fp32_final) |
| GSM8K-1319 | exact-match | **0.905** | 0.902 | +0.003 | `gsm8k_exact_match` |
| HumanEval-164 | pass@1 (sandbox) | **0.823** | 0.780 | +0.043 | function-exec pass@1 |
| MT-Bench-80×2 | LLM-judge 1–10 | **6.763** | 6.844 | −0.081 | Claude-Opus single-answer |

**Verdict: statistical parity.** Deltas are both signs and tiny — the folded ops
reorder matmuls (~1e-6), which on greedy decode occasionally flips a near-tie
argmax (sometimes right, sometimes wrong — a wash). The obfuscation has **no
measurable utility cost**; do not claim it improves accuracy.

MT-Bench detail: turn-1 7.34/7.35, turn-2 6.19/6.34; per-category deltas within
±0.4, mixed sign. (Judge is Claude-Opus, not GPT-4 → absolute ~6.8 is not
comparable to the GPT-4 leaderboard; only the OURS−plaintext gap under the same
judge is meaningful.)

## 2. Throughput / latency

| | tokens/sec | notes |
|---|---|---|
| **OURS** (trusted_shortcut, fp32 folded, optimized) | **25.6** | stable 25.35–25.80 across all 4 benchmarks |
| **plaintext, fp32** (matched precision) | **48.0** | fresh clean measurement, same H800 |
| **plaintext, bf16** (native) | 62.8 | fresh clean measurement |

- **Matched-precision (fp32 vs fp32): OURS is ~1.88× slower than plaintext.** This
  is the fair comparison — both sides pay the fp32 2× memory-bandwidth cost.
- vs plaintext's *native* bf16: ~2.45× (but that mixes precision — not
  apples-to-apples).
- Per-token, the trusted boundary adds only ~0.8 ms; the gap is the untrusted
  worker's folded fp32 forward + the masking/lift, which at batch-1 decode is
  memory-bandwidth bound (see the latency floor analysis in
  `results/ours_amulet/generation_backend_eval/`).

## 3. Summary

| dimension | result |
|---|---|
| **accuracy** | parity on IFEval / GSM8K / HumanEval / MT-Bench (deltas ≤ noise) |
| **throughput** | ~1.88× slower than plaintext at matched fp32 (25.6 vs 48.0 tok/s) |
| **correctness** | bit-identical to `current`; token-identical to `A_rightmul`; matches plaintext at the argmax level (parity) |
| **precision** | fp32 throughout (required; bf16 fold causes a −7pt degeneration artifact — see aaai_fp32_final ablation) |

Sources: `results/ours_amulet/aaai_ts_optimized/` (OURS, this run),
`results/ours_a_rightmul/aaai_fp32_final/` (plaintext responses + MT-Bench Claude-Opus judge),
`results/ours_a_rightmul/humaneval/` (HumanEval plaintext).
