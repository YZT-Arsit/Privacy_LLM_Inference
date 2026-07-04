# AAAI fp32 apples-to-apples results — OURS (folded_remote A_rightmul) vs plaintext

**Model:** Qwen2.5-7B-Instruct · **Precision:** fp32 both sides · **Decoding:** greedy, seq_len 1024, max_new_tokens 512

- **OURS** = TDX-guest trusted boundary + H800 untrusted folded GPU worker (`A_rightmul`, single TEE
  session, zero nonlinear TEE crossings, attested). fp32 forced via the worker flag
  `--fold-dtype-override float32` (numerical-only; design/security/TEE-crossings unchanged).
- **PLAINTEXT** = H800 `plaintext_local`, fp32, same model/decoding.

## Headline

Under matched fp32, **OURS is at statistical parity with plaintext on all three datasets.**
The obfuscation protocol has **no measurable utility cost**. Report as *parity*, not improvement —
the small deltas (both signs) are numerical noise, not a real quality change.

| Dataset | n | Metric | PLAINTEXT-fp32 | OURS-fp32 | Δ | Scorer |
|---|---|---|---|---|---|---|
| IFEval | 541 | prompt-strict acc | 66.17% | 68.76% | +2.59 pt | official google `instruction_following_eval` (prior run) |
| GSM8K | 1319 | exact-match | 90.22% | 90.52% | +0.30 pt | `gsm8k_exact_match` (re-derived here) |
| MT-Bench | 80×2 | LLM-judge 1–10 | 6.844 | 6.763 | −0.08 | **Claude Opus** single-answer |

MT-Bench turns: turn-1 7.34 / 7.35, turn-2 6.19 / 6.34. Per-category deltas within ±0.4, mixed sign.

### Why do two deltas favor OURS?
OURS ≠ bit-identical to plaintext even in fp32: the folded ops ((X−T)N_in, right-multiply nonlinear
islands, recover scale/perm) reorder matmuls → ~1e-6 perturbation. On greedy decode this occasionally
flips a **near-tie argmax**, sometimes to a right token, sometimes a wrong one — a wash. +0.30pt on
GSM8K = 4 questions of 1319 (noise). Neither path is "more correct"; both approximate the real-number
computation. **Do not claim obfuscation improves accuracy.**

### MT-Bench caveats
- Judge is **Claude Opus, not GPT-4** → absolute scores (~6.8) are NOT comparable to the GPT-4-judged
  leaderboard (~8.5). Only the OURS-vs-PLAINTEXT gap under the *same* judge is meaningful.
- max_new_tokens=512 truncates some long answers (esp. turn-2), lowering absolute scores on both sides.

## bf16 → fp32 ablation (IFEval)

| Precision | degenerate responses (≥50 same-token run) | avg tokens | prompt-strict acc |
|---|---|---|---|
| bf16 (old default) | **18** | 230.6 | ≈61.7% (−7pt artifact) |
| fp32 (fixed) | **0** | 242.6 | 68.76% |

16-bit folded-op rounding flips near-tie argmax → degeneration loops + a ~7pt accuracy artifact; fp32
removes it. Hard reproducible metric here = degeneration count **18 → 0** (accuracy % from prior official
scorer). This is the ablation supporting "why fp32 is required for the folded path".

## Files

```
ours/            OURS-fp32 responses + generation reports (ifeval, mt_bench, gsm8k)
plaintext/       PLAINTEXT-fp32 responses (ifeval, mt_bench, gsm8k)
bf16_ablation/   OURS bf16 ifeval responses (pre-fix, for the ablation)
scores/
  gsm8k_exact_match.json        re-derived GSM8K accuracy both sides
  mtbench_judge_scores.json     per-question + aggregate Claude-judge MT-Bench scores
  mtbench_judge_input.json      exact judge inputs (q + ref + both answers) for reproducibility
  mtbench_batches/              raw per-batch judge outputs (8 × 10 questions)
  bf16_vs_fp32_ablation.json    degeneration ablation
COMPILED_RESULTS.json           machine-readable summary of everything above
```

## Provenance / reproduction

- OURS responses pulled from TDX `39.96.4.252:/root/privacy_llm_tee_artifacts/aaai_full_fp32/...`.
- PLAINTEXT responses pulled from H800 `.../plaintext_local_fp32/...`.
- GSM8K scored with `src/pllo/benchmarks/generation_datasets.py::gsm8k_exact_match` vs `final_answer`.
- MT-Bench judged by 8 parallel Claude-Opus agents, official single-answer rubric, reference used for
  math/reasoning/coding. Inputs and per-question scores saved for audit.
- IFEval prompt-strict is from the prior session's official google IFEval scorer (that package is not
  installed in the current environment, so it was not re-derived here; degeneration 18→0 was).
- Code: worker flag `--fold-dtype-override` in commit `86ac1c0` (committed + pushed to origin).
