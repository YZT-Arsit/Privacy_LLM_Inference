# MT-Bench evaluation for AAAI (protected-inference utility preservation)

MT-Bench has only **80 questions** (two turns each → 160 generations). For the
AAAI submission we therefore **run all 80, never sample**, and the primary
evaluation is **utility preservation** of protected greedy decoding versus
plaintext greedy decoding — NOT an absolute MT-Bench leaderboard score.

## Why preservation is the primary metric

Ours (A_rightmul) computes the *same* deterministic greedy generation as plaintext
on the GPU; the obfuscation only changes the masked representation the untrusted
GPU sees. So the right question is: **does protected inference change the output?**
If token-exact-match is ~100%, obfuscation provably did not alter generation.

## Three evaluation layers

### (a) Primary — preservation vs plaintext (always run, no external deps)
For every `question_id`, `turn1`, `turn2`, compare plaintext vs ours:
- exact response match, token exact-match rate
- normalized edit distance, length ratio
- ROUGE-L, chrF
- finish-reason match
- latency delta

Computed by `scripts/validate_aaai_generation_results.py --dataset mt_bench`
(per-turn tables via `utility_preservation_per_turn`). Implemented in
`src/pllo/benchmarks/aaai_preservation.py`.

### (b) Optional — local judge (reserved hook, not required)
A **local** judge model (no external API) can score plaintext vs ours pairwise.
This is a reserved optional hook: it is **not wired into the server path** and
never gates the paper-facing verdict. When added, it must run a LOCAL model only
and record `local_judge_model`. The FastChat export in (c) is the supported way to
judge today.

### (c) Optional — official MT-Bench judge export
The runner writes a **FastChat/MT-Bench-compatible** answer file
(`mt_bench_judge.jsonl`: `{question_id, category, turns, responses}`) so an
official GPT-4/Claude judge can be run **later, offline, in a network-capable
environment**. This is NOT a required server step (the H800 has no internet).

## Divergence case studies

When ours ≠ plaintext, `validate_aaai_generation_results.py` emits
`divergence_case_studies` (worst edit distance first): `question_id`, turn,
plaintext response, ours response, edit distance, token mismatch. These go in the
appendix.

## Completeness + resume

All 80 questions must complete. The runner
(`scripts/run_aaai_generation_benchmark.py --dataset mt_bench`) is two-turn,
crash-safe, and resumable; failed questions are listed in `failed_records` and can
be re-run with the same command (`--resume`). The validator fails if
`completed_examples != num_examples` (from the dataset card) unless `--allow-failed`.
