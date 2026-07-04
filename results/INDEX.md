# Results organization

Curated results are split into four categories. Source (`src/`) and scripts
(`scripts/`) are **not** moved — they are referenced by path in runbooks and the
H800 workflow. This file maps both results and the relevant scripts to categories.

## `results/ours_amulet/` — OUR scheme, Amulet nonlinear (`trusted_shortcut`)
The current mainline: nonlinear migrated via the Amulet selector-lift
(`op_backend = amulet_migrated`), fp32 folded-remote.
- `aaai_ts_optimized/` — IFEval/GSM8K/HumanEval/MT-Bench + OURS-vs-plaintext comparison (throughput/latency/accuracy)
- `generation_backend_eval/` — current vs trusted_shortcut backend eval + latency optimization (amulet cache, keep-alive, precompute-embed)
- Scripts: `run_generation_backend_eval.py`, `run_aaai_generation_benchmark.py`, `run_gpu_worker_server.py`, `run_amulet_lifted_probe.py`, `build_qwen7b_folded_package.py`, `build_qwen7b_embedding_artifact.py`

## `results/ours_a_rightmul/` — OUR scheme, single-right-multiply (`A_rightmul`)
The compatible-right-multiply nonlinear design (`compatible_right_multiply`), the
paper-facing backend. Bit-identical outputs to `trusted_shortcut` at fp32.
- `aaai_fp32_final/` — A_rightmul vs plaintext, all fp32, MT-Bench Claude-Opus judge, bf16→fp32 ablation
- `humaneval/` — HumanEval pass@1 (A_rightmul) + plaintext
- Scripts: `compare_plaintext_vs_ours_aaai.py`, `evaluate_humaneval_pass1.py`, `evaluate_generation_preservation.py`, `run_amulet_right_mask_nonlinear_experiments.py`

## `results/attacks/` — security attack experiments (PAUSED)
Our attacks against our own scheme (go/no-go security self-audit).
- `ours_gram/` (self-Gram residual-mask recovery), `ours_pad/` (pad recovery),
  `real_attention_fingerprint_qwen7b.json`, `attack_suite_ours.json`,
  `task2*` (layer-0 guardrail), `FINDINGS.md`, `THREAT_MODEL_AND_THEORY.md`
- `outputs/attacks/` — attack experiment outputs (adaptive-island, activation, fingerprint, stronger-attackers, leakage)

## `results/baselines/` — prior-work / other methods
Baseline privacy methods compared against ours (NOT our scheme).
- `conjformer/` (CONJFORMER repro), `obfuscatune_*`, `cryptogen_he_latency.json`,
  `guardrail_latency.json`, `fp_*` (fingerprint baselines), `BASELINES.md`,
  `registry/privacy_inference_methods.yaml`
- `outputs/baselines/` — prior-work comparison tables

## Other trees (not part of the 4-way split)
- `outputs/lora/` — private-LoRA training/inference stage (~25 `run_lora_*.py`)
- `outputs/paper/` — paper tables/ablations/claims audits
- `outputs/intermediate/` — intermediate correctness/probe scratch (gpt2/llama/hf/nonlinear-island/decoder probes)
- `outputs/linear_boundary_pad/`, `outputs/_cpu_validation/`, `outputs/_summary_runs/` — kept as-is
- `paper_draft/`, `paper_results/` — paper writeup

**Removed** (outdated, 2026-07-04): `outputs/examples_e3_e5`, `examples_e6_e8`, `examples_e9_e13`.
