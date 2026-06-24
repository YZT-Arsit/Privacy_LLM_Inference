# Evaluation E3–E5 — methodology and reproduction

This document describes experiments **E3–E5** for the privacy-preserving folded-
package inference system for Qwen2.5-7B. E1/E2 (standalone masked-compute
correctness + token scaling) and the deployment validation (H800 local executable
package, H800 remote HTTP decode, TDX-lite, TDX-attested) are covered elsewhere;
E3–E5 add a scaling sweep, a setup-cost report, and a consolidated comparison.

All E3–E5 runners **reuse** the validated decode path
(`scripts/run_tee_gpu_protocol_demo.py::build_remote_folded_package_decode_report`)
and the existing package/artifact metadata — no protocol or masking logic is
duplicated, and the H800/TDX deployment path is unchanged. Reports are honest by
construction: a missing input is reported as `not_provided` / `null`, never
assumed, and no TDX attestation is claimed unless it is present in the consumed
TDX-attested JSON.

## E3 — remote package-backed decode scaling

**Goal.** Reproducible sweep of `max_new_tokens` (and optionally `seq_len`) over
the already-working remote folded-package decode.

- Code: `scripts/run_e3_remote_decode_scaling.py`,
  `src/pllo/experiments/e3_remote_decode_scaling.py`.
- Modes: **H800 full-reference** (`--model-path` + `--folded-package-path`) or
  **TDX-lite** (`--embedding-path --skip-reference` with `--input-ids-file` and
  `--expected-token-ids`/`--expected-token-ids-file`).
- Correctness note: greedy decode is a deterministic prefix, so an `n`-token row
  is compared against the **first `n`** expected tokens of a (≥`n`) reference run.
- Per-row fields: `experiment`, `stage`, `boundary_mode`, `gpu_worker_remote`,
  `gpu_backend`, `max_new_tokens`, `seq_len`, `tokens_exact_match`,
  `token_match_rate`, `package_backed_prefill`, `package_backed_decode`,
  `folded_package_loaded`, `folded_package_valid`, `worker_has_mask_secrets`,
  `tee_used_on_gpu`, `gpu_visible_plaintext_fields`, `leaked_secret_fields`,
  `audit_passed`, `latency_s`, `latency_per_token_s`, `trusted_bytes`,
  `gpu_bytes`, `boundary_calls`, `gpu_calls`, `peak_gpu_memory_mb`.
- Summary: pass/fail per N + latency / bytes / boundary-call / security tables.

```
python scripts/run_e3_remote_decode_scaling.py \
  --gpu-worker-url http://127.0.0.1:18082 --gpu-backend qwen7b_folded_package \
  --model-name Qwen2.5-7B-Instruct \
  --embedding-path /root/.../qwen7b_boundary_artifact_cuda \
  --input-ids-file outputs/qwen7b_folded_remote_decode_reference_for_tdx.json \
  --expected-token-ids-file outputs/qwen7b_folded_remote_decode_reference_for_tdx.json \
  --max-new-tokens-list 1,4,8,16 --seq-len 128 --dtype bfloat16 --device cpu \
  --audit true --output-json outputs/e3_remote_decode_scaling.json \
  --output-csv outputs/e3_remote_decode_scaling.csv \
  --output-md  outputs/e3_remote_decode_scaling.md
```

Reported to date (status): `max_new_tokens ∈ {1,4,8,16}` all pass,
`token_match_rate=1.0`, no GPU-visible plaintext, no leaked secrets,
`worker_has_mask_secrets=False`, `tee_used_on_gpu=False`.

## E4 — setup / provisioning cost + amortization

**Goal.** Consolidate one-time setup facts (package generation/size/load,
boundary embedding artifact size/hash) and estimate transfer + amortized cost.

- Code: `scripts/run_e4_setup_cost_report.py`,
  `src/pllo/experiments/e4_setup_cost.py` (pure parsing/arithmetic; no torch).
- Sources: package manifest on disk and/or prior `build`/`verify`/`inspection`/
  `load-probe` JSONs; boundary artifact `boundary_meta.json`.
- Key facts: `folded_package_size_gb` (≈26.339369, F32),
  `folded_package_size_if_bf16_gb` (≈13.17), `num_layers`=28, `num_shards`=29,
  `manifest_hash`, `generation_time_s`, `package_verify_passed`,
  `package_load_time_s`, `boundary_embedding_artifact_size_gb` (≈1.066),
  `boundary_embedding_artifact_hash`, `contains_mask_secrets=True`,
  `trusted_only=True`, transfer time per bandwidth, amortized cost per session.
- Size note (recorded in every report): folded operators are stored in **float32
  for numerical fidelity**, so the measured ~26.34GB is ~2× a bf16 store
  (~13.17GB); bf16 is smaller but is not the current measured artifact.

```
python scripts/run_e4_setup_cost_report.py \
  --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \
  --embedding-artifact-path /root/.../qwen7b_boundary_artifact_cuda \
  --build-json outputs/folded_build_full.json \
  --verify-json outputs/folded_verify_full.json \
  --inspection-json outputs/folded_full_inspection.json \
  --load-probe-json outputs/qwen7b_folded_full_load_probe.json \
  --bandwidth-mbps-list 100,500,1000,5000 \
  --amortize-sessions-list 1,10,100,1000 \
  --output-json outputs/e4_setup_cost_report.json \
  --output-md outputs/e4_setup_cost_report.md \
  --output-csv outputs/e4_setup_cost_report.csv
```

## E5 — final comparison + security matrix

**Goal.** Paper-ready consolidation: correctness, deployment, security matrix,
cost, and limitations, rendered as JSON + Markdown + LaTeX.

- Code: `scripts/run_e5_final_comparison_report.py`,
  `src/pllo/experiments/e5_final_comparison.py` (pure parsing; no torch).
- Sections:
  1. **Correctness** — H800 local prefill (k=28), one-step logits, short decode;
     H800 remote HTTP scaling (E3); TDX-lite; TDX-attested.
  2. **Deployment** — boundary location, package location, whether the boundary
     needs the full checkpoint / the full 26GB package, whether TDX attestation
     is bound, whether tokens match.
  3. **Security matrix** — 11 assets (input_ids, prompt embeddings, masked
     embeddings, N₀/mask secrets, vocab mask, folded package, raw Qwen weights,
     recovered logits, sampled token ids, KV cache, boundary embedding artifact)
     × {TDX boundary visible, H800 worker visible, GPU-visible, protected, notes},
     cross-checked against the audit results in the provided decode reports.
  4. **Cost** — latency, trusted/gpu bytes, boundary calls, peak GPU memory,
     package size, plus the E4 setup costs.
  5. **Limitations** — attested run validates `max_new_tokens=4`; longer scaling
     is measured in remote/H800 mode; the boundary artifact holds trusted mask
     tensors and must stay in TDX; the package is F32 for fidelity; HTTP transport
     is a research prototype; no formal cryptographic security is claimed.

```
python scripts/run_e5_final_comparison_report.py \
  --e1-json outputs/e1_qwen_no_lora.json --e2-json outputs/e2_qwen_scaling.json \
  --local-prefill-json outputs/qwen7b_folded_full_prefill_28layer_probe.json \
  --local-logits-json  outputs/qwen7b_folded_full_onestep_logits_probe.json \
  --local-decode-json  outputs/qwen7b_folded_full_decode_probe.json \
  --remote-scaling-json outputs/e3_remote_decode_scaling.json \
  --tdx-lite-json outputs/tdx_qwen7b_folded_remote_lite_decode_probe_cuda_artifact.json \
  --tdx-attested-json outputs/tdx_attested_qwen7b_folded_remote_decode_probe.json \
  --setup-cost-json outputs/e4_setup_cost_report.json \
  --output-json outputs/e5_final_comparison_report.json \
  --output-md  outputs/e5_final_comparison_report.md \
  --output-tex outputs/e5_final_comparison_table.tex
```

The E5 runner also writes `outputs/paper_ready_final_evaluation.md`.

## Example artifacts (before the next live run)

`outputs/examples_e3_e5/` holds example inputs encoding the **reported** H800/TDX
numbers, plus generated `e4_setup_cost_report.*` / `e5_final_comparison_report.*`,
so the report generators and `outputs/paper_ready_final_evaluation.md` can be
inspected now. These are clearly marked examples; regenerate the canonical reports
from real `outputs/*.json` after the next H800/TDX run.

## Testing

`tests/test_e3_e4_e5_reports.py` exercises the E3/E4/E5 aggregation + rendering
with tiny in-memory dicts and JSON fixtures (no H800/TDX/CUDA/checkpoint), plus a
torch-gated E3 integration test that drives the runner against a live tiny CPU
worker. E3 (runner + library) is included in the Python-3.6 API scan
(`tests/test_no_python39_only_apis.py`) since it may run on the TDX boundary.
