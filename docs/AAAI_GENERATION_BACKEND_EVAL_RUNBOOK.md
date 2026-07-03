# Generation-task eval: `current` vs `trusted_shortcut` (Amulet `amulet_migrated`)

Goal of this stage (NOT security): verify the Amulet-style nonlinear migration
(`trusted_shortcut` → op_backend `amulet_migrated`) is *really executed* on
generation tasks, with a full dimension audit, and compare it to the `current`
trusted-boundary baseline. A_rightmul is intentionally out of scope this round.

## What is implemented

| piece | path |
|---|---|
| eval library (backend verify, nonlinear evidence, dimension audit, quality, compare, writers) | `src/pllo/benchmarks/generation_backend_eval.py` |
| approximate IFEval strict/loose scorer (coverage-reported) | `src/pllo/benchmarks/ifeval_scoring.py` |
| driver CLI | `scripts/run_generation_backend_eval.py` |
| base-package verify CLI | `scripts/verify_qwen7b_folded_package.py` |
| smoke prompts (10, zh/en/math/long/multi-turn/code/instruction) | `data/eval/generation_smoke_prompts.jsonl` |
| tests (6 required + extras, all pass) | `tests/test_generation_backend_eval.py` |

Key facts baked in:
- The nonlinear backend is **fixed at worker launch** (`run_gpu_worker_server.py
  --nonlinear-backend`), so `current` vs `trusted_shortcut` = two worker configs
  (each with its own folded package). The eval **connects** to a running worker
  and **verifies via `/health`** that the served backend matches `--expected-…`;
  a `trusted_shortcut` request served by a `current` worker is a hard
  `silent_fallback_detected` failure (exit 2).
- For `trusted_shortcut`, genuine `amulet_migrated` lift evidence is required
  (`report_has_amulet_execution`: op_backend amulet_migrated + amulet_lift_executed
  + lifted_nonlinear_ops_count>0 + lift_k≥2 + lifted_gpu_bytes>0). Missing → exit 3
  with diagnostics (unless `--allow-missing-evidence`). The lift counter fires on
  the **SwiGLU SiLU** lift, so the MLP path must run for evidence to populate.
- Dimension audit fields are tagged `measured` (client-side captured shapes) vs
  `derived_from_config` / `derived_from_exec_metadata` — never faked.

## Step 0 — tests (local, no GPU)

```
python -m pytest tests/test_generation_backend_eval.py -q
```

## Step 1 — build the `trusted_shortcut` seq1024 package (if not present)

`trusted_shortcut` is non-paper-facing, so `--allow-unwired-nonlinear` is
required and `--paper-facing` must NOT be passed. Pad is on by default.

```
python scripts/build_qwen7b_folded_package.py \
  --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
  --output-dir /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_trusted_shortcut_seq1024_pad \
  --seq-len 1024 --num-layers 28 --dtype bfloat16 \
  --nonlinear-backend trusted_shortcut --allow-unwired-nonlinear \
  --linear-boundary-pad \
  --device cuda \
  --output-json /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_trusted_shortcut_seq1024_pad/build_report.json
```

Verify:

```
python scripts/verify_qwen7b_folded_package.py \
  --folded-package-path /root/.../qwen7b_folded_full_trusted_shortcut_seq1024_pad \
  --expected-nonlinear-backend trusted_shortcut \
  --output-json /root/.../verify_report.json
# expect: package_valid=true, nonlinear_backend_ok=true,
#         contains_mask_secrets=false, hash_mismatches=[], missing_shards=[]
```

## Step 2 — launch a worker per backend

```
# current baseline worker
python scripts/run_gpu_worker_server.py --port 18082 \
  --backend qwen7b_folded_package \
  --folded-package-path /root/.../qwen7b_folded_full_current_seq1024_pad \
  --nonlinear-backend current --device cuda --dtype bfloat16

# trusted_shortcut worker (separate port OR restart on the same port)
python scripts/run_gpu_worker_server.py --port 18083 \
  --backend qwen7b_folded_package \
  --folded-package-path /root/.../qwen7b_folded_full_trusted_shortcut_seq1024_pad \
  --nonlinear-backend trusted_shortcut --allow-unwired-nonlinear \
  --device cuda --dtype bfloat16
```

## Step 3 — smoke (10 prompts) per backend

```
python scripts/run_generation_backend_eval.py \
  --backend current \
  --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
  --folded-package-path /root/.../qwen7b_folded_full_current_seq1024_pad \
  --embedding-path /root/.../qwen7b_boundary_artifact_current_cuda \
  --gpu-worker-url http://127.0.0.1:18082 \
  --prompt-file data/eval/generation_smoke_prompts.jsonl \
  --max-new-tokens 64 --expected-nonlinear-backend current \
  --output-dir outputs/generation_backend_eval/current_smoke

python scripts/run_generation_backend_eval.py \
  --backend trusted_shortcut \
  --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \
  --folded-package-path /root/.../qwen7b_folded_full_trusted_shortcut_seq1024_pad \
  --embedding-path /root/.../qwen7b_boundary_artifact_trusted_shortcut_cuda \
  --gpu-worker-url http://127.0.0.1:18083 \
  --prompt-file data/eval/generation_smoke_prompts.jsonl \
  --max-new-tokens 64 \
  --expected-nonlinear-backend trusted_shortcut \
  --expected-op-backend amulet_migrated \
  --compare-baseline-dir outputs/generation_backend_eval/current_smoke \
  --output-dir outputs/generation_backend_eval/trusted_shortcut_smoke
```

## Step 3b — REAL TDX attestation (required for paper-facing runs)

To bind the run to a genuine TD Quote (not mock), (1) generate the quote on the
TDX VM, then (2) pass `--worker-backend tdx_attested_remote` + the quote +
`--require-tdx`:

```
# on the TDX VM (39.96.4.252): produce the quote evidence bound to THIS design
python scripts/generate_alibaba_tdx_quote_evidence.py \
  --nonlinear-backend trusted_shortcut \
  --output-json /root/tdx_quote_trusted_shortcut.json    # copy back to client

# client run, real-TDX gated
python scripts/run_generation_backend_eval.py \
  --backend trusted_shortcut --worker-backend tdx_attested_remote \
  --attestation-evidence-json ./tdx_quote_trusted_shortcut.json \
  --expected-mr-td <MR_TD_HEX> --require-tdx \
  --model-path .../Qwen2___5-7B-Instruct \
  --folded-package-path .../qwen7b_folded_full_trusted_shortcut_seq1024_pad \
  --embedding-path .../qwen7b_boundary_artifact_trusted_shortcut_cuda \
  --gpu-worker-url http://127.0.0.1:18083 \
  --prompt-file data/eval/generation_smoke_prompts.jsonl \
  --max-new-tokens 64 --expected-nonlinear-backend trusted_shortcut \
  --expected-op-backend amulet_migrated \
  --output-dir outputs/generation_backend_eval/trusted_shortcut_smoke_tdx
```

The run writes `attestation.json` (`boundary_attested`, `mr_td`,
`runtime_hash_bound`, `binding_mismatch_reason`) and `report.md` §0. `--require-tdx`
makes a non-attested boundary a hard failure (exit 6). NOTE: the runtime hash is
bound to the nonlinear design, so re-generate the quote when switching `current`
↔ `trusted_shortcut`.

## Step 4 — IFEval subset (20 → 100 → 541)

```
python scripts/run_generation_backend_eval.py --backend <b> ... \
  --prompt-file data/aaai/ifeval.jsonl --limit 20  --max-new-tokens 512 \
  --output-dir outputs/generation_backend_eval/<b>_ifeval20
# then --limit 100, finally no --limit for the full 541
```

IFEval strict/loose (approximate reimpl, coverage-reported) is scored
automatically when the prompt file carries `meta.instruction_id_list`
(`ifeval_scores.json`). Sanity: on the existing fp32 responses the scorer gives
OURS strict-prompt 0.676 vs plaintext 0.641 (cov 96.3%).

## Outputs (per run dir)

`config.json`, `manifest_audit.json`, `dimension_audit.json`,
`nonlinear_execution_evidence.json`, `generations.jsonl`, `metrics.csv`,
`per_layer_dimensions.csv`, `report.md`, `run_summary.json`, and (when a
baseline dir is given) `comparison_current_vs_trusted_shortcut.json`.

## Acceptance (this stage = usability, not security)

1. current smoke generates; 2. trusted_shortcut smoke generates;
3. no silent fallback (backend_verification_passed);
4. nonlinear evidence non-empty + shows amulet_migrated lift;
5. dimension_audit covers model/attention/KV/MLP/nonlinear/LM-head/mask+pad;
6. readable current-vs-trusted_shortcut comparison;
7. token/logits mismatch → first-mismatch + error reported;
8. any trusted_shortcut failure → explicit cause.

Do NOT write "trusted_shortcut is secure" / Amulet-privacy / pad claims here —
this round reports only generation usability, nonlinear-path evidence, dimension
audit, and correctness/perf comparison.
