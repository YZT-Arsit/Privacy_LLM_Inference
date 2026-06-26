# Real H800 + TDX full-evaluation runbook (master)

The next expensive server session should be a **pure execution session**: every
artifact is prepared offline and dry-run validated. This master runbook drives the
whole no-LoRA + LoRA + public-benchmark + security + performance + claim-validation
flow end to end. For LoRA-only or benchmark-only sessions see
[`REAL_H800_TDX_LORA_RUNBOOK.md`](REAL_H800_TDX_LORA_RUNBOOK.md) and
[`REAL_PUBLIC_BENCHMARK_RUNBOOK.md`](REAL_PUBLIC_BENCHMARK_RUNBOOK.md).

> **Nonlinear design dimension.** Every build/verify/probe/decode/E9/E10/latency
> script accepts `--nonlinear-backend current|trusted_shortcut` (aliases:
> `amulet_migrated`, `tee_shortcut_nonlinear`); these commands default to
> `current`. To produce results for BOTH designs (advisor requirement) use
> [`REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md`](REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md):
> it namespaces artifacts per design and **regenerates the TDX runtime hash +
> TD Quote per design** (the runtime hash binds the design, so design-A evidence
> cannot be replayed for design B). Pass the matching `--expected-nonlinear-backend`
> to verifiers and tag required claims as `claim[<design>]`.

Threat model is unchanged: the untrusted H800 worker holds only the public base
folded package + folded-LoRA operators + masked activations + public metadata;
input ids, mask secrets, raw LoRA, optimizer state, labels, and logit recovery
stay trusted-side (H800 trusted process or the TDX guest).

## Phase map (run in order)

The evaluation is split into four phases so a server session has a clear scope.
Health checks use `curl /health` / a Python socket probe — **`ss` is not
installed on H800; do not use it.**

- **A. H800 core matrix** — build+verify both folded packages, boundary
  artifacts, local probes, remote decode, E3 short matrix (max_new_tokens
  1,4,8,16), E9 (plaintext + folded) on the converted real data, pairwise +
  aggregate, security transcript scan + negative tests, latency. Per design.
  Validate with `python scripts/check_h800_stage_outputs.py`.
- **B. H800 extended decode/context scaling** — extended decode-length sweep
  (`outputs/e3_extended/e3_decode_len_scaling_<design>.json`) and context sweep
  (`outputs/e3_context/e3_context_<design>_seq{128,256,512,1024}.json`), then
  `python scripts/render_extended_latency_context.py`.
- **C. TDX-attested minimum deployment evidence** — see
  [`REAL_TDX_MINIMUM_DEPLOYMENT_EVIDENCE_RUNBOOK.md`](REAL_TDX_MINIMUM_DEPLOYMENT_EVIDENCE_RUNBOOK.md)
  (runtime hash + TD Quote per design; attested decode; cross-design negative
  control).
- **D. Final packaging / final gate** — `package_final_artifacts.py`,
  `validate_paper_claims.py`, `final_submission_gate.py`.

For BOTH nonlinear designs end to end, drive phases A–D from
[`REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md`](REAL_DUAL_NONLINEAR_FULL_EVAL_RUNBOOK.md).

```
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
PKGS=/root/autodl-tmp/privacy_llm_packages
BASE=$PKGS/qwen7b_folded_full
ART=$PKGS/qwen7b_boundary_artifact_cuda
LORA=$PKGS/qwen7b_lora_folded_synth_r4
URL=http://127.0.0.1:18083
PORT=18083
TM=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
MRTD=<expected_mr_td>
```

## 0. Preflight (before spending server time)

```
python scripts/preflight_real_eval.py \
  --model-path $MODEL --base-folded-package-path $BASE \
  --embedding-artifact-path $ART --lora-folded-package-path $LORA \
  --gpu-worker-url $URL --backend tdx_attested_remote \
  --attestation-evidence outputs/attestation_evidence.json --expected-mr-td $MRTD \
  --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --output-json outputs/preflight.json --output-md outputs/preflight.md
```

`preflight_passed=false` lists blockers (missing model/base/artifact/evidence,
`runtime_hash != evidence.report_data`, `--require-real` would fall back, output
dir not writable). Clear them before opening the worker.

## 1. Start the H800 worker (base [+ LoRA])

Add `--nonlinear-backend trusted_shortcut` for design B so the worker actually
executes the Amulet lift (and `/health` carries the measured execution evidence);
omit it (or pass `current`) for design A.

```
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package --nonlinear-backend current \
  --folded-package-path $BASE --folded-lora-package-path $LORA \
  --listen-host 0.0.0.0 --listen-port $PORT \
  --device cuda --dtype bfloat16 --audit true
```

## 2. Check H800 package health

```
curl -s $URL/health        # expect status ok, peak_gpu_memory_mb, gpu_backend
```

## 3. Start the TDX SSH tunnel (from the TDX guest)

```
# forward the TDX-guest localhost:$PORT to the H800 worker
ssh -N -L $PORT:127.0.0.1:$PORT <h800-user>@<h800-host> &
curl -s $URL/health        # from inside the TDX guest, confirm reachability
```

## 3b. IFEval generation from the TDX boundary (folded_remote) — MAIN run

> **The paper's main generation benchmark MUST be launched from the TDX guest**
> as a `folded_remote` boundary client. The trusted TDX guest holds only the
> tokenizer/config + embedding artifact + masking/recovery/sampling; the
> untrusted H800 holds the resident folded weights and does the GPU compute:
>
> ```
> TDX trusted boundary  --tunnel-->  http://127.0.0.1:18082  -->  H800 GPU worker
> ```
>
> **A benchmark launched on the H800 itself is NOT a real TDX result.** Only a
> run launched from the TDX guest with `--tdx-boundary-client` and TDX evidence
> attached counts as the main experiment.

The earlier hang was `--precompute-obfuscation-schedule` materializing 541×1024
CUDA secret tensors before any GPU call. Those tensors are only consumed for
freshness *accounting* (never read on the decode path), so the default is now
`--schedule-proof-mode online_deterministic` (slot metadata only, proven full
coverage). Heavy precompute is opt-in (`--schedule-proof-mode
precompute_secret_tensors --enable-secret-tensor-precompute --allow-secret-persist`,
always on `--schedule-precompute-device cpu`).

### Sync the small inputs to the TDX guest (NO full 7B weights)

```
mkdir -p /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted
scp h800-new:/root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl \
  /root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/
mkdir -p /root/autodl-tmp/privacy_llm_packages
rsync -av --progress \
  h800-new:/root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current/ \
  /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current/
mkdir -p /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.json merges.txt; do
  scp h800-new:/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct/$f \
    /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct/ 2>/dev/null || true
done   # tokenizer/config ONLY — do NOT sync *.safetensors to the TDX guest
```

### Preflight the TDX -> H800 bridge

```
python scripts/preflight_tdx_h800_bridge.py \
  --gpu-worker-url http://127.0.0.1:18082 --h800-worker-ssh-alias h800-new \
  --input-jsonl $IFEVAL --embedding-path $EMBED --model-meta-path $MODEL_META \
  --tdx-measurement-log outputs/tdx_evidence/tdx_measurement_coverage.log \
  --output-json outputs/tdx_evidence/preflight.json \
  --output-md   outputs/tdx_evidence/preflight.md
# preflight_passed=true requires: ssh ok, worker /health ok, tee_used_on_gpu=false,
# input + embedding present, tokenizer/config present, measurement coverage OK.
```

### TDX smoke (2 examples)

```
cd /root/Privacy_LLM_Inference
source /root/miniconda3/bin/activate tdx310
export PYTHONPATH=$PWD/src:$PYTHONPATH
export IFEVAL=/root/autodl-tmp/datasets/privacy_llm_benchmarks/converted/ifeval_prompts.jsonl
export EMBED=/root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current
export MODEL_META=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
mkdir -p outputs/tdx_ifeval
PYTHONUNBUFFERED=1 python scripts/run_ifeval_generation.py \
  --tdx-boundary-client --tee-mode real_tdx --trusted-runtime tdx_guest \
  --h800-worker-ssh-alias h800-new \
  --input-jsonl "$IFEVAL" --backend folded_remote \
  --model-name Qwen2.5-7B-Instruct --model-path "$MODEL_META" \
  --gpu-worker-url http://127.0.0.1:18082 --embedding-path "$EMBED" \
  --seq-len 1024 --max-new-tokens 64 --dtype bfloat16 --device cpu --audit \
  --nonlinear-backend current --use-chat-template --require-real \
  --max-examples 2 --progress --progress-every 1 --stream-responses \
  --schedule-proof-mode online_deterministic --trace-worker-timings \
  --align-generation-config --repetition-penalty 1.05 \
  --tdx-measurement-log outputs/tdx_evidence/tdx_measurement_coverage.log \
  --output-response-jsonl outputs/tdx_ifeval/tdx_folded_smoke2_responses.jsonl \
  --output-report-json    outputs/tdx_ifeval/tdx_folded_smoke2_generation.json \
  2>&1 | tee outputs/tdx_ifeval/tdx_folded_smoke2_generation.log
```

Within seconds the log shows flushed per-example progress
(`[ifeval] example 1/2 ... phase=start|generate_start|done`) and the responses
JSONL gains one line per example immediately; the H800 worker log shows
`/init /prefill /decode`.

### TDX full run (541 examples)

Same command with `--max-examples 541 --max-new-tokens 512`, the evidence
attached, and `tdx_folded_ifeval541_*` output names:

```
PYTHONUNBUFFERED=1 python scripts/run_ifeval_generation.py \
  --tdx-boundary-client --tee-mode real_tdx --trusted-runtime tdx_guest \
  --h800-worker-ssh-alias h800-new \
  --input-jsonl "$IFEVAL" --backend folded_remote \
  --model-name Qwen2.5-7B-Instruct --model-path "$MODEL_META" \
  --gpu-worker-url http://127.0.0.1:18082 --embedding-path "$EMBED" \
  --seq-len 1024 --max-new-tokens 512 --dtype bfloat16 --device cpu --audit \
  --nonlinear-backend current --use-chat-template --require-real \
  --max-examples 541 --progress --progress-every 1 --stream-responses \
  --schedule-proof-mode online_deterministic --trace-worker-timings \
  --align-generation-config --repetition-penalty 1.05 \
  --attestation-evidence-json outputs/tdx_evidence/attestation_evidence.json \
  --deployment-truth-json     outputs/tdx_evidence/deployment_truth.json \
  --tdx-measurement-log       outputs/tdx_evidence/tdx_measurement_coverage.log \
  --output-response-jsonl outputs/tdx_ifeval/tdx_folded_ifeval541_responses.jsonl \
  --output-report-json    outputs/tdx_ifeval/tdx_folded_ifeval541_generation.json \
  2>&1 | tee outputs/tdx_ifeval/tdx_folded_ifeval541_generation.log
```

A run counts as the main TDX result only when the report shows
`tdx_boundary_client=true`, `tee_mode=real_tdx`,
`full_model_weights_loaded_in_trusted_runtime=false`,
`h800_worker_tee_used_on_gpu=false`, `dry_run=false`, and `tdx_claim_ready=true`
(evidence attached + measurement coverage OK). See the focused
[`REAL_TDX_IFEVAL_BOUNDARY_RUNBOOK.md`](REAL_TDX_IFEVAL_BOUNDARY_RUNBOOK.md) for
the full field list.

## 4. Run the no-LoRA remote / TDX eval

```
# no-LoRA TDX-lite remote decode (boundary holds only the embedding artifact)
python scripts/run_tee_gpu_protocol_demo.py --mode boundary_client \
  --gpu-backend qwen7b_folded_package --gpu-worker-url $URL \
  --embedding-path $ART --skip-reference true \
  --input-ids-file outputs/tdx_input_ids.json \
  --expected-token-ids "$(python -c "import json;print(','.join(map(str,json.load(open('outputs/tdx_expected_tokens.json'))['expected_token_ids'])))")" \
  --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cpu --audit true \
  --record-transcript outputs/transcript_nolora_tdx.jsonl \
  --output-json outputs/qwen7b_folded_remote_decode.json
```

## 5. Run the LoRA remote / TDX eval

See [`REAL_H800_TDX_LORA_RUNBOOK.md`](REAL_H800_TDX_LORA_RUNBOOK.md) (build/verify
the folded-LoRA package, local + remote LoRA decode, TDX-lite LoRA replay). The
one-command path:

```
python scripts/run_e6_lora_real_h800_pipeline.py \
  --model-path $MODEL --base-folded-package-path $BASE \
  --embedding-artifact-path $ART --lora-mode synthetic \
  --lora-rank 4 --lora-alpha 8 --target-modules $TM \
  --output-lora-package $LORA --gpu-worker-url $URL --listen-port $PORT \
  --seq-len 128 --max-new-tokens 4 --dtype bfloat16 --device cuda --audit true \
  --output-json outputs/e6_lora_real_h800_pipeline.json
```

LoRA **attested** remote decode (the wrapper now attaches + verifies the binding;
without `--attestation-evidence` it makes no attestation claim):

```
python scripts/run_qwen7b_lora_folded_remote_decode_probe.py \
  --gpu-worker-url $URL --embedding-path $ART \
  --input-ids-file outputs/qwen7b_lora_folded_local_probe.json \
  --expected-token-ids-file outputs/qwen7b_lora_folded_local_probe.json \
  --max-new-tokens 4 --seq-len 128 --dtype bfloat16 --device cpu --audit true \
  --attestation-evidence outputs/attestation_evidence.json --expected-mr-td $MRTD \
  --write-runtime-manifest outputs/lora_runtime_manifest.json \
  --output-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json
```

The output JSON includes `attestation`, `boundary_attested`, `runtime_hash`,
`expected_runtime_hash`, `evidence_report_data`, `runtime_hash_bound`,
`binding_mismatch_reason`, and `mr_td`. The wrapper exits non-zero if the binding
does not verify.

## 6. Prepare public benchmark JSONL files

From LOCAL dataset files (no downloads). See
[`PUBLIC_BENCHMARK_PREPARATION.md`](PUBLIC_BENCHMARK_PREPARATION.md).

```
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path /root/datasets/mmlu_test.csv --dataset-name mmlu --split test \
  --max-examples 300 --seed 2035 \
  --output-jsonl outputs/bench/mmlu_300.jsonl \
  --dataset-card-json outputs/bench/mmlu_300_card.json
```

## 7. Run public benchmark subsets

```
# plaintext baseline + no-LoRA TDX-attested candidate (use --require-real so a
# missing model/worker hard-fails instead of emitting a stub dry_run report)
python scripts/run_e9_task_utility_benchmark.py --require-real \
  --dataset-jsonl outputs/bench/mmlu_300.jsonl --task-type multiple_choice \
  --backend plaintext_local --model-path $MODEL \
  --seq-len 512 --max-new-tokens 8 --dtype bfloat16 --device cuda --audit true \
  --output-json outputs/e9_mmlu_plaintext_local.json

python scripts/run_e9_task_utility_benchmark.py --require-real \
  --dataset-jsonl outputs/bench/mmlu_300.jsonl --task-type multiple_choice \
  --backend tdx_attested_remote --model-name Qwen2.5-7B-Instruct \
  --model-path $MODEL --gpu-worker-url $URL --embedding-path $ART \
  --attestation-evidence outputs/attestation_evidence.json --expected-mr-td $MRTD \
  --seq-len 512 --max-new-tokens 8 --dtype bfloat16 --device cpu --audit true \
  --output-json outputs/e9_mmlu_tdx_attested_remote.json --output-md outputs/e9_mmlu.md \
  --output-csv outputs/e9_mmlu.csv

# pairwise + aggregate utility preservation (this — not a single E9 metric — is
# what backs the public_benchmark_utility_preserved claim)
python scripts/run_e9_pairwise_utility_preservation.py \
  --baseline-json outputs/e9_mmlu_plaintext_local.json \
  --candidate-json outputs/e9_mmlu_tdx_attested_remote.json \
  --max-abs-drop 0.02 --max-rel-drop 0.05 --dataset mmlu \
  --output-json outputs/e9_mmlu_pairwise.json
# ... repeat for gsm8k/boolq/sst2, then:
python scripts/run_e9_pairwise_utility_preservation.py --aggregate \
  --pairwise-json outputs/e9_mmlu_pairwise.json \
  --pairwise-json outputs/e9_gsm8k_pairwise.json \
  --pairwise-json outputs/e9_boolq_pairwise.json \
  --pairwise-json outputs/e9_sst2_pairwise.json \
  --required-datasets mmlu,gsm8k,boolq,sst2 \
  --output-json outputs/e9_aggregate_utility.json \
  --output-md outputs/e9_aggregate_utility.md

# LoRA utility preservation (E10): base / plaintext-LoRA / folded-LoRA on e.g. SST-2
python scripts/run_e10_lora_utility_benchmark.py \
  --dataset-name sst2 --task-type classification --metric-name accuracy \
  --base-json outputs/e9_sst2_base.json \
  --plaintext-lora-json outputs/e9_sst2_plaintext_lora.json \
  --folded-lora-json outputs/e9_sst2_folded_lora_remote.json \
  --lora-verify-json outputs/qwen7b_lora_folded_verify.json \
  --no-lora-decode-json outputs/qwen7b_folded_remote_decode.json \
  --lora-decode-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json outputs/e10_lora_utility.json --output-md outputs/e10_lora_utility.md
```

## 8. Regenerate the TDX runtime hash

```
python scripts/check_tdx_measurement_coverage.py --verbose   # must exit 0
python scripts/write_tee_boundary_runtime_hash.py \
  --boundary-backend process --gpu-backend qwen7b --expected-mr-td $MRTD \
  --output outputs/runtime_hash.txt --manifest outputs/runtime_manifest.json
```

## 9. Generate the TDX quote + Alibaba attestation (on the TDX VM)

```
python scripts/generate_tdx_attestation_evidence.py \
  --report-data $(cat outputs/runtime_hash.txt) \
  --output outputs/attestation_evidence.json
```

## 10. Validate report_data == runtime_hash

Re-run the attested decode (step 4 / LoRA step 5) with `--attestation-evidence
outputs/attestation_evidence.json --expected-mr-td $MRTD` and confirm
`runtime_hash_bound=True`, `boundary_attested=True`, `binding_mismatch_reason=null`.
The demo recomputes the hash with the SAME flags and verifies the binding.

## 11. Run the security transcript scan

```
python scripts/scan_security_transcript.py \
  --transcript-jsonl outputs/transcript_nolora_tdx.jsonl \
  --output-json outputs/security_transcript_scan.json \
  --output-md outputs/security_transcript_scan.md --fail-on-leak true
```

## 12. Run the security negative tests

```
python scripts/run_security_negative_tests.py \
  --output-json outputs/security_negative_tests.json \
  --output-md outputs/security_negative_tests.md      # all 14 must be caught
```

## 13. Run the final claim validator

```
python scripts/validate_paper_claims.py \
  --result-json outputs/qwen7b_folded_remote_decode.json \
  --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --result-json outputs/tdx_attested_qwen7b_lora_folded_remote_decode.json \
  --result-json outputs/e9_aggregate_utility.json \
  --result-json outputs/e10_lora_utility.json \
  --result-json outputs/security_negative_tests.json \
  --required-claims no_lora_tdx_attested_remote_package_decode,public_benchmark_utility_preserved \
  --output-json outputs/paper_claim_validation.json \
  --output-md outputs/paper_claim_validation.md
```

Note: `public_benchmark_utility_preserved` is backed ONLY by the aggregate/pairwise
preservation report — passing a single `e9_*` metric report is refused and flagged
as an overclaim risk.

Then consolidate everything (E13):

```
python scripts/check_deployment_truth.py \
  --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --result-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json outputs/deployment_truth.json --output-md outputs/deployment_truth.md

python scripts/run_e12_latency_baselines.py \
  --plaintext-h800-json outputs/e_plaintext_h800.json \
  --folded-h800-remote-json outputs/qwen7b_folded_remote_decode.json \
  --tdx-attested-remote-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --folded-lora-remote-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --output-json outputs/e12_latency_baselines.json \
  --output-csv outputs/e12_latency_baselines.csv \
  --output-md outputs/e12_latency_baselines.md \
  --output-tex outputs/e12_latency_baselines.tex

python scripts/run_e13_final_evaluation_report.py \
  --correctness-json outputs/qwen7b_folded_remote_decode.json \
  --correctness-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \
  --correctness-json outputs/qwen7b_lora_folded_remote_decode_probe.json \
  --e8-json outputs/e8_lora_final_report.json \
  --e9-json outputs/e9_mmlu_tdx_attested.json \
  --e10-json outputs/e10_lora_utility.json \
  --latency-json outputs/e12_latency_baselines.json \
  --security-negative-json outputs/security_negative_tests.json \
  --output-json outputs/e13_final_evaluation.json \
  --output-md docs/paper_draft/evaluation_full.md
```

## 14. Package final artifacts

```
python scripts/package_final_artifacts.py \
  --outputs-dir outputs --tee-artifacts-dir /root/privacy_llm_tee_artifacts \
  --base-folded-package-path $BASE --embedding-artifact-path $ART \
  --lora-folded-package-path $LORA \
  --extra-file outputs/runtime_hash.txt \
  --output-tar /root/privacy_llm_final_artifacts.tar.gz
```

## 15. Cleanup

```
pkill -f "run_tee_gpu_protocol_demo.py --mode gpu_worker_server" || true
pkill -f "ssh -N -L $PORT" || true
rm -rf outputs/e6_pipeline
nvidia-smi      # confirm no lingering GPU processes
# do NOT delete $BASE / $ART unless you intend to rebuild them (expensive setup)
```
