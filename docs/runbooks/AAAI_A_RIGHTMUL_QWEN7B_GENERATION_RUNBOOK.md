# AAAI Runbook — A_rightmul + Qwen2.5-7B generation (TDX boundary + H800 worker)

AAAI comparison: **plaintext-GPU vs ours (A_rightmul, folded remote + TDX/H800)**.
No pure-TEE, no LoRA, no `amulet_secure_R` (a baseline slot is reserved for future
Slalom-style/pure-TEE work but is not run here).

Models: **Qwen2.5-7B-Instruct** (main), **Llama-7B** (structure generalization),
**GPT-2** (correctness sanity). Datasets: **IFEval, GSM8K, MT-Bench**. Config:
`seq_len=1024`, `max_new_tokens=512`, **EOS on**, greedy, batch 1.

Network model: the **H800 has NO internet** — it runs only local code / local
models / local datasets (`--local-files-only`). The **Alibaba TDX guest CAN reach
GitHub** — code pulls, dataset prep, and conversions happen there, then are
`rsync`'d to the H800. The TDX guest is the trusted **boundary client** and
produces the real attestation evidence binding the A_rightmul runtime hash.

Placeholders (never hard-code secrets / SSH keys / passwords):
`<H800_HOST>`, `<TDX_HOST>`, `<MRTD>`, `<MODEL_QWEN_PATH>`, `<MODEL_LLAMA_PATH>`,
`<MODEL_GPT2_PATH>`, `<EMB_ARTIFACT>`, `<H800_WORKER_URL>`, `<DATA_DIR>`,
`<PKG_DIR>`, `<EVIDENCE_JSON>`.

---

## 1. TDX: pull code, prepare data, sync to H800

```bash
# (TDX guest — has GitHub access)
cd <REPO_DIR> && git pull
export PYTHONPATH=$PWD/src:$PYTHONPATH
# raw datasets must already be present locally (no auto-download); normalise:
python scripts/prepare_aaai_generation_datasets.py --dataset all \
  --ifeval-input <RAW_IFEVAL> --gsm8k-input <RAW_GSM8K> \
  --mt-bench-input <RAW_MTBENCH> --output-dir <DATA_DIR>
# sync normalised data + cards + code to the H800 (SSH key auth; no passwords)
rsync -avz <DATA_DIR>/ <H800_HOST>:<DATA_DIR>/
rsync -avz <REPO_DIR>/ <H800_HOST>:<REPO_DIR>/   # or git on a mirror
```

## 2. H800: safe cleanup of stale outputs (dry-run first)

```bash
# (H800) dry-run: see what WOULD be archived (nothing is moved/deleted)
python scripts/cleanup_stale_experiments.py outputs
# execute: MOVE stale outputs to outputs/archive_<ts>/ (models/packages/raw kept)
python scripts/cleanup_stale_experiments.py outputs --execute
```

## 3. H800: build the Qwen2.5-7B A_rightmul folded package (paper-facing)

```bash
# (H800; offline; CUDA + local checkpoint)
python scripts/build_qwen7b_folded_package.py \
  --model-path <MODEL_QWEN_PATH> --model-name Qwen2.5-7B-Instruct \
  --output-dir <PKG_DIR> --seq-len 1024 --num-layers 28 --dtype bfloat16 \
  --device cuda --nonlinear-backend A_rightmul --paper-facing \
  --mask-mode signed_permutation --attention-mask-family pairwise_rotation \
  --linear-boundary-pad --mr-td <MRTD> --output-json outputs/build_qwen_A_rightmul.json
# MUST report: compatible_masks_verified=True, residual_mask_is_signed_permutation=True,
# attention_qk_scores_preserved=True, swiglu_shared_channel_permutation=True,
# arbitrary_dense_mask_rejected=True, paper_facing=True, all 8 pad modules covered.
```

## 4. H800: start the GPU worker

```bash
python scripts/run_gpu_worker_server.py --backend qwen7b_folded_package \
  --folded-package-path <PKG_DIR> --nonlinear-backend A_rightmul \
  --device cuda --dtype bfloat16 --host 0.0.0.0 --port 18082
# the worker REFUSES A_rightmul unless the package certifies compatible_masks_verified=True
```

## 5. TDX: generate the A_rightmul runtime hash + real TD quote

```bash
# (TDX guest) bind the quote to the A_rightmul design
python scripts/generate_tdx_attestation_evidence.py --nonlinear-backend A_rightmul \
  --expected-mr-td <MRTD> --quote-command '<REAL_TDX_QUOTE_CMD>' \
  --attest-command '<REAL_TDX_ATTEST_CMD>' \
  --output-evidence <EVIDENCE_JSON>
# real quote required: tee=tdx, debug=false, 3-part JWT, report_data==runtime_hash,
# runtime_hash_binds_nonlinear_backend=True, mr_td==<MRTD>. --simulate => non-paper-facing.
```

## 6. TDX boundary client: run ours (folded_remote) — per dataset

```bash
# (TDX guest; tunnel to the H800 worker over an SSH-key tunnel, no passwords)
ssh -i ~/.ssh/<KEY> -N -L 18082:127.0.0.1:18082 <H800_HOST> &
OURS="--backend folded_remote --model-name Qwen2.5-7B-Instruct \
  --model-path <MODEL_QWEN_PATH> --gpu-worker-url http://127.0.0.1:18082 \
  --embedding-path <EMB_ARTIFACT> --nonlinear-backend A_rightmul \
  --seq-len 1024 --max-new-tokens 512 --use-chat-template --require-real \
  --tdx-boundary-client --attestation-evidence-json <EVIDENCE_JSON> \
  --expected-mr-td <MRTD> --paper-facing-aaai --resume \
  --max-retries-per-example 3 --retry-sleep-sec 2 --heartbeat-interval-sec 10"

for DS in ifeval gsm8k mt_bench; do
  python scripts/run_aaai_generation_benchmark.py --dataset $DS \
    --dataset-jsonl <DATA_DIR>/$DS.jsonl $OURS --run-id qwen_${DS}_ours \
    --output-response-jsonl outputs/aaai/qwen/folded_remote/$DS/responses.jsonl \
    --output-report-json    outputs/aaai/qwen/folded_remote/$DS/report.json \
    --status-json    outputs/status/qwen_${DS}_ours.status.json \
    --heartbeat-json outputs/status/qwen_${DS}_ours.heartbeat.json
done
```

## 7. H800: run plaintext baseline (folded_remote's counterpart)

```bash
# (H800; full model loads on the H800, never on the TDX guest)
PLAIN="--backend plaintext_local --model-name Qwen2.5-7B-Instruct \
  --model-path <MODEL_QWEN_PATH> --seq-len 1024 --max-new-tokens 512 \
  --use-chat-template --require-real --paper-facing-aaai --resume"
for DS in ifeval gsm8k mt_bench; do
  python scripts/run_aaai_generation_benchmark.py --dataset $DS \
    --dataset-jsonl <DATA_DIR>/$DS.jsonl $PLAIN --run-id qwen_${DS}_plain \
    --output-response-jsonl outputs/aaai/qwen/plaintext_local/$DS/responses.jsonl \
    --output-report-json    outputs/aaai/qwen/plaintext_local/$DS/report.json \
    --status-json    outputs/status/qwen_${DS}_plain.status.json \
    --heartbeat-json outputs/status/qwen_${DS}_plain.heartbeat.json
done
```

## 8. IFEval / GSM8K / MT-Bench notes

- **IFEval**: responses are preserved + the prompt `meta` (instruction ids/kwargs)
  is in the normalised JSONL for the official offline checker.
- **GSM8K**: the runner extracts the numeric answer (`#### ` / last number) and
  records `exact_match`; the report carries `gsm8k_exact_match_accuracy`.
- **MT-Bench**: two-turn, all 80 run; see `docs/MT_BENCH_AAAI_EVALUATION.md`. A
  FastChat-compatible answer file is produced for later offline judging.

## 9. Resume + monitor

```bash
# resume: just re-run the SAME command (completed ids are skipped automatically)
# live run progress:
python scripts/monitor_aaai_run.py --status-json outputs/status/qwen_gsm8k_ours.status.json
# live worker health:
python scripts/monitor_h800_worker.py --gpu-worker-url http://127.0.0.1:18082 \
  --run-id qwen --output-dir outputs/status
```

## 10. Validate

```bash
for DS in ifeval gsm8k mt_bench; do
  python scripts/validate_aaai_generation_results.py --dataset $DS \
    --plaintext-report   outputs/aaai/qwen/plaintext_local/$DS/report.json \
    --plaintext-responses outputs/aaai/qwen/plaintext_local/$DS/responses.jsonl \
    --ours-report        outputs/aaai/qwen/folded_remote/$DS/report.json \
    --ours-responses     outputs/aaai/qwen/folded_remote/$DS/responses.jsonl \
    --dataset-card <DATA_DIR>/cards/${DS}_card.json \
    --attestation-evidence-json <EVIDENCE_JSON> --expected-mr-td <MRTD> \
    --run-id qwen_$DS \
    --output-json outputs/aaai/qwen/validation/aaai_validation_$DS.json \
    --output-md   outputs/aaai/qwen/validation/aaai_validation_$DS.md \
    --output-csv  outputs/aaai/qwen/validation/aaai_tables_$DS.csv
done
# or drive the whole matrix:
python scripts/run_aaai_experiment_matrix.py --run-mode validate-only \
  --dataset-dir <DATA_DIR> --models qwen2.5-7b-instruct \
  --attestation-evidence-json <EVIDENCE_JSON> --expected-mr-td <MRTD>
```

## 11. Llama-7B structure generalization (smaller scale)

Same as §3–§10 with `--model-path <MODEL_LLAMA_PATH>`, `--model-name Llama-7B`,
`--num-layers 32` (build), and `--models llama-7b` in the matrix. Llama-7B is the
structure-generalization result, not the headline.

## 12. GPT-2 correctness sanity (not a headline number)

```bash
# build (paper-facing compatible masks; small model)
python scripts/build_qwen7b_folded_package.py --model-path <MODEL_GPT2_PATH> \
  --model-name gpt2 --output-dir <PKG_GPT2> --seq-len 1024 --num-layers <N> \
  --dtype float32 --device cuda --nonlinear-backend A_rightmul --paper-facing \
  --mask-mode signed_permutation --attention-mask-family pairwise_rotation \
  --linear-boundary-pad
# generation: drop --paper-facing-aaai (sanity, not an AAAI headline result)
python scripts/run_aaai_generation_benchmark.py --dataset gsm8k \
  --dataset-jsonl <DATA_DIR>/gsm8k.jsonl --backend folded_remote \
  --model-name gpt2 --model-path <MODEL_GPT2_PATH> \
  --gpu-worker-url http://127.0.0.1:18082 --embedding-path <EMB_GPT2> \
  --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \
  --require-real --tdx-boundary-client --attestation-evidence-json <EVIDENCE_JSON> \
  --resume --output-response-jsonl outputs/aaai/gpt2/folded_remote/gsm8k/responses.jsonl \
  --output-report-json outputs/aaai/gpt2/folded_remote/gsm8k/report.json
```

---

**Security notes:** no passwords or SSH private keys in any script or log; all
cross-host access uses SSH keys / aliases. The H800 needs no internet. The TDX
quote binds the A_rightmul runtime hash; simulated / off-TDX evidence is rejected
by `--paper-facing-aaai` and by `validate_aaai_generation_results.py`.
