# AAAI Runbook — code-gen / sensitive-prompt / long-prompt + GPU-staged schedule

Extends the main AAAI runbook (`AAAI_A_RIGHTMUL_QWEN7B_GENERATION_RUNBOOK.md`)
with HumanEval (MBPP optional), SensitivePrompt-1024 security stress,
LongBench-1024-lite long-prompt stress, and the GPU-staged (non-secret)
obfuscation schedule.

AAAI comparison stays: **plaintext-GPU vs ours (A_rightmul, folded remote +
TDX/H800)**, now with an additional **staged** variant
(`folded_remote_staged`). NO amulet / amulet_secure_R, NO LoRA, NO pure-TEE.

Network model unchanged: **H800 has no internet** (local code/models/data only);
**TDX** pulls datasets + makes the real attestation. Placeholders only — no
passwords, no SSH private keys in any script/log.

---

## 1. TDX: fetch HumanEval / MBPP / LongBench

```bash
# (TDX guest; has network). Reads a LOCAL raw file (preferred) or downloads here.
python scripts/fetch_aaai_optional_datasets.py --dataset humaneval \
  --raw-input <RAW_HUMANEVAL_JSONL> --output-dir <CONVERTED_DIR>
python scripts/fetch_aaai_optional_datasets.py --dataset mbpp --include-mbpp \
  --raw-input <RAW_MBPP_JSONL> --output-dir <CONVERTED_DIR>            # optional
python scripts/fetch_aaai_optional_datasets.py --dataset longbench_1024_lite \
  --raw-input <RAW_LONGBENCH_JSONL> --max-seq-len 1024 --output-dir <CONVERTED_DIR>
```

## 2. TDX: build SensitivePrompt-1024 (synthetic, no real PII)

```bash
python scripts/build_sensitive_prompt_stress_set.py --num-per-bucket 20 \
  --buckets 128,512,1024 \
  --output-jsonl <CONVERTED_DIR>/sensitive_prompt_1024.jsonl \
  --card-json <CONVERTED_DIR>/cards/sensitive_prompt_1024_card.json
```

## 3. TDX -> H800 sync

```bash
rsync -avz <CONVERTED_DIR>/ <H800_HOST>:<DATA_DIR>/      # SSH key auth; no passwords
```

## 4. H800: verify dataset cards (no fetch on H800)

```bash
for ds in humaneval sensitive_prompt_1024 longbench_1024_lite; do
  python -c "import json;d=json.load(open('<DATA_DIR>/cards/${ds}_card.json'));print(d['dataset_name'],d['num_examples'],d.get('output_sha256')[:12])"
done
```

## 5. H800: plaintext baseline

```bash
PLAIN="--backend plaintext_local --model-name Qwen2.5-7B-Instruct \
  --model-path <MODEL_QWEN_PATH> --seq-len 1024 --max-new-tokens 512 \
  --use-chat-template --require-real --paper-facing-aaai --resume"
# code-gen
python scripts/run_code_generation_benchmark.py --dataset humaneval \
  --dataset-jsonl <DATA_DIR>/humaneval.jsonl $PLAIN --run-id he_plain \
  --output-response-jsonl outputs/aaai/qwen/plaintext_local/humaneval/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/plaintext_local/humaneval/report.json
# sensitive
python scripts/run_sensitive_prompt_benchmark.py \
  --dataset-jsonl <DATA_DIR>/sensitive_prompt_1024.jsonl $PLAIN --run-id sp_plain \
  --output-response-jsonl outputs/aaai/qwen/plaintext_local/sensitive/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/plaintext_local/sensitive/report.json
```

## 6. Pre-stage the GPU schedule (TEE side; non-secret)

```bash
python scripts/prestage_gpu_obfuscation_schedule.py \
  --folded-package-path <PKG_DIR> --embedding-artifact-path <EMB_ARTIFACT> \
  --nonlinear-backend A_rightmul --seq-len 1024 --max-new-tokens 512 \
  --num-layers 28 --output-dir <STAGED_DIR> --paper-facing-aaai \
  --output-json outputs/staged/qwen_prestage.json
# MUST report: gpu_staged_schedule_contains_raw_masks=False, _raw_pad=False,
# _plaintext_input=False, _token_ids=False, audit_passed=True.
```

## 7. TDX+H800: ours A_rightmul — unstaged then staged

```bash
OURS="--backend folded_remote --model-name Qwen2.5-7B-Instruct \
  --model-path <MODEL_QWEN_PATH> --gpu-worker-url <H800_WORKER_URL> \
  --embedding-path <EMB_ARTIFACT> --nonlinear-backend A_rightmul \
  --seq-len 1024 --max-new-tokens 512 --use-chat-template --require-real \
  --tdx-boundary-client --attestation-evidence-json <EVIDENCE_JSON> \
  --expected-mr-td <MRTD> --paper-facing-aaai --resume"

# unstaged (HumanEval)
python scripts/run_code_generation_benchmark.py --dataset humaneval \
  --dataset-jsonl <DATA_DIR>/humaneval.jsonl $OURS --run-id he_ours_unstaged \
  --output-response-jsonl outputs/aaai/qwen/folded_remote_unstaged/humaneval/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/folded_remote_unstaged/humaneval/report.json

# staged (HumanEval): add the staged-schedule flags
python scripts/run_code_generation_benchmark.py --dataset humaneval \
  --dataset-jsonl <DATA_DIR>/humaneval.jsonl $OURS --run-id he_ours_staged \
  --use-gpu-staged-schedule --require-staged-schedule \
  --gpu-staged-schedule-dir <STAGED_DIR> \
  --staged-schedule-audit-json outputs/aaai/qwen/folded_remote_staged/humaneval/staged_audit.json \
  --output-response-jsonl outputs/aaai/qwen/folded_remote_staged/humaneval/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/folded_remote_staged/humaneval/report.json

# sensitive (ours, record transcript for the leakage scan)
python scripts/run_sensitive_prompt_benchmark.py \
  --dataset-jsonl <DATA_DIR>/sensitive_prompt_1024.jsonl $OURS --run-id sp_ours \
  --record-transcript \
  --transcript-jsonl outputs/aaai/qwen/folded_remote_unstaged/sensitive/transcript.jsonl \
  --output-response-jsonl outputs/aaai/qwen/folded_remote_unstaged/sensitive/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/folded_remote_unstaged/sensitive/report.json

# long-prompt stress (NOT an official LongBench score)
python scripts/run_long_prompt_stress.py --dataset longbench_1024_lite \
  --dataset-jsonl <DATA_DIR>/longbench_1024_lite.jsonl $OURS --run-id lb_ours \
  --output-response-jsonl outputs/aaai/qwen/folded_remote_unstaged/longbench/responses.jsonl \
  --output-report-json    outputs/aaai/qwen/folded_remote_unstaged/longbench/report.json
```

## 8. Monitor + resume

```bash
python scripts/monitor_aaai_run.py --status-json outputs/status/he_ours_unstaged.status.json
python scripts/monitor_h800_worker.py --gpu-worker-url <H800_WORKER_URL> --run-id qwen
# resume: re-run the SAME command (completed ids auto-skipped)
```

## 9. HumanEval pass@1 (sandboxed, CPU)

```bash
python scripts/evaluate_code_generation.py --dataset humaneval \
  --dataset-jsonl <DATA_DIR>/humaneval.jsonl \
  --plaintext-responses outputs/aaai/qwen/plaintext_local/humaneval/responses.jsonl \
  --ours-responses outputs/aaai/qwen/folded_remote_unstaged/humaneval/responses.jsonl \
  --timeout 10 --output-json outputs/aaai/qwen/code_eval_humaneval.json
# reports pass@1_plaintext / pass@1_ours / pass@1_delta + completion preservation.
```

## 10. SensitivePrompt leakage scan

```bash
python scripts/evaluate_sensitive_prompt_security.py \
  --dataset-jsonl <DATA_DIR>/sensitive_prompt_1024.jsonl \
  --response-jsonl outputs/aaai/qwen/folded_remote_unstaged/sensitive/responses.jsonl \
  --transcript-jsonl outputs/aaai/qwen/folded_remote_unstaged/sensitive/transcript.jsonl \
  --report-json outputs/aaai/qwen/folded_remote_unstaged/sensitive/report.json \
  --output-json outputs/aaai/qwen/sensitive_security_report.json \
  --output-md   outputs/aaai/qwen/sensitive_security_report.md
# leakage_pass MUST be true: no sensitive span / forbidden field on any
# GPU-visible channel. (The trusted-side final response may contain spans -- the
# GPU never sees it -- that is not a leak.)
```

## 11. LongBench-1024-lite — NOT an official LongBench score

The long-prompt run is **stress / scaling / security only**: inputs are capped to
1024 tokens. The report stamps `official_longbench_score=false` and
`reason="seq_len fixed to 1024; used for stress/scaling/security only"`. Do not
report it as a LongBench leaderboard number.

## 12. Staged vs unstaged latency + staged-schedule security

```bash
python scripts/compare_staged_vs_unstaged_latency.py \
  --unstaged-report outputs/aaai/qwen/folded_remote_unstaged/humaneval/report.json \
  --staged-report   outputs/aaai/qwen/folded_remote_staged/humaneval/report.json \
  --output-json outputs/aaai/qwen/staged_vs_unstaged_humaneval.json
```

**Staged-schedule security (engineering optimisation, NOT a model change):**
- `raw_input_protected = true`
- raw masks (`N`, `N_in`, `N_out`, `N_inv`) are **not** on the GPU
- raw pad / `T` are **not** on the GPU
- only **composed, non-secret / masked-basis** artifacts are staged:
  `xpad_tilde = T·N_in`, `cpad_tilde = T·W·N_out`, folded `*_tilde` refs, public
  commitments/shapes
- **sampling + logits recovery remain trusted-side**: `sampling_location =
  trusted_boundary`, `plaintext_logits_on_gpu = false`, `sampled_token_on_gpu =
  false`
- if the staged schedule is missing or fails the no-secret audit, the
  paper-facing run **fails** — there is no fallback to an unsafe path.

## 13. Cleanup

```bash
python scripts/cleanup_stale_experiments.py outputs            # dry-run
python scripts/cleanup_stale_experiments.py outputs --execute  # archive non-AAAI/stale
# also archives: amulet/secure_R/LoRA/pure-TEE backends under AAAI dirs,
# "claims staged but audit not passed", contains_raw_mask/pad, seq_len!=1024,
# max_new_tokens!=512, EOS disabled, dry_run, simulated_unsigned.
```

---

**Security summary:** the GPU never sees the raw prompt, token ids, plaintext
embedding/hidden/logits, raw mask/inverse, or raw pad — staged or not. The staged
schedule only moves NON-SECRET composed artifacts to the GPU to cut online
TEE<->GPU round-trips. This protects user INPUT, not only model weights.
