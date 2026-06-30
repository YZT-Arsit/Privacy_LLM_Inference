# AAAI Runbook (DEPRECATED) — see AAAI_A_RIGHTMUL_QWEN7B_GENERATION_RUNBOOK.md

> **DEPRECATED / SUPERSEDED.** This file is kept only to avoid breaking links.
> The single canonical AAAI runbook is
> [`AAAI_A_RIGHTMUL_QWEN7B_GENERATION_RUNBOOK.md`](AAAI_A_RIGHTMUL_QWEN7B_GENERATION_RUNBOOK.md).
> It previously documented the legacy `run_generation_benchmark.py` /
> `run_ifeval_generation.py` pipeline with the `--paper-facing-generation` gate;
> that path is **debug-only** and is **not** an AAAI paper-facing source.

## Canonical AAAI mainline

The AAAI comparison is **plaintext-GPU vs ours (A_rightmul + linear boundary pad +
TDX boundary client + H800 folded worker)**. No LoRA, no `amulet_secure_R`, no
pure-TEE. Use ONLY these scripts, with the paper-facing flag **`--paper-facing-aaai`**:

| Step | Script |
|---|---|
| Generation (plaintext + ours) | `scripts/run_aaai_generation_benchmark.py` |
| Validation (resume-aware) | `scripts/validate_aaai_generation_results.py` |
| Full matrix (plan/run/validate) | `scripts/run_aaai_experiment_matrix.py` |
| Real TDX quote (Alibaba) | `scripts/generate_alibaba_tdx_quote_evidence.{sh,py}` |

See the canonical runbook §1–§12 for the end-to-end order (TDX `git pull` →
`rsync` to the offline H800 → cleanup dry-run → build → worker → tunnel → re-quote
→ ours → plaintext → validate). The full execution order, resume/monitor, and the
hard rules live there.

### Legacy runners (debug only — NOT AAAI paper-facing)

- `scripts/run_ifeval_generation.py` — single-dataset debug runner; its
  `--paper-facing-generation` flag is a legacy per-runner gate, not the AAAI gate.
- `scripts/run_generation_benchmark.py` — legacy adapter over the above.

Do not cite results from these for AAAI; use `run_aaai_generation_benchmark.py`
with `--paper-facing-aaai`.

### Staged backend

`folded_remote_staged` is **not** in the default matrix (the base runner still
performs the online remask/pad, so it is freshness-coverage only and its report
carries `do_not_use_as_latency_claim=true`). Run it only with
`run_aaai_experiment_matrix.py --include-staged` (experimental).

## Auxiliary evaluation (operate on the AAAI runner outputs)

These reuse the response JSONLs written by `run_aaai_generation_benchmark.py`
(`outputs/aaai/qwen/{plaintext_local,folded_remote}/<DS>/responses.jsonl`):

```bash
# utility preservation (GSM8K plaintext/ours/delta; MT-Bench turn1/2 + FastChat judge)
python scripts/evaluate_generation_preservation.py --dataset gsm8k \
  --dataset-jsonl <DATA_DIR>/gsm8k.jsonl \
  --plaintext-responses outputs/aaai/qwen/plaintext_local/gsm8k/responses.jsonl \
  --ours-responses      outputs/aaai/qwen/folded_remote/gsm8k/responses.jsonl \
  --output-json outputs/aaai/qwen/preservation/gsm8k_preservation.json

# sandboxed HumanEval pass@1 (CPU; never on H800)
python scripts/evaluate_humaneval_pass1.py --dataset-jsonl <DATA_DIR>/humaneval.jsonl \
  --plaintext-responses outputs/aaai/qwen/plaintext_local/humaneval/responses.jsonl \
  --ours-responses      outputs/aaai/qwen/folded_remote/humaneval/responses.jsonl \
  --output-json outputs/aaai/qwen/preservation/humaneval_pass1.json

# SensitivePrompt GPU-visible leakage scan (must pass) + headline compare
python scripts/evaluate_sensitive_prompt_security.py \
  --dataset-jsonl <DATA_DIR>/sensitive_prompt_1024.jsonl \
  --response-jsonl outputs/aaai/qwen/folded_remote/sensitive/responses.jsonl \
  --transcript-jsonl outputs/aaai/qwen/folded_remote/sensitive/transcript.jsonl \
  --report-json outputs/aaai/qwen/folded_remote/sensitive/report.json \
  --output-json outputs/aaai/qwen/sensitive_security_report.json
python scripts/compare_plaintext_vs_ours_aaai.py \
  --plaintext-dir outputs/aaai/qwen/plaintext_local \
  --ours-dir      outputs/aaai/qwen/folded_remote \
  --datasets ifeval gsm8k mt_bench humaneval sensitive_prompt_1024 \
  --attestation-evidence-json <EVIDENCE_JSON> \
  --output-dir outputs/aaai/qwen/main_results
```

For the code / sensitive / long-prompt stress tasks (HumanEval, MBPP,
SensitivePrompt-1024, LongBench-1024-lite) and the GPU-staged schedule, see
[`AAAI_OPTIONAL_CODE_SENSITIVE_LONGPROMPT_RUNBOOK.md`](AAAI_OPTIONAL_CODE_SENSITIVE_LONGPROMPT_RUNBOOK.md).

## Hard rules

- AAAI mainline = **plaintext_local vs folded_remote (unstaged) A_rightmul**; no
  LoRA, no `amulet_secure_R`, no pure-TEE.
- Paper-facing flag is **`--paper-facing-aaai`** (the AAAI gate).
- **Re-generate the TDX quote on every code change** — an old quote is stale.
- No passwords / SSH keys in any script, arg, or log.
- The GPU never sees the raw prompt / token ids / plaintext embedding / hidden /
  logits / raw mask `N`/`N_inv` / raw pad `T`; SensitivePrompt runs persist only a
  `prompt_sha256`, and failed records / error logs are sanitized.
