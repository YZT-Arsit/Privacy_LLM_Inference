# Generation benchmark runbook (privacy-preserving autoregressive generation)

The paper's core claim is privacy-preserving **autoregressive generation**, so the
short-output E9 matrix (MMLU / BoolQ / AG News stay as short-output utility
preservation) is augmented here with **generation-focused** benchmarks:

- **GSM8K** with longer decoding budgets (e.g. `--max-new-tokens 64` and `128`),
  scored by `numeric_exact_match` (+ `extracted_number`).
- **CNN/DailyMail** and **XSum** small summarization subsets, scored by
  `rouge1 / rouge2 / rougeL` (`rouge_score` if importable, else a pure-Python
  LCS / n-gram fallback flagged with `rouge_unavailable=true` — never downloaded).
- Optional **custom open-ended generation** JSONL, scored by `exact_text_match` +
  `normalized_edit_similarity` vs a reference (when present).

**Scope / honesty rails (enforced in code):**

- **CURRENT design only.** `--nonlinear-backend current`; `trusted_shortcut` is
  refused (exit 3) until it is paper-facing in the real path.
- **No downloads.** Datasets are staged manually under
  `/root/autodl-tmp/datasets/privacy_llm_benchmarks/raw` and converted under
  `.../converted`. `rouge_score` is only used if already importable.
- **No dry-run / fixture / tiny data in paper-ready outputs.** Use `--require-real`;
  stub runs are labeled `dry_run=true, paper_ready=false`.
- **No LLM judge, no subjective quality scoring** — every metric is an objective
  string/token comparison.
- Plaintext generation passes the **attention mask**.
- Every report carries `max_new_tokens`, `seq_len`, `model_name`, `backend`,
  `nonlinear_backend`, `paper_ready`, `dry_run`, `audit_passed`; outputs are
  JSON + CSV + MD.

```
RAW=/root/autodl-tmp/datasets/privacy_llm_benchmarks/raw
CONV=/root/autodl-tmp/datasets/privacy_llm_benchmarks/converted
OUT=outputs/generation
MODEL=/root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct
URL=http://127.0.0.1:18080          # folded GPU worker (started with --nonlinear-backend current)
ART=/root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current
mkdir -p $OUT
```

## 0. Convert staged raw data -> normalized generation JSONL (+ dataset cards)

No downloads — point `--input` at the manually staged raw file.

```
# GSM8K raw {question, answer}
python scripts/prepare_generation_benchmark_jsonl.py --dataset gsm8k \
  --input $RAW/gsm8k_test.jsonl --output $CONV/gsm8k_gen.jsonl \
  --max-examples 200 --seed 0

# CNN/DailyMail raw {article, highlights}
python scripts/prepare_generation_benchmark_jsonl.py --dataset cnndm \
  --input $RAW/cnndm_test.jsonl --output $CONV/cnndm_gen.jsonl \
  --max-examples 100 --seed 0

# XSum raw {document, summary}
python scripts/prepare_generation_benchmark_jsonl.py --dataset xsum \
  --input $RAW/xsum_test.jsonl --output $CONV/xsum_gen.jsonl \
  --max-examples 100 --seed 0

# Optional custom open-ended {id, prompt[, reference][, category]}
python scripts/prepare_generation_benchmark_jsonl.py --dataset custom \
  --input $RAW/custom_opengen.jsonl --output $CONV/custom_gen.jsonl
```

Each writes `<output>.card.json` (input/output sha256, counts, prompt template,
task_type, metric, `no_downloads=true`).

## 1. Start the folded GPU worker (CURRENT design)

The folded path reuses the existing real predictor / boundary path; start the
worker exactly as in the dual-nonlinear runbook, **with `--nonlinear-backend
current`** (the protocol is unchanged):

```
python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \
  --gpu-backend qwen7b_folded_package --nonlinear-backend current \
  --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current \
  --device cuda --dtype bfloat16 --listen-port 18080 --audit true
curl -fsS $URL/health && echo
```

## 2. GSM8K-128 — plaintext and folded current

Run GSM8K at both budgets (64 and 128). Example for 128:

```
python scripts/run_generation_utility_benchmark.py \
  --dataset-jsonl $CONV/gsm8k_gen.jsonl --backend plaintext_local \
  --model-path $MODEL --require-real --seq-len 1024 --max-new-tokens 128 \
  --output-json $OUT/gsm8k128_plaintext.json --output-csv $OUT/gsm8k128_plaintext.csv \
  --output-md $OUT/gsm8k128_plaintext.md

python scripts/run_generation_utility_benchmark.py \
  --dataset-jsonl $CONV/gsm8k_gen.jsonl --backend folded_remote \
  --nonlinear-backend current --model-path $MODEL --gpu-worker-url $URL \
  --embedding-path $ART --require-real --seq-len 1024 --max-new-tokens 128 \
  --output-json $OUT/gsm8k128_folded.json --output-csv $OUT/gsm8k128_folded.csv \
  --output-md $OUT/gsm8k128_folded.md
```

For the 64-token budget repeat with `--max-new-tokens 64` and `gsm8k64_*` names.

## 3. CNN/DailyMail (small) — plaintext and folded current

```
python scripts/run_generation_utility_benchmark.py \
  --dataset-jsonl $CONV/cnndm_gen.jsonl --backend plaintext_local \
  --model-path $MODEL --require-real --seq-len 1536 --max-new-tokens 128 \
  --output-json $OUT/cnndm_plaintext.json --output-csv $OUT/cnndm_plaintext.csv \
  --output-md $OUT/cnndm_plaintext.md

python scripts/run_generation_utility_benchmark.py \
  --dataset-jsonl $CONV/cnndm_gen.jsonl --backend folded_remote \
  --nonlinear-backend current --model-path $MODEL --gpu-worker-url $URL \
  --embedding-path $ART --require-real --seq-len 1536 --max-new-tokens 128 \
  --output-json $OUT/cnndm_folded.json --output-csv $OUT/cnndm_folded.csv \
  --output-md $OUT/cnndm_folded.md
```

(XSum is identical with `$CONV/xsum_gen.jsonl` and `xsum_*` names; the custom
open-ended set with `$CONV/custom_gen.jsonl`.)

## 4. Pairwise preservation reports (plaintext vs folded current)

```
python scripts/run_generation_pairwise_preservation.py \
  --baseline-json $OUT/gsm8k128_plaintext.json \
  --candidate-json $OUT/gsm8k128_folded.json \
  --output-json $OUT/gsm8k128_pairwise.json --output-csv $OUT/gsm8k128_pairwise.csv \
  --output-md $OUT/gsm8k128_pairwise.md

python scripts/run_generation_pairwise_preservation.py \
  --baseline-json $OUT/cnndm_plaintext.json \
  --candidate-json $OUT/cnndm_folded.json \
  --output-json $OUT/cnndm_pairwise.json --output-md $OUT/cnndm_pairwise.md
```

Each pairwise report carries `metric_abs_drop`, `metric_rel_drop`,
`exact_output_match_rate`, `length_delta_mean`, `latency_ratio`, `audit_passed`,
and `utility_preserved` (drop within thresholds **and** both inputs real
**and** the candidate audit not failed).

## 5. Final summary table

Combine the per-dataset pairwise reports into one table:

```
python scripts/run_generation_pairwise_preservation.py \
  --summary-input $OUT/gsm8k64_pairwise.json \
  --summary-input $OUT/gsm8k128_pairwise.json \
  --summary-input $OUT/cnndm_pairwise.json \
  --summary-input $OUT/xsum_pairwise.json \
  --output-json $OUT/generation_summary.json \
  --output-csv $OUT/generation_summary.csv \
  --output-md  $OUT/generation_summary.md
```

`generation_summary.md` is the paper-facing table: one row per dataset/budget with
the metric drop, exact-output rate, length delta, latency ratio, audit, and the
`utility_preserved` decision, plus `all_utility_preserved` / `all_paper_ready`.

## Artifact invalidation

- The folded **packages** and existing **current** decode / E3 / E9 artifacts are
  NOT affected (no protocol change; no tensor-format change). These generation
  benchmark outputs are NEW files.
- The only existing outputs that should be regenerated are **older plaintext E9
  generation outputs produced WITHOUT an attention mask** (if any are still
  present) — re-run them through these scripts so the plaintext baseline is
  correct. `trusted_shortcut` is not paper-facing here, so nothing trusted_shortcut
  needs regenerating for this benchmark.
