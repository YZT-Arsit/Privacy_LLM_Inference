# Public benchmark preparation runbook

This runbook converts **local** public-benchmark files into the normalized JSONL
schema used by the task-utility runner (E9), plus a provenance dataset card.

> **No downloads.** You supply the local file already extracted from the public
> dataset. **Licenses are your responsibility** — each dataset has its own terms;
> this tooling never fetches data and records only sha256 provenance.

## Pipeline

```
local file (CSV/TSV/JSONL)
  -> scripts/prepare_public_benchmark_jsonl.py   (convert + validate + sample)
  -> normalized JSONL  +  dataset card JSON
  -> scripts/run_e9_task_utility_benchmark.py    (score; stub by default)
```

The prepare step validates **every** example against the schema
(`src/pllo/benchmarks/task_schemas.py`), shuffles deterministically with
`--seed`, takes the first `--max-examples`, and writes a dataset card with the
input and output file sha256.

## Recommended paper-facing subset sizes

| dataset            | name (`--dataset-name`) | task_type         | metric             | recommended N |
| ------------------ | ----------------------- | ----------------- | ------------------ | ------------- |
| MMLU               | `mmlu`                  | multiple_choice   | accuracy           | 200–500       |
| GSM8K              | `gsm8k`                 | generation_exact  | numeric_exact_match| 100–200       |
| BoolQ              | `boolq`                 | yes_no            | accuracy           | 200–500       |
| AG News            | `agnews`                | classification    | macro_f1 (+acc)    | 500–1000      |
| SST-2              | `sst2`                  | classification    | accuracy           | 500–1000      |
| C-Eval (optional)  | `ceval`                 | multiple_choice   | accuracy           | 200–500       |
| CMMLU (optional)   | `cmmlu`                 | multiple_choice   | accuracy           | 200–500       |
| CNN/DailyMail (opt)| `cnndm`                 | summarization     | rouge_l            | 100–500       |
| XSum (optional)    | `xsum`                  | summarization     | rouge_l            | 100–500       |

## Expected local input formats

- **MMLU / C-Eval / CMMLU** — CSV with header `question,A,B,C,D,answer`
  (`answer` is the letter `A`–`D`).
- **GSM8K** — JSONL objects `{"question": ..., "answer": ...}`
  (`answer` may contain a `#### 42` marker; the numeric is extracted).
- **BoolQ** — JSONL objects `{"passage": ..., "question": ..., "answer": ...}`
  (`answer` is a bool or `yes`/`no`).
- **AG News** — CSV with header `label,title,description`
  (`label` is `1`–`4` or the class name; text = `title + ". " + description`).
- **SST-2** — TSV or CSV with header `sentence,label` (`label` is `0`/`1`).
- **CNN/DailyMail / XSum** — JSONL objects `{"document": ..., "summary": ...}`.

## Example commands

MMLU (multiple choice, 200 examples):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/mmlu_test.csv \
  --dataset-name mmlu --split test \
  --max-examples 200 --seed 0 \
  --output-jsonl outputs/bench/mmlu_test.jsonl \
  --dataset-card-json outputs/bench/mmlu_test.card.json
```

GSM8K (numeric exact match, 100 examples):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/gsm8k_test.jsonl \
  --dataset-name gsm8k --split test \
  --max-examples 100 --seed 0 \
  --output-jsonl outputs/bench/gsm8k_test.jsonl \
  --dataset-card-json outputs/bench/gsm8k_test.card.json
```

BoolQ (yes/no, 300 examples):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/boolq_val.jsonl \
  --dataset-name boolq --split validation \
  --max-examples 300 --seed 0 \
  --output-jsonl outputs/bench/boolq_val.jsonl \
  --dataset-card-json outputs/bench/boolq_val.card.json
```

AG News (classification, macro-F1, 1000 examples):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/agnews_test.csv \
  --dataset-name agnews --split test \
  --max-examples 1000 --seed 0 \
  --output-jsonl outputs/bench/agnews_test.jsonl \
  --dataset-card-json outputs/bench/agnews_test.card.json
```

SST-2 (classification, accuracy, 872 examples):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/sst2_dev.tsv \
  --dataset-name sst2 --split validation \
  --max-examples 872 --seed 0 \
  --output-jsonl outputs/bench/sst2_dev.jsonl \
  --dataset-card-json outputs/bench/sst2_dev.card.json
```

C-Eval / CMMLU (optional, Chinese MCQ):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/ceval_val.csv \
  --dataset-name ceval --split val \
  --max-examples 300 --seed 0 \
  --output-jsonl outputs/bench/ceval_val.jsonl \
  --dataset-card-json outputs/bench/ceval_val.card.json
```

CNN/DailyMail or XSum (optional, summarization, ROUGE-L):

```bash
python scripts/prepare_public_benchmark_jsonl.py \
  --input-path data/cnndm_test.jsonl \
  --dataset-name cnndm --split test \
  --max-examples 200 --seed 0 \
  --output-jsonl outputs/bench/cnndm_test.jsonl \
  --dataset-card-json outputs/bench/cnndm_test.card.json
```

## Scoring (E9)

By default the runner uses a deterministic **stub** predictor (no model), so the
report is labeled `dry_run=true, paper_ready=false`. This is intended for CI and
for validating the pipeline on a laptop.

```bash
python scripts/run_e9_task_utility_benchmark.py \
  --dataset-jsonl outputs/bench/mmlu_test.jsonl \
  --backend plaintext_local --task-type multiple_choice \
  --output-json outputs/e9_mmlu.json \
  --output-md   outputs/e9_mmlu.md \
  --output-csv  outputs/e9_mmlu.csv
```

Backends: `plaintext_local`, `folded_remote`, `tdx_lite_remote`,
`tdx_attested_remote`, `folded_lora_remote`, `tdx_attested_folded_lora_remote`.

To produce a **paper-ready** (`paper_ready=true`) report you must supply a real
model path / GPU worker URL and the operator-provided predictor for the chosen
backend; otherwise the run falls back to the stub and stays `dry_run=true`.
```
