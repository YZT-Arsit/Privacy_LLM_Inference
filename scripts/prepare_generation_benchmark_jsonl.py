"""Convert manually-staged raw generation datasets -> normalized JSONL + card.

NO DOWNLOADS. The user stages raw files under
``/root/autodl-tmp/datasets/privacy_llm_benchmarks/raw`` and writes converted
files under ``.../converted``. This script reads a local raw JSONL and emits the
normalized generation schema the generation benchmark consumes:

    {id, dataset_name, task_type, prompt, reference[, numeric_reference][, category]}

Supported ``--dataset``:
* ``gsm8k``  -- raw {question, answer}      -> task_type=generation_exact
* ``cnndm``  -- raw {article, highlights}   -> task_type=summarization
* ``xsum``   -- raw {document, summary}     -> task_type=summarization
* ``custom`` -- raw {id, prompt[, reference][, category]} -> task_type=open_ended

A provenance dataset card JSON (input/output sha256, counts, prompt template,
metric) is written next to the converted file (or ``--card-output``). Sampling is
deterministic (seeded shuffle). License compliance is the user's responsibility.

Example::

    python scripts/prepare_generation_benchmark_jsonl.py --dataset cnndm \\
        --input  $RAW/cnndm_test.jsonl \\
        --output $CONV/cnndm_gen.jsonl --max-examples 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_metrics import extract_number  # noqa: E402
from pllo.benchmarks.public_dataset_converters import (  # noqa: E402
    build_dataset_card,
    deterministic_sample,
    iter_jsonl,
    sha256_file,
)

# Deterministic, dataset-agnostic prompt templates (no model-specific tokens).
GSM8K_PROMPT = ("Solve the following math problem. Show your reasoning and end "
                "with the final numeric answer.\n\nProblem: {q}\nAnswer:")
SUMMARY_PROMPT = ("Summarize the following article in a few sentences.\n\n"
                  "Article:\n{doc}\n\nSummary:")

DATASETS = ("gsm8k", "cnndm", "xsum", "custom")
_TASK = {"gsm8k": "generation_exact", "cnndm": "summarization",
         "xsum": "summarization", "custom": "open_ended"}
_METRIC = {"gsm8k": "numeric_exact_match", "cnndm": "rougeL",
           "xsum": "rougeL", "custom": "normalized_edit_similarity"}


def _eid(dataset: str, split: str, i: int) -> str:
    return "%s-%s-%d" % (dataset, split, i)


def _convert(dataset: str, rows, split: str):
    if dataset == "gsm8k":
        for i, r in enumerate(rows):
            q = str(r.get("question", r.get("prompt", ""))).strip()
            ans = str(r.get("answer", r.get("reference", ""))).strip()
            yield {"id": r.get("id") or _eid("gsm8k", split, i),
                   "dataset_name": "gsm8k", "task_type": "generation_exact",
                   "prompt": GSM8K_PROMPT.format(q=q),
                   "reference": ans, "numeric_reference": extract_number(ans)}
    elif dataset in ("cnndm", "xsum"):
        doc_key = "article" if dataset == "cnndm" else "document"
        ref_key = "highlights" if dataset == "cnndm" else "summary"
        for i, r in enumerate(rows):
            doc = str(r.get(doc_key, r.get("document",
                                           r.get("article", "")))).strip()
            ref = str(r.get(ref_key, r.get("summary",
                                           r.get("highlights", "")))).strip()
            yield {"id": r.get("id") or _eid(dataset, split, i),
                   "dataset_name": dataset, "task_type": "summarization",
                   "prompt": SUMMARY_PROMPT.format(doc=doc), "reference": ref}
    else:  # custom
        for i, r in enumerate(rows):
            out = {"id": r.get("id") or _eid("custom", split, i),
                   "dataset_name": r.get("dataset_name", "custom"),
                   "task_type": "open_ended",
                   "prompt": str(r.get("prompt", "")).strip()}
            if r.get("reference") is not None:
                out["reference"] = str(r.get("reference"))
            if r.get("category") is not None:
                out["category"] = str(r.get("category"))
            yield out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--input", required=True, help="staged raw JSONL (no download)")
    ap.add_argument("--output", required=True, help="converted normalized JSONL")
    ap.add_argument("--card-output", default=None,
                    help="dataset card JSON (default: <output>.card.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print("ERROR: raw input not found: %s (stage it manually, no downloads)"
              % in_path, file=sys.stderr)
        return 3

    rows = list(iter_jsonl(in_path))
    examples = list(_convert(args.dataset, rows, args.split))
    examples = deterministic_sample(
        examples, args.max_examples if args.max_examples > 0 else None,
        seed=args.seed)
    # drop empty-prompt rows defensively (never silently emit bad examples)
    examples = [e for e in examples if str(e.get("prompt", "")).strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    card = build_dataset_card(
        source_name=args.dataset, split=args.split, sample_count=len(examples),
        task_type=_TASK[args.dataset], metric=_METRIC[args.dataset],
        sampling_seed=args.seed,
        input_file_sha256=sha256_file(in_path),
        output_file_sha256=sha256_file(out_path),
        dataset_name=args.dataset,
        prompt_template=(GSM8K_PROMPT if args.dataset == "gsm8k"
                         else SUMMARY_PROMPT if args.dataset in ("cnndm", "xsum")
                         else "<custom prompt passed through>"),
        converted_path=str(out_path), raw_path=str(in_path),
        no_downloads=True, benchmark_family="generation")
    card_path = Path(args.card_output) if args.card_output \
        else out_path.with_suffix(out_path.suffix + ".card.json")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print("=== prepared generation benchmark (%s) ===" % args.dataset)
    print("examples=%d task_type=%s metric=%s"
          % (len(examples), _TASK[args.dataset], _METRIC[args.dataset]))
    print("converted=%s" % out_path)
    print("card=%s" % card_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
