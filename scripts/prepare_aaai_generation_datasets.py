"""Prepare + localise the AAAI generation datasets (IFEval / GSM8K / MT-Bench).

Runs on the TDX guest or a workstation (which CAN reach GitHub/HF mirrors), reads
RAW local files (already downloaded -- this script never downloads), normalises
each dataset to a unified ``{id, dataset, prompt, ...}`` JSONL, and writes a
reproducibility CARD with sha256s. The normalised JSONL + cards are then
``scp``/``rsync``'d to the H800 (which has NO internet) for the real generation.

No network access is performed here. All inputs are local paths.

Outputs (default under ``--output-dir``):
  ifeval.jsonl  gsm8k.jsonl  mt_bench.jsonl
  cards/ifeval_card.json  cards/gsm8k_card.json  cards/mt_bench_card.json

Examples::

    python scripts/prepare_aaai_generation_datasets.py --dataset ifeval \\
        --input <RAW_IFEVAL> --output-dir /root/autodl-tmp/datasets/.../aaai
    python scripts/prepare_aaai_generation_datasets.py --dataset gsm8k \\
        --input <RAW_GSM8K> --gsm8k-prompt-style zero_shot --output-dir <DIR>
    python scripts/prepare_aaai_generation_datasets.py --dataset all \\
        --ifeval-input <...> --gsm8k-input <...> --mt-bench-input <...> \\
        --output-dir <DIR>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_datasets import (  # noqa: E402
    build_dataset_card, load_gsm8k, load_ifeval, load_mt_bench)


def _write_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prepare_one(dataset, raw_input, out_dir, *, split, max_examples,
                 gsm8k_prompt_style) -> dict:
    out_dir = Path(out_dir)
    cards = out_dir / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    if dataset == "ifeval":
        rows = load_ifeval(raw_input)
    elif dataset == "gsm8k":
        rows = load_gsm8k(raw_input, prompt_style=gsm8k_prompt_style)
    elif dataset == "mt_bench":
        rows = load_mt_bench(raw_input)
    else:
        raise ValueError("unknown dataset %r" % dataset)
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]
    out_jsonl = out_dir / ("%s.jsonl" % dataset)
    _write_jsonl(rows, out_jsonl)
    card = build_dataset_card(dataset_name=dataset, split=split, rows=rows,
                              source_path=raw_input, output_path=out_jsonl)
    card_path = cards / ("%s_card.json" % dataset)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("[prep] %s: %d examples -> %s (card %s)"
          % (dataset, len(rows), out_jsonl, card_path))
    return {"dataset": dataset, "num_examples": len(rows),
            "output_jsonl": str(out_jsonl), "card": str(card_path),
            "card_data": card}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=["ifeval", "gsm8k", "mt_bench", "all"])
    ap.add_argument("--input", default=None, help="raw input for a single dataset")
    ap.add_argument("--ifeval-input", default=None)
    ap.add_argument("--gsm8k-input", default=None)
    ap.add_argument("--mt-bench-input", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--gsm8k-prompt-style", default="zero_shot",
                    choices=["zero_shot", "few_shot_cot"])
    ap.add_argument("--output-json", default=None,
                    help="optional summary JSON of what was prepared")
    args = ap.parse_args()

    results = []
    if args.dataset == "all":
        plan = [("ifeval", args.ifeval_input), ("gsm8k", args.gsm8k_input),
                ("mt_bench", args.mt_bench_input)]
        for ds, raw in plan:
            if not raw:
                print("WARNING: --dataset all but no --%s-input; skipping %s"
                      % (ds.replace("_", "-"), ds), file=sys.stderr)
                continue
            results.append(_prepare_one(
                ds, raw, args.output_dir, split=args.split,
                max_examples=args.max_examples,
                gsm8k_prompt_style=args.gsm8k_prompt_style))
    else:
        raw = args.input
        if raw is None:
            print("ERROR: --input is required for a single dataset",
                  file=sys.stderr)
            return 3
        results.append(_prepare_one(
            args.dataset, raw, args.output_dir, split=args.split,
            max_examples=args.max_examples,
            gsm8k_prompt_style=args.gsm8k_prompt_style))

    if not results:
        print("ERROR: nothing prepared (no inputs provided)", file=sys.stderr)
        return 3
    summary = {"stage": "prepare_aaai_generation_datasets",
               "output_dir": str(args.output_dir), "prepared": results}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
