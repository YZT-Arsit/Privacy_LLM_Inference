"""Convert a LOCAL public-benchmark file into normalized JSONL + a dataset card.

No internet / no downloads: you supply a local CSV/TSV/JSONL extracted from the
public dataset (license compliance is your responsibility). Every example is
validated against the normalized schema; sampling is deterministic (seeded).

Example::

    python scripts/prepare_public_benchmark_jsonl.py \\
        --input-path data/mmlu_test.csv \\
        --dataset-name mmlu --split test \\
        --max-examples 200 --seed 0 \\
        --output-jsonl outputs/bench/mmlu_test.jsonl \\
        --dataset-card-json outputs/bench/mmlu_test.card.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.public_dataset_converters import (  # noqa: E402
    CONVERTERS,
    build_dataset_card,
    deterministic_sample,
    sha256_file,
)
from pllo.benchmarks.task_schemas import assert_valid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--dataset-name", required=True, choices=sorted(CONVERTERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--dataset-card-json", default=None)
    args = ap.parse_args()

    in_path = Path(args.input_path)
    if not in_path.is_file():
        print("ERROR: input file not found: %s" % in_path, file=sys.stderr)
        return 2

    converter = CONVERTERS[args.dataset_name]
    try:
        examples = list(converter(in_path, split=args.split))
    except TypeError:
        examples = list(converter(in_path))

    for ex in examples:
        assert_valid(ex)

    sampled = deterministic_sample(
        examples, args.max_examples if args.max_examples > 0 else None,
        seed=args.seed)
    for ex in sampled:
        assert_valid(ex)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in sampled:
            fh.write(json.dumps(ex, ensure_ascii=False, default=str))
            fh.write("\n")

    task_type = sampled[0]["task_type"] if sampled else None
    metric = sampled[0]["metric"] if sampled else None
    card = build_dataset_card(
        source_name=args.dataset_name, split=args.split,
        sample_count=len(sampled), task_type=task_type, metric=metric,
        sampling_seed=args.seed,
        input_file_sha256=sha256_file(in_path),
        output_file_sha256=sha256_file(out_path))

    if args.dataset_card_json:
        cp = Path(args.dataset_card_json)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")

    print("=== prepare public benchmark JSONL ===")
    print("dataset=%s split=%s task_type=%s metric=%s"
          % (args.dataset_name, args.split, task_type, metric))
    print("examples_in=%d sampled=%d seed=%d"
          % (len(examples), len(sampled), args.seed))
    print("output_jsonl=%s" % out_path)
    print("input_sha256=%s" % card["input_file_sha256"])
    print("output_sha256=%s" % card["output_file_sha256"])
    print("\nJSONL WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
