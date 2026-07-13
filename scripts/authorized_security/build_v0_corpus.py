#!/usr/bin/env python3
"""Strip evaluator-only generation fields into a strict V0 JSONL corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from view_contracts import View, ViewRecord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="-", help="raw evaluator JSONL or - for stdin")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--task", default="e2e_nlg")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    src = sys.stdin if args.input == "-" else open(args.input)
    records = []
    try:
        for line in src:
            raw = json.loads(line)
            record = ViewRecord.build(View.V0, {
                "sample_id": f"{args.task}:{args.split}:{int(raw['sample_id']):06d}",
                "run_id": args.run_id,
                "task": args.task,
                "split": args.split,
                "output_text": raw["generated_text"],
                "output_token_ids": raw["token_ids"],
            })
            records.append(record.export())
    finally:
        if src is not sys.stdin:
            src.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite V0 corpus: {args.output}")
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps({"view": "V0", "records": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
