"""Build the synthetic SensitivePrompt-1024 stress set (no real PII).

Generates deterministic synthetic privacy-stress prompts (128/512/1024-token
buckets) with fabricated ``sensitive_spans`` for the GPU-visible leakage scan.
Runs anywhere (no network); the output JSONL + card are synced to the H800.

Example::

    python scripts/build_sensitive_prompt_stress_set.py \\
      --num-per-bucket 20 --output-jsonl <DIR>/sensitive_prompt_1024.jsonl \\
      --card-json <DIR>/cards/sensitive_prompt_1024_card.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_datasets import build_dataset_card  # noqa: E402
from pllo.benchmarks.sensitive_prompts import (  # noqa: E402
    LENGTH_BUCKETS, build_sensitive_prompt_set)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-per-bucket", type=int, default=10)
    ap.add_argument("--buckets", default="128,512,1024")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--card-json", default=None)
    args = ap.parse_args()

    buckets = tuple(int(b) for b in args.buckets.split(",") if b.strip())
    rows = build_sensitive_prompt_set(num_per_bucket=args.num_per_bucket,
                                      buckets=buckets or LENGTH_BUCKETS,
                                      seed=args.seed)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[sensitive] wrote %d prompts -> %s" % (len(rows), out))
    if args.card_json:
        card = build_dataset_card(dataset_name="sensitive_prompt_1024",
                                  split="synthetic", rows=rows,
                                  source_path="(synthetic)", output_path=out)
        card["synthetic"] = True
        card["contains_real_pii"] = False
        card["buckets"] = list(buckets)
        Path(args.card_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.card_json).write_text(json.dumps(card, indent=2),
                                        encoding="utf-8")
        print("[sensitive] card -> %s" % args.card_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
