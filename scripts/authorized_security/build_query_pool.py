#!/usr/bin/env python3
"""Freeze the researcher-controlled public E2E query pool (no references)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-input", type=Path, required=True)
    ap.add_argument("--private-train-jsonl", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.generation_input.read_text())
    private_prompts = set()
    with args.private_train_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            private_prompts.add(tuple(row["input_ids"][: row["sup_start"]]))
    rows = []
    exact_overlap = 0
    for item in raw:
        prompt_ids = list(item["prompt_ids"])
        exact_overlap += tuple(prompt_ids) in private_prompts
        rows.append({
            "sample_id": f"e2e_nlg:test:{int(item['sample_id']):06d}",
            "task": "e2e_nlg", "split": "test",
            "query_text": item["meaning_representation"],
            "query_prompt_token_ids": prompt_ids,
        })
    if exact_overlap:
        raise RuntimeError(f"query pool has {exact_overlap} exact prompt overlaps with private train")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen query pool: {args.output}")
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    manifest = {
        "schema": "authorized_security_query_pool", "version": "1.0",
        "records": len(rows), "exact_private_train_prompt_overlap": exact_overlap,
        "source_generation_input_sha256": sha256(args.generation_input),
        "private_train_split_sha256": sha256(args.private_train_jsonl),
        "query_pool_sha256": sha256(args.output),
        "references_included": False,
    }
    mp = args.output.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
