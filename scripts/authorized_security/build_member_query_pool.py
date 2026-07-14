#!/usr/bin/env python3
"""Build a deterministic prompt-only member query pool from a private split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite member pool: {args.output}")
    rows = torch.load(args.train_data, map_location="cpu", weights_only=False)
    if args.samples > len(rows):
        raise ValueError("requested more samples than the training split contains")
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[: args.samples]
    queries = []
    for index in selected:
        row = rows[index]
        sup_start = int(row["sup_start"])
        prompt_ids = [int(token) for token in row["input_ids"][:sup_start]]
        queries.append({"sample_id": int(row["sample_id"]), "prompt_ids": prompt_ids})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queries, sort_keys=True) + "\n")
    manifest = {
        "schema": "authorized_security_member_query_pool",
        "version": "1.0",
        "split": "train",
        "records": len(queries),
        "selection_seed": args.seed,
        "source_train_data_sha256": sha256(args.train_data),
        "output_sha256": sha256(args.output),
        "labels_or_targets_included": False,
        "selection_indices_sha256": hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
