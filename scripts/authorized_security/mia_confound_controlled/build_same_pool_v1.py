#!/usr/bin/env python3
"""Build a deterministic 1,000-sample same-source E2E membership pool, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


HEADER = "User:\nGenerate a natural-language description for the following restaurant attributes:\n"
TRAILER = "\n\nAssistant:"
SHADOWS = (7, 1234, 2025)
PATTERNS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))


def digest(value) -> str:
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def parse_mr(prompt: str) -> tuple[str, list[str]]:
    text = prompt
    if text.startswith(HEADER):
        text = text[len(HEADER):]
    if TRAILER in text:
        text = text.split(TRAILER, 1)[0]
    text = " ".join(text.split())
    fields = [match.strip() for match in re.findall(r"(?:^|,)\s*([^\[,]+)\s*\[", text)]
    return text, sorted(set(fields))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite pool directory: {args.output}")
    args.output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source_rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]

    groups: dict[str, list[dict]] = defaultdict(list)
    decoded: dict[str, tuple[str, list[str], str]] = {}
    for row in source_rows:
        prompt_ids = row["input_ids"][:int(row["sup_start"])]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        mr, fields = parse_mr(prompt)
        normalized_mr = normalize(mr)
        group_id = digest(normalized_mr)
        groups[group_id].append(row)
        decoded[group_id] = (mr, fields, normalized_mr)
    if len(groups) < args.pool_size:
        raise RuntimeError(f"only {len(groups)} unique MR groups for pool size {args.pool_size}")

    ordered_groups = sorted(groups, key=lambda key: digest(f"{args.selection_seed}:{key}"))[:args.pool_size]
    pool: list[dict] = []
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    semantic_groups: dict[str, list[str]] = defaultdict(list)
    for index, group_id in enumerate(ordered_groups):
        candidates = groups[group_id]
        chosen = min(candidates, key=lambda row: digest(row["input_ids"][int(row["sup_start"]):]))
        split = int(chosen["sup_start"])
        prompt_ids, target_ids = chosen["input_ids"][:split], chosen["input_ids"][split:]
        mr, fields, normalized_mr = decoded[group_id]
        sample_id = f"ccmia_{index:04d}"
        reference_hash = digest(target_ids)
        duplicate_id = digest({"input": normalized_mr, "target": target_ids})
        semantic_id = f"mr:{group_id}"
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        lexical_tokens = re.findall(r"\w+", normalized_mr)
        record = {
            "sample_id": sample_id,
            "source_population": "official_e2e_train_derived_frozen_subset",
            "original_sample_id": chosen["sample_id"],
            "prompt_ids": prompt_ids, "target_ids": target_ids,
            "normalized_input_hash": digest(normalized_mr), "reference_hash": reference_hash,
            "input_token_length": len(prompt_ids), "target_token_length": len(target_ids),
            "mr_field_set": fields, "mr_field_count": len(fields),
            "lexical_token_count": len(lexical_tokens),
            "lexical_unique_count": len(set(lexical_tokens)),
            "lexical_character_count": len(normalized_mr),
            "duplicate_group_id": f"exact:{duplicate_id}",
            "semantic_group_id": semantic_id,
            "meaning_representation": mr, "reference_text": target_text,
        }
        pool.append(record)
        duplicate_groups[record["duplicate_group_id"]].append(sample_id)
        semantic_groups[semantic_id].append(sample_id)

    rng = random.Random(args.selection_seed + 1)
    shuffled = list(range(args.pool_size))
    rng.shuffle(shuffled)
    assignments = {row["sample_id"]: {} for row in pool}
    for pair_start in range(0, len(shuffled), 2):
        left, right = shuffled[pair_start:pair_start + 2]
        pattern = PATTERNS[rng.randrange(len(PATTERNS))]
        complement = tuple(1 - item for item in pattern)
        for shadow, value in zip(SHADOWS, pattern):
            assignments[pool[left]["sample_id"]][str(shadow)] = value
        for shadow, value in zip(SHADOWS, complement):
            assignments[pool[right]["sample_id"]][str(shadow)] = value
    assignment_rows = [{"sample_id": sample_id, "membership_by_shadow": values}
                       for sample_id, values in assignments.items()]
    for shadow in SHADOWS:
        if sum(row["membership_by_shadow"][str(shadow)] for row in assignment_rows) != args.pool_size // 2:
            raise RuntimeError(f"shadow {shadow} assignment is not balanced")
    if any(sum(row["membership_by_shadow"].values()) not in (1, 2) for row in assignment_rows):
        raise RuntimeError("every sample must cross membership states across three shadows")

    write_jsonl(args.output / "candidate_pool.jsonl", pool)
    write_jsonl(args.output / "membership_assignments.jsonl", assignment_rows)
    for shadow in SHADOWS:
        rows = [{"sample_id": row["sample_id"], "member": bool(row["membership_by_shadow"][str(shadow)])}
                for row in assignment_rows]
        write_jsonl(args.output / f"shadow_s{shadow}_membership.jsonl", rows)
    (args.output / "duplicate_groups.json").write_text(json.dumps(duplicate_groups, indent=2) + "\n")
    (args.output / "semantic_groups.json").write_text(json.dumps(semantic_groups, indent=2) + "\n")
    manifest = {
        "schema": "confound_controlled_same_pool", "version": "1.0",
        "source": str(args.source), "source_sha256": digest(args.source.read_bytes()),
        "tokenizer": str(args.tokenizer), "selection_seed": args.selection_seed,
        "source_rows": len(source_rows), "source_unique_mr_groups": len(groups),
        "pool_size": len(pool), "unique_normalized_inputs": len({row["normalized_input_hash"] for row in pool}),
        "unique_references": len({row["reference_hash"] for row in pool}),
        "shadow_seeds": list(SHADOWS), "members_per_shadow": args.pool_size // 2,
        "membership_constraint": "each sample is member in 1 or 2 of 3 shadows; every shadow is 500/500",
        "official_split_used": "train only", "membership_defined_by_official_split": False,
    }
    (args.output / "candidate_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = []
    for path in sorted(args.output.iterdir()):
        if path.name != "pool_hashes.sha256":
            hashes.append(f"{digest(path.read_bytes())}  {path.name}")
    (args.output / "pool_hashes.sha256").write_text("\n".join(hashes) + "\n")


if __name__ == "__main__":
    main()
