#!/usr/bin/env python3
"""Prepare, but never launch, three same-pool shadow-run input bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = (7, 1234, 2025)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite prepared shadow inputs: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    pool = [json.loads(line) for line in (args.pool / "candidate_pool.jsonl").read_text().splitlines() if line.strip()]
    assignments = {row["sample_id"]: row["membership_by_shadow"] for row in
                   (json.loads(line) for line in (args.pool / "membership_assignments.jsonl").read_text().splitlines() if line.strip())}
    if len(pool) != 1000 or set(assignments) != {row["sample_id"] for row in pool}:
        raise RuntimeError("pool or assignment cardinality mismatch")
    plan = []
    for seed in SEEDS:
        root = args.output / f"shadow_s{seed}"
        root.mkdir()
        members = [row for row in pool if assignments[row["sample_id"]][str(seed)] == 1]
        if len(members) != 500:
            raise RuntimeError(f"shadow {seed} has {len(members)} members")
        write_jsonl(root / "train_members.jsonl", [{"sample_id": row["sample_id"],
                    "input_ids": row["prompt_ids"] + row["target_ids"],
                    "sup_start": len(row["prompt_ids"])} for row in members])
        write_jsonl(root / "evaluation_queries.jsonl", [{"sample_id": row["sample_id"],
                    "prompt_ids": row["prompt_ids"]} for row in pool])
        write_jsonl(root / "evaluator_membership_metadata.jsonl", [{"sample_id": row["sample_id"],
                    "member": bool(assignments[row["sample_id"]][str(seed)])} for row in pool])
        config = {
            "schema": "confound_controlled_shadow_config", "version": "1.0",
            "status": "PREPARED_NOT_LAUNCHED", "shadow_seed": seed,
            "pool_manifest_sha256": sha256(args.pool / "candidate_pool_manifest.json"),
            "membership_sha256": sha256(args.pool / f"shadow_s{seed}_membership.jsonl"),
            "model": "Qwen2.5-0.5B", "base_package_root_hash": "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
            "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "training": {"targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         "rank": 8, "alpha": 16, "lr": 0.0002, "steps": 750,
                         "batch_size": 16, "dtype": "fp32", "members": 500},
            "C0_INFERENCE_ONLY": {"model_mode": "eval", "forward_only": True, "samples": 1000,
                                  "labels_exposed": False, "loss": False, "gradients": False,
                                  "dlogits": False, "optimizer_state": False,
                                  "per_prompt_fresh_transform": True, "dtype": "fp32",
                                  "same_batch_padding_tokenizer_extractor": True},
            "V0": {"same_samples_and_shadow_model": True, "generated_text": True,
                   "final_label": True, "probabilities": False},
            "launch_allowed": False,
        }
        (root / "shadow_config.json").write_text(json.dumps(config, indent=2) + "\n")
        hashes = [f"{sha256(path)}  {path.name}" for path in sorted(root.iterdir()) if path.name != "hashes.sha256"]
        (root / "hashes.sha256").write_text("\n".join(hashes) + "\n")
        plan.append({"shadow_seed": seed, "status": "PREPARED_NOT_LAUNCHED",
                     "members": 500, "evaluation_samples": 1000, "directory": str(root)})
    (args.output / "prepared_shadow_plan.json").write_text(json.dumps({
        "schema": "prepared_confound_controlled_shadow_plan", "version": "1.0",
        "role": "OFFLINE_MIA_ANALYSIS_ONLY", "jobs_launched": False,
        "estimated_gpu_tdx_hours": "17-18 total for three shadows", "shadows": plan,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
