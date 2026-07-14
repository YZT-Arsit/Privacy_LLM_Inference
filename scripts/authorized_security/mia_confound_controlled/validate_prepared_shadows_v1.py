#!/usr/bin/env python3
"""CPU-only validator for prepared shadow bundles; performs no remote actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results = []
    states: dict[str, list[bool]] = {}
    for seed in (7, 1234, 2025):
        root = args.root / f"shadow_s{seed}"
        train, queries, metadata = rows(root / "train_members.jsonl"), rows(root / "evaluation_queries.jsonl"), rows(root / "evaluator_membership_metadata.jsonl")
        config = json.loads((root / "shadow_config.json").read_text())
        query_ids = [row["sample_id"] for row in queries]
        member_ids = {row["sample_id"] for row in train}
        labels = {row["sample_id"]: bool(row["member"]) for row in metadata}
        checks = {
            "train_members_500": len(train) == 500,
            "evaluation_queries_1000": len(queries) == 1000,
            "metadata_1000": len(metadata) == 1000,
            "query_ids_unique": len(set(query_ids)) == 1000,
            "labels_outside_feature_queries": all("member" not in row for row in queries),
            "member_ids_match_metadata": member_ids == {item for item, value in labels.items() if value},
            "eval_mode": config["C0_INFERENCE_ONLY"]["model_mode"] == "eval",
            "forward_only": config["C0_INFERENCE_ONLY"]["forward_only"] is True,
            "no_training_signals": all(config["C0_INFERENCE_ONLY"][key] is False for key in
                                       ("labels_exposed", "loss", "gradients", "dlogits", "optimizer_state")),
            "fresh_transform": config["C0_INFERENCE_ONLY"]["per_prompt_fresh_transform"] is True,
            "not_launchable": config["launch_allowed"] is False,
        }
        for sample_id, value in labels.items():
            states.setdefault(sample_id, []).append(value)
        results.append({"shadow_seed": seed, "passed": all(checks.values()), "checks": checks})
    cross = all(any(values) and not all(values) for values in states.values()) and len(states) == 1000
    report = {"schema": "prepared_shadow_validation", "version": "1.0",
              "all_passed": all(row["passed"] for row in results) and cross,
              "every_sample_crosses_membership_state": cross, "remote_actions": False,
              "shadows": results}
    output = args.root / "validation_report.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite validation report: {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise RuntimeError("prepared shadow validation failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
