#!/usr/bin/env python3
"""Create and fail-closed verify future confound-controlled shadow adapter manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


REQUIRED = ("adapter_sha256", "adapter_size_bytes", "shadow_seed", "pool_manifest_sha256",
            "membership_assignment_sha256", "base_package_root_hash", "tokenizer_sha256",
            "training_config_sha256", "target_set", "rank", "alpha", "steps", "dtype",
            "collection_mode", "model_mode")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(manifest: dict, expected: dict) -> list[str]:
    errors = []
    for field in REQUIRED:
        if field not in manifest:
            errors.append(f"missing:{field}")
    for field, value in expected.items():
        if manifest.get(field) != value:
            errors.append(f"mismatch:{field}")
    if manifest.get("collection_mode") != "C0_INFERENCE_ONLY":
        errors.append("forbidden_collection_mode")
    if manifest.get("model_mode") != "eval":
        errors.append("model_not_eval")
    if manifest.get("labels_in_feature_file") is not False:
        errors.append("labels_in_feature_file")
    if manifest.get("runtime_metadata_in_features") is not False:
        errors.append("runtime_metadata_in_features")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--adapter", type=Path, required=True)
    create.add_argument("--pool-manifest", type=Path, required=True)
    create.add_argument("--membership", type=Path, required=True)
    create.add_argument("--training-config", type=Path, required=True)
    create.add_argument("--context", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create":
        context = json.loads(args.context.read_text())
        manifest = {
            "schema": "confound_controlled_shadow_adapter_manifest", "version": "1.0",
            "adapter_sha256": sha256(args.adapter), "adapter_size_bytes": args.adapter.stat().st_size,
            "pool_manifest_sha256": sha256(args.pool_manifest),
            "membership_assignment_sha256": sha256(args.membership),
            "training_config_sha256": sha256(args.training_config), **context,
            "collection_mode": "C0_INFERENCE_ONLY", "model_mode": "eval",
            "labels_in_feature_file": False, "runtime_metadata_in_features": False,
        }
        errors = verify(manifest, manifest)
        if errors:
            raise RuntimeError(errors)
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite manifest: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        return

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite self-test directory: {args.output}")
    args.output.mkdir(parents=True)
    valid = {
        "adapter_sha256": "a" * 64, "adapter_size_bytes": 123, "shadow_seed": 7,
        "pool_manifest_sha256": "b" * 64, "membership_assignment_sha256": "c" * 64,
        "base_package_root_hash": "d" * 64, "tokenizer_sha256": "e" * 64,
        "training_config_sha256": "f" * 64,
        "target_set": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "rank": 8, "alpha": 16, "steps": 750, "dtype": "fp32",
        "collection_mode": "C0_INFERENCE_ONLY", "model_mode": "eval",
        "labels_in_feature_file": False, "runtime_metadata_in_features": False,
    }
    cases = []
    cases.append({"name": "valid", "expected_pass": True, "manifest": valid, "expected": valid})
    mutations = {
        "wrong_pool": ("pool_manifest_sha256", "0" * 64),
        "wrong_membership": ("membership_assignment_sha256", "0" * 64),
        "wrong_checkpoint": ("base_package_root_hash", "0" * 64),
        "wrong_shadow": ("shadow_seed", 1234),
        "training_mode": ("model_mode", "train"),
        "training_transcript_mixed": ("collection_mode", "C1_TRAINING_TRANSCRIPT"),
        "label_in_features": ("labels_in_feature_file", True),
        "runtime_metadata_in_features": ("runtime_metadata_in_features", True),
    }
    for name, (field, value) in mutations.items():
        changed = deepcopy(valid); changed[field] = value
        cases.append({"name": name, "expected_pass": False, "manifest": changed, "expected": valid})
    missing = deepcopy(valid); missing.pop("adapter_sha256")
    cases.append({"name": "missing_adapter_hash", "expected_pass": False,
                  "manifest": missing, "expected": valid})
    results = []
    for case in cases:
        errors = verify(case["manifest"], case["expected"])
        passed = not errors
        results.append({"name": case["name"], "expected_pass": case["expected_pass"],
                        "actual_pass": passed, "test_passed": passed == case["expected_pass"],
                        "errors": errors})
    report = {"schema": "adapter_manifest_negative_controls", "version": "1.0",
              "all_tests_passed": all(row["test_passed"] for row in results), "tests": results}
    (args.output / "adapter_manifest_negative_controls.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Adapter Manifest Negative Controls", "",
             "| Test | Expected accept | Actual accept | Harness pass |", "|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['name']} | {row['expected_pass']} | {row['actual_pass']} | {row['test_passed']} |")
    (args.output / "adapter_manifest_negative_controls.md").write_text("\n".join(lines) + "\n")
    if not report["all_tests_passed"]:
        raise RuntimeError("adapter manifest negative-control self-test failed")


if __name__ == "__main__":
    main()
