#!/usr/bin/env python3
"""Append-only, read-only ingestion of terminal T1--T4 protected bundles."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from canonical_io import BUNDLE, REPO, sha256

GL = REPO / "results/aaai_private_base/generative_lora"
RUNS = GL / "protected_runs"
GENERATIONS = GL / "generations"
METRICS = GL / "extension/metrics"
CELLS = {
    "T1": ("e2e_T1_qv_s1234_protected", ("q_proj", "v_proj"), "t1_qv_s1234_protected"),
    "T2": ("e2e_T2_qkvo_s1234_protected", ("q_proj", "k_proj", "v_proj", "o_proj"), "t2_qkvo_s1234_protected"),
    "T3": ("e2e_T3_mlp_s1234_protected", ("gate_proj", "up_proj", "down_proj"), "t3_mlp_s1234_protected"),
    "T4": ("e2e_T4_all7_s1234_protected", ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), "t4_all7_s1234_protected"),
}
TRUSTED_A = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
TRUSTED_B = {"q_proj", "k_proj"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tensor_inventory(path: Path, targets: tuple[str, ...]) -> dict:
    adapter = torch.load(path, map_location="cpu", weights_only=True)
    total = trusted = gpu = factors = trusted_a = trusted_b = 0
    for key, value in adapter.items():
        projection = key.split(".", 1)[1]
        if projection not in targets or not isinstance(value, tuple) or len(value) != 2:
            raise RuntimeError(f"unexpected adapter entry {key}")
        a, b = value
        total += a.numel() + b.numel(); factors += 2
        if projection in TRUSTED_A:
            trusted += a.numel(); trusted_a += 1
        else:
            gpu += a.numel()
        if projection in TRUSTED_B:
            trusted += b.numel(); trusted_b += 1
        else:
            gpu += b.numel()
    if total != trusted + gpu:
        raise RuntimeError("factor ownership accounting mismatch")
    return {"trainable_parameter_count": total, "total_factor_count": factors,
            "trusted_A_factor_count": trusted_a, "trusted_B_factor_count": trusted_b,
            "gpu_local_factor_count": factors - trusted_a - trusted_b,
            "trusted_parameter_count": trusted, "gpu_local_parameter_count": gpu,
            "tdx_optimizer_tensor_state_bytes": trusted * 3 * 4,
            "projected_raw_tensor_bytes_each_direction_per_step": trusted * 4}


def inspect_cell(cell: str, tag: str, targets: tuple[str, ...], metric_dir: str) -> dict:
    paths = {
        "completion": METRICS / metric_dir / "completion_validation.json",
        "metrics": METRICS / metric_dir / "aggregate_metrics.csv",
        "result": RUNS / f"{tag}.result.json",
        "training": RUNS / f"{tag}.json",
        "adapter": RUNS / f"{tag}.adapter.pt",
        "train_counters": RUNS / f"{tag}.training_tdx_counters.json",
        "generation_counters": RUNS / f"{tag}.generation_tdx_counters.json",
        "attestation": RUNS / f"{tag}.attestation.json",
        "session": RUNS / f"{tag}.tdx_session.json",
        "generation": GENERATIONS / f"g2_protected_{tag}.jsonl",
        "generation_profile": GENERATIONS / f"g2_protected_{tag}.jsonl.profile.json",
    }
    missing = [str(path.relative_to(REPO)) for path in paths.values() if not path.is_file()]
    base = {"cell": cell, "run_tag": tag, "target_set": list(targets),
            "status": "PENDING", "terminal": False, "missing_required_paths": missing}
    if missing:
        return base
    completion, result, training = load_json(paths["completion"]), load_json(paths["result"]), load_json(paths["training"])
    rows = sum(1 for line in paths["generation"].open() if line.strip())
    terminal_checks = {
        "completion_steps_750": completion.get("steps_run") == 750,
        "completion_generation_500": completion.get("generation_records") == 500,
        "completion_unique_ids_500": completion.get("unique_sample_ids") == 500,
        "completion_all_finite": completion.get("all_steps_finite") is True,
        "completion_attested": completion.get("attestation_verified") is True,
        "result_completed": result.get("completed") is True,
        "result_steps_750": result.get("max_steps") == 750,
        "training_steps_750": training.get("steps_run") == 750,
        "generation_rows_500": rows == 500,
        "target_set_exact": tuple(training.get("lora_targets", ())) == targets,
    }
    if not all(terminal_checks.values()):
        return {**base, "terminal_checks": terminal_checks}
    for relative, expected in completion.get("file_hashes", {}).items():
        path = REPO / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"terminal hash mismatch: {relative}")
    with paths["metrics"].open() as handle:
        quality = next(csv.DictReader(handle))
    profile, attestation, session = load_json(paths["generation_profile"]), load_json(paths["attestation"]), load_json(paths["session"])
    inventory = tensor_inventory(paths["adapter"], targets)
    trajectory = training["trajectory"]
    mean_total_step = sum(float(row["wall"]) for row in trajectory) / len(trajectory)
    mean_roundtrip = sum(float(row["net_ce"]) + float(row["net_adamw"]) for row in trajectory) / len(trajectory)
    binding = session["binding_manifest"]
    return {**base, "status": "TERMINAL", "terminal": True, "missing_required_paths": [],
            "terminal_checks": terminal_checks,
            "quality": {key: quality.get(key) for key in ("BLEU", "chrF", "ROUGE_L", "invalid_rate_pct", "repetition_pct", "avg_len", "n")},
            **inventory, "adapter_bytes": paths["adapter"].stat().st_size,
            "trusted_roundtrip_time_sec_per_step": mean_roundtrip,
            "trusted_compute_time_sec_per_step": None,
            "transport_only_time_sec_per_step": None,
            "total_step_time_sec": mean_total_step,
            "protected_generation_tokens_per_sec": profile.get("tokens_per_sec"),
            "attestation_status": "PASS" if attestation.get("attestation_verified") and attestation.get("overall_appraisal_result") == "SUCCESS" else "FAIL",
            "source_revision": binding.get("source_revision"),
            "source_bundle_hash": binding.get("code_revision_or_worktree_hash"),
            "source_hashes": binding.get("source_hashes", {}),
            "config_hashes": {key: session.get(key) for key in ("package_root_hash", "model_config_hash", "service_hash", "template_hash", "tokenizer_hash", "label_schema_hash")},
            "artifact_hashes": {name: sha256(path) for name, path in paths.items()},
            "provenance": {
                "quality": "MEASURED", "trainable_parameter_count": "MEASURED",
                "adapter_bytes": "MEASURED", "factor_counts": "MEASURED",
                "tdx_optimizer_tensor_state_bytes": "PROJECTED",
                "projected_raw_tensor_bytes_each_direction_per_step": "PROJECTED",
                "trusted_roundtrip_time_sec_per_step": "MEASURED",
                "trusted_compute_time_sec_per_step": "MISSING", "transport_only_time_sec_per_step": "MISSING",
                "total_step_time_sec": "MEASURED", "protected_generation_tokens_per_sec": "MEASURED",
                "attestation_status": "MEASURED"
            },
            "notes": ["Raw tensor-byte projection excludes serialization, framing, and authentication overhead.",
                      "This run did not retain the profiling split needed to separate trusted compute from transport; both remain MISSING."]}


def main() -> None:
    events = [inspect_cell(cell, *spec) for cell, spec in CELLS.items()]
    stamp = datetime.now(timezone.utc).isoformat()
    ledger = BUNDLE / "target_ablation_ingestion_v2.jsonl"
    previous = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                old = json.loads(line); old.pop("observed_at", None); previous[old["cell"]] = old
    with ledger.open("a") as handle:
        for event in events:
            if previous.get(event["cell"]) != event:
                handle.write(json.dumps({"observed_at": stamp, **event}, sort_keys=True) + "\n")
    (BUNDLE / "target_ablation_snapshot_v2.json").write_text(json.dumps({
        "schema": "target_ablation_snapshot", "version": "2.0", "observed_at": stamp,
        "admission_rule": "Only 750-step, 500-generation, finite, attested, hash-verified terminal bundles are admitted. Smoke runs are excluded.",
        "cells": events}, indent=2) + "\n")
    fields = ["cell", "status", "target_set", "BLEU", "chrF", "ROUGE_L", "invalid_rate_pct", "repetition_pct",
              "trainable_parameter_count", "adapter_bytes", "trusted_A_factor_count", "trusted_B_factor_count",
              "gpu_local_factor_count", "tdx_optimizer_tensor_state_bytes", "projected_raw_tensor_bytes_each_direction_per_step",
              "trusted_roundtrip_time_sec_per_step", "trusted_compute_time_sec_per_step", "transport_only_time_sec_per_step",
              "total_step_time_sec", "protected_generation_tokens_per_sec", "attestation_status", "provenance_summary"]
    with (BUNDLE / "target_ablation_table_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for event in events:
            quality = event.get("quality", {})
            row = {key: event.get(key) for key in fields}
            for key in ("BLEU", "chrF", "ROUGE_L", "invalid_rate_pct", "repetition_pct"):
                row[key] = quality.get(key)
            row["target_set"] = "+".join(event["target_set"])
            row["provenance_summary"] = json.dumps(event.get("provenance", {}), sort_keys=True)
            writer.writerow(row)
    lines = ["# Target Ablation Table (v2)", "",
             "Only full terminal cells are populated. Completed smoke runs are deliberately excluded.", "",
             "| Cell | Status | Targets | BLEU | chrF | ROUGE-L | invalid% | repetition% | Trainable params | Adapter bytes | Trusted A/B factors | GPU-local factors | TDX optimizer state bytes | Raw bytes/direction/step | Trusted roundtrip s/step | Trusted compute | Transport-only | Step s | tok/s | Attestation |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    def show(value): return "MISSING" if value is None else str(value)
    for event in events:
        q = event.get("quality", {})
        lines.append(f"| {event['cell']} | {event['status']} | {'+'.join(event['target_set'])} | "
                     f"{show(q.get('BLEU'))} | {show(q.get('chrF'))} | {show(q.get('ROUGE_L'))} | "
                     f"{show(q.get('invalid_rate_pct'))} | {show(q.get('repetition_pct'))} | "
                     f"{show(event.get('trainable_parameter_count'))} | {show(event.get('adapter_bytes'))} | "
                     f"{show(event.get('trusted_A_factor_count'))}/{show(event.get('trusted_B_factor_count'))} | "
                     f"{show(event.get('gpu_local_factor_count'))} | {show(event.get('tdx_optimizer_tensor_state_bytes'))} | "
                     f"{show(event.get('projected_raw_tensor_bytes_each_direction_per_step'))} | "
                     f"{show(event.get('trusted_roundtrip_time_sec_per_step'))} | {show(event.get('trusted_compute_time_sec_per_step'))} | "
                     f"{show(event.get('transport_only_time_sec_per_step'))} | {show(event.get('total_step_time_sec'))} | "
                     f"{show(event.get('protected_generation_tokens_per_sec'))} | {show(event.get('attestation_status'))} |")
    lines += ["", "Provenance tags are stored per field in the JSON snapshot and CSV. Trainable and factor counts are MEASURED by enumerating the hash-verified adapter. TDX state and raw bytes/direction/step are PROJECTED from tensor ownership and exclude service/framing overhead. Trusted compute and transport-only time are MISSING because the terminal T1 run retained roundtrip timing but not the profiling split."]
    (BUNDLE / "target_ablation_table_v2.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
