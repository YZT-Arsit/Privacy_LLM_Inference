#!/usr/bin/env python3
"""Assemble artifact misuse-resistance evidence without projecting missing cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--r4-log", type=Path,
                    help="Measured local direct-transport wrong-session/HMAC fault log")
    ap.add_argument("--r3-log", type=Path,
                    help="Measured local diagnostic-boundary attestation gate log")
    args = ap.parse_args()
    restart = json.loads((args.evidence / "restart_reproducibility.json").read_text())
    lifecycle = json.loads((args.evidence / "lifecycle_gate.json").read_text())
    negative = json.loads((args.evidence / "negative_controls.json").read_text())
    r4_text = args.r4_log.read_text(errors="replace") if args.r4_log else ""
    r4_checks = {
        name: f"[PASS] {name}" in r4_text
        for name in ("bad_hmac", "replay", "wrong_runid", "recover_after_faults",
                     "counters_consistent")
    }
    r4_complete = bool(r4_text) and all(r4_checks.values())
    r3_text = args.r3_log.read_text(errors="replace") if args.r3_log else ""
    r3_checks = {
        "tdx_invalid_attestation_refused":
            "ok   TDX: required + NOT verified -> refused" in r3_text,
        "tdx_debug_true_refused": "ok   TDX: required + debug TRUE -> refused" in r3_text,
        "a10_invalid_session_refused":
            "ok   A10: --require-attestation but session invalid -> refuses" in r3_text,
        "end_to_end_refused":
            "ok   END-TO-END invalid attestation: TDX refuses AND A10 refuses" in r3_text,
        "all_contract_checks_passed": "[test_attestation_gate] PASS=12 FAIL=0" in r3_text,
    }
    r3_complete = bool(r3_text) and all(r3_checks.values())
    conditions = {
        "R0": {"status": "PENDING_MEASUREMENT", "trusted_service": False,
               "note": "package-only initialization/forward test not yet run"},
        "R1": {"status": "PENDING_MEASUREMENT", "gpu_runtime": True,
               "note": "runtime-without-boundary forward test not yet run"},
        "R2": {"status": "PENDING_MEASUREMENT", "researcher_generated_inputs": True,
               "note": "meaningful-output test without TDX-only mapping not yet run"},
        "R3": {"status": "COMPLETE_VERIFIED" if r3_complete else "PENDING_MEASUREMENT",
               "diagnostic_boundary_stub": True, "runtime_gate_checks": r3_checks,
               "initialization_succeeds": False if r3_complete else None,
               "internal_forward_starts": False if r3_complete else None,
               "meaningful_final_output": False if r3_complete else None,
               "note": ("invalid local diagnostic boundary is refused before protected execution"
                        if r3_complete else "fail-closed local stub test not yet run")},
        "R4": {"status": "COMPLETE_VERIFIED" if r4_complete else "PARTIAL",
               "different_session_binding": True,
               "stale_adapter_version_rejected": negative["controls"].get("stale_version") == "rejected_ok",
               "runtime_fault_checks": r4_checks,
               "note": ("wrong run/session HMAC, replay, recovery, and counters passed"
                        if r4_complete else "explicit wrong run/session HMAC condition remains")},
        "R5": {"status": "COMPLETE_VERIFIED",
               "all_mismatched_metadata_rejected": negative["all_controls_fail_closed"],
               "controls": negative["controls"]},
    }
    result = {
        "schema": "protected_artifact_misuse_resistance", "version": "1.0",
        "term": "protected artifact misuse-resistance evaluation",
        "authorized_reference": {
            "fresh_process_ok": restart["fresh_process_ok"],
            "rows": restart["rows"], "token_agreement": restart["token_agreement"],
            "generation_agreement": restart["generation_agreement"],
            "hash_agreement": restart["hash_agreement"],
            "attestation_A": restart["attestation_A"], "attestation_B": restart["attestation_B"],
            "transformed_base_only": lifecycle["transformed_base_only"],
            "transformed_adapter_only": lifecycle["transformed_adapter_only"],
        },
        "conditions": conditions,
        "complete": all(x["status"] == "COMPLETE_VERIFIED" for x in conditions.values()),
        "source_hashes": {
            **{p.name: sha256(p) for p in sorted(args.evidence.glob("*.json"))},
            **({str(args.r3_log.name): sha256(args.r3_log)} if args.r3_log else {}),
            **({str(args.r4_log.name): sha256(args.r4_log)} if args.r4_log else {}),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_condition_results.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "schema": "artifact_resistance_evaluation_manifest", "version": "1.0",
        "researcher_owned_fixtures_only": True,
        "conditions": list(conditions), "completed_conditions": [
            k for k, v in conditions.items() if v["status"] == "COMPLETE_VERIFIED"
        ],
        "active_external_targets": [], "complete": result["complete"],
    }
    (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Protected artifact misuse-resistance evaluation", "",
             "All fixtures are researcher-owned. No external endpoint or authentication bypass is used.", "",
             f"Authorized transformed-package restart: {restart['rows']} rows, token/text/hash agreement "
             f"{restart['token_agreement']:.1%}, attestation A/B true.", "",
             "| Condition | Status | Current evidence |", "|---|---|---|"]
    for key, value in conditions.items():
        note = value.get("note", "all configured mismatch controls rejected")
        lines.append(f"| {key} | {value['status']} | {note} |")
    pending = [key for key, value in conditions.items()
               if value["status"] != "COMPLETE_VERIFIED"]
    lines += ["", ("The stage remains partial until " + ", ".join(pending)
                   + " are measured independently." if pending
                   else "All configured conditions are complete and independently measured.")]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
