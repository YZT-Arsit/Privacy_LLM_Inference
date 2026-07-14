#!/usr/bin/env python3
"""Build and verify the append-only CPU handoff for the three real-TDX shadows."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
HANDOFF = ROOT / "cpu_handoff"
FINAL = ROOT / "final_summary"
UPSTREAM = ROOT.parent
PROHIBITED_SOURCE = REPO / "results/aaai_private_base/authorized_security_eval/mia_v2_handoff/prohibited_feature_list.json"

EXPECTED_ADAPTERS = {
    1: "848dc99b4fa08e638f68bb42c6b293913dfb241d7e25fd02de917b0e91520fb2",
    2: "7c46d98316c5f09266cade307ecd2653f81c5fcfa021eef5536e326d45547ec9",
    3: "1990fef5376210989d3302cba30206993d280a098557981bd59680f897828c37",
}
EXPECTED_MEMBERSHIPS = {
    1: "01eb983e80078c81ae48abff3838f9671ec4225cf53eedcc84869f63fa151676",
    2: "bf98dedcf37a42a9439d9d9fd949263e44440de43b47f7ad87eba632f7bc222a",
    3: "6bc88e7e92f07aac642759ca870eaec19abd9d4cdfc7209ecbb0ee93547cad4a",
}
RUN_IDS = {
    1: "real-tdx-three-shadow-v1-s1-1784043541-281e99b3",
    2: "real-tdx-three-shadow-v1-s2-1784047313-bc98d6bc",
    3: "real-tdx-three-shadow-v1-s3-1784048319-75354d0b",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def require_new(paths: list[Path]) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise SystemExit("append-only refusal: output already exists: " + ", ".join(existing))


def command_for_shadow(shadow: int, masked: str) -> str:
    # The historical session key is intentionally excluded from the handoff.
    return (
        "python3 /root/real_tdx_three_shadow_v1/bin/collect_real_tdx_common.py "
        f"--session /root/real_tdx_three_shadow_v1/shadow_{shadow}/a10_session.json "
        "--tdx 172.30.25.153 --key /root/passkey.pem "
        "--service-cmd '/root/real_tdx_three_shadow_v1/launch_service.sh "
        f"/root/real_tdx_three_shadow_v1/shadow_{shadow}/tdx_session.json' "
        f"--masked-adapter /root/real_tdx_three_shadow_v1/shadow_{shadow}/masked_adapter/{masked}/adapter_model.safetensors "
        "--queries /root/real_tdx_three_shadow_v1/inputs/evaluation_queries.jsonl "
        "--schema /root/real_tdx_three_shadow_v1/inputs/common_v2_feature_schema.json "
        f"--output-dir /root/real_tdx_three_shadow_v1/shadow_{shadow}/collection "
        "--max-new-tokens 96 --require-attestation"
    )


def main() -> None:
    outputs = [
        HANDOFF / "README.md",
        HANDOFF / "handoff_manifest.json",
        HANDOFF / "artifact_hashes.json",
        HANDOFF / "collection_validation.md",
        HANDOFF / "per_shadow_summary.csv",
        HANDOFF / "schema_manifest.json",
        HANDOFF / "environment_manifest.json",
        HANDOFF / "prohibited_feature_list.json",
        FINAL / "final_status.json",
        FINAL / "final_status.md",
    ]
    require_new(outputs)
    HANDOFF.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    schema_path = ROOT / "inputs/common_v2_feature_schema.json"
    schema = load(schema_path)
    joint_path = ROOT / "collection_validation/joint/joint_validation_report.json"
    joint = load(joint_path)
    prohibited = load(PROHIBITED_SOURCE)
    write_json(HANDOFF / "prohibited_feature_list.json", prohibited)

    if joint["status"] != "PASS" or not all(joint["gates"].values()):
        raise SystemExit("joint validation did not pass")
    if schema["feature_count"] != 610 or sha256(schema_path) != joint["schema_sha256"]:
        raise SystemExit("schema gate failed")

    shadows = []
    for i in (1, 2, 3):
        adapter = UPSTREAM / f"shadow_{i}/training/adapter/adapter_model.safetensors"
        membership = ROOT / f"shadow_{i}/membership_join.jsonl"
        collection = ROOT / f"shadow_{i}/collection"
        manifest_path = collection / "collection_manifest.json"
        validity_path = ROOT / f"collection_validation/shadow_{i}/validity_report.json"
        manifest, validity = load(manifest_path), load(validity_path)
        transform_name = "trusted_adapter_transform_manifest.json" if i == 2 else "trusted_transform_manifest.json"
        transform = ROOT / f"shadow_{i}/tdx_evidence/{transform_name}"
        checks = {
            "source_adapter_hash": sha256(adapter) == EXPECTED_ADAPTERS[i] == validity["source_adapter_sha256"],
            "membership_hash": sha256(membership) == EXPECTED_MEMBERSHIPS[i] == validity["membership_sha256"],
            "manifest_pass": manifest["status"] == "PASS",
            "validity_pass": validity["status"] == f"SHADOW_{i}_COLLECTION_VALID" and all(validity["gates"].values()),
            "records": manifest["records"] == validity["records"] == 1000,
            "balance": validity["members"] == validity["nonmembers"] == 500,
            "features": manifest["feature_columns"] == validity["feature_columns"] == 610,
            "attestation": manifest["attestation_verified"] is True,
            "no_fallback": manifest["silent_fallbacks"] == 0,
            "no_disjoint_ranges": validity["disjoint_range_feature_count"] == 0,
            "transform_manifest_present": transform.is_file(),
        }
        if not all(checks.values()):
            raise SystemExit(f"shadow {i} failed: {checks}")
        shadows.append({
            "shadow": i,
            "status": validity["status"],
            "run_id": RUN_IDS[i],
            "adapter": {"path": rel(adapter), "sha256": sha256(adapter)},
            "membership": {"path": rel(membership), "sha256": sha256(membership)},
            "collection_manifest": rel(manifest_path),
            "v0_matrix": rel(collection / "real_tdx_v0_outputs.jsonl"),
            "v2_matrix": rel(collection / "real_tdx_v2_features.csv"),
            "sample_id_join": rel(collection / "sample_index.json"),
            "transform_commitments": rel(collection / "transform_commitments.jsonl"),
            "validity_report": rel(validity_path),
            "attestation_evidence": rel(ROOT / f"shadow_{i}/tdx_evidence/attestation"),
            "tdx_session_counters": rel(ROOT / f"shadow_{i}/tdx_evidence/session_counters.json"),
            "trusted_transform_manifest": rel(transform),
            "masked_adapter_sha256": manifest["masked_adapter_sha256"],
            "records": 1000,
            "members": 500,
            "nonmembers": 500,
            "feature_columns": 610,
            "dtype": manifest["dtype"],
            "wall_sec": manifest["wall_sec"],
            "tdx_decode_calls": manifest["tdx_decode_calls"],
            "collection_command": command_for_shadow(i, manifest["masked_adapter_sha256"]),
            "command_secret_policy": "session JSON supplied the historical key; session files are deliberately excluded",
            "checks": checks,
        })

    environment = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "collection_host": {
            "role": "A10 inference host",
            "public_ip": "39.107.123.173",
            "private_ip": "172.30.25.154",
            "hostname": "iZ2zeajy3dc0ssoyi2cf3vZ",
            "kernel": "Linux 5.15",
            "python": "3.10.12",
            "torch": "2.8.0+cu128",
            "numpy": "2.2.6",
            "gpu": "NVIDIA A10",
            "gpu_uuid": "GPU-0729b83b-60dd-c88d-2b9c-2049e0127eae",
            "driver": "580.126.09",
            "final_state": "idle; 0 MiB used and 0% utilization at final audit",
        },
        "trusted_host": {
            "role": "Intel TDX service host",
            "public_ip": "39.96.43.122",
            "private_ip": "172.30.25.153",
            "hostname": "iZ2zebdihkqll10ragqp0cZ",
            "kernel": "Linux 5.10",
            "python": "3.10.20",
            "tdx_device": "/dev/tdx_guest",
            "final_state": "idle; no collection service or additional listening data-plane port",
        },
        "frozen_runtime_hashes": {
            "transformed_package_root": "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
            "model_config": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
            "tokenizer": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "tdx_service": "c2b32d13d778747cc5054294a5563d56c02150b4148cb00723293ddbe68e796d",
            "collector": sha256(ROOT / "bin/collect_real_tdx_common.py"),
        },
        "resource_policy": {
            "retraining_performed": False,
            "shared_registry_modified": False,
            "commit_created": False,
            "concurrent_shadow_collections": False,
        },
    }
    write_json(HANDOFF / "environment_manifest.json", environment)

    schema_manifest = {
        "schema": schema["schema"],
        "version": schema["version"],
        "collection_mode": "REAL_TDX_BACKED_VIEW",
        "collection_scope": "frozen 1000-sample candidate pool per shadow",
        "source_schema_scope_note": "The immutable schema file records the earlier 250-sample fidelity subset; only its 610 frozen feature definitions are reused here.",
        "feature_count": 610,
        "feature_order_frozen": True,
        "schema_path": rel(schema_path),
        "schema_sha256": sha256(schema_path),
        "canonical_feature_definitions_sha256": schema["canonical_feature_definitions_sha256"],
        "excluded_features": schema["excluded_features"],
        "prohibited_feature_list": rel(HANDOFF / "prohibited_feature_list.json"),
        "prohibited_feature_list_sha256": sha256(HANDOFF / "prohibited_feature_list.json"),
    }
    write_json(HANDOFF / "schema_manifest.json", schema_manifest)

    handoff_manifest = {
        "schema": "real_tdx_three_shadow_cpu_handoff_v1",
        "status": "PASS",
        "final_collection_status": "REAL_TDX_THREE_SHADOW_COLLECTION_COMPLETE",
        "collection_mode": "REAL_TDX_BACKED_VIEW",
        "scope": "three frozen 1000-sample candidate pools; 500 members and 500 nonmembers per shadow",
        "feature_schema": "common-semantics schema v2 (610 columns)",
        "sample_pool_sha256": "b65e8b4a6fc531187164472b0245260b6c85816a1040c6959cf0cbeff1cebcae",
        "queries_sha256": "cfcd454e2a394bbeef046b997c4fe8db7978e3227e30f9097e7f9599381f2826",
        "ordered_sample_ids_sha256": "d077509179a93d8873bb5254993040ad336b08f48ed0704a5c9de0d1e59b845c",
        "schema_sha256": sha256(schema_path),
        "shadows": shadows,
        "joint_validation_report": rel(joint_path),
        "joint_validation_status": joint["status"],
        "cpu_owner_next_actions": [
            "real-TDX leave-one-shadow-out classification",
            "V0, V2, and V0+V2 comparisons",
            "group-aware sensitivity analysis where supported",
            "paired bootstrap intervals",
            "feature-family ablations",
            "negative-control verification",
        ],
        "final_paper_facing_mia_executed": False,
        "excluded_sensitive_files": [
            "shadow_*/a10_session.json",
            "shadow_*/tdx_session.json",
        ],
        "exclusion_reason": "historical session key material; not required for CPU analysis",
    }
    write_json(HANDOFF / "handoff_manifest.json", handoff_manifest)

    with (HANDOFF / "per_shadow_summary.csv").open("w", newline="") as f:
        fields = ["shadow", "status", "source_adapter_sha256", "masked_adapter_sha256", "membership_sha256",
                  "records", "members", "nonmembers", "feature_columns", "dtype", "wall_sec", "tdx_decode_calls",
                  "attestation_verified", "silent_fallbacks", "shuffle_auc", "pseudo_membership_auc",
                  "top_univariate_symmetric_auc", "disjoint_range_feature_count"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in shadows:
            validity = load(REPO / s["validity_report"])
            controls = {x["control"]: x["auc"] for x in validity["controls"]}
            w.writerow({
                "shadow": s["shadow"], "status": s["status"], "source_adapter_sha256": s["adapter"]["sha256"],
                "masked_adapter_sha256": s["masked_adapter_sha256"], "membership_sha256": s["membership"]["sha256"],
                "records": s["records"], "members": s["members"], "nonmembers": s["nonmembers"],
                "feature_columns": s["feature_columns"], "dtype": s["dtype"], "wall_sec": s["wall_sec"],
                "tdx_decode_calls": s["tdx_decode_calls"], "attestation_verified": True, "silent_fallbacks": 0,
                "shuffle_auc": controls["shuffled_membership"], "pseudo_membership_auc": controls["pseudo_membership"],
                "top_univariate_symmetric_auc": validity["top_univariate_symmetric_auc"],
                "disjoint_range_feature_count": validity["disjoint_range_feature_count"],
            })

    validation_lines = [
        "# Real-TDX three-shadow collection validation", "",
        "Status: **PASS** (`REAL_TDX_THREE_SHADOW_COLLECTION_COMPLETE`).", "",
        "Each shadow contains exactly 1,000 rows (500 members / 500 nonmembers), 610 fp32 features, a matched V0 output, a complete sample-ID join, valid fresh TDX attestation, zero silent fallbacks, and no disjoint member/nonmember ranges.", "",
        "| Shadow | Wall time (s) | Decode calls | Shuffle AUC | Pseudo AUC | Top univariate symmetric AUC |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in shadows:
        validity = load(REPO / s["validity_report"])
        controls = {x["control"]: x["auc"] for x in validity["controls"]}
        validation_lines.append(f"| {s['shadow']} | {s['wall_sec']} | {s['tdx_decode_calls']} | {controls['shuffled_membership']:.6f} | {controls['pseudo_membership']:.6f} | {validity['top_univariate_symmetric_auc']:.6f} |")
    d = joint["diagnostics"]
    validation_lines += [
        "", "Joint gates: all passed. The joint top univariate symmetric AUC is "
        f"{d['top_joint_univariate_symmetric_auc']:.6f}; mean shuffle AUC is {d['mean_shuffle_auc']:.6f}; "
        f"mean pseudo-membership AUC is {d['mean_pseudo_auc']:.6f}; disjoint-range count is {d['disjoint_range_count']}.", "",
        "The shadow-identity diagnostic reaches 1.0 accuracy, but the matrix contains no run/session/shadow metadata. The separation is therefore documented as legitimate model-specific behavior and does not invalidate collection.", "",
        "The final paper-facing MIA classifier was not run in this collection session.", "",
    ]
    (HANDOFF / "collection_validation.md").write_text("\n".join(validation_lines))

    readme = """# CPU handoff: three real-TDX shadow collections

This handoff is complete and hash-verified. It contains the three terminal collection manifests, frozen adapters and membership maps by path and hash, V0/V2 matrices, sample joins, the 610-column schema, prohibited-field policy, fresh TDX attestation evidence, per-shadow validity reports, joint validation, environment metadata, and recorded collection commands.

Start with `handoff_manifest.json`, verify every entry in `artifact_hashes.json`, then run the frozen offline CPU pipeline. The intended next analysis is leave-one-shadow-out MIA with V0, V2, and V0+V2 comparisons plus the listed controls and intervals.

The collection session did **not** run the final paper-facing MIA classifier. Files `shadow_*/a10_session.json` and `shadow_*/tdx_session.json` are intentionally excluded because they contain historical session secrets and are not analysis inputs.

Collection mode: `REAL_TDX_BACKED_VIEW`. Scope: three 1,000-sample pools, balanced 500/500 per shadow. Schema: common-semantics schema v2, 610 frozen feature columns.
"""
    (HANDOFF / "README.md").write_text(readme)

    registered: list[Path] = [
        HANDOFF / "README.md", HANDOFF / "handoff_manifest.json", HANDOFF / "collection_validation.md",
        HANDOFF / "per_shadow_summary.csv", HANDOFF / "schema_manifest.json", HANDOFF / "environment_manifest.json",
        HANDOFF / "prohibited_feature_list.json", schema_path, ROOT / "inputs/evaluation_queries.jsonl",
        ROOT / "collection_validation/joint/joint_validation_report.json",
        ROOT / "collection_validation/joint/joint_diagnostics.json",
        ROOT / "collection_validation/joint/joint_univariate_auc_ks.csv",
        ROOT / "collection_validation/joint/summary.md",
        ROOT / "resource_audit/stale_scp_cleanup.json",
        ROOT / "resource_audit/shadow_2_prelaunch_comparison.json",
        ROOT / "resource_audit/shadow_3_prelaunch_comparison.json",
    ]
    registered += sorted((ROOT / "bin").glob("*.py"))
    for i, s in enumerate(shadows, 1):
        registered += [
            REPO / s["adapter"]["path"], REPO / s["membership"]["path"],
            REPO / s["collection_manifest"], REPO / s["v0_matrix"], REPO / s["v2_matrix"],
            REPO / s["sample_id_join"], REPO / s["transform_commitments"], REPO / s["validity_report"],
            ROOT / f"shadow_{i}/collection.log", ROOT / f"shadow_{i}/tdx_evidence/session_counters.json",
            REPO / s["trusted_transform_manifest"],
            ROOT / f"collection_validation/shadow_{i}/summary.md",
            ROOT / f"collection_validation/shadow_{i}/validity_controls.csv",
            ROOT / f"collection_validation/shadow_{i}/univariate_auc_ks.csv",
        ]
        registered += sorted((ROOT / f"shadow_{i}/tdx_evidence/attestation").glob("*"))
    missing = [str(p) for p in registered if not p.is_file()]
    if missing:
        raise SystemExit("missing registered artifacts: " + ", ".join(missing))
    artifacts = {rel(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(set(registered))}
    hashes = {"schema": "sha256_artifact_manifest_v1", "status": "PASS", "artifact_count": len(artifacts), "artifacts": artifacts}
    write_json(HANDOFF / "artifact_hashes.json", hashes)
    mismatches = [p for p, m in artifacts.items() if sha256(REPO / p) != m["sha256"] or (REPO / p).stat().st_size != m["bytes"]]
    if mismatches:
        raise SystemExit("post-write hash verification failed: " + ", ".join(mismatches))

    final = {
        "status": "REAL_TDX_THREE_SHADOW_COLLECTION_COMPLETE",
        "shadows_complete": [1, 2, 3],
        "records_per_shadow": 1000,
        "feature_columns": 610,
        "per_shadow_validation": "PASS",
        "joint_validation": "PASS",
        "cpu_handoff": "PASS",
        "registered_artifact_count": len(artifacts),
        "registered_artifacts_verified": True,
        "artifact_hash_manifest": rel(HANDOFF / "artifact_hashes.json"),
        "artifact_hash_manifest_sha256": sha256(HANDOFF / "artifact_hashes.json"),
        "final_paper_facing_mia_executed": False,
        "gpu_final_state": environment["collection_host"]["final_state"],
        "tdx_final_state": environment["trusted_host"]["final_state"],
    }
    write_json(FINAL / "final_status.json", final)
    (FINAL / "final_status.md").write_text(
        "# REAL_TDX_THREE_SHADOW_COLLECTION_COMPLETE\n\n"
        "All three 1,000-sample collections, all per-shadow gates, joint validation, CPU handoff generation, and registered-artifact SHA-256 verification passed. The final paper-facing MIA classifier remains for the offline CPU owner.\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
