#!/usr/bin/env python3
"""CPU-only provenance and schema audit for the frozen V2 captures.

This script never modifies the frozen corpora.  It separates schema validity
from validity of the historical source-comparison cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reductions(prefix, values, family, source_tensor, layer, source_stat, allowed=True, rationale=""):
    x = np.asarray(values, dtype=float)
    out = []
    for stat, value in zip(("mean", "std", "min", "max"),
                           (x.mean(), x.std(), x.min(), x.max())):
        out.append({"feature": f"{prefix}.{stat}", "value": float(value), "family": family,
                    "source_tensor": source_tensor, "layer": layer,
                    "extraction_function": f"{stat}({source_stat})", "allowed": allowed,
                    "rationale": rationale})
    return out


def feature_items(row: dict) -> list[dict]:
    out = []
    hidden = row["transformed_hidden"]
    out.append({"feature": "hidden.final.last_l2", "value": float(hidden["final_last_l2"]),
                "family": "hidden", "source_tensor": "transformed_hidden.final",
                "layer": "final", "extraction_function": "l2(last_token)", "allowed": True,
                "rationale": "orthogonal residual transform preserves L2 norm; no raw coordinate"})
    for item in hidden["per_layer"]:
        layer = int(item["layer"])
        out.append({"feature": f"hidden.layer_{layer:02d}.last_l2", "value": float(item["last_l2"]),
                    "family": "hidden", "source_tensor": f"transformed_hidden.layer_{layer:02d}",
                    "layer": layer, "extraction_function": "l2(last_token_pre_attention)",
                    "allowed": True,
                    "rationale": "orthogonal residual transform preserves L2 norm; no raw coordinate"})
    for field, label in (("masked_k", "k"), ("masked_v", "v")):
        for item in row[field]:
            layer = int(item["layer"])
            out += reductions(f"kv.{label}.layer_{layer:02d}.last_l2_by_head",
                              item["last_l2_by_head"], "kv", f"{field}.layer_{layer:02d}", layer,
                              "per_head_l2(last_token)", True,
                              "order-insensitive reduction of per-head invariant norms")
    for item in row["attention_scores"]:
        layer = int(item["layer"])
        for key in ("last_entropy_by_head", "last_max_by_head",
                    "true_score_last_mean_by_head", "true_score_last_std_by_head"):
            out += reductions(f"attention.layer_{layer:02d}.{key}", item[key], "attention",
                              f"attention_scores.layer_{layer:02d}", layer, key, True,
                              "order-insensitive head reduction; Q/K transform cancels in scores")
        seq = float(row["tensor_shapes"]["sequence_length"])
        normalized = np.asarray(item["last_argmax_by_head"], dtype=float) / max(seq - 1, 1)
        out += reductions(f"attention.layer_{layer:02d}.normalized_last_argmax_by_head", normalized,
                          "attention", f"attention_scores.layer_{layer:02d}", layer,
                          "argmax_position/(sequence_length-1)", False,
                          "raw positional coordinate can encode ordering/effective-length artifacts")
    logits = row["masked_logits"]
    for key in ("mean", "std", "l2", "min", "max"):
        out.append({"feature": f"logit.{key}", "value": float(logits[key]), "family": "logit",
                    "source_tensor": "masked_logits.last_token", "layer": "final",
                    "extraction_function": key, "allowed": True,
                    "rationale": "permutation-invariant scalar from protocol-exposed masked logits"})
    top = np.asarray(logits["top_values"], dtype=float)
    out += reductions("logit.top_values", top, "logit", "masked_logits.top_values", "final",
                      "top32_values", True,
                      "order-insensitive reductions of protocol-exposed values; indices excluded")
    out.append({"feature": "logit.top1_top2_margin", "value": float(top[0] - top[1]),
                "family": "logit", "source_tensor": "masked_logits.top_values", "layer": "final",
                "extraction_function": "top1_value-top2_value", "allowed": True,
                "rationale": "permutation-invariant margin genuinely available from exposed masked logits"})
    return out


def stats(x: np.ndarray, prefix: str) -> dict:
    finite = x[np.isfinite(x)]
    return {f"{prefix}_min": float(finite.min()) if finite.size else math.nan,
            f"{prefix}_max": float(finite.max()) if finite.size else math.nan,
            f"{prefix}_unique_values": int(np.unique(finite).size),
            f"{prefix}_missing_values": int(x.size - finite.size),
            f"{prefix}_dtype": str(x.dtype), f"{prefix}_shape": json.dumps(list(x.shape))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-a", type=Path, required=True)
    ap.add_argument("--source-b", type=Path, required=True)
    ap.add_argument("--profile-a", type=Path, required=True)
    ap.add_argument("--profile-b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # The directory is preregistered with this script; generated artifacts must be absent.
    for name in ("feature_provenance_report.md", "feature_column_audit.csv",
                 "feature_family_summary.json", "leakage_source_analysis.md", "v2_feature_schema.json"):
        if (args.output / name).exists():
            raise RuntimeError(f"refusing to overwrite {name}")
    a, b = load(args.source_a), load(args.source_b)
    pa, pb = json.loads(args.profile_a.read_text()), json.loads(args.profile_b.read_text())
    items = [feature_items(r) for r in a + b]
    names = [x["feature"] for x in items[0]]
    if any([x["feature"] for x in row] != names for row in items):
        raise RuntimeError("feature ordering differs between rows")
    matrix = np.asarray([[x["value"] for x in row] for row in items], dtype=np.float64)
    y = np.r_[np.ones(len(a), int), np.zeros(len(b), int)]
    rows = []
    for col, descriptor in enumerate(items[0]):
        xa, xb = matrix[:len(a), col], matrix[len(a):, col]
        auc = float(roc_auc_score(y, matrix[:, col])) if np.unique(matrix[:, col]).size > 1 else .5
        direct = descriptor["feature"].lower()
        direct_identifier = any(x in direct for x in
                                ("sample_id", "run_id", "session", "adapter", "checkpoint", "split", "source"))
        status = "INVALID" if not descriptor["allowed"] or direct_identifier else "DIAGNOSTIC_ONLY"
        rows.append({"feature_index": col, "feature": descriptor["feature"],
                     "feature_family": descriptor["family"], "source_tensor": descriptor["source_tensor"],
                     "layer": descriptor["layer"], "extraction_function": descriptor["extraction_function"],
                     "status": status, "schema_allowed": descriptor["allowed"],
                     "rationale": descriptor["rationale"],
                     "collection_path_a": str(args.source_a), "collection_path_b": str(args.source_b),
                     "batch_metadata": "batch_size=1 inferred from capture loop; not recorded in profile",
                     "run_identifier_equal": pa.get("run_id") == pb.get("run_id"),
                     "adapter_identifier_equal": pa.get("adapter_sha256") == pb.get("adapter_sha256"),
                     "checkpoint_identifier_equal": pa.get("base_package_root_hash") == pb.get("base_package_root_hash"),
                     "session_identifier_present": False,
                     "preprocessing_artifact": "different frozen query/tokenization files; no feature scaling before capture",
                     "ordering_artifact": "source A uses train-ID order; source B uses test-ID order",
                     "sample_alignment": "NO_COMMON_SAMPLE_IDS",
                     "direct_collection_source_identifier": direct_identifier,
                     **stats(xa, "source_a"), **stats(xb, "source_b"),
                     "roc_auc_source_higher": auc, "roc_auc_separability": max(auc, 1-auc),
                     "ks_statistic": float(ks_2samp(xa, xb).statistic),
                     "disjoint_ranges": bool(xa.max() < xb.min() or xb.max() < xa.min())})
    with (args.output / "feature_column_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    ids_a, ids_b = [r["sample_id"] for r in a], [r["sample_id"] for r in b]
    meta = {
        "source_a": {"records": len(a), "sha256": sha(args.source_a), "profile_sha256": sha(args.profile_a),
                     "split_values": sorted({r.get("split") for r in a}),
                     "run_ids": sorted({r.get("run_id") for r in a}),
                     "adapter_ids": sorted({r["package_metadata"].get("adapter_sha256") for r in a}),
                     "package_ids": sorted({r["package_metadata"].get("base_package_root_hash") for r in a})},
        "source_b": {"records": len(b), "sha256": sha(args.source_b), "profile_sha256": sha(args.profile_b),
                     "split_values": sorted({r.get("split") for r in b}),
                     "run_ids": sorted({r.get("run_id") for r in b}),
                     "adapter_ids": sorted({r["package_metadata"].get("adapter_sha256") for r in b}),
                     "package_ids": sorted({r["package_metadata"].get("base_package_root_hash") for r in b})},
        "common_sample_ids": len(set(ids_a) & set(ids_b)),
        "ordered_sample_ids_equal": ids_a == ids_b,
        "same_capture_profile_fields": {k: pa.get(k) == pb.get(k) for k in
                                        ("schema", "version", "run_id", "base_package_root_hash",
                                         "adapter_sha256", "device", "dtype", "victim_runs", "prefill_per_sample")},
        "different_query_hashes": pa.get("queries_sha256") != pb.get("queries_sha256"),
        "raw_metadata_direct_source_identifiers": ["sample_id namespace", "split"],
        "raw_metadata_indirect_identifiers": ["queries_sha256 in profile"],
        "historical_source_test_valid": False,
        "invalidity_reason": "source label is aliased with dataset split/query set and there are zero shared sample IDs",
    }
    family_summary = []
    for family in ("attention", "kv", "hidden", "logit", "combined"):
        z = rows if family == "combined" else [r for r in rows if r["feature_family"] == family]
        family_summary.append({"family": family, "columns": len(z),
                               "schema_allowed_columns": sum(bool(r["schema_allowed"]) for r in z),
                               "invalid_columns": sum(r["status"] == "INVALID" for r in z),
                               "diagnostic_only_columns": sum(r["status"] == "DIAGNOSTIC_ONLY" for r in z),
                               "max_source_auc_separability": max(r["roc_auc_separability"] for r in z),
                               "max_ks": max(r["ks_statistic"] for r in z),
                               "disjoint_columns": sum(bool(r["disjoint_ranges"]) for r in z),
                               "historical_evidence_status": "DIAGNOSTIC_ONLY"})
    summary = {"schema": "feature_family_provenance_audit_v2", "metadata": meta,
               "feature_columns": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
               "families": family_summary,
               "conclusion": "historical AUC cannot identify a collection-code confound because source is aliased with dataset split/query set"}
    (args.output / "feature_family_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    schema_features = []
    for d in items[0]:
        schema_features.append({"feature_name": d["feature"], "feature_family": d["family"],
                                "source_tensor": d["source_tensor"], "layer": d["layer"],
                                "extraction_function": d["extraction_function"],
                                "status": "ALLOWED" if d["allowed"] else "FORBIDDEN",
                                "rationale": d["rationale"]})
    prohibited = ["sample_id", "run_id", "session_id", "adapter_id", "adapter_sha256",
                  "checkpoint_id", "base_package_root_hash", "split", "source_file", "timestamp",
                  "generation_order", "tensor_shapes", "sequence_length", "top_indices",
                  "raw_tensor_coordinates", "labels", "loss", "gradients", "dlogits", "optimizer_state"]
    schema = {"schema": "metadata_free_invariant_v2", "version": "2.0",
              "collection_mode": "SIMULATED_PROTOCOL_VIEW", "feature_order_frozen": True,
              "features": schema_features, "allowed_feature_count": sum(d["allowed"] for d in items[0]),
              "forbidden_feature_count": sum(not d["allowed"] for d in items[0]),
              "prohibited_fields": prohibited,
              "release_contract": {"identifiers_used_for_evaluator_join_only": ["sample_id"],
                                   "identifiers_visible_to_model": [],
                                   "preprocessing_fit_scope": "attack-training split only",
                                   "raw_coordinates_released": False}}
    (args.output / "v2_feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    report = ["# V2 feature provenance report", "",
              "**Historical feature-source result: DIAGNOSTIC_ONLY. It is not privacy leakage evidence.**", "",
              f"The two frozen corpora contain {len(a)} and {len(b)} records but have **{meta['common_sample_ids']} shared sample IDs**. "
              "The source label is exactly aliased with train/test split, sample-ID namespace, query-file hash, and content distribution. "
              "Run ID, adapter hash, transformed-package root, capture schema, dtype, device, and one-prefill loop are equal.", "",
              "| Family | Columns | Schema allowed | Invalid | Diagnostic only | Max source AUC | Disjoint |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for x in family_summary:
        report.append(f"| {x['family']} | {x['columns']} | {x['schema_allowed_columns']} | {x['invalid_columns']} | {x['diagnostic_only_columns']} | {x['max_source_auc_separability']:.4f} | {x['disjoint_columns']} |")
    report += ["", "Normalized attention argmax-position features are marked INVALID because they retain raw positional coordinates. "
               "All otherwise invariant numerical columns remain DIAGNOSTIC_ONLY until the same frozen samples are collected twice through the identical path.", "",
               "Batch size is inferred as one from the capture loop but was not recorded in the profiles. No session ID is present. Raw metadata contains `split`, sample-ID namespace, run ID, adapter hash, and package root; none is permitted in the corrected feature matrix."]
    (args.output / "feature_provenance_report.md").write_text("\n".join(report) + "\n")
    analysis = ["# Leakage/source analysis", "",
                "The previous AUC=1 finding does not isolate collection source. The compared rows come from different query files and disjoint train/test samples, so source, dataset split, semantic distribution, prompt distribution, and sample namespace all change together.", "",
                "Direct raw identifiers: `split` and the `sample_id` namespace. Indirect profile identifier: `queries_sha256`. Run, adapter, base package, dtype, device, capture schema, and prefill count are equal.", "",
                "Required correction: use one fixed-length frozen sample list twice, identical capture code/config/batch/dtype/package/adapter, strip identifiers before fitting, group train/test by sample ID, and keep only schema-allowed invariant features. If that paired repeat remains separable, stop before MIA."]
    (args.output / "leakage_source_analysis.md").write_text("\n".join(analysis) + "\n")
    print(json.dumps({"columns": len(rows), "allowed": schema["allowed_feature_count"],
                      "common_ids": meta["common_sample_ids"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
