#!/usr/bin/env python3
"""Strict feature-level audit for the exact-length-matched V2 MIA cohort."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_v2_membership_pilot import (  # noqa: E402
    exact_length_matched, feature_items, load, sha256,
)

MATCH_SEED = 20260714
MI_SEED = 20260714


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def class_stats(values: np.ndarray, prefix: str) -> dict:
    observed = finite(values)
    missing_rate = float(1.0 - observed.size / values.size)
    if observed.size == 0:
        return {
            f"{prefix}_min": math.nan, f"{prefix}_max": math.nan,
            f"{prefix}_mean": math.nan, f"{prefix}_std": math.nan,
            f"{prefix}_missing_rate": missing_rate, f"{prefix}_zero_rate": math.nan,
            f"{prefix}_unique_count": 0,
        }
    return {
        f"{prefix}_min": float(observed.min()),
        f"{prefix}_max": float(observed.max()),
        f"{prefix}_mean": float(observed.mean()),
        f"{prefix}_std": float(observed.std(ddof=1)) if observed.size > 1 else 0.0,
        f"{prefix}_missing_rate": missing_rate,
        f"{prefix}_zero_rate": float(np.mean(observed == 0.0)),
        f"{prefix}_unique_count": int(np.unique(observed).size),
    }


def field_paths(row: dict) -> set[str]:
    paths: set[str] = set()
    for family in ("transformed_hidden", "masked_logits", "masked_q", "masked_k",
                   "masked_v", "attention_scores", "tensor_shapes"):
        if family in row:
            paths.add(family)
    for family in ("masked_q", "masked_k", "masked_v", "attention_scores"):
        for layer in row.get(family, []):
            paths.add(f"{family}.layer_{int(layer['layer']):02d}")
    for layer in row.get("transformed_hidden", {}).get("per_layer", []):
        paths.add(f"transformed_hidden.layer_{int(layer['layer']):02d}")
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=Path, required=True)
    parser.add_argument("--nonmember", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw_member, raw_nonmember = load(args.member), load(args.nonmember)
    member, nonmember = exact_length_matched(raw_member, raw_nonmember, MATCH_SEED)
    if len(member) != len(nonmember):
        raise RuntimeError("exact-length cohort is not balanced")

    all_rows = member + nonmember
    item_rows = [feature_items(row, "v2_all") for row in all_rows]
    names = [item["feature"] for item in item_rows[0]]
    if any([item["feature"] for item in items] != names for items in item_rows):
        raise RuntimeError("feature columns differ across collection paths")
    matrix = np.asarray([[item["value"] for item in items] for items in item_rows], dtype=float)
    labels = np.asarray([1] * len(member) + [0] * len(nonmember), dtype=int)

    member_paths = set.intersection(*(field_paths(row) for row in member))
    nonmember_paths = set.intersection(*(field_paths(row) for row in nonmember))
    mi = mutual_info_classif(matrix, labels, discrete_features=False,
                             random_state=MI_SEED)

    audit_rows: list[dict] = []
    for index, descriptor in enumerate(item_rows[0]):
        m = matrix[:len(member), index]
        n = matrix[len(member):, index]
        mf, nf = finite(m), finite(n)
        usable = mf.size > 0 and nf.size > 0
        y_obs = np.r_[np.ones(mf.size, dtype=int), np.zeros(nf.size, dtype=int)]
        x_obs = np.r_[mf, nf]
        combined = np.r_[m, n]
        combined_finite = finite(combined)
        auc = float(roc_auc_score(y_obs, x_obs)) if usable and np.unique(x_obs).size > 1 else 0.5
        ks = float(ks_2samp(mf, nf, method="auto").statistic) if usable else math.nan
        member_below = bool(usable and mf.max() < nf.min())
        nonmember_below = bool(usable and nf.max() < mf.min())
        family_layer = descriptor["tensor_family"]
        if isinstance(descriptor["layer"], int):
            family_layer += f".layer_{descriptor['layer']:02d}"
        available_member = family_layer in member_paths
        available_nonmember = family_layer in nonmember_paths
        available_both = available_member and available_nonmember
        row = {
            "feature_index": index,
            "feature": descriptor["feature"],
            "tensor_family": descriptor["tensor_family"],
            "layer": descriptor["layer"],
            "statistic": descriptor["statistic"],
            "invariant_under_per_prompt_transform": descriptor["invariant_under_per_prompt_transform"],
            "derived_from_runtime_metadata": descriptor["derived_from_runtime_metadata"],
            "available_in_member_path": available_member,
            "available_in_nonmember_path": available_nonmember,
            "available_in_both_collection_paths": available_both,
            **class_stats(m, "member"),
            **class_stats(n, "nonmember"),
            "combined_missing_rate": float(1.0 - combined_finite.size / combined.size),
            "combined_zero_rate": (float(np.mean(combined_finite == 0.0))
                                   if combined_finite.size else math.nan),
            "combined_unique_count": int(np.unique(combined_finite).size),
            "roc_auc_member_higher": auc,
            "roc_auc_separability": max(auc, 1.0 - auc),
            "ks_statistic": ks,
            "mutual_information_nats": float(mi[index]),
            "disjoint_ranges": member_below or nonmember_below,
            "range_order": ("member_below_nonmember" if member_below else
                            "nonmember_below_member" if nonmember_below else "overlap"),
        }
        audit_rows.append(row)

    disjoint = [row for row in audit_rows if row["disjoint_ranges"]]
    direct_feature_names = " ".join(names).lower()
    explicit_checks = {
        "source_file": not any(x in direct_feature_names for x in ("source_file", "source_path")),
        "run_or_session_id": not any(x in direct_feature_names for x in ("run_id", "session_id")),
        "checkpoint_id": not any(x in direct_feature_names for x in ("checkpoint", "adapter_sha", "root_hash")),
        "training_or_eval_mode": not any(x in direct_feature_names for x in ("training_mode", "eval_mode")),
        "step_or_epoch": not any(x in direct_feature_names for x in ("step", "epoch")),
        "path_dependent_tensor_presence_flag": not any("presence" in x or "present" in x for x in names),
        "member_label_or_split_identity": not any(x in direct_feature_names for x in ("member", "label", "split")),
    }
    same_run = {row.get("run_id") for row in member} == {row.get("run_id") for row in nonmember}
    same_package = ({json.dumps(row.get("package_metadata", {}), sort_keys=True) for row in member}
                    == {json.dumps(row.get("package_metadata", {}), sort_keys=True) for row in nonmember})
    all_available = all(row["available_in_both_collection_paths"] for row in audit_rows)

    leakage_rows = [
        {"forbidden_signal": "source_file", "directly_selected": False,
         "empirically_encoded": bool(disjoint), "pass": not bool(disjoint),
         "evidence": f"{len(disjoint)} disjoint-range numerical features identify the aligned source cohort"},
        {"forbidden_signal": "run_or_session_id", "directly_selected": not explicit_checks["run_or_session_id"],
         "empirically_encoded": False, "pass": explicit_checks["run_or_session_id"] and same_run,
         "evidence": f"not selected; same run_id sets across paths={same_run}"},
        {"forbidden_signal": "checkpoint_id", "directly_selected": not explicit_checks["checkpoint_id"],
         "empirically_encoded": False, "pass": explicit_checks["checkpoint_id"] and same_package,
         "evidence": f"not selected; identical package metadata across paths={same_package}"},
        {"forbidden_signal": "training_or_eval_mode", "directly_selected": not explicit_checks["training_or_eval_mode"],
         "empirically_encoded": False, "pass": explicit_checks["training_or_eval_mode"],
         "evidence": "no training/eval-mode feature or field is selected"},
        {"forbidden_signal": "step_or_epoch", "directly_selected": not explicit_checks["step_or_epoch"],
         "empirically_encoded": False, "pass": explicit_checks["step_or_epoch"],
         "evidence": "no step/epoch feature or field is selected"},
        {"forbidden_signal": "path_dependent_tensor_presence_flag",
         "directly_selected": not explicit_checks["path_dependent_tensor_presence_flag"],
         "empirically_encoded": False,
         "pass": explicit_checks["path_dependent_tensor_presence_flag"] and all_available,
         "evidence": f"all {len(audit_rows)} columns are available in both paths={all_available}"},
        {"forbidden_signal": "member_label_or_split_identity",
         "directly_selected": not explicit_checks["member_label_or_split_identity"],
         "empirically_encoded": bool(disjoint), "pass": not bool(disjoint),
         "evidence": f"{len(disjoint)} features have disjoint member/nonmember ranges"},
    ]
    verdict = "FAIL" if any(not row["pass"] for row in leakage_rows) else "PASS"

    write_csv(args.output / "feature_audit.csv", audit_rows)
    write_csv(args.output / "disjoint_range_features.csv", disjoint)
    write_csv(args.output / "leakage_rule_audit.csv", leakage_rows)

    top = sorted(audit_rows, key=lambda row: (row["roc_auc_separability"], row["ks_statistic"]),
                 reverse=True)[:20]
    family_summary = []
    for family in dict.fromkeys(row["tensor_family"] for row in audit_rows):
        selected = [row for row in audit_rows if row["tensor_family"] == family]
        raw_layers = {row["layer"] for row in selected}
        numeric_layers = sorted(layer for layer in raw_layers if isinstance(layer, int))
        named_layers = sorted(str(layer) for layer in raw_layers if not isinstance(layer, int))
        layer_parts = []
        if numeric_layers:
            contiguous = numeric_layers == list(range(numeric_layers[0], numeric_layers[-1] + 1))
            layer_parts.append((f"{numeric_layers[0]}-{numeric_layers[-1]}" if contiguous
                                else ",".join(map(str, numeric_layers))))
        layer_parts.extend(named_layers)
        family_summary.append({
            "tensor_family": family,
            "features": len(selected),
            "layers": ", ".join(layer_parts),
            "transform_invariant": sum(bool(row["invariant_under_per_prompt_transform"])
                                       for row in selected),
            "runtime_metadata_derived": sum(bool(row["derived_from_runtime_metadata"])
                                            for row in selected),
            "available_in_both_paths": sum(bool(row["available_in_both_collection_paths"])
                                           for row in selected),
            "disjoint_ranges": sum(bool(row["disjoint_ranges"]) for row in selected),
        })
    summary = {
        "schema": "authorized_v2_mia_feature_audit", "version": "1.0",
        "verdict": verdict, "cohort": "exact_length_matched",
        "matching_seed": MATCH_SEED, "records_per_class": len(member),
        "feature_count": len(audit_rows), "disjoint_range_feature_count": len(disjoint),
        "all_features_available_in_both_collection_paths": all_available,
        "member_sha256": sha256(args.member), "nonmember_sha256": sha256(args.nonmember),
        "statistics": {"std": "sample (ddof=1)", "missing": "non-finite value",
                       "zero_rate_denominator": "finite observations",
                       "roc_auc": "raw member-higher and direction-free separability",
                       "mutual_information": "sklearn kNN estimator, nats, random_state=20260714"},
        "failed_rules": [row["forbidden_signal"] for row in leakage_rows if not row["pass"]],
        "family_summary": family_summary,
        "top_features": [{key: row[key] for key in
                          ("feature", "roc_auc_member_higher", "roc_auc_separability",
                           "ks_statistic", "mutual_information_nats", "disjoint_ranges")}
                         for row in top],
    }
    (args.output / "feature_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = args.output / "report.md"
    existing = report.read_text().rstrip() if report.exists() else "# V2 Membership Privacy Pilot"
    marker = "\n\n## Exact-length feature audit"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    lines = [existing, "", "## Exact-length feature audit", "",
             f"**MIA audit verdict: {verdict}.** The cohort contains {len(member)} member and "
             f"{len(nonmember)} nonmember records and {len(audit_rows)} feature columns. "
             f"{len(disjoint)} columns have disjoint member/nonmember ranges, so a threshold on any "
             "one of those columns identifies split/source-cohort identity perfectly on this cohort. "
             "This fails the strict leakage rule even though forbidden metadata fields are not directly selected.", "",
             "All per-feature descriptive statistics, missing/zero rates, unique counts, univariate "
             "ROC-AUC, KS, mutual information, disjoint-range flags, and provenance are in "
             "`feature_audit.csv`. The disjoint subset is in `disjoint_range_features.csv`; the "
             "rule-by-rule decision is in `leakage_rule_audit.csv`.", "",
             "### Leakage-rule checks", "",
             "| Forbidden signal | Directly selected | Empirically encoded | Pass | Evidence |",
             "|---|---:|---:|---:|---|"]
    for row in leakage_rows:
        lines.append(f"| {row['forbidden_signal']} | {row['directly_selected']} | "
                     f"{row['empirically_encoded']} | {row['pass']} | {row['evidence']} |")
    lines += ["", "### Feature provenance summary", "",
              "| Tensor family | Features | Layers | Transform-invariant | Runtime metadata | Both paths | Disjoint |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in family_summary:
        lines.append(f"| {row['tensor_family']} | {row['features']} | {row['layers']} | "
                     f"{row['transform_invariant']} | {row['runtime_metadata_derived']} | "
                     f"{row['available_in_both_paths']} | {row['disjoint_ranges']} |")
    lines += ["", "### Strongest univariate features", "",
              "| Feature | Member-higher AUC | Direction-free AUC | KS | MI (nats) | Disjoint |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in top:
        lines.append(f"| `{row['feature']}` | {row['roc_auc_member_higher']:.6f} | "
                     f"{row['roc_auc_separability']:.6f} | {row['ks_statistic']:.6f} | "
                     f"{row['mutual_information_nats']:.6f} | {row['disjoint_ranges']} |")
    lines += ["", "Statistics use sample standard deviation (`ddof=1`). Missing means non-finite. "
              "Zero rates are over finite observations. Mutual information is the sklearn continuous "
              "k-nearest-neighbor estimate in nats with fixed seed 20260714. Every feature is marked "
              "transform-invariant because it is a norm, an order-insensitive logit statistic under "
              "the permutation-only vocabulary mask, an attention quantity under Q/K mask cancellation, "
              "or sequence length. Only sequence length is runtime-metadata-derived."]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
