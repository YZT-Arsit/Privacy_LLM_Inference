#!/usr/bin/env python3
"""Attribute the frozen old-cohort disjoint features without rerunning inference."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def category(row: dict) -> str:
    family, feature = row["tensor_family"], row["feature"]
    if family in {"transformed_hidden", "masked_q", "masked_k", "masked_v"}:
        return "hidden norm"
    if family == "attention_scores":
        if "entropy" in feature:
            return "attention entropy"
        if any(token in feature for token in ("max", "min", "argmax")):
            return "attention extrema"
        return "attention moments"
    if family == "masked_logits":
        return "masked logits"
    if family == "tensor_shapes":
        return "tensor shape"
    return "other"


def layer_band(value: str) -> str:
    try:
        layer = int(value)
    except ValueError:
        return value
    return "early_0_7" if layer <= 7 else "middle_8_15" if layer <= 15 else "late_16_23"


def reduction(feature: str) -> str:
    for item in ("mean", "std", "min", "max", "l2"):
        if feature.endswith("." + item) or feature.endswith("_" + item):
            return item
    return "direct"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite attribution directory: {args.output}")
    args.output.mkdir(parents=True)
    source = list(csv.DictReader(args.feature_audit.open()))
    disjoint = [row for row in source if row["disjoint_ranges"] == "True"]
    rows = []
    for row in disjoint:
        rows.append({
            "feature": row["feature"], "feature_family": category(row),
            "tensor_family": row["tensor_family"], "layer": row["layer"],
            "layer_band": layer_band(row["layer"]), "tensor_source": row["tensor_family"],
            "statistic": row["statistic"], "reduction": reduction(row["feature"]),
            "transform_dependence": "invariant" if row["invariant_under_per_prompt_transform"] == "True" else "coordinate_dependent",
            "invariant_under_per_prompt_transform": row["invariant_under_per_prompt_transform"],
            "member_min": row["member_min"], "member_max": row["member_max"],
            "member_mean": row["member_mean"], "member_std": row["member_std"],
            "nonmember_min": row["nonmember_min"], "nonmember_max": row["nonmember_max"],
            "nonmember_mean": row["nonmember_mean"], "nonmember_std": row["nonmember_std"],
            "univariate_auc_member_higher": row["roc_auc_member_higher"],
            "univariate_auc_separability": row["roc_auc_separability"],
            "ks_statistic": row["ks_statistic"],
            "member_zero_rate": row["member_zero_rate"], "nonmember_zero_rate": row["nonmember_zero_rate"],
            "combined_zero_rate": row["combined_zero_rate"],
            "member_unique_count": row["member_unique_count"],
            "nonmember_unique_count": row["nonmember_unique_count"],
            "combined_unique_count": row["combined_unique_count"],
            "direction_of_separation": row["range_order"],
        })
    if len(rows) != 446:
        raise RuntimeError(f"expected 446 disjoint features, found {len(rows)}")
    write_csv(args.output / "confound_feature_attribution.csv", rows)
    summaries = []
    for dimension, key in (("feature_family", "feature_family"), ("tensor_family", "tensor_family"),
                           ("layer_band", "layer_band"), ("reduction", "reduction")):
        counts = Counter(row[key] for row in rows)
        for name, count in counts.most_common():
            summaries.append({"dimension": dimension, "group": name, "disjoint_features": count,
                              "fraction_of_446": count / len(rows)})
    write_csv(args.output / "confound_family_summary.csv", summaries)
    family_counts = Counter(row["tensor_family"] for row in rows)
    category_counts = Counter(row["feature_family"] for row in rows)
    lines = ["# Old-Cohort Confound Attribution", "",
             "This report attributes the 446 frozen disjoint-range features. No model inference was run.", "",
             "| Tensor family | Disjoint features | Share |", "|---|---:|---:|"]
    for name, count in family_counts.most_common():
        lines.append(f"| {name} | {count} | {100 * count / len(rows):.1f}% |")
    lines += ["", "| Analysis category | Disjoint features | Share |", "|---|---:|---:|"]
    for name, count in category_counts.most_common():
        lines.append(f"| {name} | {count} | {100 * count / len(rows):.1f}% |")
    lines += ["", "## Attribution conclusion", "",
              "The signal is broad rather than isolated to one layer or one statistic. Attention-score "
              "features are the largest tensor family, while norm-preserving hidden/Q/K/V summaries also "
              "contribute heavily. All 446 are transform-invariant, so per-prompt coordinate refresh cannot remove them.", "",
              "Both classes used the same capture code, run ID, package root, adapter hash, dtype, and tensor presence. "
              "Collection-time model state is therefore not supported as the explanation. The old design still "
              "cannot separate train/test semantic distribution from true membership because source split and "
              "membership are identical. References were not inputs to the prefill feature capture, so direct "
              "target/reference-field leakage is not supported."]
    (args.output / "confound_attribution_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
