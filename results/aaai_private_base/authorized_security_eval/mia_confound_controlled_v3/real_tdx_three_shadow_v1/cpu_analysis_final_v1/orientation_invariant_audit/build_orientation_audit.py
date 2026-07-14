#!/usr/bin/env python3
"""Append-only orientation-invariant audit from persisted aggregate results only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MODE = "REAL_TDX_BACKED_VIEW"
EVALUATION = "three-shadow leave-one-shadow-out"
SCHEMA = "common-semantics v2, 610 columns"
META = {"collection_mode": MODE, "evaluation": EVALUATION, "feature_schema": SCHEMA}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(f"append-only refusal: {path}")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text.rstrip() + "\n")


def md_header(title: str) -> str:
    return (f"# {title}\n\nCollection mode: {MODE}  \nEvaluation: {EVALUATION}  \n"
            f"Feature schema: {SCHEMA}\n")


def main() -> None:
    existing = [p for p in HERE.iterdir() if p.name not in {"build_orientation_audit.py", "__pycache__"}]
    if existing:
        raise RuntimeError(f"append-only refusal; outputs exist: {existing}")

    per_path = PARENT / "per_fold_metrics.csv"
    group_path = PARENT / "group_disjoint_metrics.csv"
    ci_path = PARENT / "confidence_intervals.json"
    per = pd.read_csv(per_path)
    group = pd.read_csv(group_path)

    # Exact deterministic transform of the already persisted raw AUCs; no fit occurs.
    per["raw_auc_primary_preserved"] = per["roc_auc"]
    per["symmetric_auc"] = per["roc_auc"].map(lambda x: max(x, 1.0 - x))
    per_out = per[["collection_mode", "evaluation", "feature_schema", "protocol", "held_out_shadow",
                   "view", "classifier", "raw_auc_primary_preserved", "symmetric_auc"]]
    per_out.to_csv(HERE / "per_fold_symmetric_auc.csv", index=False)

    macro_rows = []
    fold_rows = per_out[per_out["protocol"].isin(["primary_loso", "leave_one_family_out"])]
    for keys, frame in fold_rows.groupby(["protocol", "view", "classifier"], dropna=False):
        macro_rows.append({**META, "protocol": keys[0], "view": keys[1], "classifier": keys[2],
                           "folds": len(frame), "raw_auc_macro_mean": frame["raw_auc_primary_preserved"].mean(),
                           "raw_auc_macro_std": frame["raw_auc_primary_preserved"].std(ddof=0),
                           "symmetric_auc_macro_mean": frame["symmetric_auc"].mean(),
                           "symmetric_auc_macro_std": frame["symmetric_auc"].std(ddof=0)})
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(HERE / "macro_symmetric_auc.csv", index=False)

    group["raw_auc_primary_preserved"] = group["roc_auc"]
    group["symmetric_auc"] = group["roc_auc"].map(lambda x: max(x, 1.0 - x))
    group_out = group[["collection_mode", "evaluation", "feature_schema", "protocol", "held_out_shadow",
                       "view", "classifier", "raw_auc_primary_preserved", "symmetric_auc"]]
    group_out.to_csv(HERE / "group_disjoint_symmetric_auc.csv", index=False)

    primary = per_out[(per_out["protocol"] == "primary_loso") & (per_out["classifier"] == "logistic_l2")]
    paired_rows = []
    for view in ("V2_combined", "V0_plus_V2"):
        comparison = f"{view}_minus_V0"
        effects = []
        for held in (1, 2, 3):
            a = primary[(primary["held_out_shadow"].astype(str) == str(held)) & (primary["view"] == view)].iloc[0]
            b = primary[(primary["held_out_shadow"].astype(str) == str(held)) & (primary["view"] == "V0")].iloc[0]
            effect = float(a["symmetric_auc"] - b["symmetric_auc"]); effects.append(effect)
            paired_rows.append({**META, "comparison": comparison, "scope": f"held_out_shadow_{held}",
                                "symmetric_auc_difference": effect, "ci95_low": None, "ci95_high": None,
                                "bootstrap_replicates": 0, "bootstrap_status": "WITHHELD_NO_PREDICTION_LEVEL_INPUT"})
        paired_rows.append({**META, "comparison": comparison, "scope": "equal_weight_macro",
                            "symmetric_auc_difference": sum(effects) / 3.0, "ci95_low": None, "ci95_high": None,
                            "bootstrap_replicates": 0, "bootstrap_status": "WITHHELD_NO_PREDICTION_LEVEL_INPUT"})
    pd.DataFrame(paired_rows).to_csv(HERE / "paired_symmetric_auc_differences.csv", index=False)

    primary_macro = macro[(macro["protocol"] == "primary_loso") & (macro["classifier"] == "logistic_l2")]
    group_macro = group_out.groupby(["view", "classifier"], as_index=False).agg(
        raw_auc_macro_mean=("raw_auc_primary_preserved", "mean"),
        symmetric_auc_macro_mean=("symmetric_auc", "mean"))
    summary = {}
    for view in ("V0", "V2_combined", "V0_plus_V2"):
        row = primary_macro[primary_macro["view"] == view].iloc[0]
        grow = group_macro[(group_macro["view"] == view) & (group_macro["classifier"] == "logistic_l2")].iloc[0]
        summary[view] = {"raw_auc_macro": float(row["raw_auc_macro_mean"]),
                         "symmetric_auc_macro": float(row["symmetric_auc_macro_mean"]),
                         "group_disjoint_raw_auc_macro": float(grow["raw_auc_macro_mean"]),
                         "group_disjoint_symmetric_auc_macro": float(grow["symmetric_auc_macro_mean"])}

    bootstrap = {
        **META,
        "status": "WITHHELD_NO_PREDICTION_LEVEL_INPUT",
        "requested_replicates": 2000,
        "completed_replicates": 0,
        "no_retraining_performed": True,
        "reason": "The frozen analysis persisted only aggregate raw-AUC differences and percentile endpoints; held-out y/score arrays and bootstrap replicate values were not persisted.",
        "why_not_derivable": "symmetric_AUC=max(AUC,1-AUC) is nonlinear, so a paired symmetric-AUC interval cannot be reconstructed from the raw difference interval or its endpoints.",
        "minimum_additional_artifact": "For every held-out shadow and view, persist the original sample order, binary y, and frozen decision score; alternatively persist all paired per-replicate AUCs for both views.",
        "forbidden_workaround_not_used": "Classifiers were not refit and predictions were not regenerated.",
        "source_confidence_intervals_sha256": sha(ci_path),
    }
    write(HERE / "bootstrap_status.json", bootstrap)

    table = [md_header("Orientation-invariant MIA diagnostic"),
             "Raw AUC remains the primary cross-shadow generalization result. Symmetric AUC is a secondary conservative diagnostic that permits post-hoc score inversion.", "",
             "| View | Raw AUC | Symmetric AUC | Group-disjoint raw AUC | Group-disjoint symmetric AUC |",
             "|---|---:|---:|---:|---:|"]
    for view in ("V0", "V2_combined", "V0_plus_V2"):
        z = summary[view]
        table.append(f"| {view} | {z['raw_auc_macro']:.3f} | {z['symmetric_auc_macro']:.3f} | {z['group_disjoint_raw_auc_macro']:.3f} | {z['group_disjoint_symmetric_auc_macro']:.3f} |")
    paired = pd.DataFrame(paired_rows)
    table += ["", "## Paired symmetric-AUC differences", "", "| Comparison | Fold effects | Equal-weight macro | 95% CI |",
              "|---|---|---:|---|"]
    for comparison in ("V2_combined_minus_V0", "V0_plus_V2_minus_V0"):
        q = paired[paired["comparison"] == comparison]
        folds = ", ".join(f"{x:+.3f}" for x in q[q["scope"].str.startswith("held")]["symmetric_auc_difference"])
        macro_effect = q[q["scope"] == "equal_weight_macro"]["symmetric_auc_difference"].iloc[0]
        table.append(f"| {comparison} | {folds} | {macro_effect:+.3f} | withheld: prediction-level scores absent |")
    table += ["", "## Interpretation", "",
              "- Raw AUC below 0.5 indicates unstable or reversed score orientation on a held-out shadow.",
              "- Symmetric AUC is a conservative diagnostic allowing score inversion after observing orientation.",
              "- Neither raw nor symmetric AUC permits interpreting V2 below 0.5 as active protection.",
              "- The supported conclusion is only that V2 does not outperform the external V0 view.",
              "- Paired bootstrap CIs are withheld because their exact calculation requires the frozen sample-level predictions, which were not persisted. No classifier was retrained."]
    write(HERE / "audit_report.md", "\n".join(table))

    readme = (md_header("Orientation-invariant audit provenance") +
              "\nThis directory is append-only and derives deterministic symmetric-AUC point estimates from the previously frozen CSV metrics. Existing analysis files were not modified. No classifier fit, prediction regeneration, GPU operation, or data recollection occurred.\n\n"
              f"Inputs:\n\n- `per_fold_metrics.csv`: `{sha(per_path)}`\n- `group_disjoint_metrics.csv`: `{sha(group_path)}`\n- `confidence_intervals.json`: `{sha(ci_path)}`\n")
    write(HERE / "README.md", readme)

    outputs = ["README.md", "per_fold_symmetric_auc.csv", "macro_symmetric_auc.csv",
               "group_disjoint_symmetric_auc.csv", "paired_symmetric_auc_differences.csv",
               "bootstrap_status.json", "audit_report.md", "build_orientation_audit.py"]
    hashes = {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in outputs}
    write(HERE / "artifact_hashes.json", {**META, "status": "PASS_FOR_AVAILABLE_INPUTS", "files": hashes})
    bad = [name for name, meta in hashes.items() if sha(HERE / name) != meta["sha256"]]
    if bad: raise RuntimeError(f"hash verification failed: {bad}")
    print(json.dumps({"status": "PARTIAL_BOOTSTRAP_WITHHELD", "summary": summary,
                      "paired_macro": {c: float(paired[(paired.comparison == c) & (paired.scope == 'equal_weight_macro')].symmetric_auc_difference.iloc[0]) for c in paired.comparison.unique()},
                      "hashes": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
