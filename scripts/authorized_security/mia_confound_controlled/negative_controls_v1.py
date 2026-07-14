#!/usr/bin/env python3
"""CPU-only label/source controls for the frozen invalid V2 cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


CONTROL_SEEDS = (7, 17, 29, 41, 73, 101, 313, 809, 1234, 2025)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_mia_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(value) -> str:
    if not isinstance(value, bytes):
        value = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size == 2:
            values.append(roc_auc_score(y[indices], score[indices]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def split_indices(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed + offset)
        train, test = next(splitter.split(np.zeros(len(y)), y, groups))
        if np.unique(y[train]).size == 2 and np.unique(y[test]).size == 2:
            return train, test
    raise RuntimeError("could not construct a two-class group split")


def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    train, test = split_indices(y, groups, seed)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    model.fit(x[train], y[train])
    score = model.predict_proba(x[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], score)
    auc = float(roc_auc_score(y[test], score))
    ci = bootstrap_auc(y[test], score, seed)
    return {
        "train_records": len(train), "test_records": len(test),
        "roc_auc": auc, "roc_auc_ci95_low": ci[0], "roc_auc_ci95_high": ci[1],
        "accuracy": float(accuracy_score(y[test], score >= .5)),
        "attack_advantage_max_tpr_minus_fpr": float(np.max(tpr - fpr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=Path, required=True)
    parser.add_argument("--nonmember", type=Path, required=True)
    parser.add_argument("--member-queries", type=Path, required=True)
    parser.add_argument("--nonmember-queries", type=Path, required=True)
    parser.add_argument("--pilot-script", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite negative-control directory: {args.output}")
    args.output.mkdir(parents=True)

    pilot = load_module(args.pilot_script)
    raw_member, raw_nonmember = pilot.load(args.member), pilot.load(args.nonmember)
    member, nonmember = pilot.exact_length_matched(raw_member, raw_nonmember, 20260714)
    rows = member + nonmember
    items = [pilot.feature_items(row, "v2_all") for row in rows]
    names = [item["feature"] for item in items[0]]
    descriptors = items[0]
    if any([item["feature"] for item in current] != names for current in items):
        raise RuntimeError("feature schemas differ")
    x = np.asarray([[item["value"] for item in current] for current in items], dtype=float)
    source_y = np.asarray([1] * len(member) + [0] * len(nonmember), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    member_queries = json.loads(args.member_queries.read_text())
    nonmember_queries = [json.loads(line) for line in args.nonmember_queries.read_text().splitlines() if line.strip()]
    member_map = {f"e2e_nlg:train:{int(row['sample_id']):06d}": row["prompt_ids"] for row in member_queries}
    nonmember_map = {str(row["sample_id"]): row["query_prompt_token_ids"] for row in nonmember_queries}
    prompt_ids = []
    for row in member:
        prompt_ids.append(member_map[row["sample_id"]])
    for row in nonmember:
        prompt_ids.append(nonmember_map[row["sample_id"]])
    exact_groups = np.asarray([digest(value) for value in prompt_ids])
    semantic_groups = np.asarray([digest(normalize(tokenizer.decode(value, skip_special_tokens=True)))
                                  for value in prompt_ids])

    families = np.asarray([item["tensor_family"] for item in descriptors])
    feature_sets = {
        "F0_shape_only": np.flatnonzero(families == "tensor_shapes"),
        "F2_invariant_hidden": np.flatnonzero(np.isin(families, ["transformed_hidden", "masked_q", "masked_k", "masked_v"])),
        "F3_attention_no_shape": np.flatnonzero(families == "attention_scores"),
        "F4_masked_logits": np.flatnonzero(families == "masked_logits"),
        "F5_combined_metadata_free": np.flatnonzero(families != "tensor_shapes"),
    }
    control_rows = []
    for seed in CONTROL_SEEDS:
        rng = np.random.default_rng(seed)
        definitions = {
            "N0_label_permutation": (x, rng.permutation(source_y), semantic_groups),
            "N1_pseudo_within_members": (x[:len(member)], rng.permutation(np.arange(len(member)) % 2),
                                          semantic_groups[:len(member)]),
            "N2_pseudo_within_nonmembers": (x[len(member):], rng.permutation(np.arange(len(nonmember)) % 2),
                                             semantic_groups[len(member):]),
            "N3_source_train_vs_test": (x, source_y, semantic_groups),
        }
        for control, (control_x, control_y, control_groups) in definitions.items():
            for feature_set, indices in feature_sets.items():
                result = evaluate(control_x[:, indices], np.asarray(control_y, dtype=int), control_groups, seed)
                control_rows.append({"control": control, "feature_set": feature_set,
                                     "seed": seed, "features": len(indices), **result})
    write_csv(args.output / "negative_controls.csv", control_rows)

    summaries = []
    for control in dict.fromkeys(row["control"] for row in control_rows):
        for feature_set in feature_sets:
            selected = [row for row in control_rows if row["control"] == control and row["feature_set"] == feature_set]
            aucs = np.asarray([row["roc_auc"] for row in selected])
            summaries.append({"control": control, "feature_set": feature_set,
                              "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std(ddof=1)),
                              "materially_above_chance": bool(control != "N3_source_train_vs_test" and aucs.mean() > .60)})
    write_csv(args.output / "negative_control_summary.csv", summaries)

    member_exact = {digest(row["prompt_ids"]) for row in member_queries}
    nonmember_exact = {digest(row["query_prompt_token_ids"]) for row in nonmember_queries}
    member_semantic = {digest(normalize(tokenizer.decode(row["prompt_ids"], skip_special_tokens=True))) for row in member_queries}
    nonmember_semantic = {digest(normalize(tokenizer.decode(row["query_prompt_token_ids"], skip_special_tokens=True))) for row in nonmember_queries}
    pseudo_failures = [row for row in summaries if row["materially_above_chance"]]
    pipeline = {
        "schema": "frozen_v2_negative_control_pipeline_audit", "version": "1.0",
        "status": "FAIL" if pseudo_failures else "PASS",
        "failure_threshold": "mean raw ROC-AUC > 0.60 over seeds for N0/N1/N2",
        "fit_scope": {
            "group_split_before_fit": True, "group_key": "normalized decoded prompt hash",
            "imputer": "median; fit attack-train only", "scaler": "StandardScaler; fit attack-train only",
            "pca": "not used", "feature_selection": "not used",
            "classifier": "LogisticRegression; fit attack-train only",
        },
        "N4_sample_order_file_origin": {
            "raw_concatenation_order_encodes_source": True,
            "order_or_file_origin_included_as_feature": False,
            "member_file_records": len(raw_member), "nonmember_file_records": len(raw_nonmember),
            "audit_split_shuffles_groups": True,
        },
        "N5_duplicate_near_duplicate": {
            "member_unique_exact_prompts": len(member_exact),
            "nonmember_unique_exact_prompts": len(nonmember_exact),
            "cross_source_exact_prompt_overlap": len(member_exact & nonmember_exact),
            "cross_source_normalized_prompt_overlap": len(member_semantic & nonmember_semantic),
            "duplicate_group_crosses_attack_split": False,
        },
        "failed_pseudo_controls": pseudo_failures,
    }
    (args.output / "pipeline_leakage_audit.json").write_text(json.dumps(pipeline, indent=2) + "\n")
    lines = ["# Frozen-Cohort Negative Controls", "",
             f"Pipeline verdict for shuffled/pseudo controls: **{pipeline['status']}**.", "",
             "| Control | Feature set | AUC mean ± SD | Materially above chance |",
             "|---|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['control']} | {row['feature_set']} | "
                     f"{row['auc_mean']:.3f} ± {row['auc_std']:.3f} | {row['materially_above_chance']} |")
    lines += ["", "N3 is deliberately a source-label classifier. High N3 performance confirms that the "
              "old membership task is source classification; it is not a pipeline-control failure.", "",
              f"Exact cross-source prompt overlap: {len(member_exact & nonmember_exact)}; normalized decoded "
              f"overlap: {len(member_semantic & nonmember_semantic)}. Preprocessing is encapsulated in an "
              "attack-train-only sklearn Pipeline."]
    (args.output / "negative_control_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
