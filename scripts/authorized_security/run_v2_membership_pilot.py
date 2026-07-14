#!/usr/bin/env python3
"""Measured V2 membership pilot with a sequence-length-matched control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values) -> list[float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max())]


def feature_items(row: dict, variant: str) -> list[dict]:
    """Return named features with transform/runtime provenance.

    The summaries deliberately preserve the original pilot's column order.  The
    transform-invariance flags follow the package algebra: residual/Q/K/V masks
    are orthogonal, attention scores cancel the shared Q/K mask, and the vocab
    mask is permutation-only.
    """
    seq = float(row["tensor_shapes"]["sequence_length"])
    out: list[dict] = []

    def add(name: str, value: float, family: str, layer: int | str,
            statistic: str, runtime_metadata: bool = False) -> None:
        out.append({
            "feature": name,
            "value": float(value),
            "tensor_family": family,
            "layer": layer,
            "statistic": statistic,
            "invariant_under_per_prompt_transform": True,
            "derived_from_runtime_metadata": runtime_metadata,
        })

    def add_reductions(prefix: str, values, family: str, layer: int | str,
                       source_statistic: str) -> None:
        for reduction, value in zip(("mean", "std", "min", "max"), stats(values)):
            add(f"{prefix}.{reduction}", value, family, layer,
                f"{reduction}({source_statistic})")

    if variant == "shape_only":
        add("tensor_shapes.sequence_length", seq, "tensor_shapes", "global",
            "sequence_length", runtime_metadata=True)
        return out
    if variant in {"v2_no_attention", "v2_all"}:
        hidden = row["transformed_hidden"]
        add("transformed_hidden.final.last_l2", hidden["final_last_l2"],
            "transformed_hidden", "final", "last_token_l2")
        for item in hidden["per_layer"]:
            layer = int(item["layer"])
            add(f"transformed_hidden.layer_{layer:02d}.last_l2", item["last_l2"],
                "transformed_hidden", layer, "pre_attention_last_token_l2")
        logits = row["masked_logits"]
        for key in ["mean", "std", "l2", "min", "max"]:
            add(f"masked_logits.{key}", logits[key], "masked_logits", "final", key)
        add_reductions("masked_logits.top_values", logits["top_values"],
                       "masked_logits", "final", "top_32_values")
        for field in ["masked_q", "masked_k", "masked_v"]:
            for layer in row[field]:
                layer_id = int(layer["layer"])
                add_reductions(f"{field}.layer_{layer_id:02d}.last_l2_by_head",
                               layer["last_l2_by_head"], field, layer_id,
                               "last_token_l2_by_head")
    if variant in {"attention_only", "v2_all"}:
        for layer in row["attention_scores"]:
            layer_id = int(layer["layer"])
            for key in ["last_entropy_by_head", "last_max_by_head",
                        "true_score_last_mean_by_head", "true_score_last_std_by_head"]:
                add_reductions(f"attention_scores.layer_{layer_id:02d}.{key}",
                               layer[key], "attention_scores", layer_id, key)
            argmax = np.asarray(layer["last_argmax_by_head"], dtype=float) / max(seq - 1.0, 1.0)
            add_reductions(f"attention_scores.layer_{layer_id:02d}.normalized_last_argmax_by_head",
                           argmax, "attention_scores", layer_id,
                           "last_argmax_by_head/(sequence_length-1)")
    if variant == "v2_all":
        add("tensor_shapes.sequence_length", seq, "tensor_shapes", "global",
            "sequence_length", runtime_metadata=True)
    return out


def features(row: dict, variant: str) -> list[float]:
    return [item["value"] for item in feature_items(row, variant)]


def exact_length_matched(member: list[dict], nonmember: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_m: dict[int, list[dict]] = {}
    by_n: dict[int, list[dict]] = {}
    for row in member:
        by_m.setdefault(int(row["tensor_shapes"]["sequence_length"]), []).append(row)
    for row in nonmember:
        by_n.setdefault(int(row["tensor_shapes"]["sequence_length"]), []).append(row)
    left, right = [], []
    for length in sorted(set(by_m) & set(by_n)):
        rng.shuffle(by_m[length]); rng.shuffle(by_n[length])
        count = min(len(by_m[length]), len(by_n[length]))
        left.extend(by_m[length][:count]); right.extend(by_n[length][:count])
    return left, right


def tpr_at(fpr: np.ndarray, tpr: np.ndarray, limit: float) -> float:
    eligible = tpr[fpr <= limit]
    return float(eligible.max()) if eligible.size else 0.0


def bootstrap_auc(y: np.ndarray, score: np.ndarray, seed: int, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed); values = []
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def evaluate(member: list[dict], nonmember: list[dict], cohort: str, variant: str, seed: int) -> dict:
    rows = member + nonmember
    y = np.asarray([1] * len(member) + [0] * len(nonmember), dtype=int)
    x = np.asarray([features(row, variant) for row in rows], dtype=float)
    train, test = train_test_split(np.arange(len(y)), test_size=.4, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    model.fit(x[train], y[train]); score = model.predict_proba(x[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], score); auc = float(roc_auc_score(y[test], score))
    return {
        "cohort": cohort, "variant": variant, "seed": seed,
        "member_records": len(member), "nonmember_records": len(nonmember),
        "train_records": len(train), "test_records": len(test), "features": x.shape[1],
        "roc_auc": auc, "roc_auc_ci95": bootstrap_auc(y[test], score, seed),
        "attack_advantage": float(np.max(tpr - fpr)),
        "tpr_at_1pct_fpr": tpr_at(fpr, tpr, .01),
        "tpr_at_0_1pct_fpr": tpr_at(fpr, tpr, .001),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", type=Path, required=True)
    ap.add_argument("--nonmember", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite MIA output directory: {args.output}")
    args.output.mkdir(parents=True)
    member, nonmember = load(args.member), load(args.nonmember)
    if len(member) != len(nonmember):
        raise RuntimeError("raw cohort must be balanced")
    matched_m, matched_n = exact_length_matched(member, nonmember, 20260714)
    if len(matched_m) < 100:
        raise RuntimeError("too few exact-length-matched records")
    rows = []
    variants = ["shape_only", "v2_no_attention", "attention_only", "v2_all"]
    for cohort, left, right in [("raw_balanced", member, nonmember),
                                ("exact_length_matched", matched_m, matched_n)]:
        for variant in variants:
            for seed in [7, 1234, 2025]:
                rows.append(evaluate(left, right, cohort, variant, seed))
    fields = ["cohort", "variant", "seed", "member_records", "nonmember_records",
              "train_records", "test_records", "features", "roc_auc", "roc_auc_ci95",
              "attack_advantage", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr"]
    with (args.output / "per_run_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for cohort in ["raw_balanced", "exact_length_matched"]:
        for variant in variants:
            selected = [row for row in rows if row["cohort"] == cohort and row["variant"] == variant]
            aggregate.append({
                "cohort": cohort, "variant": variant, "seeds": len(selected),
                "records_per_class": selected[0]["member_records"],
                "roc_auc_mean": float(np.mean([x["roc_auc"] for x in selected])),
                "roc_auc_std": float(np.std([x["roc_auc"] for x in selected], ddof=1)),
                "advantage_mean": float(np.mean([x["attack_advantage"] for x in selected])),
                "tpr_at_1pct_fpr_mean": float(np.mean([x["tpr_at_1pct_fpr"] for x in selected])),
                "tpr_at_0_1pct_fpr_mean": float(np.mean([x["tpr_at_0_1pct_fpr"] for x in selected])),
            })
    with (args.output / "aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    result = {
        "schema": "authorized_v2_membership_pilot", "version": "1.0",
        "status": "diagnostic_until_aligned_v0_member_outputs_exist",
        "member_sha256": sha256(args.member), "nonmember_sha256": sha256(args.nonmember),
        "raw_records_per_class": len(member), "exact_length_matched_records_per_class": len(matched_m),
        "classifier": "standardized_logistic_regression", "seeds": [7, 1234, 2025],
        "aggregate": aggregate,
        "limitations": [
            "V0 member outputs are not yet collected, so the preregistered V2-minus-V0 comparison is incomplete.",
            "Raw train/test results may include dataset-split distribution shift; exact-length matching removes only sequence-length shift.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# V2 Membership Privacy Pilot", "",
             "Measured on 500 member and 500 non-member V2 records. The exact-length-matched cohort is the stronger control.", "",
             "| Cohort | Features | n/class | ROC-AUC mean±sd | Advantage | TPR@1%FPR | TPR@0.1%FPR |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append(f"| {row['cohort']} | {row['variant']} | {row['records_per_class']} | "
                     f"{row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} | {row['advantage_mean']:.4f} | "
                     f"{row['tpr_at_1pct_fpr_mean']:.4f} | {row['tpr_at_0_1pct_fpr_mean']:.4f} |")
    lines += ["", "This is not the final MIA conclusion because matched V0 member outputs are still missing."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
