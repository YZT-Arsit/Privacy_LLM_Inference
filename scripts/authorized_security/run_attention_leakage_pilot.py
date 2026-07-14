#!/usr/bin/env python3
"""Narrow V0/V2 attention-leakage pilot over the frozen E2E corpus.

The attacker features are read only from the immutable view corpora.  Prompt
category and true prompt length are evaluator-side labels joined by sample_id;
they are never added to either attacker view.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_numbers(value) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            if key in {"layer", "last_argmax_by_head", "top_indices"}:
                continue
            out.extend(flatten_numbers(value[key]))
        return out
    return []


def v2_features(row: dict, include_attention: bool) -> list[float]:
    fields = ["transformed_hidden", "masked_q", "masked_k", "masked_v",
              "masked_logits", "tensor_shapes"]
    if include_attention:
        fields.append("attention_scores")
    out = []
    for field in fields:
        out.extend(flatten_numbers(row[field]))
    return out


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2 or np.std(values, ddof=1) == 0:
        return float(np.mean(values)), float(np.mean(values))
    mean = float(np.mean(values)); sem = stats.sem(values)
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - half, mean + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", default="7,1234,2025")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("ordered_sample_ids_match") is not True:
        raise RuntimeError("V0/V2 ordered sample IDs are not frozen and aligned")
    entries = {x["view"]: x for x in manifest["entries"]}
    if set(entries) != {"V0", "V2"}:
        raise RuntimeError("pilot requires exactly frozen V0 and V2 entries")
    paths = {view: Path(entry["path"]) for view, entry in entries.items()}
    for view, entry in entries.items():
        if sha256(paths[view]) != entry["sha256"]:
            raise RuntimeError(f"{view} corpus hash mismatch")
    v0 = load_jsonl(paths["V0"]); v2 = load_jsonl(paths["V2"])
    queries = json.loads(args.queries.read_text())
    qmap = {f"e2e_nlg:test:{int(q['sample_id']):06d}": q for q in queries}
    ids = [row["sample_id"] for row in v0]
    if ids != [row["sample_id"] for row in v2] or any(x not in qmap for x in ids):
        raise RuntimeError("sample-id join failed")

    # Three reasonably populated prompt-structure categories: <=6, 7, or 8 attributes.
    attr_counts = np.asarray([qmap[x]["meaning_representation"].count("], ") + 1 for x in ids])
    category = np.asarray(["le6" if x <= 6 else str(x) for x in attr_counts])
    prompt_len = np.asarray([len(qmap[x]["prompt_ids"]) for x in ids], dtype=float)
    text = np.asarray([row["output_text"] for row in v0])
    x_v2_no = np.asarray([v2_features(row, False) for row in v2], dtype=float)
    x_v2_attn = np.asarray([v2_features(row, True) for row in v2], dtype=float)
    if not np.isfinite(x_v2_no).all() or not np.isfinite(x_v2_attn).all():
        raise RuntimeError("non-finite V2 feature")

    seeds = [int(x) for x in args.seeds.split(",")]
    runs = []
    for seed in seeds:
        train, test = train_test_split(np.arange(len(ids)), test_size=0.4,
                                       random_state=seed, stratify=category)
        views = {
            "V0": (text, TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                          max_features=5000)),
            "V2_without_attention": (x_v2_no, StandardScaler()),
            "V2_with_attention": (x_v2_attn, StandardScaler()),
        }
        for view, (features, scaler) in views.items():
            cat_model = make_pipeline(
                scaler,
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            )
            cat_model.fit(features[train], category[train])
            pred = cat_model.predict(features[test])
            runs.extend([
                {"seed": seed, "view": view, "task": "prompt_category",
                 "metric": "accuracy", "value": accuracy_score(category[test], pred),
                 "n_train": len(train), "n_test": len(test)},
                {"seed": seed, "view": view, "task": "prompt_category",
                 "metric": "macro_f1", "value": f1_score(category[test], pred, average="macro"),
                 "n_train": len(train), "n_test": len(test)},
            ])
            reg_model = make_pipeline(scaler, Ridge(alpha=1.0))
            reg_model.fit(features[train], prompt_len[train])
            reg = reg_model.predict(features[test])
            rounded = np.rint(reg)
            runs.extend([
                {"seed": seed, "view": view, "task": "sequence_length",
                 "metric": "mae_tokens", "value": mean_absolute_error(prompt_len[test], reg),
                 "n_train": len(train), "n_test": len(test)},
                {"seed": seed, "view": view, "task": "sequence_length",
                 "metric": "exact_rounded", "value": accuracy_score(prompt_len[test], rounded),
                 "n_train": len(train), "n_test": len(test)},
                {"seed": seed, "view": view, "task": "sequence_length",
                 "metric": "within_1_token", "value": float(np.mean(np.abs(prompt_len[test] - rounded) <= 1)),
                 "n_train": len(train), "n_test": len(test)},
            ])

    grouped = {}
    for row in runs:
        grouped.setdefault((row["view"], row["task"], row["metric"]), []).append(row["value"])
    aggregate = []
    for (view, task, metric), values in sorted(grouped.items()):
        lo, hi = ci95(values)
        aggregate.append({"view": view, "task": task, "metric": metric,
                          "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                          "ci95_low": lo, "ci95_high": hi, "seeds": len(values)})

    def mean(view: str, task: str, metric: str) -> float:
        return float(np.mean(grouped[(view, task, metric)]))
    increments = []
    for task, metric in (("prompt_category", "accuracy"),
                         ("prompt_category", "macro_f1"),
                         ("sequence_length", "mae_tokens"),
                         ("sequence_length", "exact_rounded"),
                         ("sequence_length", "within_1_token")):
        paired = [a - b for a, b in zip(grouped[("V2_with_attention", task, metric)],
                                        grouped[("V2_without_attention", task, metric)])]
        lo, hi = ci95(paired)
        increments.append({"task": task, "metric": metric,
                           "v2_with_minus_without_attention": float(np.mean(paired)),
                           "ci95_low": lo, "ci95_high": hi})

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "view", "task", "metric", "value", "n_train", "n_test"]
    with (args.output / "per_run_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(runs)
    with (args.output / "aggregate_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregate[0])); w.writeheader(); w.writerows(aggregate)
    result = {
        "schema": "authorized_attention_leakage_pilot", "version": "1.0",
        "manifest_sha256": sha256(args.manifest), "queries_sha256": sha256(args.queries),
        "seeds": seeds, "records": len(ids), "evaluator_labels_not_in_attack_views": True,
        "increments": increments, "complete_tasks": ["prompt_category", "sequence_length"],
        "pending_tasks": ["phrase_token_relation", "repeated_input_linking", "membership",
                          "attention_assisted_surrogate_fidelity"],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Attention/runtime leakage pilot", "",
             "All attacker features come from the immutable V0/V2 view manifest. Labels are joined",
             "by the trusted evaluator and are not present in either attacker corpus.", "",
             "| View | Category accuracy | Category macro-F1 | Length MAE | Exact length | Within 1 token |",
             "|---|---:|---:|---:|---:|---:|"]
    for view in ("V0", "V2_without_attention", "V2_with_attention"):
        lines.append(f"| {view} | {mean(view,'prompt_category','accuracy'):.4f} | "
                     f"{mean(view,'prompt_category','macro_f1'):.4f} | "
                     f"{mean(view,'sequence_length','mae_tokens'):.4f} | "
                     f"{mean(view,'sequence_length','exact_rounded'):.4f} | "
                     f"{mean(view,'sequence_length','within_1_token'):.4f} |")
    lines += ["", "This is a measured pilot, not the complete attention-ablation phase. "
              "Repeated-input, membership, relation, and surrogate-fidelity cells remain pending."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
