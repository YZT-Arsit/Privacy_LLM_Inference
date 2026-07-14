#!/usr/bin/env python3
"""Metadata-free exact-length attention/KV source-confound diagnostic.

The binary target is collection source, not membership. Membership inference is
explicitly withheld until corrected shadow captures are complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = [7, 1234, 2025]
FAMILIES = {
    "shape_only": {"tensor_shapes"},
    "attention": {"attention_scores"},
    "kv": {"masked_k", "masked_v"},
    "hidden": {"transformed_hidden"},
    "logit": {"masked_logits"},
    "combined_metadata_free": {
        "attention_scores", "masked_k", "masked_v", "transformed_hidden", "masked_logits"
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stats(values) -> list[float]:
    x = np.asarray(values, dtype=float)
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max())]


def descriptors(row: dict) -> list[dict]:
    seq = float(row["tensor_shapes"]["sequence_length"])
    out = [{"name": "tensor_shapes.sequence_length", "value": seq,
            "family": "tensor_shapes", "layer": "global", "statistic": "sequence_length",
            "invariant": True, "runtime_metadata": True}]

    def reductions(prefix, values, family, layer, source):
        for reduction, value in zip(("mean", "std", "min", "max"), stats(values)):
            out.append({"name": f"{prefix}.{reduction}", "value": value, "family": family,
                        "layer": layer, "statistic": f"{reduction}({source})",
                        "invariant": True, "runtime_metadata": False})

    hidden = row["transformed_hidden"]
    out.append({"name": "transformed_hidden.final.last_l2", "value": hidden["final_last_l2"],
                "family": "transformed_hidden", "layer": "final", "statistic": "last_token_l2",
                "invariant": True, "runtime_metadata": False})
    for item in hidden["per_layer"]:
        out.append({"name": f"transformed_hidden.layer_{item['layer']:02d}.last_l2",
                    "value": item["last_l2"], "family": "transformed_hidden",
                    "layer": int(item["layer"]), "statistic": "pre_attention_last_token_l2",
                    "invariant": True, "runtime_metadata": False})
    logits = row["masked_logits"]
    for key in ("mean", "std", "l2", "min", "max"):
        out.append({"name": f"masked_logits.{key}", "value": logits[key], "family": "masked_logits",
                    "layer": "final", "statistic": key, "invariant": True, "runtime_metadata": False})
    reductions("masked_logits.top_values", logits["top_values"], "masked_logits", "final", "top_values")
    for family in ("masked_k", "masked_v"):
        for layer in row[family]:
            reductions(f"{family}.layer_{layer['layer']:02d}.last_l2_by_head",
                       layer["last_l2_by_head"], family, int(layer["layer"]), "last_l2_by_head")
    for layer in row["attention_scores"]:
        lid = int(layer["layer"])
        for key in ("last_entropy_by_head", "last_max_by_head", "true_score_last_mean_by_head",
                    "true_score_last_std_by_head"):
            reductions(f"attention_scores.layer_{lid:02d}.{key}", layer[key],
                       "attention_scores", lid, key)
        normalized = np.asarray(layer["last_argmax_by_head"], float) / max(seq - 1, 1)
        reductions(f"attention_scores.layer_{lid:02d}.normalized_last_argmax_by_head", normalized,
                   "attention_scores", lid, "last_argmax/(sequence_length-1)")
    return out


def exact_match(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict], int]:
    rng = np.random.default_rng(20260714)
    aa, bb = {}, {}
    for row in a:
        aa.setdefault(int(row["tensor_shapes"]["sequence_length"]), []).append(row)
    for row in b:
        bb.setdefault(int(row["tensor_shapes"]["sequence_length"]), []).append(row)
    fixed_length = max(set(aa) & set(bb), key=lambda length: (min(len(aa[length]), len(bb[length])), -length))
    ia, ib = rng.permutation(len(aa[fixed_length])), rng.permutation(len(bb[fixed_length]))
    n = min(len(ia), len(ib))
    left = [aa[fixed_length][i] for i in ia[:n]]
    right = [bb[fixed_length][i] for i in ib[:n]]
    return left, right, fixed_length


def bootstrap_auc(y, score, seed, n=1000):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-a", required=True, type=Path)
    ap.add_argument("--source-b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    a, b, fixed_length = exact_match(load(args.source_a), load(args.source_b))
    if len(a) != len(b) or len(a) < 20:
        raise RuntimeError("insufficient balanced exact-length cohort")
    rows = a + b; y = np.r_[np.ones(len(a), int), np.zeros(len(b), int)]
    desc = [descriptors(r) for r in rows]
    names = [d["name"] for d in desc[0]]
    if any([d["name"] for d in x] != names for x in desc):
        raise RuntimeError("feature path mismatch")
    xall = np.asarray([[d["value"] for d in x] for x in desc], float)
    if not np.isfinite(xall).all():
        raise RuntimeError("non-finite features")
    lengths_a = sorted(int(x["tensor_shapes"]["sequence_length"]) for x in a)
    lengths_b = sorted(int(x["tensor_shapes"]["sequence_length"]) for x in b)
    if lengths_a != lengths_b:
        raise RuntimeError("length multiset mismatch")

    runs = []
    for view, allowed in FAMILIES.items():
        cols = [i for i, d in enumerate(desc[0]) if d["family"] in allowed]
        x = xall[:, cols]
        for seed in SEEDS:
            tr, te = train_test_split(np.arange(len(y)), test_size=.4, random_state=seed, stratify=y)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=4000, class_weight="balanced", random_state=seed))
            model.fit(x[tr], y[tr]); score = model.predict_proba(x[te])[:, 1]
            auc = float(roc_auc_score(y[te], score))
            runs.append({"view": view, "seed": seed, "control": "true_source_label",
                         "features": len(cols), "roc_auc": auc,
                         "accuracy": float(accuracy_score(y[te], score >= .5)),
                         "roc_auc_ci95": json.dumps(bootstrap_auc(y[te], score, seed))})
            rng = np.random.default_rng(seed + 9000); shuffled = rng.permutation(y[tr])
            control = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=4000, class_weight="balanced", random_state=seed))
            control.fit(x[tr], shuffled); cscore = control.predict_proba(x[te])[:, 1]
            runs.append({"view": view, "seed": seed, "control": "train_label_shuffle",
                         "features": len(cols), "roc_auc": float(roc_auc_score(y[te], cscore)),
                         "accuracy": float(accuracy_score(y[te], cscore >= .5)),
                         "roc_auc_ci95": json.dumps(bootstrap_auc(y[te], cscore, seed + 1))})

    univariate = []
    for i, d in enumerate(desc[0]):
        if d["family"] == "tensor_shapes":
            views = ["shape_only"]
        else:
            views = [v for v, fs in FAMILIES.items() if d["family"] in fs]
        va, vb = xall[:len(a), i], xall[len(a):, i]
        auc = float(roc_auc_score(y, xall[:, i])) if len(np.unique(xall[:, i])) > 1 else .5
        univariate.append({"feature": d["name"], "family": d["family"], "layer": d["layer"],
                           "statistic": d["statistic"], "invariant_under_per_prompt_transform": d["invariant"],
                           "derived_from_runtime_metadata": d["runtime_metadata"],
                           "views": ";".join(views), "roc_auc_source_higher": auc,
                           "roc_auc_separability": max(auc, 1 - auc),
                           "ks_statistic": float(ks_2samp(va, vb).statistic),
                           "disjoint_ranges": bool(va.max() < vb.min() or vb.max() < va.min())})
    aggregate = []
    for view in FAMILIES:
        for control in ("true_source_label", "train_label_shuffle"):
            selected = [x for x in runs if x["view"] == view and x["control"] == control]
            vals = np.asarray([x["roc_auc"] for x in selected])
            aggregate.append({"view": view, "control": control, "features": selected[0]["features"],
                              "roc_auc_mean": float(vals.mean()),
                              "roc_auc_std": float(vals.std(ddof=1))})
    for path, data in ((args.out / "per_run_metrics.csv", runs),
                       (args.out / "aggregate_metrics.csv", aggregate),
                       (args.out / "univariate_source_audit.csv", univariate)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    forbidden = ["source_file", "run_id", "session_id", "checkpoint_id", "training_mode",
                 "eval_mode", "step", "epoch", "presence", "member", "split", "label"]
    selected_names = " ".join(names).lower()
    metadata_free = [d for d in desc[0] if d["family"] != "tensor_shapes"]
    result = {
        "schema": "fixed_length_metadata_free_view_audit_v1",
        "view_label": "SIMULATED_PROTOCOL_VIEW",
        "records_per_source": len(a), "fixed_sequence_length": fixed_length,
        "unique_sequence_lengths": sorted(set(lengths_a + lengths_b)),
        "exact_length_multisets_identical": lengths_a == lengths_b,
        "collection_controls": {"batch_size": 1, "padding_side": "not_applicable_unpadded_single_sample_prefill",
                                "effective_length_selected_as_feature": False, "dtype": "fp32",
                                "collection_point": "single prefill", "mask_refresh_policy": "same frozen package/capture path"},
        "source_a_sha256": sha(args.source_a), "source_b_sha256": sha(args.source_b),
        "forbidden_fields_selected": [x for x in forbidden if x in selected_names],
        "metadata_free_feature_count": len(metadata_free),
        "membership_result": "WITHHELD",
        "membership_reason": "corrected shadow plans exist but completed corrected shadow feature captures do not",
        "interpretation": "source-identity/confound diagnostic only; not a membership attack result",
        "aggregate": aggregate,
        "top_univariate": sorted(univariate, key=lambda x: x["roc_auc_separability"], reverse=True)[:20],
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Fixed-length attention/KV audit", "", "**View: SIMULATED_PROTOCOL_VIEW.**", "",
             f"The cohort contains {len(a)} records per source, all at sequence length {fixed_length}. "
             "The target below is collection source, not membership; membership results are withheld pending completed corrected shadow captures.", "",
             "| View | Features | Source AUC | Label-shuffle AUC |", "|---|---:|---:|---:|"]
    for view in FAMILIES:
        t = next(x for x in aggregate if x["view"] == view and x["control"] == "true_source_label")
        s = next(x for x in aggregate if x["view"] == view and x["control"] == "train_label_shuffle")
        lines.append(f"| {view} | {t['features']} | {t['roc_auc_mean']:.4f} | {s['roc_auc_mean']:.4f} |")
    lines += ["", "No source file, run/session/checkpoint identifier, train/eval mode, step/epoch, tensor-presence flag, split, label, or member field is selected. Any above-chance source AUC therefore diagnoses numerical collection/source confounding in the simulated captures; it must not be reported as MIA evidence."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"records_per_source": len(a), "out": str(args.out)}))


if __name__ == "__main__":
    main()
