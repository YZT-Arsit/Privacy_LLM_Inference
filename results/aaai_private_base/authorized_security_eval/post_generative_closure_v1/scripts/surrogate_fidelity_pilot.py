#!/usr/bin/env python3
"""Small, bounded SURROGATE_FIDELITY_PILOT over frozen aligned V0/V2 views.

This predicts a coarse generated-output length class. It is an output-property
probe, not model extraction and not exact text reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


BUDGETS = [.01, .05, .10, .20]
SEEDS = [7, 1234, 2025]
INPUT_DIM = 1024
HIDDEN = 64
STEPS = 300


def accuracy_score(y, pred) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def f1_score(y, pred, average="macro") -> float:
    if average != "macro": raise ValueError("only macro supported")
    vals = []
    for c in range(3):
        tp = np.sum((y == c) & (pred == c)); fp = np.sum((y != c) & (pred == c)); fn = np.sum((y == c) & (pred != c))
        vals.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    return float(np.mean(vals))


def log_loss(y, prob, labels=None) -> float:
    del labels
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())


def _stats(values) -> list[float]:
    x = np.asarray(values, dtype=float)
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max())]


def descriptors(row: dict) -> list[dict]:
    """The pinned 706-feature, shape-free subset used by the closure audit."""
    seq = float(row["tensor_shapes"]["sequence_length"]); out = []
    def add(name, value, family, layer, statistic):
        out.append({"name": name, "value": float(value), "family": family,
                    "layer": layer, "statistic": statistic})
    def reductions(prefix, values, family, layer, source):
        for reduction, value in zip(("mean", "std", "min", "max"), _stats(values)):
            add(f"{prefix}.{reduction}", value, family, layer, f"{reduction}({source})")
    hidden = row["transformed_hidden"]
    add("transformed_hidden.final.last_l2", hidden["final_last_l2"], "transformed_hidden", "final", "last_token_l2")
    for item in hidden["per_layer"]:
        add(f"transformed_hidden.layer_{item['layer']:02d}.last_l2", item["last_l2"],
            "transformed_hidden", int(item["layer"]), "pre_attention_last_token_l2")
    logits = row["masked_logits"]
    for key in ("mean", "std", "l2", "min", "max"):
        add(f"masked_logits.{key}", logits[key], "masked_logits", "final", key)
    reductions("masked_logits.top_values", logits["top_values"], "masked_logits", "final", "top_values")
    for family in ("masked_k", "masked_v"):
        for layer in row[family]:
            reductions(f"{family}.layer_{layer['layer']:02d}.last_l2_by_head", layer["last_l2_by_head"],
                       family, int(layer["layer"]), "last_l2_by_head")
    for layer in row["attention_scores"]:
        lid = int(layer["layer"])
        for key in ("last_entropy_by_head", "last_max_by_head", "true_score_last_mean_by_head", "true_score_last_std_by_head"):
            reductions(f"attention_scores.layer_{lid:02d}.{key}", layer[key], "attention_scores", lid, key)
        normalized = np.asarray(layer["last_argmax_by_head"], float) / max(seq - 1, 1)
        reductions(f"attention_scores.layer_{lid:02d}.normalized_last_argmax_by_head", normalized,
                   "attention_scores", lid, "last_argmax/(sequence_length-1)")
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def hash_text(text: str, dim: int = 256) -> np.ndarray:
    x = np.zeros(dim, np.float32)
    normalized = " ".join(text.lower().split())
    for n in (2, 3, 4):
        for i in range(max(0, len(normalized) - n + 1)):
            token = normalized[i:i+n].encode()
            h = hashlib.blake2b(token, digest_size=8).digest()
            value = int.from_bytes(h, "little")
            x[value % dim] += 1.0 if value >> 63 else -1.0
    norm = np.linalg.norm(x)
    return x / norm if norm else x


def output_class(row: dict) -> int:
    # Fixed thresholds chosen before model fitting; roughly balanced on E2E generations.
    n = len(row["output_token_ids"])
    return 0 if n <= 27 else 1 if n <= 34 else 2


def pad(x: np.ndarray) -> np.ndarray:
    if x.size > INPUT_DIM:
        raise RuntimeError(f"feature vector {x.size} exceeds pinned input dimension")
    return np.pad(x.astype(np.float32), (0, INPUT_DIM - x.size))


def balanced_prefix(indices: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); by = []
    for c in range(3):
        z = indices[y[indices] == c].copy(); rng.shuffle(z); by.append(list(z))
    out = []
    while len(out) < n and any(by):
        for c in range(3):
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return np.asarray(out, int)


def leakage_safe_split(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep exact and high-overlap output groups on only one side of the split."""
    normalized = [" ".join(r["output_text"].lower().split()) for r in rows]
    grams = []
    for text in normalized:
        toks = text.split(); grams.append(set(zip(toks, toks[1:])))
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for i in range(len(rows)):
        for j in range(i):
            inter = len(grams[i] & grams[j]); union_size = len(grams[i] | grams[j])
            if normalized[i] == normalized[j] or (union_size and inter / union_size >= .90):
                union(i, j)
    groups = {}
    for i in range(len(rows)): groups.setdefault(find(i), []).append(i)
    ordered = list(groups.values()); np.random.default_rng(20260714).shuffle(ordered)
    test, validation, query = [], [], []
    for group in ordered:
        target = test if len(test) < 180 else validation if len(validation) < 40 else query
        target.extend(group)
    # If grouping makes a boundary overshoot, that is acceptable; groups never cross.
    sides = {}
    for name, idx in (("test", test), ("validation", validation), ("query", query)):
        for i in idx: sides[find(i)] = name
    cross = sum(len({sides[find(i)] for i in group}) > 1 for group in ordered)
    return np.asarray(test), np.asarray(validation), np.asarray(query), {
        "near_duplicate_threshold": "word-bigram Jaccard >= 0.90",
        "connected_components": len(ordered), "cross_split_components": cross,
    }


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(INPUT_DIM, HIDDEN), torch.nn.ReLU(),
                                       torch.nn.Linear(HIDDEN, 3))
    def forward(self, x):
        return self.net(x)


def fit_score(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              seed: int, device: str, shuffled: bool = False) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    # Train-only preprocessing. Zero-variance dimensions remain unchanged.
    mean = x[train].mean(0); std = x[train].std(0); std[std < 1e-6] = 1
    ztrain = torch.tensor((x[train] - mean) / std, device=device)
    ztest = torch.tensor((x[test] - mean) / std, device=device)
    target = y[train].copy()
    if shuffled:
        target = np.random.default_rng(seed + 10000).permutation(target)
    target_t = torch.tensor(target, dtype=torch.long, device=device)
    model = Student().to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(ztrain), target_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.softmax(model(ztest), -1).cpu().numpy()
    return prob, {"preprocessing_fit_records": len(train), "optimizer_steps": STEPS,
                  "parameter_count": sum(p.numel() for p in model.parameters())}


def bootstrap(y, pred, seed, samples=1000):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y)); vals.append(accuracy_score(y[idx], pred[idx]))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True, type=Path)
    ap.add_argument("--v2", required=True, type=Path)
    ap.add_argument("--source-audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    v0, v2 = load(args.v0), load(args.v2)
    ids0, ids2 = [x["sample_id"] for x in v0], [x["sample_id"] for x in v2]
    if ids0 != ids2 or len(ids0) != len(set(ids0)):
        raise RuntimeError("ordered alignment/duplicate control failed")
    y = np.asarray([output_class(x) for x in v0], int)
    v0x = np.asarray([hash_text(x["output_text"]) for x in v0])
    v2x = np.asarray([[d["value"] for d in descriptors(x)
                      if d["family"] != "tensor_shapes"] for x in v2], np.float32)
    views = {
        "V0_output_only": np.asarray([pad(x) for x in v0x]),
        "V2_protocol_faithful_simulated_gpu": np.asarray([pad(x) for x in v2x]),
        "V0_plus_V2": np.asarray([pad(np.r_[a, b]) for a, b in zip(v0x, v2x)]),
        "plaintext_label_positive_control": np.asarray([pad(np.eye(3, dtype=np.float32)[c]) for c in y]),
    }
    views["shape_only_negative_control"] = np.asarray([
        pad(np.asarray([x["tensor_shapes"]["sequence_length"]], np.float32)) for x in v2])
    views["random_gaussian_matched_control"] = np.random.default_rng(20260714).normal(
        size=(len(y), INPUT_DIM)).astype(np.float32)
    perm = np.random.default_rng(20260715).permutation(len(y))
    views["V2_shuffled_sample_feature_control"] = views["V2_protocol_faithful_simulated_gpu"][perm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test, validation, query_pool, duplicate_audit = leakage_safe_split(v0)
    if min(len(test), len(validation), len(query_pool)) == 0:
        raise RuntimeError("group-isolated split is empty")
    runs = []
    for budget in BUDGETS:  # 1% is intentionally executed first.
        n = max(3, int(math.ceil(len(y) * budget)))
        for seed in SEEDS:
            train = balanced_prefix(query_pool, y, n, seed)
            for view, x in views.items():
                prob, meta = fit_score(x, y, train, test, seed, device)
                pred = prob.argmax(1)
                runs.append({"budget_fraction": budget, "queries": len(train), "seed": seed,
                             "view": view, "control": "aligned", "accuracy": accuracy_score(y[test], pred),
                             "macro_f1": f1_score(y[test], pred, average="macro"),
                             "cross_entropy": log_loss(y[test], prob, labels=[0,1,2]),
                             "accuracy_ci95": json.dumps(bootstrap(y[test], pred, seed)), **meta})
                # Required label-shuffle control, same architecture and budget.
                cprob, cmeta = fit_score(x, y, train, test, seed, device, shuffled=True)
                cpred = cprob.argmax(1)
                runs.append({"budget_fraction": budget, "queries": len(train), "seed": seed,
                             "view": view, "control": "label_shuffle", "accuracy": accuracy_score(y[test], cpred),
                             "macro_f1": f1_score(y[test], cpred, average="macro"),
                             "cross_entropy": log_loss(y[test], cprob, labels=[0,1,2]),
                             "accuracy_ci95": json.dumps(bootstrap(y[test], cpred, seed + 1)), **cmeta})
    aggregate = []
    for budget in BUDGETS:
        for view in views:
            for control in ("aligned", "label_shuffle"):
                z = [r for r in runs if r["budget_fraction"] == budget and r["view"] == view
                     and r["control"] == control]
                aggregate.append({"budget_fraction": budget, "queries": z[0]["queries"], "view": view,
                                  "control": control, "accuracy_mean": float(np.mean([r["accuracy"] for r in z])),
                                  "accuracy_std": float(np.std([r["accuracy"] for r in z], ddof=1)),
                                  "macro_f1_mean": float(np.mean([r["macro_f1"] for r in z])),
                                  "cross_entropy_mean": float(np.mean([r["cross_entropy"] for r in z]))})
    for path, data in ((args.out / "per_run_metrics.csv", runs), (args.out / "aggregate_metrics.csv", aggregate)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    source_audit = json.loads(args.source_audit.read_text())
    source_auc = max(x["roc_auc_mean"] for x in source_audit["aggregate"]
                     if x["control"] == "true_source_label" and x["view"] != "shape_only")
    aligned_v2 = [x for x in aggregate if x["view"] == "V2_protocol_faithful_simulated_gpu"
                  and x["control"] == "aligned"]
    aligned_v0 = [x for x in aggregate if x["view"] == "V0_output_only" and x["control"] == "aligned"]
    gains = [{"budget_fraction": a["budget_fraction"],
              "v2_minus_v0_accuracy": a["accuracy_mean"] - b["accuracy_mean"]}
             for a, b in zip(aligned_v2, aligned_v0)]
    shuffled_material = any(x["accuracy_mean"] > .50 for x in aggregate
                            if x["control"] == "label_shuffle")
    fail_reasons = []
    if shuffled_material: fail_reasons.append("at_least_one_label_shuffle_control_above_0.50_accuracy")
    if source_auc >= .99: fail_reasons.append("source_identity_perfectly_separable_in_fixed_length_audit")
    if duplicate_audit["cross_split_components"]: fail_reasons.append("near_duplicate_component_crosses_split")
    result = {
        "schema": "surrogate_fidelity_pilot_v1", "view_label": "SIMULATED_PROTOCOL_VIEW",
        "scope_label": "SURROGATE_FIDELITY_PILOT", "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "target": "coarse generated-output token-length class (<=27, 28-34, >=35)",
        "claims_boundary": "output-property agreement only; not exact text reconstruction or full model extraction",
        "exact_string_agreement": "NOT_APPLICABLE_TO_THIS_PILOT",
        "v0_sha256": sha(args.v0), "v2_sha256": sha(args.v2), "records": len(y),
        "pilot_verdict": "FAIL" if fail_reasons else "PASS", "fail_reasons": fail_reasons,
        "fixed_evaluator_records": len(test), "fixed_validation_records": len(validation),
        "query_pool_records": len(query_pool),
        "ordered_ids_match": ids0 == ids2, "duplicate_ids": len(ids0) - len(set(ids0)),
        "split_overlap": len((set(test) | set(validation)) & set(query_pool)),
        "duplicate_and_near_duplicate_audit": duplicate_audit,
        "preprocessing": "fit on queried train prefix only; validation/evaluator excluded",
        "architecture": {"input_dim": INPUT_DIM, "hidden": HIDDEN, "classes": 3, "steps": STEPS,
                         "random_initialization": True, "identical_across_views": True},
        "negative_controls": ["label_shuffle", "shuffled_sample_to_feature_mapping",
                              "random_gaussian_matched_dimension", "shape_only",
                              "source_identity_classifier", "pseudo_view_labels_via_label_shuffle"],
        "source_identity_max_auc": source_auc,
        "v2_minus_v0_fidelity_gain": gains,
        "kl_divergence": "NOT_REPORTED: interface does not expose a genuine target distribution",
        "categorical_exact_match": "reported as accuracy",
        "corpus_output_agreement": "NOT_APPLICABLE: pilot target is a coarse output property",
        "aggregate": aggregate,
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    one = [x for x in aggregate if x["budget_fraction"] == .01]
    lines = ["# SURROGATE_FIDELITY_PILOT", "", "**SIMULATED_PROTOCOL_VIEW — not REAL_TDX_BACKED_VIEW.**", "",
             f"**Pilot verdict: {result['pilot_verdict']}.** " + "; ".join(fail_reasons) + ".", "",
             "This bounded pilot measures agreement on a coarse generated-output length property. It does not reconstruct exact text and is not evidence of full model extraction.", "",
             "## 1% query budget (executed first)", "", "| View | Queries | Agreement | Macro-F1 | Cross-entropy |", "|---|---:|---:|---:|---:|"]
    for view in views:
        z = next(x for x in one if x["view"] == view and x["control"] == "aligned")
        lines.append(f"| {view} | {z['queries']} | {z['accuracy_mean']:.4f} | {z['macro_f1_mean']:.4f} | {z['cross_entropy_mean']:.4f} |")
    lines += ["", "Every cell uses a random-init 1024→64→3 MLP with the same 300-step optimization budget. Train, validation, and evaluator groups are disjoint under exact text and word-bigram Jaccard ≥0.90 grouping; IDs are ordered/aligned with no duplicates; preprocessing is fit only on each queried training prefix. Label shuffle, sample-feature shuffle, Gaussian, shape-only, and source-identity controls are recorded. Because the fixed-length source classifier is perfect, this pilot fails and no V2 utility claim is permitted."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"device": device, "records": len(y), "out": str(args.out)}))


if __name__ == "__main__":
    main()
