#!/usr/bin/env python3
"""Fail-closed CPU evaluator for matched confound-controlled V0/V2 MIA bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DISALLOWED = {"member", "membership", "label", "split", "source_file", "run_id", "session_id",
              "checkpoint_id", "training_mode", "eval_mode", "step", "epoch", "tensor_present"}
SEEDS = (7, 1234, 2025)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as exc: raise RuntimeError(f"invalid JSON: {path}:{number}") from exc
    return rows


def validate_condition(root: Path, seed: int, condition: str) -> tuple[list[dict], dict]:
    folder = root / f"shadow_s{seed}" / condition.lower()
    manifest_path, feature_path = folder / "manifest.json", folder / "features.jsonl"
    if not manifest_path.is_file() or not feature_path.is_file():
        raise RuntimeError(f"missing {condition} collection for shadow {seed}")
    manifest = json.loads(manifest_path.read_text())
    required = {"schema", "version", "shadow_seed", "condition", "collection_mode", "eval_mode",
                "forward_only", "sample_pool_sha256", "query_ids_sha256", "collector_path_identity_sha256",
                "records_sha256", "feature_columns", "feature_families", "adapter_manifest_sha256"}
    if required - set(manifest): raise RuntimeError(f"shadow {seed} {condition}: incomplete manifest")
    if manifest["shadow_seed"] != seed or manifest["condition"] != condition:
        raise RuntimeError(f"shadow {seed} {condition}: manifest identity mismatch")
    if manifest["collection_mode"] != "C0" or manifest["eval_mode"] is not True or manifest["forward_only"] is not True:
        raise RuntimeError(f"shadow {seed} {condition}: collection path is not C0/eval/forward-only")
    if sha256(feature_path) != manifest["records_sha256"]:
        raise RuntimeError(f"shadow {seed} {condition}: feature hash mismatch")
    columns = list(manifest["feature_columns"])
    if len(columns) != len(set(columns)) or any(name.lower() in DISALLOWED for name in columns):
        raise RuntimeError(f"shadow {seed} {condition}: disallowed or duplicate feature column")
    if set(manifest["feature_families"]) != set(columns):
        raise RuntimeError(f"shadow {seed} {condition}: incomplete feature-family map")
    rows = jsonl(feature_path)
    if len(rows) != 1000 or len({row.get("sample_id") for row in rows}) != 1000:
        raise RuntimeError(f"shadow {seed} {condition}: expected 1,000 unique samples")
    allowed = {"sample_id", "duplicate_group_id", "semantic_group_id", "features"}
    for row in rows:
        if set(row) != allowed or set(row["features"]) != set(columns):
            raise RuntimeError(f"shadow {seed} {condition}: record/schema mismatch")
        if any(name.lower() in DISALLOWED for name in row["features"]):
            raise RuntimeError(f"shadow {seed} {condition}: disallowed field")
        for value in row["features"].values():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise RuntimeError(f"shadow {seed} {condition}: features must be numeric or null")
    return rows, manifest


def validate_and_load(root: Path) -> dict[str, list[dict]]:
    loaded = {"V0": [], "V2": []}; schemas = {}
    for seed in SEEDS:
        condition_data = {}
        for condition in ("V0", "V2"):
            rows, manifest = validate_condition(root, seed, condition); condition_data[condition] = (rows, manifest)
            signature = (tuple(manifest["feature_columns"]), tuple(sorted(manifest["feature_families"].items())))
            if condition in schemas and schemas[condition] != signature:
                raise RuntimeError(f"{condition}: feature schema differs across shadows")
            schemas[condition] = signature
        v0, m0 = condition_data["V0"]; v2, m2 = condition_data["V2"]
        if m0["sample_pool_sha256"] != m2["sample_pool_sha256"] or m0["query_ids_sha256"] != m2["query_ids_sha256"]:
            raise RuntimeError(f"shadow {seed}: V0/V2 pool mismatch")
        if m0["collector_path_identity_sha256"] != m2["collector_path_identity_sha256"]:
            raise RuntimeError(f"shadow {seed}: V0/V2 collection paths differ")
        map0, map2 = {r["sample_id"]: r for r in v0}, {r["sample_id"]: r for r in v2}
        if set(map0) != set(map2): raise RuntimeError(f"shadow {seed}: V0/V2 sample IDs differ")
        for sid in map0:
            for key in ("duplicate_group_id", "semantic_group_id"):
                if map0[sid][key] != map2[sid][key]: raise RuntimeError(f"shadow {seed}: group mismatch")
        metadata = root / f"shadow_s{seed}" / "evaluator_membership_metadata.jsonl"
        labels = {r["sample_id"]: int(r["member"]) for r in jsonl(metadata)}
        if set(labels) != set(map0) or set(labels.values()) != {0, 1}:
            raise RuntimeError(f"shadow {seed}: invalid separate membership metadata")
        for condition, records in (("V0", v0), ("V2", v2)):
            for row in records:
                loaded[condition].append({**row, "shadow_seed": seed, "membership": labels[row["sample_id"]],
                                          "families": condition_data[condition][1]["feature_families"]})
    return loaded


def matrix(rows: list[dict], families: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = sorted(rows[0]["features"])
    if families is not None: columns = [c for c in columns if rows[0]["families"][c] in families]
    if not columns: raise RuntimeError("empty feature-family selection")
    x = np.asarray([[np.nan if row["features"][c] is None else float(row["features"][c]) for c in columns] for row in rows])
    y = np.asarray([row["membership"] for row in rows], dtype=int)
    groups = np.asarray([row["semantic_group_id"] for row in rows])
    return x, y, groups, columns


def pipeline(n_features: int, select_k: int | None = None) -> Pipeline:
    k = "all" if select_k is None else min(select_k, n_features)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("selector", SelectKBest(f_classif, k=k)),
                     ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def operating_points(y: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, score)
    def at(limit: float) -> float:
        eligible = tpr[fpr <= limit]
        return float(eligible.max()) if len(eligible) else 0.0
    return {"roc_auc": float(roc_auc_score(y, score)),
            "attack_advantage": float(np.max(tpr - fpr)),
            "attack_advantage_definition": "maximum empirical test-set TPR(tau)-FPR(tau) over score thresholds tau",
            "tpr_at_1pct_fpr": at(0.01), "tpr_at_0.1pct_fpr": at(0.001),
            "negative_count": int((y == 0).sum()),
            "fpr_resolution": 1.0 / int((y == 0).sum())}


def group_bootstrap(y: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int, n: int = 2000) -> dict:
    rng = np.random.default_rng(seed); unique = np.unique(groups); values = defaultdict(list)
    by_group = {g: np.where(groups == g)[0] for g in unique}
    for _ in range(n):
        chosen = rng.choice(unique, len(unique), replace=True); idx = np.concatenate([by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2: continue
        point = operating_points(y[idx], score[idx])
        for key in ("roc_auc", "attack_advantage", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr"):
            values[key].append(point[key])
    return {key + "_ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for key, v in values.items()}


def fit_score(train: list[dict], test: list[dict], families: set[str] | None, seed: int,
              shuffled_train_labels: bool = False) -> dict:
    xtr, ytr, _, columns = matrix(train, families); xte, yte, groups, _ = matrix(test, families)
    if shuffled_train_labels: ytr = np.random.default_rng(seed).permutation(ytr)
    model = pipeline(xtr.shape[1], select_k=min(256, xtr.shape[1])); model.fit(xtr, ytr)
    score = model.predict_proba(xte)[:, 1]; point = operating_points(yte, score)
    return {**point, **group_bootstrap(yte, score, groups, seed), "feature_count": len(columns),
            "preprocessing_fit_scope": "attack training fold only"}


def matched_condition_delta(v0: list[dict], v2: list[dict]) -> list[dict]:
    """Primary paired V2-minus-V0 comparison on identical held-out shadow samples."""
    out = []
    for heldout in SEEDS:
        scores = {}
        test_keys = None
        y_reference = groups_reference = None
        for condition, rows in (("V0", v0), ("V2", v2)):
            train = sorted((r for r in rows if r["shadow_seed"] != heldout), key=lambda r: (r["shadow_seed"], r["sample_id"]))
            test = sorted((r for r in rows if r["shadow_seed"] == heldout), key=lambda r: r["sample_id"])
            keys = [r["sample_id"] for r in test]
            if test_keys is not None and keys != test_keys: raise RuntimeError("V0/V2 held-out alignment failure")
            test_keys = keys
            xtr, ytr, _, _ = matrix(train); xte, yte, groups, _ = matrix(test)
            model = pipeline(xtr.shape[1], min(256, xtr.shape[1])); model.fit(xtr, ytr)
            scores[condition] = model.predict_proba(xte)[:, 1]
            if y_reference is not None and (not np.array_equal(yte, y_reference) or not np.array_equal(groups, groups_reference)):
                raise RuntimeError("V0/V2 label/group alignment failure")
            y_reference, groups_reference = yte, groups
        p0, p2 = operating_points(y_reference, scores["V0"]), operating_points(y_reference, scores["V2"])
        keys = ("roc_auc", "attack_advantage", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr")
        rng = np.random.default_rng(71000 + heldout); unique = np.unique(groups_reference)
        by_group = {g: np.where(groups_reference == g)[0] for g in unique}; boot = defaultdict(list)
        for _ in range(2000):
            chosen = rng.choice(unique, len(unique), replace=True); idx = np.concatenate([by_group[g] for g in chosen])
            if len(np.unique(y_reference[idx])) < 2: continue
            a = operating_points(y_reference[idx], scores["V0"][idx]); b = operating_points(y_reference[idx], scores["V2"][idx])
            for key in keys: boot[key].append(b[key] - a[key])
        out.append({"evaluation": "matched_leave_one_shadow_run_out_v2_minus_v0", "heldout": heldout,
                    "differences": {key: {"point": p2[key] - p0[key],
                        "paired_group_bootstrap_ci95": [float(np.quantile(boot[key], .025)), float(np.quantile(boot[key], .975))]}
                        for key in keys}, "fpr_resolution": p2["fpr_resolution"]})
    return out


def evaluate_condition(rows: list[dict], condition: str) -> list[dict]:
    families = sorted(set(rows[0]["families"].values())); selections = [("all", None)] + [(f, {f}) for f in families]
    out = []
    for name, selected in selections:
        for heldout in SEEDS:
            train = [r for r in rows if r["shadow_seed"] != heldout]; test = [r for r in rows if r["shadow_seed"] == heldout]
            out.append({"condition": condition, "evaluation": "leave_one_shadow_run_out", "heldout": heldout,
                        "feature_family": name, **fit_score(train, test, selected, 31000 + heldout)})
            out.append({"condition": condition, "evaluation": "label_shuffle_control", "heldout": heldout,
                        "feature_family": name, **fit_score(train, test, selected, 41000 + heldout, True)})
        x, y, groups, _ = matrix(rows, selected)
        split = GroupShuffleSplit(n_splits=5, test_size=.25, random_state=51000)
        for fold, (train_idx, test_idx) in enumerate(split.split(x, y, groups)):
            train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
            out.append({"condition": condition, "evaluation": "leave_samples_out_semantic_groups", "heldout": fold,
                        "feature_family": name, **fit_score(train, test, selected, 51000 + fold)})
    return out


def pseudo_membership_controls(rows: list[dict]) -> list[dict]:
    out = []
    for true_class, name in ((1, "pseudo_within_members"), (0, "pseudo_within_nonmembers")):
        subset = [dict(row) for row in rows if row["membership"] == true_class]
        rng = np.random.default_rng(61000 + true_class)
        pseudo = np.tile([0, 1], (len(subset) + 1) // 2)[:len(subset)]; rng.shuffle(pseudo)
        for row, label in zip(subset, pseudo): row["membership"] = int(label)
        x, y, groups, _ = matrix(subset)
        splitter = GroupShuffleSplit(n_splits=5, test_size=.25, random_state=62000 + true_class)
        for fold, (a, b) in enumerate(splitter.split(x, y, groups)):
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2: continue
            out.append({"condition": "V2", "evaluation": name, "heldout": fold, "feature_family": "all",
                        **fit_score([subset[i] for i in a], [subset[i] for i in b], None, 63000 + fold)})
    return out


def self_test() -> None:
    rng = np.random.default_rng(1); y = np.tile([0, 1], 100); score = y * .8 + rng.normal(0, .1, len(y))
    point = operating_points(y, score)
    assert point["roc_auc"] > .99 and point["attack_advantage"] > .9
    groups = np.asarray([f"g{i}" for i in range(len(y))]); ci = group_bootstrap(y, score, groups, 2, 100)
    assert "roc_auc_ci95" in ci
    model = pipeline(3, 2); model.fit(rng.normal(size=(200, 3)), y)
    assert hasattr(model.named_steps["imputer"], "statistics_")
    print("corrected_mia_evaluator_v2 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--collections", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if args.collections is None or args.output is None: parser.error("--collections and --output are required")
    if args.output.exists(): raise RuntimeError(f"refusing to overwrite: {args.output}")
    loaded = validate_and_load(args.collections)
    results = evaluate_condition(loaded["V0"], "V0") + evaluate_condition(loaded["V2"], "V2")
    results += pseudo_membership_controls(loaded["V2"])
    deltas = matched_condition_delta(loaded["V0"], loaded["V2"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "corrected_mia_results", "version": "2.0",
        "attack_advantage_definition": "max_tau TPR(tau)-FPR(tau) on the stated test fold",
        "fpr_resolution_guard": "TPR@0.1% FPR is resolution-limited when fewer than 1,000 test negatives; negative count and resolution are always reported.",
        "results": results, "matched_v2_minus_v0": deltas}, indent=2) + "\n")


if __name__ == "__main__": main()
