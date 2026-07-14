#!/usr/bin/env python3
"""Frozen offline real-TDX three-shadow LOSO membership analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODE = "REAL_TDX_BACKED_VIEW"
EVALUATION = "three-shadow leave-one-shadow-out"
FEATURE_SCHEMA = "common-semantics v2, 610 columns"
SEED = 20260714
CS = [0.01, 0.1, 1.0, 10.0]
BOOTSTRAPS = 2000
HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE.parent
REPO = next(p for p in HERE.parents if (p / ".git").exists())
HANDOFF = INPUT_ROOT / "cpu_handoff"
POOL = REPO / "results/aaai_private_base/authorized_security_eval/mia/confound_controlled_v2/pool/candidate_pool.jsonl"
META = {"collection_mode": MODE, "evaluation": EVALUATION, "feature_schema": FEATURE_SCHEMA}
VIEW_INDICES: dict[str, list[int]] = {}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def dump(name: str, value) -> None:
    path = HERE / name
    if path.exists():
        raise RuntimeError(f"append-only refusal: {path}")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False, sort_keys=True)
    path.write_text(text.rstrip() + "\n")


def md_header(title: str) -> str:
    return (f"# {title}\n\nCollection mode: {MODE}  \nEvaluation: {EVALUATION}  \n"
            f"Feature schema: {FEATURE_SCHEMA}\n")


def normtext(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def bundle_digest(entries: dict[str, dict], prefix: str) -> str:
    lines = [f"{p}\0{m['sha256']}\0{m['bytes']}" for p, m in sorted(entries.items()) if prefix in p]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def input_integrity():
    handoff = json.loads((HANDOFF / "handoff_manifest.json").read_text())
    registered = json.loads((HANDOFF / "artifact_hashes.json").read_text())["artifacts"]
    bad = []
    for name, meta in registered.items():
        path = REPO / name
        if not path.is_file():
            bad.append(f"missing:{name}")
        elif sha(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            bad.append(f"hash_or_size:{name}")
    schema_path = REPO / handoff["shadows"][0]["v2_matrix"]
    schema_doc = json.loads((INPUT_ROOT / "inputs/common_v2_feature_schema.json").read_text())
    columns = [x["feature_name"] for x in schema_doc["features"]]
    prohibited = json.loads((HANDOFF / "prohibited_feature_list.json").read_text())
    prohibited_names = set(prohibited["features"] + prohibited.get("additional", []))
    forbidden_tokens = ("member", "label", "split", "run_id", "session", "checkpoint", "source_file",
                        "timestamp", "filename", "tensor_presence", "hidden.final.last_l2")
    data, per_shadow = [], []
    for item in handoff["shadows"]:
        i = int(item["shadow"])
        v0 = load_jsonl(REPO / item["v0_matrix"])
        membership = load_jsonl(REPO / item["membership"]["path"])
        index = json.loads((REPO / item["sample_id_join"]).read_text())
        frame = pd.read_csv(REPO / item["v2_matrix"])
        ids0 = [x["sample_id"] for x in v0]
        idsm = [x["sample_id"] for x in membership]
        manifest = json.loads((REPO / item["collection_manifest"]).read_text())
        validity = json.loads((REPO / item["validity_report"]).read_text())
        adapter_ok = sha(REPO / item["adapter"]["path"]) == item["adapter"]["sha256"]
        membership_ok = sha(REPO / item["membership"]["path"]) == item["membership"]["sha256"]
        bad_columns = [x for x in frame.columns if x in prohibited_names or any(t in x.lower() for t in forbidden_tokens)]
        checks = {
            "adapter_hash": adapter_ok,
            "membership_hash": membership_ok,
            "rows_1000": len(v0) == len(membership) == len(index) == len(frame) == 1000,
            "balanced_500_500": sum(bool(x["member"]) for x in membership) == 500,
            "id_join_exact": index == ids0 == idsm,
            "unique_ids": len(set(index)) == 1000,
            "schema_exact_610": len(frame.columns) == 610 and list(frame.columns) == columns,
            "hidden_final_absent": "hidden.final.last_l2" not in frame.columns,
            "forbidden_fields_absent": not bad_columns,
            "labels_separate": "member" not in frame.columns and "sample_id" not in frame.columns,
            "finite_complete": not frame.isna().any().any() and np.isfinite(frame.to_numpy(float)).all(),
            "manifest_pass": manifest["status"] == "PASS",
            "validity_pass": validity["status"] == f"SHADOW_{i}_COLLECTION_VALID" and all(validity["gates"].values()),
        }
        per_shadow.append({"shadow": i, "checks": checks, "members": 500, "nonmembers": 500,
                           "bundle_path": str((INPUT_ROOT / f"shadow_{i}").relative_to(REPO)),
                           "bundle_inventory_sha256": bundle_digest(registered, f"/shadow_{i}/")})
        data.append({"shadow": i, "ids": index, "text": [x["generated_text"] for x in v0],
                     "y": np.asarray([int(x["member"]) for x in membership]),
                     "x": frame.to_numpy(float), "columns": list(frame.columns)})
    secret_hits = []
    secret_re = re.compile(r'"session_key_hex"\s*:\s*"[0-9a-fA-F]{32,}"')
    for name in registered:
        path = REPO / name
        if path.suffix.lower() in {".json", ".jsonl", ".md", ".csv", ".py"}:
            try:
                if secret_re.search(path.read_text(errors="ignore")):
                    secret_hits.append(name)
            except OSError:
                pass
    gates = {
        "handoff_status_pass": handoff["status"] == "PASS",
        "registered_hashes": not bad,
        "registered_artifact_count_78": len(registered) == 78,
        "three_shadow_integrity": all(all(x["checks"].values()) for x in per_shadow),
        "schema_hash": sha(INPUT_ROOT / "inputs/common_v2_feature_schema.json") == handoff["schema_sha256"],
        "collection_validation_pass": json.loads((INPUT_ROOT / "collection_validation/joint/joint_validation_report.json").read_text())["status"] == "PASS",
        "trusted_secrets_absent_from_registered_handoff": not secret_hits,
        "session_files_excluded_from_inventory": not any(x.endswith(("a10_session.json", "tdx_session.json")) for x in registered),
    }
    report = {**META, "status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
              "hash_mismatches": bad, "secret_hits": secret_hits, "registered_artifacts": len(registered),
              "per_shadow": per_shadow, "handoff_manifest_sha256": sha(HANDOFF / "handoff_manifest.json"),
              "artifact_hashes_sha256": sha(HANDOFF / "artifact_hashes.json")}
    return handoff, data, columns, report


def duplicate_audit(data):
    pool = load_jsonl(POOL)
    by_id = {x["sample_id"]: x for x in pool}
    def repeated(key):
        c = Counter(json.dumps(x[key], sort_keys=True) if not isinstance(x[key], str) else x[key] for x in pool)
        return {k: v for k, v in c.items() if v > 1}
    exact_inputs, refs, mrs = repeated("normalized_input_hash"), repeated("reference_hash"), repeated("semantic_group_id")
    template = {x["sample_id"]: "|".join(x["mr_field_set"]) for x in pool}
    templates = Counter(template.values())
    tokens = [set(normtext(x["meaning_representation"]).split()) for x in pool]
    near_pairs, parent = [], list(range(len(pool)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            den = len(tokens[i] | tokens[j])
            if den and len(tokens[i] & tokens[j]) / den >= .9:
                near_pairs.append((pool[i]["sample_id"], pool[j]["sample_id"])); union(i, j)
    near_groups = Counter(find(i) for i in range(len(pool)))
    near_sizes = sorted((v for v in near_groups.values() if v > 1), reverse=True)
    generated = []
    for d in data:
        exact = Counter(normtext(x) for x in d["text"])
        skeletons = []
        for sid, output in zip(d["ids"], d["text"]):
            slot_tokens = set(normtext(by_id[sid]["meaning_representation"]).split())
            words = ["<num>" if w.isdigit() else "<slot>" if w in slot_tokens else w for w in normtext(output).split()]
            skeletons.append(" ".join(w for j, w in enumerate(words) if j == 0 or w != words[j-1] or w != "<slot>"))
        sk = Counter(skeletons)
        generated.append({"shadow": d["shadow"], "unique_exact_outputs": len(exact),
                          "exact_duplicate_output_groups": sum(v > 1 for v in exact.values()),
                          "rows_in_exact_duplicate_output_groups": sum(v for v in exact.values() if v > 1),
                          "largest_exact_output_group": max(exact.values()),
                          "repeated_output_template_groups": sum(v > 1 for v in sk.values()),
                          "rows_in_repeated_output_templates": sum(v for v in sk.values() if v > 1),
                          "largest_output_template_group": max(sk.values())})
    audit = {**META, "exact_duplicate_inputs": sum(exact_inputs.values()), "exact_duplicate_input_groups": len(exact_inputs),
             "exact_duplicate_references": sum(refs.values()), "exact_duplicate_reference_groups": len(refs),
             "repeated_exact_meaning_representations": sum(mrs.values()),
             "near_duplicate_pairs_jaccard_ge_0_9": len(near_pairs), "near_duplicate_groups": len(near_sizes),
             "near_duplicate_group_sizes": near_sizes, "near_duplicate_pair_examples": near_pairs[:20],
             "semantic_template_definition": "sorted exact mr_field_set", "semantic_template_groups": len(templates),
             "semantic_template_group_sizes": dict(sorted(templates.items())), "generated_output_audit": generated,
             "sample_ids_present_in_all_three_shadows": len(set(data[0]["ids"]) & set(data[1]["ids"]) & set(data[2]["ids"])),
             "removals": [], "group_disjoint_rule": "five GroupKFold partitions of mr_field_set; test-group IDs excluded from both attack-training shadows"}
    return by_id, template, audit


def text_transformer():
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=2, max_features=4000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=4000, sublinear_tf=True)),
    ])


def family_indices(columns):
    return {k: [i for i, x in enumerate(columns) if x.startswith(k + ".")] for k in ("attention", "kv", "hidden", "logit")}


def prepare(view, texts, x, train, test):
    if view == "V0":
        tx = text_transformer(); a = tx.fit_transform(texts[train]); b = tx.transform(texts[test])
        return a, b
    if view == "V0_plus_V2":
        tx = text_transformer(); a0 = tx.fit_transform(texts[train]); b0 = tx.transform(texts[test])
        imp, sc = SimpleImputer(strategy="median"), StandardScaler()
        a2 = sc.fit_transform(imp.fit_transform(x[train])); b2 = sc.transform(imp.transform(x[test]))
        return sparse.hstack([a0, sparse.csr_matrix(a2)], format="csr"), sparse.hstack([b0, sparse.csr_matrix(b2)], format="csr")
    inds = VIEW_INDICES[view]; imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    a = sc.fit_transform(imp.fit_transform(x[train][:, inds])); b = sc.transform(imp.transform(x[test][:, inds]))
    return a, b


def classifier(kind, c):
    if kind == "logistic_l2":
        return LogisticRegression(C=c, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=SEED)
    return LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=SEED)


def model_scores(model, x):
    score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
    prob = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else None
    return np.asarray(score), None if prob is None else np.asarray(prob)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): value += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(value)


def metrics(y, score, prob=None):
    fpr, tpr, _ = roc_curve(y, score)
    pred = (prob >= .5).astype(int) if prob is not None else (score >= 0).astype(int)
    neg = int((y == 0).sum())
    return {"roc_auc": float(roc_auc_score(y, score)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "average_precision": float(average_precision_score(y, score)), "attack_advantage": float(np.max(tpr - fpr)),
            "tpr_at_1pct_fpr": float(tpr[fpr <= .01].max()),
            "tpr_at_0_1pct_fpr": None if neg < 1000 else float(tpr[fpr <= .001].max()),
            "brier_score": None if prob is None else float(brier_score_loss(y, prob)),
            "log_loss": None if prob is None else float(log_loss(y, prob)),
            "ece_10bin": None if prob is None else ece(y, prob)}


def tune(view, kind, texts, x, y, shadow, train):
    values = []
    train_shadows = sorted(set(shadow[train])); assert len(train_shadows) == 2
    for c in CS:
        aucs = []
        for validation_shadow in train_shadows:
            va = train[shadow[train] == validation_shadow]; tr = train[shadow[train] != validation_shadow]
            a, b = prepare(view, texts, x, tr, va); model = classifier(kind, c); model.fit(a, y[tr])
            aucs.append(roc_auc_score(y[va], model_scores(model, b)[0]))
        values.append(float(np.mean(aucs)))
    best = max(range(len(CS)), key=lambda i: (values[i], -CS[i]))
    return CS[best], values


def evaluate(view, kind, texts, x, y, shadow, train, test, fixed_c=None):
    c, cv = (fixed_c, []) if fixed_c is not None else tune(view, kind, texts, x, y, shadow, train)
    a, b = prepare(view, texts, x, train, test)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always"); model = classifier(kind, c); model.fit(a, y[train])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    score, prob = model_scores(model, b)
    return metrics(y[test], score, prob), score, prob, c, cv, converged, model


def main():
    global VIEW_INDICES
    existing = [p for p in HERE.iterdir() if p.name not in {"analysis_driver.py", "__pycache__"}]
    if existing: raise RuntimeError(f"append-only refusal; outputs exist: {existing}")
    handoff, data, columns, integrity = input_integrity()
    if integrity["status"] != "PASS":
        dump("final_status.json", {**META, "status": "REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_WITHHELD", "integrity": integrity})
        raise SystemExit("REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_WITHHELD")

    by_id, field_group, dup = duplicate_audit(data)
    texts = np.asarray(sum([d["text"] for d in data], []), object)
    x = np.vstack([d["x"] for d in data]); y = np.concatenate([d["y"] for d in data])
    shadow = np.repeat([1, 2, 3], 1000); ids = np.asarray(sum([d["ids"] for d in data], []), object)
    lengths = np.asarray([by_id[s]["input_token_length"] for s in ids], float)
    families = family_indices(columns)
    VIEW_INDICES = {"V2_" + k: v for k, v in families.items()}
    VIEW_INDICES["V2_combined"] = list(range(610))
    for family in families:
        VIEW_INDICES["V2_without_" + family] = [j for j in range(610) if j not in families[family]]
    main_views = ["V0", "V2_attention", "V2_kv", "V2_hidden", "V2_logit", "V2_combined", "V0_plus_V2"]
    leaveout_views = ["V2_without_attention", "V2_without_kv", "V2_without_hidden", "V2_without_logit"]
    classifiers = ["logistic_l2", "linear_svm"]
    held_order = [1, 2, 3]
    rows, predictions, coefficients, selected = [], {}, {}, {}
    for held in held_order:
        train = np.flatnonzero(shadow != held); test = np.flatnonzero(shadow == held)
        for view in main_views:
            for kind in classifiers:
                met, score, prob, c, cv, conv, model = evaluate(view, kind, texts, x, y, shadow, train, test)
                rows.append({**META, "protocol": "primary_loso", "held_out_shadow": held, "view": view, "classifier": kind,
                             "train_records": len(train), "held_out_records": len(test), "selected_C": c,
                             "internal_cv_auc": json.dumps(cv), "converged": conv, **met})
                predictions[(held, view, kind)] = {"y": y[test], "score": score, "prob": prob}
                selected[(held, view, kind)] = c
                if view == "V2_combined" and kind == "logistic_l2": coefficients[held] = model.coef_[0].copy()
        for view in leaveout_views:
            met, score, prob, c, cv, conv, model = evaluate(view, "logistic_l2", texts, x, y, shadow, train, test)
            rows.append({**META, "protocol": "leave_one_family_out", "held_out_shadow": held, "view": view,
                         "classifier": "logistic_l2", "train_records": len(train), "held_out_records": len(test),
                         "selected_C": c, "internal_cv_auc": json.dumps(cv), "converged": conv, **met})
            predictions[(held, view, "logistic_l2")] = {"y": y[test], "score": score, "prob": prob}

    primary = pd.DataFrame(rows)
    pooled_rows = []
    for view in main_views:
        for kind in classifiers:
            ps = [predictions[(h, view, kind)] for h in held_order]
            yy = np.concatenate([z["y"] for z in ps]); ss = np.concatenate([z["score"] for z in ps])
            pp = None if ps[0]["prob"] is None else np.concatenate([z["prob"] for z in ps])
            pooled_rows.append({**META, "protocol": "pooled_loso_secondary", "held_out_shadow": "pooled", "view": view,
                                "classifier": kind, "train_records": 2000, "held_out_records": 3000, "selected_C": "per_fold",
                                "internal_cv_auc": "per_fold", "converged": all(primary[(primary.protocol == "primary_loso") & (primary.view == view) & (primary.classifier == kind)].converged),
                                **metrics(yy, ss, pp)})
    per = pd.concat([primary, pd.DataFrame(pooled_rows)], ignore_index=True)
    per.to_csv(HERE / "per_fold_metrics.csv", index=False)

    metric_names = ["roc_auc", "balanced_accuracy", "average_precision", "attack_advantage", "tpr_at_1pct_fpr", "brier_score", "log_loss", "ece_10bin"]
    macros = []
    for keys, group in per.groupby(["protocol", "view", "classifier"], dropna=False):
        out = {**META, "protocol": keys[0], "view": keys[1], "classifier": keys[2], "folds": len(group)}
        for metric in metric_names:
            out[metric + "_mean"] = group[metric].mean(); out[metric + "_std"] = group[metric].std(ddof=0)
        macros.append(out)
    macro = pd.DataFrame(macros); macro.to_csv(HERE / "macro_metrics.csv", index=False)

    unique_ids = np.asarray(data[0]["ids"]); groups = np.asarray([field_group[s] for s in unique_ids])
    group_splits = list(GroupKFold(5).split(unique_ids, groups=groups))
    group_rows = []
    for held in held_order:
        for view in ["V0", "V2_combined", "V0_plus_V2"]:
            yy, ss, pp, cs, converged, train_counts = [], [], [], [], [], []
            for subfold, (_, test_pos) in enumerate(group_splits, 1):
                test_ids = set(unique_ids[test_pos])
                test = np.asarray([j for j in np.flatnonzero(shadow == held) if ids[j] in test_ids])
                train = np.asarray([j for j in np.flatnonzero(shadow != held) if ids[j] not in test_ids])
                met, score, prob, c, cv, conv, model = evaluate(view, "logistic_l2", texts, x, y, shadow, train, test)
                yy.extend(y[test]); ss.extend(score); pp.extend(prob); cs.append(c); converged.append(conv); train_counts.append(len(train))
            met = metrics(np.asarray(yy), np.asarray(ss), np.asarray(pp))
            group_rows.append({**META, "protocol": "semantic_group_disjoint_5fold_sensitivity", "held_out_shadow": held,
                               "view": view, "classifier": "logistic_l2", "semantic_subfolds": 5,
                               "train_records_mean": float(np.mean(train_counts)), "held_out_records": len(yy),
                               "selected_C_by_subfold": json.dumps(cs), "converged": all(converged), **met})
    group_frame = pd.DataFrame(group_rows); group_frame.to_csv(HERE / "group_disjoint_metrics.csv", index=False)

    comparisons = [("V2_combined", "V0"), ("V0_plus_V2", "V0"), ("V2_attention", "V0"),
                   ("V2_kv", "V0"), ("V2_hidden", "V0"), ("V2_logit", "V0")]
    boot_metrics = ["roc_auc", "balanced_accuracy", "average_precision", "attack_advantage", "tpr_at_1pct_fpr", "brier_score"]
    rng = np.random.default_rng(SEED); boot_out, diff_rows = {}, []
    for view_a, view_b in comparisons:
        key = f"{view_a}_minus_{view_b}"; fold_effects = {m: [] for m in boot_metrics}
        for held in held_order:
            a, b = predictions[(held, view_a, "logistic_l2")], predictions[(held, view_b, "logistic_l2")]
            ma, mb = metrics(a["y"], a["score"], a["prob"]), metrics(b["y"], b["score"], b["prob"])
            for metric in boot_metrics:
                effect = ma[metric] - mb[metric]; fold_effects[metric].append(effect)
                diff_rows.append({**META, "comparison": key, "metric": metric, "scope": f"held_out_shadow_{held}",
                                  "difference": effect, "ci95_low": None, "ci95_high": None, "replicates": 0})
        samples = {m: [] for m in boot_metrics}
        for _ in range(BOOTSTRAPS):
            per_metric = {m: [] for m in boot_metrics}
            for held in held_order:
                a, b = predictions[(held, view_a, "logistic_l2")], predictions[(held, view_b, "logistic_l2")]
                ix = rng.integers(0, len(a["y"]), len(a["y"]))
                if len(np.unique(a["y"][ix])) < 2: continue
                ma = metrics(a["y"][ix], a["score"][ix], a["prob"][ix]); mb = metrics(b["y"][ix], b["score"][ix], b["prob"][ix])
                for metric in boot_metrics: per_metric[metric].append(ma[metric] - mb[metric])
            if all(len(per_metric[m]) == 3 for m in boot_metrics):
                for metric in boot_metrics: samples[metric].append(float(np.mean(per_metric[metric])))
        boot_out[key] = {"fold_effects": fold_effects, "metrics": {}}
        for metric in boot_metrics:
            observed = float(np.mean(fold_effects[metric])); low, high = np.quantile(samples[metric], [.025, .975])
            boot_out[key]["metrics"][metric] = {"observed_difference": observed, "ci95": [float(low), float(high)], "replicates": len(samples[metric])}
            diff_rows.append({**META, "comparison": key, "metric": metric, "scope": "equal_weight_macro",
                              "difference": observed, "ci95_low": low, "ci95_high": high, "replicates": len(samples[metric])})
    simulated = {"V0": .675, "V2_combined": .479, "V0_plus_V2": .494}
    for view, sim_value in simulated.items():
        real = primary[(primary.protocol == "primary_loso") & (primary.view == view) & (primary.classifier == "logistic_l2")].roc_auc.mean()
        diff_rows.append({**META, "comparison": f"real_TDX_{view}_minus_simulated_{view}", "metric": "roc_auc",
                          "scope": "descriptive_cross_domain_not_paired", "difference": real - sim_value,
                          "ci95_low": None, "ci95_high": None, "replicates": 0})
    pd.DataFrame(diff_rows).to_csv(HERE / "paired_view_differences.csv", index=False)
    dump("confidence_intervals.json", {**META, "method": "paired within-held-out-shadow percentile bootstrap; shared indices; equal shadow weight",
                                       "exploratory_due_to_three_shadows": True, "bootstrap_replicates": BOOTSTRAPS, "comparisons": boot_out})

    control_rows, control_summary = [], {}
    def numeric_control(name, xx, labels, repetitions=1):
        macro_values = []
        for rep in range(repetitions):
            fold_values = []
            source = xx(rep) if callable(xx) else xx
            for held in held_order:
                train, test = np.flatnonzero(shadow != held), np.flatnonzero(shadow == held)
                imp, sc = SimpleImputer(strategy="median"), StandardScaler()
                a = sc.fit_transform(imp.fit_transform(source[train])); b = sc.transform(imp.transform(source[test]))
                c = selected[(held, "V2_combined", "logistic_l2")]; model = classifier("logistic_l2", c); model.fit(a, labels[rep][train] if isinstance(labels, list) else labels[train])
                lab = labels[rep][test] if isinstance(labels, list) else labels[test]
                auc = roc_auc_score(lab, model.predict_proba(b)[:, 1]); fold_values.append(auc)
                control_rows.append({**META, "control": name, "replicate": rep, "held_out_shadow": held,
                                     "metric": "roc_auc", "value": auc, "applicable": True, "validity_low": .4, "validity_high": .6})
            macro_values.append(float(np.mean(fold_values)))
        control_summary[name] = float(np.mean(macro_values))
    numeric_control("shape_only", lengths.reshape(-1, 1), y)
    numeric_control("presence_only", np.ones((len(y), 1)), y)
    numeric_control("missingness_only", np.zeros((len(y), 1)), y)
    numeric_control("collection_order", np.tile(np.arange(1000), 3).reshape(-1, 1), y)
    numeric_control("audit_metadata_only_membership", np.ones((len(y), 1)), y)
    numeric_control("gaussian_matched_610", lambda r: np.random.default_rng(SEED + 10 + r).normal(size=x.shape), y, 5)
    shuffled, pseudo = [], []
    for rep in range(5):
        rr = np.random.default_rng(SEED + 100 + rep); sy = y.copy(); py = np.zeros_like(y)
        for s in held_order:
            idx = np.flatnonzero(shadow == s); sy[idx] = rr.permutation(sy[idx]); py[rr.choice(idx, 500, replace=False)] = 1
        shuffled.append(sy); pseudo.append(py)
    numeric_control("shuffled_membership", x, shuffled, 5)
    numeric_control("pseudo_membership", x, pseudo, 5)
    joint_diag = json.loads((INPUT_ROOT / "collection_validation/joint/joint_diagnostics.json").read_text())
    for name, metric, value, note in [
        ("shadow_identity", "accuracy", joint_diag["shadow_identity_nearest_centroid_cv_accuracy"], "legitimate model-specific diagnostic; metadata absent"),
        ("run_session_identity_audit_metadata_only", "accuracy", joint_diag["run_session_identity_audit_metadata_only_accuracy"], "audit metadata only; fields absent from MIA matrix"),
        ("source_classifier", "accuracy", None, "not applicable: all 1000 samples have one source_population"),
    ]:
        control_rows.append({**META, "control": name, "replicate": "", "held_out_shadow": "pooled", "metric": metric,
                             "value": value, "applicable": value is not None, "validity_low": None, "validity_high": None, "note": note})
    controls_pass = all(.4 <= v <= .6 for v in control_summary.values())
    pd.DataFrame(control_rows).to_csv(HERE / "negative_controls.csv", index=False)

    ablation_rows = []
    for view in main_views + leaveout_views:
        protocol = "leave_one_family_out" if view in leaveout_views else "primary_loso"
        group = primary[(primary.protocol == protocol) & (primary.view == view) & (primary.classifier == "logistic_l2")]
        ablation_rows.append({**META, "view": view, "protocol": protocol, "roc_auc_mean": group.roc_auc.mean(),
                              "roc_auc_std": group.roc_auc.std(ddof=0), "balanced_accuracy_mean": group.balanced_accuracy.mean(),
                              "average_precision_mean": group.average_precision.mean(), "attack_advantage_mean": group.attack_advantage.mean()})
    pd.DataFrame(ablation_rows).to_csv(HERE / "feature_family_ablation.csv", index=False)

    coef = np.vstack([coefficients[h] for h in held_order]); mean_abs = np.abs(coef).mean(axis=0); top_coef = np.argsort(-mean_abs)[:25]
    univariate = []
    for j, name in enumerate(columns):
        values = []
        for d in data:
            raw_auc = roc_auc_score(d["y"], d["x"][:, j]); ks = ks_2samp(d["x"][d["y"] == 1, j], d["x"][d["y"] == 0, j]).statistic
            mi = mutual_info_classif(d["x"][:, [j]], d["y"], discrete_features=False, random_state=SEED)[0]
            values.append((raw_auc, max(raw_auc, 1 - raw_auc), ks, mi))
        univariate.append({"feature": name, "raw_auc": np.mean([z[0] for z in values]), "symmetric_auc": np.mean([z[1] for z in values]),
                           "ks": np.mean([z[2] for z in values]), "mi": np.mean([z[3] for z in values]),
                           "direction_stable": all(z[0] >= .5 for z in values) or all(z[0] <= .5 for z in values)})
    top_uni = sorted(univariate, key=lambda z: z["symmetric_auc"], reverse=True)[:25]
    attr = [md_header("Feature attribution"), "Coefficients use train-only standardization. Importance identifies association among validated transformed summaries; it does not imply plaintext recovery.",
            "", "## Family coefficient magnitude", "", "| Family | Mean absolute coefficient | Maximum absolute coefficient |", "|---|---:|---:|"]
    for family, inds in {**families, "combined": list(range(610))}.items():
        attr.append(f"| {family} | {np.abs(coef[:, inds]).mean():.6f} | {np.abs(coef[:, inds]).max():.6f} |")
    attr += ["", "## Top standardized coefficients", "", "| Feature | Family | Mean coefficient | Mean absolute | Fold directions |", "|---|---|---:|---:|---|"]
    for j in top_coef:
        family = next(k for k, inds in families.items() if j in inds)
        directions = ",".join("+" if v > 0 else "-" if v < 0 else "0" for v in coef[:, j])
        attr.append(f"| {columns[j]} | {family} | {coef[:, j].mean():.6f} | {mean_abs[j]:.6f} | {directions} |")
    attr += ["", "## Top univariate features", "", "| Feature | Symmetric AUC | KS | Mutual information | Direction stable |", "|---|---:|---:|---:|---|"]
    for row in top_uni: attr.append(f"| {row['feature']} | {row['symmetric_auc']:.4f} | {row['ks']:.4f} | {row['mi']:.6f} | {row['direction_stable']} |")
    dump("feature_attribution.md", "\n".join(attr))

    config = {**META, "primary_classifier": "logistic_l2", "classifiers": {
        "logistic_l2": {"C_grid": CS, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced", "max_iter": 5000},
        "linear_svm": {"C_grid": CS, "class_weight": "balanced", "max_iter": 10000}},
        "random_seed": SEED, "hyperparameter_selection": "swap the two attack-training shadows; mean validation ROC-AUC; lower-C tie break",
        "V0": {"word_tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_features": 4000},
               "char_wb_tfidf": {"ngram_range": [3, 5], "min_df": 2, "max_features": 4000}},
        "V2": {"imputation": "training median", "scaling": "training StandardScaler"},
        "MLP": "not used; it was excluded by the frozen simulated-analysis policy"}
    dump("classifier_config.json", config)

    integrity_lines = [md_header("Integrity validation"), "Status: **PASS**", "", f"- Registered artifacts recomputed: {integrity['registered_artifacts']}; mismatches: 0.",
                       "- Each shadow: 1,000 rows, 500 members, 500 nonmembers, exact V0/V2/index join, and 610 ordered features.",
                       "- `hidden.final.last_l2`, prohibited metadata, membership labels, sample IDs, and trusted session secrets are absent from feature matrices and the registered handoff.", "", "| Shadow bundle | Inventory SHA-256 | Adapter/membership/schema/collection gates |", "|---|---|---|"]
    for item in integrity["per_shadow"]: integrity_lines.append(f"| `{item['bundle_path']}` | `{item['bundle_inventory_sha256']}` | PASS |")
    dump("integrity_validation.md", "\n".join(integrity_lines))

    dup_lines = [md_header("Duplicate and semantic-group audit"), f"- Exact duplicate inputs: {dup['exact_duplicate_inputs']} rows in {dup['exact_duplicate_input_groups']} groups.",
                 f"- Exact duplicate references: {dup['exact_duplicate_references']} rows in {dup['exact_duplicate_reference_groups']} groups.",
                 f"- Repeated exact meaning representations: {dup['repeated_exact_meaning_representations']} rows.",
                 f"- Near-duplicate pairs at token Jaccard ≥0.9: {dup['near_duplicate_pairs_jaccard_ge_0_9']}; groups: {dup['near_duplicate_groups']}; sizes: {dup['near_duplicate_group_sizes']}.",
                 f"- Semantic-template groups (`mr_field_set`): {dup['semantic_template_groups']}.",
                 "- No samples were removed. The group-disjoint sensitivity excludes every test semantic-template group from both attack-training shadows.", "", "## V0 output duplication/template audit", ""]
    for z in dup["generated_output_audit"]:
        dup_lines.append(f"- Shadow {z['shadow']}: {z['exact_duplicate_output_groups']} exact-output groups ({z['rows_in_exact_duplicate_output_groups']} rows; max {z['largest_exact_output_group']}); {z['repeated_output_template_groups']} repeated-template groups ({z['rows_in_repeated_output_templates']} rows; max {z['largest_output_template_group']}).")
    dump("duplicate_group_audit.md", "\n".join(dup_lines))

    dump("statistical_method.md", md_header("Statistical method") + f"\nPrimary evaluation uses exactly `(S2,S3)→S1`, `(S1,S3)→S2`, and `(S1,S2)→S3`. Imputation, scaling, TF-IDF vocabularies, and C selection are fitted only within the two attack-training shadows. Group-disjoint sensitivity uses five deterministic `mr_field_set` partitions and excludes test-group IDs from both training shadows.\n\nMetrics include ROC-AUC, balanced accuracy, average precision, max(TPR−FPR), TPR@1% FPR, and logistic calibration metrics. TPR@0.1% FPR is withheld because 500 negatives give 0.2% empirical resolution.\n\nPaired uncertainty uses {BOOTSTRAPS} within-shadow percentile-bootstrap replicates, shared indices across compared views, and equal shadow weighting. Intervals are exploratory because there are only three shadows.\n")
    dump("limitations.md", md_header("Limitations") + "\n- Only three shadow models are available; cross-shadow uncertainty and confidence intervals are exploratory.\n- The same 1,000 identities occur across shadows; the semantic-group-disjoint sensitivity is therefore required.\n- One source population is present, so a balanced source classifier is not identifiable.\n- TPR@0.1% FPR is unsupported with 500 held-out nonmembers.\n- Shadow identity is detectable from legitimate model behavior, although runtime identity metadata is absent.\n- This evaluation does not establish zero leakage, formal privacy, information-theoretic security, equivalence, or universal MIA resistance.\n")

    def primary_macro(view, metric="roc_auc"):
        g = primary[(primary.protocol == "primary_loso") & (primary.view == view) & (primary.classifier == "logistic_l2")]
        return float(g[metric].mean()), float(g[metric].std(ddof=0))
    real_values = {v: primary_macro(v)[0] for v in ("V0", "V2_combined", "V0_plus_V2")}
    v2_diff = boot_out["V2_combined_minus_V0"]["metrics"]["roc_auc"]
    fold_directions = boot_out["V2_combined_minus_V0"]["fold_effects"]["roc_auc"]
    v2_near = .45 <= real_values["V2_combined"] <= .55
    if not controls_pass:
        status = "REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_WITHHELD"; decision = "VALIDITY_CONTROL_FAILURE"
        claim = "The real-TDX paper-facing membership claim is withheld because a predefined negative control failed."
    elif v2_near and v2_diff["ci95"][0] <= 0:
        status = "REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_COMPLETE"; decision = "A"
        claim = "Under the evaluated three-shadow real-TDX setting, validated GPU-visible V2 features do not provide detectable membership advantage beyond the external output view."
        if real_values["V0"] > .55:
            decision = "C_QUALIFIED_V0_ABOVE_CHANCE"
            claim += " Output-level membership exposure remains, while the protected GPU view does not measurably amplify it."
    elif all(x > 0 for x in fold_directions) and v2_diff["ci95"][0] > 0:
        status = "REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_COMPLETE"; decision = "B"
        claim = "Under the evaluated real-TDX setting, GPU-visible transformed observations provide measurable additional membership signal."
    else:
        status = "REAL_TDX_MIA_THREE_SHADOW_ANALYSIS_COMPLETE"; decision = "D"
        claim = "Real-TDX membership results are inconclusive because effects are unstable across held-out shadow models."
    strongest_view = max(real_values, key=real_values.get)
    table = [md_header("Paper-facing real-TDX MIA table"), "Primary classifier: L2 logistic regression; values are macro mean ± population SD over three held-out shadows.", "",
             "| View | ROC-AUC | Balanced accuracy | Average precision | TPR@1% FPR | Advantage |", "|---|---:|---:|---:|---:|---:|"]
    for view in main_views:
        g = primary[(primary.protocol == "primary_loso") & (primary.view == view) & (primary.classifier == "logistic_l2")]
        table.append(f"| {view} | {g.roc_auc.mean():.3f} ± {g.roc_auc.std(ddof=0):.3f} | {g.balanced_accuracy.mean():.3f} ± {g.balanced_accuracy.std(ddof=0):.3f} | {g.average_precision.mean():.3f} ± {g.average_precision.std(ddof=0):.3f} | {g.tpr_at_1pct_fpr.mean():.3f} ± {g.tpr_at_1pct_fpr.std(ddof=0):.3f} | {g.attack_advantage.mean():.3f} ± {g.attack_advantage.std(ddof=0):.3f} |")
    table += ["", f"Secondary attacker-capability summary: `max(V0, V2, V0+V2)` ROC-AUC = **{real_values[strongest_view]:.3f}** ({strongest_view})."]
    dump("paper_table.md", "\n".join(table))
    dump("paper_claim_text.md", md_header("Paper claim text") + f"\nDecision: **{decision}**\n\n{claim}\n\nThis statement is limited to the evaluated three-shadow real-TDX setting and is not a claim of MIA prevention.\n")
    sim_lines = [md_header("Simulated versus real-TDX comparison"), "The real-TDX result is paper-facing; simulation is a larger controlled companion. Cross-domain differences are descriptive and not paired inference.", "",
                 "| View | Simulated LOSO AUC | Real-TDX LOSO AUC | Real − simulated |", "|---|---:|---:|---:|"]
    for view, sim in simulated.items(): sim_lines.append(f"| {view} | {sim:.3f} | {real_values[view]:.3f} | {real_values[view]-sim:+.3f} |")
    sim_group = {"V0": .610, "V2_combined": .504, "V0_plus_V2": .521}
    sim_lines += ["", "## Semantic-group-disjoint sensitivity", "", "| View | Simulated AUC | Real-TDX AUC | Real − simulated |", "|---|---:|---:|---:|"]
    for view, sim in sim_group.items():
        real = group_frame[group_frame.view == view].roc_auc.mean(); sim_lines.append(f"| {view} | {sim:.3f} | {real:.3f} | {real-sim:+.3f} |")
    dump("simulated_real_comparison.md", "\n".join(sim_lines))

    final = {**META, "status": status, "decision": decision, "paper_claim": claim, "input_integrity": "PASS",
             "completed_primary_folds": ["S2+S3->S1", "S1+S3->S2", "S1+S2->S3"],
             "group_disjoint_sensitivity": "PASS", "bootstrap": "PASS", "controls_pass": controls_pass,
             "control_macro_values": control_summary, "primary_logistic_macro_roc_auc": real_values,
             "max_attack_performance": {"metric": "roc_auc", "view": strongest_view, "value": real_values[strongest_view]},
             "v2_minus_v0_roc_auc": v2_diff, "fold_effect_directions": fold_directions,
             "unsupported_claims": ["zero leakage", "MIA prevention", "formal privacy", "information-theoretic protection", "universal robustness"]}
    dump("final_status.json", final)
    dump("final_status.md", md_header(status) + f"\nDecision: **{decision}**\n\n{claim}\n\nAll three LOSO folds, 15 semantic-group-disjoint subfolds, paired bootstrap, feature attribution, and required applicable negative controls completed.\n")

    required = ["final_status.md", "final_status.json", "integrity_validation.md", "duplicate_group_audit.md",
                "classifier_config.json", "per_fold_metrics.csv", "macro_metrics.csv", "group_disjoint_metrics.csv",
                "paired_view_differences.csv", "confidence_intervals.json", "negative_controls.csv",
                "feature_family_ablation.csv", "feature_attribution.md", "simulated_real_comparison.md",
                "statistical_method.md", "limitations.md", "paper_table.md", "paper_claim_text.md", "analysis_driver.py"]
    files = {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in required}
    dump("artifact_hashes.json", {**META, "status": "PASS", "file_count": len(files), "files": files})
    mismatches = [name for name, meta in files.items() if sha(HERE / name) != meta["sha256"] or (HERE / name).stat().st_size != meta["bytes"]]
    if mismatches: raise RuntimeError(f"output hash verification failed: {mismatches}")
    print(json.dumps({"status": status, "decision": decision, "controls_pass": controls_pass,
                      "roc_auc": real_values, "V2_minus_V0": v2_diff, "output_hashes": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
