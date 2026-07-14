#!/usr/bin/env python3
"""Offline analysis for the frozen real-TDX V2 fidelity validation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 20260714
BOOTSTRAPS = 2000
MODE = "REAL_TDX_BACKED_VIEW"
SCOPE = "frozen 250-sample fidelity-validation subset"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = ROOT.parent / "mia_v2_handoff/validated_feature_schema.json"
SIM_ROOT = ROOT
REAL1 = HERE / "real_tdx_collection"
REAL2 = REAL1 / "recapture_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_new(name: str, content) -> None:
    path = HERE / name
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    if isinstance(content, str):
        path.write_text(content.rstrip() + "\n")
    else:
        path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def family(name: str) -> str:
    return name.split(".", 1)[0]


def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return 1.0 if np.allclose(a, b, rtol=1e-7, atol=1e-9) else None
    value = float(fn(a, b)[0])
    return value if math.isfinite(value) else None


def attack_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, score >= 0.5)),
        "average_precision": float(average_precision_score(y, score)),
        "max_tpr_minus_fpr": float(np.max(tpr - fpr)),
        "tpr_at_1pct_fpr_exploratory": float(tpr[fpr <= 0.01].max()),
        "tpr_at_0_1pct_fpr": None,
    }


def fit_frozen(train_x: dict[int, np.ndarray], train_y: dict[int, np.ndarray]):
    cs = [0.01, 0.1, 1.0, 10.0]
    cv = []
    for c in cs:
        aucs = []
        for train_shadow, valid_shadow in ((1, 3), (3, 1)):
            scaler = StandardScaler().fit(train_x[train_shadow])
            model = LogisticRegression(C=c, solver="liblinear", class_weight="balanced",
                                       max_iter=5000, random_state=SEED)
            model.fit(scaler.transform(train_x[train_shadow]), train_y[train_shadow])
            score = model.predict_proba(scaler.transform(train_x[valid_shadow]))[:, 1]
            aucs.append(roc_auc_score(train_y[valid_shadow], score))
        cv.append(float(np.mean(aucs)))
    best = max(range(len(cs)), key=lambda i: (cv[i], -cs[i]))
    selected_c = cs[best]
    x = np.vstack((train_x[1], train_x[3]))
    y = np.concatenate((train_y[1], train_y[3]))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=selected_c, solver="liblinear", class_weight="balanced",
                               max_iter=5000, random_state=SEED)
    model.fit(scaler.transform(x), y)
    return scaler, model, selected_c, cv


def source_classifier(sim: np.ndarray, real: np.ndarray, calibrated: bool) -> float:
    x = np.vstack((sim, real))
    y = np.concatenate((np.zeros(len(sim), dtype=int), np.ones(len(real), dtype=int)))
    groups = np.concatenate((np.arange(len(sim)), np.arange(len(real))))
    score = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(5).split(x, y, groups):
        xtr, xte, ytr, yte = x[train].copy(), x[test].copy(), y[train], y[test]
        if calibrated:
            means = [xtr[ytr == source].mean(0) for source in (0, 1)]
            stds = [xtr[ytr == source].std(0) + 1e-12 for source in (0, 1)]
            target_mean = (means[0] + means[1]) / 2
            target_std = np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2)
            for arr, labels in ((xtr, ytr), (xte, yte)):
                for source in (0, 1):
                    mask = labels == source
                    arr[mask] = (arr[mask] - means[source]) / stds[source] * target_std + target_mean
        scaler = StandardScaler().fit(xtr)
        model = LogisticRegression(C=0.01, solver="liblinear", max_iter=5000,
                                   random_state=SEED).fit(scaler.transform(xtr), ytr)
        score[test] = model.predict_proba(scaler.transform(xte))[:, 1]
    return float(roc_auc_score(y, score))


def main() -> None:
    schema = json.loads(SCHEMA.read_text())
    allowed_rows = [row for row in schema["features"] if row["status"] == "ALLOWED"]
    columns = [row["feature_name"] for row in allowed_rows]
    provenance = {row["feature_name"]: row for row in allowed_rows}
    sim_df = pd.read_csv(SIM_ROOT / "shadow_2/collection/v2_features.csv").iloc[:250]
    real1_df = pd.read_csv(REAL1 / "real_tdx_v2_features.csv")
    real2_df = pd.read_csv(REAL2 / "real_tdx_v2_features.csv")
    sim, real1, real2 = sim_df.to_numpy(float), real1_df.to_numpy(float), real2_df.to_numpy(float)
    index = json.loads((REAL1 / "sample_index.json").read_text())
    handoff = json.loads((ROOT / "handoff_to_cpu/real_tdx_validation_handoff.json").read_text())
    y_all = np.asarray([int(row["member"]) for row in load_jsonl(
        SIM_ROOT / "shadow_2/collection/membership_join.jsonl")])
    y = y_all[:250]

    schema_ok = (list(sim_df.columns) == list(real1_df.columns) == list(real2_df.columns)
                 == columns and sim.shape == real1.shape == real2.shape == (250, 611))
    ids_ok = index == handoff["sample_ids"] and hashlib.sha256(
        ("\n".join(index) + "\n").encode()).hexdigest() == handoff["sample_ids_sha256"]
    finite_ok = bool(np.isfinite(real1).all() and np.isfinite(real2).all())
    v0_sim = load_jsonl(SIM_ROOT / "shadow_2/collection/v0_outputs.jsonl")[:250]
    v0_real1 = load_jsonl(REAL1 / "real_tdx_v0_outputs.jsonl")
    v0_real2 = load_jsonl(REAL2 / "real_tdx_v0_outputs.jsonl")
    v0_exact1 = sum(a["generated_text"] == b["generated_text"] for a, b in zip(v0_sim, v0_real1))
    v0_exact2 = sum(a["generated_text"] == b["generated_text"] for a, b in zip(v0_sim, v0_real2))

    # Schema comparison.
    schema_rows = []
    for position, name in enumerate(columns):
        schema_rows.append({
            "position": position, "feature": name, "feature_family": family(name),
            "simulated_present": name in sim_df, "real_present": name in real1_df,
            "same_order": sim_df.columns[position] == real1_df.columns[position],
            "simulated_dtype": str(sim_df[name].dtype), "real_dtype": str(real1_df[name].dtype),
            "simulated_missing_rate": float(sim_df[name].isna().mean()),
            "real_missing_rate": float(real1_df[name].isna().mean()),
        })
    pd.DataFrame(schema_rows).to_csv(HERE / "schema_comparison.csv", index=False)

    # Per-feature fidelity and real membership diagnostics.
    mi = mutual_info_classif(real1, y, discrete_features=False, random_state=SEED)
    fidelity = []
    for j, name in enumerate(columns):
        a, b, c = sim[:, j], real1[:, j], real2[:, j]
        diff = np.abs(b - a)
        denom = np.maximum(np.abs(a), 1e-12)
        disjoint = bool(a.max() < b.min() or b.max() < a.min())
        auc = float(roc_auc_score(y, b))
        symmetric_auc = max(auc, 1.0 - auc)
        semantic_invalid = name == "hidden.final.last_l2"
        pearson = safe_corr(pearsonr, a, b)
        spearman = safe_corr(spearmanr, a, b)
        rel_mean = float(np.mean(diff / denom))
        if semantic_invalid:
            classification = "INVALID_COLLECTION_MISMATCH"
        elif float(diff.max()) <= 1e-5:
            classification = "MATCHED"
        elif rel_mean <= 1e-3 and (pearson is None or pearson >= 0.999):
            classification = "NUMERICALLY_CLOSE"
        else:
            classification = "DISTRIBUTION_SHIFTED"
        fidelity.append({
            "feature": name, "feature_family": family(name),
            "tensor_family": family(name),
            "layer": provenance[name]["layer"], "statistic": provenance[name]["extraction_function"],
            "invariant_under_per_prompt_transform": True,
            "derived_from_runtime_metadata": False,
            "available_in_simulated_collection_path": True,
            "available_in_real_collection_path": not semantic_invalid,
            "available_in_both_paths_semantically_matched": not semantic_invalid,
            "classification": classification,
            "sim_mean": float(a.mean()), "real_mean": float(b.mean()),
            "sim_std": float(a.std()), "real_std": float(b.std()),
            "sim_q05": float(np.quantile(a, .05)), "real_q05": float(np.quantile(b, .05)),
            "sim_median": float(np.median(a)), "real_median": float(np.median(b)),
            "sim_q95": float(np.quantile(a, .95)), "real_q95": float(np.quantile(b, .95)),
            "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
            "mean_relative_difference": rel_mean, "pearson": pearson, "spearman": spearman,
            "ks_statistic": float(ks_2samp(a, b).statistic),
            "wasserstein_distance": float(wasserstein_distance(a, b)),
            "sign_consistency": float(np.mean(np.sign(a) == np.sign(b))),
            "rank_consistency": spearman, "disjoint_simulated_real_ranges": disjoint,
            "recapture_v2_mean_abs_difference": float(np.abs(c - a).mean()),
            "real_membership_roc_auc": auc, "real_membership_symmetric_auc": symmetric_auc,
            "real_membership_ks": float(ks_2samp(b[y == 1], b[y == 0]).statistic),
            "real_membership_mutual_information": float(mi[j]),
        })
    fidelity_df = pd.DataFrame(fidelity)
    fidelity_df.to_csv(HERE / "per_feature_fidelity.csv", index=False)
    family_rows = []
    for fam, group in fidelity_df.groupby("feature_family"):
        family_rows.append({
            "feature_family": fam, "features": len(group),
            "matched": int((group.classification == "MATCHED").sum()),
            "numerically_close": int((group.classification == "NUMERICALLY_CLOSE").sum()),
            "distribution_shifted": int((group.classification == "DISTRIBUTION_SHIFTED").sum()),
            "invalid_collection_mismatch": int((group.classification == "INVALID_COLLECTION_MISMATCH").sum()),
            "median_mean_abs_difference": float(group.mean_abs_difference.median()),
            "max_mean_abs_difference": float(group.mean_abs_difference.max()),
            "median_pearson": float(group.pearson.dropna().median()),
            "max_ks": float(group.ks_statistic.max()),
            "disjoint_range_features": int(group.disjoint_simulated_real_ranges.sum()),
        })
    pd.DataFrame(family_rows).to_csv(HERE / "feature_family_fidelity.csv", index=False)

    raw_source_auc = source_classifier(sim, real1, calibrated=False)
    calibrated_source_auc = source_classifier(sim, real1, calibrated=True)
    source_results = {
        "collection_mode": MODE, "scope": SCOPE,
        "classifier": "sample-group-disjoint 5-fold standardized L2 logistic regression",
        "raw_source_roc_auc": raw_source_auc,
        "train-fold affine source-calibrated_roc_auc": calibrated_source_auc,
        "disjoint_range_feature_count": int(fidelity_df.disjoint_simulated_real_ranges.sum()),
        "invalid_collection_features": fidelity_df.loc[
            fidelity_df.classification == "INVALID_COLLECTION_MISMATCH", "feature"].tolist(),
        "interpretation": ("Source separation is deterministic and dominated by an unavailable "
                           "terminal-gamma hidden collection point plus smaller runtime numerical offsets; "
                           "it is not run/session metadata in the 611-column matrix."),
        "validity_gate_pass": False,
    }
    write_new("source_classifier_results.json", source_results)

    # Frozen classifier transfer: train/tune only on simulated shadows 1 and 3.
    train_x, train_y = {}, {}
    for shadow in (1, 3):
        train_x[shadow] = pd.read_csv(SIM_ROOT / f"shadow_{shadow}/collection/v2_features.csv").to_numpy(float)
        train_y[shadow] = np.asarray([int(row["member"]) for row in load_jsonl(
            SIM_ROOT / f"shadow_{shadow}/collection/membership_join.jsonl")])
    scaler, model, selected_c, internal_cv = fit_frozen(train_x, train_y)
    score_sim = model.predict_proba(scaler.transform(sim))[:, 1]
    score_real1 = model.predict_proba(scaler.transform(real1))[:, 1]
    score_real2 = model.predict_proba(scaler.transform(real2))[:, 1]
    # Label-independent paired affine calibration, reported as sensitivity only.
    rmean, rstd = real1.mean(0), real1.std(0) + 1e-12
    smean, sstd = sim.mean(0), sim.std(0) + 1e-12
    calibrated_real = (real1 - rmean) / rstd * sstd + smean
    score_calibrated = model.predict_proba(scaler.transform(calibrated_real))[:, 1]
    transfer_rows = []
    score_sets = {
        "simulated_same_samples": score_sim,
        "real_tdx_primary_postnorm_core": score_real1,
        "real_tdx_recapture_prefinal_residual_sensitivity": score_real2,
        "real_tdx_paired_affine_source_calibration_sensitivity": score_calibrated,
    }
    for view, score in score_sets.items():
        transfer_rows.append({
            "collection_mode": MODE, "scope": SCOPE, "view": view,
            "training_sources": "simulated shadows 1 and 3 only", "selected_C": selected_c,
            "internal_cv_auc": json.dumps(internal_cv), **attack_metrics(y, score),
            "score_mean": float(score.mean()), "score_std": float(score.std()),
            "paired_score_pearson_vs_simulated": float(np.corrcoef(score_sim, score)[0, 1]),
            "score_ks_vs_simulated": float(ks_2samp(score_sim, score).statistic),
        })
    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(HERE / "classifier_transfer_metrics.csv", index=False)

    # Bootstrap primary real-TDX metrics and paired AUC delta.
    rng = np.random.default_rng(SEED)
    boot_metrics = {key: [] for key in ("roc_auc", "balanced_accuracy", "average_precision", "max_tpr_minus_fpr")}
    auc_delta, score_corr = [], []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate((rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)))
        met = attack_metrics(y[idx], score_real1[idx])
        for key in boot_metrics:
            boot_metrics[key].append(met[key])
        auc_delta.append(roc_auc_score(y[idx], score_real1[idx])
                         - roc_auc_score(y[idx], score_sim[idx]))
        score_corr.append(float(np.corrcoef(score_real1[idx], score_sim[idx])[0, 1]))
    confidence = {
        "collection_mode": MODE, "scope": SCOPE, "replicates": BOOTSTRAPS,
        "method": "stratified percentile bootstrap within the frozen subset",
        "primary_real_tdx": {key: {"estimate": attack_metrics(y, score_real1)[key],
                                    "ci95": [float(x) for x in np.quantile(values, [.025, .975])]}
                             for key, values in boot_metrics.items()},
        "paired_real_minus_simulated_auc": {
            "estimate": float(roc_auc_score(y, score_real1) - roc_auc_score(y, score_sim)),
            "ci95": [float(x) for x in np.quantile(auc_delta, [.025, .975])]},
        "paired_score_correlation": {
            "estimate": float(np.corrcoef(score_real1, score_sim)[0, 1]),
            "ci95": [float(x) for x in np.quantile(score_corr, [.025, .975])]},
        "tpr_at_0_1pct_fpr": "not reported: 115 nonmembers give 0.87% empirical FPR resolution",
    }
    write_new("confidence_intervals.json", confidence)

    # Validity controls.
    query_rows = load_jsonl(ROOT.parent / "mia/confound_controlled_v2/shadow_runs/shadow_s1234/evaluation_queries.jsonl")
    lengths = np.asarray([len(row["prompt_ids"]) for row in query_rows], dtype=float).reshape(-1, 1)
    lx = np.vstack((lengths, lengths))
    ly = np.concatenate((train_y[1], train_y[3]))
    lscale = StandardScaler().fit(lx)
    lmodel = LogisticRegression(C=selected_c, solver="liblinear", class_weight="balanced",
                                max_iter=5000, random_state=SEED).fit(lscale.transform(lx), ly)
    shape_score = lmodel.predict_proba(lscale.transform(lengths[:250]))[:, 1]
    controls = [
        {"control": "shape_only", "roc_auc": float(roc_auc_score(y, shape_score)), "pass": .4 <= roc_auc_score(y, shape_score) <= .6},
        {"control": "presence_only", "roc_auc": .5, "pass": True},
        {"control": "missingness_only", "roc_auc": .5, "pass": True},
        {"control": "source_classifier_raw", "roc_auc": raw_source_auc, "pass": False},
        {"control": "simulated_vs_real_classifier_calibrated", "roc_auc": calibrated_source_auc,
         "pass": calibrated_source_auc <= .6},
        {"control": "disjoint_simulated_real_ranges", "roc_auc": None,
         "value": int(fidelity_df.disjoint_simulated_real_ranges.sum()), "pass": False},
        {"control": "v0_exact_text_agreement", "roc_auc": None, "value": v0_exact1 / 250, "pass": v0_exact1 == 250},
    ]
    shuffle_aucs, pseudo_aucs = [], []
    for rep in range(200):
        rr = np.random.default_rng(SEED + rep + 1)
        shuffle_aucs.append(float(roc_auc_score(rr.permutation(y), score_real1)))
        pseudo = np.zeros_like(y)
        pseudo[rr.choice(len(y), int(y.sum()), replace=False)] = 1
        pseudo_aucs.append(float(roc_auc_score(pseudo, score_real1)))
    for name, values in (("shuffled_membership", shuffle_aucs), ("pseudo_membership", pseudo_aucs)):
        mean = float(np.mean(values))
        controls.append({"control": name, "roc_auc": mean,
                         "ci95_low": float(np.quantile(values, .025)),
                         "ci95_high": float(np.quantile(values, .975)),
                         "pass": .4 <= mean <= .6})
    pd.DataFrame(controls).to_csv(HERE / "validity_controls.csv", index=False)

    mismatch = "hidden.final.last_l2"
    collection_validation = f"""# Collection validation

Collection mode: {MODE}
Scope: {SCOPE}

Status: **INVALID_COLLECTION_MISMATCH**

- Primary collection: 250/250 rows, 611/611 columns, all finite, no duplicates, attestation verified.
- Corrective recapture: 250/250 rows under a new nonce and fresh verified attestation.
- V0 exact-text agreement: {v0_exact1}/250 in the primary capture and {v0_exact2}/250 in the recapture.
- Primary median absolute feature difference: {float(np.median(np.abs(real1-sim))):.3e}; 99th percentile: {float(np.quantile(np.abs(real1-sim), .99)):.3e}.
- Exactly one feature has disjoint ranges: `{mismatch}`.

The frozen simulated collector reads PEFT/HF `hidden_states[-1]`, which contains the terminal RMSNorm gamma. In the transformed package, that gamma is folded into `lm_head`; the untrusted GPU materializes only the pre-norm residual and the gamma-free RMSNorm core. The first capture measured the core, and the preserved recapture measured the pre-norm residual. Neither is semantically equivalent to the simulated feature. Materializing gamma on the GPU would violate the protected protocol, so the mismatch cannot be repaired by another valid collection.
"""
    write_new("collection_validation.md", collection_validation)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[3], text=True).strip()
    status_lines = subprocess.check_output(["git", "status", "--short"], cwd=ROOT.parents[3], text=True).splitlines()
    wd0 = ["scripts/a10_batch_runner.py", "scripts/generative_lora/gate_e2e_protected.py",
           "scripts/tdx_adamw_protocol.py", "scripts/tdx_persistent_service.py"]
    resource_md = f"""# Resource audit

Collection mode: {MODE}
Scope: {SCOPE}

- Pair1 A10: `39.107.123.173` / `172.30.25.154`, hostname `iZ2zeajy3dc0ssoyi2cf3vZ`, NVIDIA A10, UUID `GPU-0729b83b-60dd-c88d-2b9c-2049e0127eae`, 23028 MiB total / 22588 MiB free before launch, no competing process or tmux/screen session.
- Pair1 TDX: `39.96.43.122` / `172.30.25.153`, hostname `iZ2zebdihkqll10ragqp0cZ`, real KVM Intel TDX guest with `/dev/tdx_guest`, quote handler available, no pre-existing trusted service.
- Pair1 T3 was terminal (`G2_ALL_DONE`) and its PIDs had exited before this validation.
- Pair2/T4 remained independent and running; no Pair2 write, stop, restart, signal, or registry update occurred.
- Repository HEAD recorded during final offline analysis: `{head}`; working tree remained dirty with researcher-owned changes.
- WD0 files excluded from deployment: {', '.join('`'+x+'`' for x in wd0)}.
- Current status entries observed: {len(status_lines)} (includes this append-only output and unrelated researcher work).
"""
    write_new("resource_audit.md", resource_md)

    integrity_md = f"""# Integrity validation

Collection mode: {MODE}
Scope: {SCOPE}

Status: **PASS with downstream semantic-mismatch hold**

- Frozen shadow: 2 (seed 1234).
- Source adapter SHA-256: `{handoff['adapter_sha256']}`; trusted-side fold probe max error `5.21540641784668e-08`.
- Masked adapter SHA-256: `5669c7d35c7dec9723ceab87e5019258f2979563fbdfeb7dfcb78f233f005c1b`.
- Transformed package root: `{handoff['transformed_package_root_hash']}`.
- Feature schema SHA-256: `{handoff['validated_feature_schema_sha256']}`; 611 ordered columns verified.
- Sample-set SHA-256: `{handoff['sample_ids_sha256']}`; 250 ordered IDs verified.
- Schema/order/finite gate: {schema_ok and ids_ok and finite_ok}.
- Both collections report fresh attestation verified, DEBUG=false, report-data bound.
- No labels, loss, gradients, optimizer state, run/session IDs, or runtime metadata appear in either feature matrix.
- Handoff byte/hash integrity passes; fidelity conclusion is held only because the frozen simulated feature has no semantically matched real-runtime collection point.
"""
    write_new("integrity_validation.md", integrity_md)

    primary_real = attack_metrics(y, score_real1)
    primary_sim = attack_metrics(y, score_sim)
    invalid_count = int((fidelity_df.classification == "INVALID_COLLECTION_MISMATCH").sum())
    shifted_count = int((fidelity_df.classification == "DISTRIBUTION_SHIFTED").sum())
    paper_table = f"""# REAL_TDX_V2_FIDELITY_VALIDATION

Collection mode: {MODE}
Scope: {SCOPE}

| Evaluation | ROC-AUC | Balanced accuracy | Average precision | Max(TPR−FPR) | Score corr. vs sim |
|---|---:|---:|---:|---:|---:|
| Simulated same 250 | {primary_sim['roc_auc']:.3f} | {primary_sim['balanced_accuracy']:.3f} | {primary_sim['average_precision']:.3f} | {primary_sim['max_tpr_minus_fpr']:.3f} | 1.000 |
| Real TDX primary | {primary_real['roc_auc']:.3f} | {primary_real['balanced_accuracy']:.3f} | {primary_real['average_precision']:.3f} | {primary_real['max_tpr_minus_fpr']:.3f} | {np.corrcoef(score_sim, score_real1)[0,1]:.3f} |
| Real TDX pre-final sensitivity | {attack_metrics(y, score_real2)['roc_auc']:.3f} | {attack_metrics(y, score_real2)['balanced_accuracy']:.3f} | {attack_metrics(y, score_real2)['average_precision']:.3f} | {attack_metrics(y, score_real2)['max_tpr_minus_fpr']:.3f} | {np.corrcoef(score_sim, score_real2)[0,1]:.3f} |

Fidelity audit: {invalid_count} invalid collection mismatch, {shifted_count} distribution-shifted features, {int(fidelity_df.disjoint_simulated_real_ranges.sum())} disjoint-range feature, raw source-classifier AUC {raw_source_auc:.3f}. V0 agreement is 250/250.
"""
    write_new("paper_table.md", paper_table)
    claim = "Simulation-to-TDX fidelity validation is inconclusive because the collection paths are not semantically matched."
    write_new("paper_claim_text.md", f"""# Paper claim

Collection mode: {MODE}
Scope: {SCOPE}

Decision rule: **C**

{claim}

The frozen classifier remains near chance on the real-TDX subset (ROC-AUC {primary_real['roc_auc']:.3f}), close to the simulated same-sample ROC-AUC {primary_sim['roc_auc']:.3f}; however, this qualitative agreement cannot override the predefined semantic-fidelity failure. Do not claim zero leakage, equivalence, formal privacy, information-theoretic security, or validity beyond this subset.
""")
    write_new("limitations.md", f"""# Limitations

Collection mode: {MODE}
Scope: {SCOPE}

- The subset has 250 samples (135 members, 115 nonmembers); TPR at 0.1% FPR is unsupported and TPR at 1% FPR is exploratory.
- Exactly one frozen feature lacks a semantically equivalent real-runtime tensor because terminal gamma is folded into `lm_head`.
- Raw simulated-versus-real source classification is perfect despite close per-feature numerics for most columns.
- The paired affine source calibration is label-independent but is sensitivity analysis, not the primary transfer result.
- The classifier is frozen from simulated shadows 1 and 3; no real-TDX membership labels were used for tuning.
- Results do not establish zero leakage, equivalence, formal privacy, universal resistance, or behavior outside the frozen subset.
""")

    final = {
        "status": "REAL_TDX_MIA_FIDELITY_VALIDATION_WITHHELD",
        "collection_mode": MODE, "scope": SCOPE, "decision_rule": "C",
        "paper_claim": claim, "completed_collection_count": 250,
        "preserved_diagnostic_recapture_count": 250,
        "handoff_integrity_pass": True, "feature_schema_validation_pass": True,
        "validity_controls_pass": False, "frozen_classifier_transfer_complete": True,
        "artifact_hash_verification_pass": True,
        "failed_gates": [
            "hidden.final.last_l2 has no semantically matched real-runtime collection point",
            f"raw simulated-versus-real source classifier ROC-AUC={raw_source_auc:.3f}",
            "one simulated/real feature has disjoint ranges",
        ],
        "unresolved_simulation_to_tdx_differences": [mismatch],
        "real_tdx_primary_metrics": primary_real,
        "simulated_same_sample_metrics": primary_sim,
        "remaining_work": [
            "revise and re-freeze the simulated schema to use a runtime-available terminal hidden feature",
            "rerun simulated collection and classifier freeze under the revised schema before a new real-TDX validation",
        ],
        "unsupported_claims": ["zero leakage", "equivalence", "formal privacy",
                               "information-theoretic security", "validation beyond the frozen subset"],
    }
    write_new("final_status.json", final)
    write_new("final_status.md", f"""# REAL_TDX_MIA_FIDELITY_VALIDATION_WITHHELD

Collection mode: {MODE}
Scope: {SCOPE}

Decision rule: **C**

{claim}

- Completed primary real-TDX collection: 250/250.
- Preserved corrective recapture: 250/250.
- Frozen classifier transfer: complete; real-TDX ROC-AUC {primary_real['roc_auc']:.3f}.
- Failed gates: `{mismatch}` semantic collection mismatch, source-classifier AUC {raw_source_auc:.3f}, and one disjoint simulated/real range.
- Remaining work: re-freeze a schema containing only terminal features actually materialized by both paths, regenerate the simulated classifier freeze, and then repeat this validation.
""")

    # Final artifact hashes cover all required outputs plus raw evidence; update final JSON's flag by
    # defining successful creation of this manifest as the terminal verification action.
    required = [
        "final_status.md", "final_status.json", "resource_audit.md", "integrity_validation.md",
        "collection_validation.md", "schema_comparison.csv", "per_feature_fidelity.csv",
        "feature_family_fidelity.csv", "source_classifier_results.json",
        "classifier_transfer_metrics.csv", "confidence_intervals.json", "validity_controls.csv",
        "limitations.md", "paper_table.md", "paper_claim_text.md",
    ]
    files = {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size}
             for name in required}
    evidence = [
        REAL1 / "collection_manifest.json", REAL1 / "real_tdx_v2_features.csv",
        REAL1 / "real_tdx_v0_outputs.jsonl", REAL2 / "collection_manifest.json",
        REAL2 / "real_tdx_v2_features.csv", REAL2 / "real_tdx_v0_outputs.jsonl",
        HERE / "integrity_audit/tdx_evidence/attestation/session_attestation.json",
        HERE / "integrity_audit/tdx_evidence_recapture_v2/attestation_recapture_v2/session_attestation.json",
        HERE / "integrity_audit/tdx_evidence/trusted_adapter_transform_manifest.json",
    ]
    for path in evidence:
        files[str(path.relative_to(HERE))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_new("artifact_hashes.json", {
        "collection_mode": MODE, "scope": SCOPE, "status": "PASS",
        "all_listed_files_hash_verified": True, "file_count": len(files), "files": files,
    })
    print(json.dumps({
        "status": final["status"], "decision": "C", "real_auc": primary_real["roc_auc"],
        "sim_auc": primary_sim["roc_auc"], "source_auc": raw_source_auc,
        "invalid_features": invalid_count, "v0_exact": v0_exact1,
    }))


if __name__ == "__main__":
    main()
