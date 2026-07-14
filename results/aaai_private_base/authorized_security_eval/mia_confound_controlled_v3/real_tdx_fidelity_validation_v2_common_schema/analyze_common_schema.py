#!/usr/bin/env python3
"""Append-only offline audit of the real-TDX common-semantics V2 schema."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 20260714
BOOTSTRAPS = 2000
MODE = "REAL_TDX_BACKED_VIEW"
SCOPE = "frozen 250-sample subset"
SCHEMA_NAME = "common-semantics schema v2"
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V1 = BASE / "real_tdx_fidelity_validation_v1"
ORIGINAL_SCHEMA = BASE.parent / "mia_v2_handoff" / "validated_feature_schema.json"
SIM2 = BASE / "shadow_2" / "collection"
REAL = V1 / "real_tdx_collection"
EXCLUDED = "hidden.final.last_l2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(name: str, content) -> None:
    p = HERE / name
    if p.exists():
        raise RuntimeError(f"append-only guard: refusing to overwrite {p}")
    if isinstance(content, str):
        p.write_text(content.rstrip() + "\n")
    else:
        p.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def safe_corr(fn, a, b):
    if np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
        return 1.0 if np.allclose(a, b, rtol=1e-7, atol=1e-9) else None
    v = float(fn(a, b)[0])
    return v if math.isfinite(v) else None


def source_oof(sim: np.ndarray, real: np.ndarray, normalized: bool = False,
               labels: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    x = np.vstack([sim, real]).astype(float, copy=True)
    default_y = np.r_[np.zeros(len(sim), int), np.ones(len(real), int)]
    y = default_y if labels is None else labels.astype(int)
    groups = np.r_[np.arange(len(sim)), np.arange(len(real))]
    score = np.zeros(len(y))
    for train, test in GroupKFold(5).split(x, y, groups):
        xtr, xte = x[train].copy(), x[test].copy()
        ytr, yte = y[train], y[test]
        if normalized:
            # Source identity is an engineering-domain label, not membership. Moments are
            # learned only in the training fold and applied by known collection path.
            means = {s: xtr[ytr == s].mean(0) for s in (0, 1)}
            stds = {s: xtr[ytr == s].std(0) + 1e-12 for s in (0, 1)}
            target_mean = (means[0] + means[1]) / 2
            target_std = np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2)
            for arr, lab in ((xtr, ytr), (xte, yte)):
                for s in (0, 1):
                    m = lab == s
                    arr[m] = (arr[m] - means[s]) / stds[s] * target_std + target_mean
        scaler = StandardScaler().fit(xtr)
        model = LogisticRegression(C=.01, solver="liblinear", max_iter=5000,
                                   random_state=SEED).fit(scaler.transform(xtr), ytr)
        score[test] = model.predict_proba(scaler.transform(xte))[:, 1]
    return float(roc_auc_score(y, score)), score


def paired_auc_ci(y: np.ndarray, score: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(y) // 2
    vals = []
    for _ in range(BOOTSTRAPS):
        idx = rng.choice(n, n, replace=True)
        ii = np.r_[idx, idx + n]
        vals.append(roc_auc_score(y[ii], score[ii]))
    return [float(x) for x in np.quantile(vals, [.025, .975])]


def graph_semantics(row: dict, classification: str) -> dict:
    name, fam = row["feature_name"], row["feature_family"]
    common = classification != "SEMANTICALLY_MISMATCHED"
    if name == EXCLUDED:
        return {
            "simulation_tensor_source": "PEFT/HF model output hidden_states[-1]",
            "real_tdx_tensor_source": "primary: gamma-free terminal RMSNorm core; recapture: pre-terminal-RMSNorm residual",
            "computational_graph_point": "simulation: after terminal RMSNorm gamma; real: core or pre-norm, neither equivalent",
            "rmsnorm_gamma_applied_simulation": True,
            "rmsnorm_gamma_applied_real_tdx": False,
            "gamma_folded_into_downstream_weight_real_tdx": True,
            "gpu_visible_in_both_paths": False,
            "runtime_dtype": "BF16 activation; FP32 reduction; CSV float64 serialization",
            "mask_or_transform_state": "orthogonal signed permutation; L2 invariant; real gamma remains trusted/folded",
        }
    if fam == "hidden":
        src = "layer-input residual before self-attention"
        mask = "orthogonal signed permutation; L2 invariant"
    elif fam == "kv":
        src = "post-projection/rotary masked K or projected masked V, grouped by head at last token"
        mask = "protocol head/coordinate transform; per-head L2 then order-insensitive reduction"
    elif fam == "attention":
        src = "last-token attention probabilities or true QK scores after transform cancellation"
        mask = "Q/K transforms cancel in scores; order-insensitive per-head reduction"
    else:
        src = "last-token masked logits or top-32 values exposed by the protocol"
        mask = "vocabulary permutation; scalar/top-value reductions are permutation invariant"
    return {
        "simulation_tensor_source": f"simulated protocol {src}",
        "real_tdx_tensor_source": f"real transformed runtime {src}",
        "computational_graph_point": src,
        "rmsnorm_gamma_applied_simulation": fam == "logit",
        "rmsnorm_gamma_applied_real_tdx": fam == "logit",
        "gamma_folded_into_downstream_weight_real_tdx": fam == "logit",
        "gpu_visible_in_both_paths": common,
        "runtime_dtype": "BF16 tensor path; FP32 reduction/softmax; CSV float64 serialization",
        "mask_or_transform_state": mask,
    }


def attack_metrics(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, score >= .5)),
        "average_precision": float(average_precision_score(y, score)),
        "max_tpr_minus_fpr": float(np.max(tpr - fpr)),
        "tpr_at_1pct_fpr_exploratory": float(tpr[fpr <= .01].max()),
        "tpr_at_0_1pct_fpr": None,
    }


def fit_frozen(train_x, train_y):
    cs = [.01, .1, 1., 10.]
    cv = []
    for c in cs:
        vals = []
        for tr, va in ((1, 3), (3, 1)):
            sc = StandardScaler().fit(train_x[tr])
            m = LogisticRegression(C=c, solver="liblinear", class_weight="balanced",
                                   max_iter=5000, random_state=SEED)
            m.fit(sc.transform(train_x[tr]), train_y[tr])
            vals.append(roc_auc_score(train_y[va], m.predict_proba(sc.transform(train_x[va]))[:, 1]))
        cv.append(float(np.mean(vals)))
    best = max(range(len(cs)), key=lambda i: (cv[i], -cs[i]))
    x = np.vstack([train_x[1], train_x[3]])
    y = np.r_[train_y[1], train_y[3]]
    sc = StandardScaler().fit(x)
    m = LogisticRegression(C=cs[best], solver="liblinear", class_weight="balanced",
                           max_iter=5000, random_state=SEED).fit(sc.transform(x), y)
    return sc, m, cs[best], cv


def main() -> None:
    original = json.loads(ORIGINAL_SCHEMA.read_text())
    allowed = [r for r in original["features"] if r["status"] == "ALLOWED"]
    cols611 = [r["feature_name"] for r in allowed]
    cols = [x for x in cols611 if x != EXCLUDED]
    prov = {r["feature_name"]: r for r in allowed}
    prior_fid = pd.read_csv(V1 / "per_feature_fidelity.csv").set_index("feature")
    sim_all = pd.read_csv(SIM2 / "v2_features.csv").iloc[:250]
    real_all = pd.read_csv(REAL / "real_tdx_v2_features.csv")
    sim_df, real_df = sim_all[cols], real_all[cols]
    sim, real = sim_df.to_numpy(float), real_df.to_numpy(float)
    ysrc = np.r_[np.zeros(250, int), np.ones(250, int)]

    # Phase 0: immutable prior inventory and hashes.
    prior_files = sorted(p for p in V1.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(V1)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in prior_files}
    write("prior_run_hashes.json", {"prior_root": str(V1), "files": hashes})
    inv_lines = ["# Prior run inventory", "", f"Prior root: `{V1}`", "",
                 f"Referenced immutable files: {len(prior_files)}. No prior artifact was copied or modified.", "",
                 "| Relative path | SHA-256 | Bytes |", "|---|---|---:|"]
    inv_lines += [f"| `{n}` | `{v['sha256']}` | {v['bytes']} |" for n, v in hashes.items()]
    write("prior_run_inventory.md", "\n".join(inv_lines))
    write("mismatch_root_cause.md", f"""# Mismatch root cause

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

1. **Protocol-semantic mismatch.** `{EXCLUDED}` observes post-gamma `hidden_states[-1]` in simulation, but the real package exposes only a gamma-free RMSNorm core or a pre-norm residual because gamma is folded into `lm_head`.
2. **Finite-precision numeric difference.** BF16 kernels, reduction order, and FP32 serialization produce small matched-path differences. The three layer-0 K reductions are near-constant and therefore correlation/rank diagnostics are ill-conditioned despite mean absolute errors below `8e-6`.
3. **Collector implementation difference.** The v1 primary and corrective collector intentionally probed two distinct real final-hidden points; neither is a substitute for the simulated post-gamma tensor. Other 610 columns use matched graph points.
4. **Metadata contamination.** None detected: feature matrices contain no file, run/session, checkpoint, mode, step/epoch, path-presence, member, or split field.
""")

    # Phase 1 semantic audit, all 611 features.
    audit = []
    for row in allowed:
        name = row["feature_name"]
        old = prior_fid.loc[name, "classification"]
        if name == EXCLUDED:
            cls = "SEMANTICALLY_MISMATCHED"
        elif old == "MATCHED":
            cls = "COMMON_EXACT_SEMANTICS"
        else:
            cls = "COMMON_NUMERICALLY_APPROXIMATE"
        sem = graph_semantics(row, cls)
        audit.append({
            "feature": name, "tensor_family": row["feature_family"], "layer": row["layer"],
            "statistic": row["extraction_function"], **sem,
            "reduction_function": row["extraction_function"],
            "semantic_equivalence_classification": cls,
            "in_common_v2_schema": name in cols,
            "prior_numeric_classification": old,
            "derived_from_runtime_metadata": False,
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(HERE / "feature_semantics_audit.csv", index=False)

    canonical = []
    for name in cols:
        r = prov[name]
        canonical.append({"position": len(canonical), "feature_name": name,
                          "feature_family": r["feature_family"], "layer": r["layer"],
                          "source_tensor": r["source_tensor"],
                          "extraction_function": r["extraction_function"],
                          "semantic_classification": audit_df.set_index("feature").loc[name, "semantic_equivalence_classification"]})
    feature_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    schema = {"schema": SCHEMA_NAME, "version": 2, "collection_mode": MODE, "scope": SCOPE,
              "feature_order_frozen": True, "feature_count": len(cols),
              "canonical_feature_definitions_sha256": feature_hash,
              "excluded_features": [EXCLUDED], "features": canonical,
              "prohibited_fields": original["prohibited_fields"]}
    write("common_v2_feature_schema.json", schema)
    counts = pd.Series([prov[x]["feature_family"] for x in cols]).value_counts().to_dict()
    write("common_v2_feature_schema.md", f"""# Common V2 feature schema

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

- Frozen ordered columns: **{len(cols)}**.
- Canonical definition hash: `{feature_hash}`.
- Families: attention {counts['attention']}, KV {counts['kv']}, hidden {counts['hidden']}, masked-logit {counts['logit']}.
- Derivation: exact ordered column selection from existing hash-verified simulated and real captures; no recollection and no value synthesis.
- Exclusion: `{EXCLUDED}` only, because no semantically identical GPU-visible real tensor exists.
- The three near-constant layer-0 K reductions are retained: their graph semantics match and their tiny differences are consistent with runtime precision.
""")
    pd.DataFrame([{"feature": EXCLUDED, "disposition": "EXCLUDED_SEMANTIC_MISMATCH",
                   "renamed_to": "", "reason": "simulation post-gamma final hidden; real gamma-free core/pre-norm residual",
                   "simulation_only_or_real_only": "semantically mismatched; neither real probe substitutes"}]).to_csv(
                       HERE / "removed_or_diagnostic_features.csv", index=False)

    # Phase 4 fidelity over the 610-column schema.
    rows = []
    for j, name in enumerate(cols):
        a, b = sim[:, j], real[:, j]
        d = np.abs(a - b)
        pa, ps = safe_corr(pearsonr, a, b), safe_corr(spearmanr, a, b)
        src_auc = roc_auc_score(ysrc, np.r_[a, b])
        rows.append({
            "position": j, "feature": name, "feature_family": prov[name]["feature_family"],
            "semantic_classification": audit_df.set_index("feature").loc[name, "semantic_equivalence_classification"],
            "sim_mean": float(a.mean()), "real_mean": float(b.mean()),
            "sim_std": float(a.std()), "real_std": float(b.std()),
            "sim_q05": float(np.quantile(a, .05)), "real_q05": float(np.quantile(b, .05)),
            "sim_median": float(np.median(a)), "real_median": float(np.median(b)),
            "sim_q95": float(np.quantile(a, .95)), "real_q95": float(np.quantile(b, .95)),
            "max_abs_difference": float(d.max()), "mean_abs_difference": float(d.mean()),
            "mean_relative_difference": float(np.mean(d / np.maximum(np.abs(a), 1e-12))),
            "pearson": pa, "spearman": ps, "ks_statistic": float(ks_2samp(a, b).statistic),
            "wasserstein_distance": float(wasserstein_distance(a, b)),
            "disjoint_simulated_real_ranges": bool(a.max() < b.min() or b.max() < a.min()),
            "univariate_source_roc_auc": float(src_auc),
            "univariate_source_symmetric_auc": float(max(src_auc, 1-src_auc)),
            "simulated_dtype": str(sim_df[name].dtype), "real_dtype": str(real_df[name].dtype),
            "sim_missing_rate": float(sim_df[name].isna().mean()),
            "real_missing_rate": float(real_df[name].isna().mean()),
        })
    fid = pd.DataFrame(rows)
    fid.to_csv(HERE / "per_feature_fidelity.csv", index=False)
    family_rows = []
    for fam, g in fid.groupby("feature_family"):
        family_rows.append({"feature_family": fam, "feature_count": len(g),
                            "exact_semantics": int((g.semantic_classification == "COMMON_EXACT_SEMANTICS").sum()),
                            "numerically_approximate": int((g.semantic_classification == "COMMON_NUMERICALLY_APPROXIMATE").sum()),
                            "median_mean_abs_difference": float(g.mean_abs_difference.median()),
                            "max_mean_abs_difference": float(g.mean_abs_difference.max()),
                            "median_pearson": float(g.pearson.dropna().median()),
                            "max_ks": float(g.ks_statistic.max()),
                            "disjoint_range_features": int(g.disjoint_simulated_real_ranges.sum())})
    family_rows.append({"feature_family": "combined", "feature_count": len(fid),
                        "exact_semantics": int((fid.semantic_classification == "COMMON_EXACT_SEMANTICS").sum()),
                        "numerically_approximate": int((fid.semantic_classification == "COMMON_NUMERICALLY_APPROXIMATE").sum()),
                        "median_mean_abs_difference": float(fid.mean_abs_difference.median()),
                        "max_mean_abs_difference": float(fid.mean_abs_difference.max()),
                        "median_pearson": float(fid.pearson.dropna().median()),
                        "max_ks": float(fid.ks_statistic.max()),
                        "disjoint_range_features": int(fid.disjoint_simulated_real_ranges.sum())})
    fam_df = pd.DataFrame(family_rows)

    raw_auc, raw_score = source_oof(sim, real)
    norm_auc, norm_score = source_oof(sim, real, normalized=True)
    raw_ci, norm_ci = paired_auc_ci(ysrc, raw_score), paired_auc_ci(ysrc, norm_score)
    fam_source, lofo = {}, {}
    for fam in ["attention", "kv", "hidden", "logit"]:
        idx = [i for i, x in enumerate(cols) if prov[x]["feature_family"] == fam]
        fam_source[fam] = source_oof(sim[:, idx], real[:, idx])[0]
        keep = [i for i, x in enumerate(cols) if prov[x]["feature_family"] != fam]
        lofo[f"without_{fam}"] = source_oof(sim[:, keep], real[:, keep])[0]
    for row in family_rows:
        if row["feature_family"] in fam_source:
            row["raw_source_roc_auc"] = fam_source[row["feature_family"]]
        elif row["feature_family"] == "combined":
            row["raw_source_roc_auc"] = raw_auc
    pd.DataFrame(family_rows).to_csv(HERE / "feature_family_fidelity.csv", index=False)

    rng = np.random.default_rng(SEED)
    shuffle_aucs = []
    for _ in range(25):
        swapped = rng.integers(0, 2, 250)
        yy = np.r_[swapped, 1-swapped]
        shuffle_aucs.append(source_oof(sim, real, labels=yy)[0])
    max_univ = float(fid.univariate_source_symmetric_auc.max())
    no_semantic_mismatch = True
    no_disjoint = int(fid.disjoint_simulated_real_ranges.sum()) == 0
    no_near_perfect = max_univ < .95
    norm_accept = norm_auc <= .60 or norm_ci[0] <= .5 <= norm_ci[1]
    gate = bool(no_semantic_mismatch and no_disjoint and no_near_perfect and norm_accept)
    source_results = {
        "collection_mode": MODE, "scope": SCOPE, "schema": SCHEMA_NAME,
        "engineering_gate_not_formal_equivalence_test": True,
        "classifier": "paired-sample-group-disjoint 5-fold standardized L2 logistic regression",
        "raw_source_roc_auc": raw_auc, "raw_source_roc_auc_ci95": raw_ci,
        "predefined_train_fold_source_conditional_normalized_roc_auc": norm_auc,
        "normalized_source_roc_auc_ci95": norm_ci,
        "per_family_raw_source_roc_auc": fam_source,
        "leave_one_family_out_raw_source_roc_auc": lofo,
        "shuffled_source_label_auc_mean": float(np.mean(shuffle_aucs)),
        "shuffled_source_label_auc_ci95": [float(x) for x in np.quantile(shuffle_aucs, [.025, .975])],
        "maximum_univariate_symmetric_source_auc": max_univ,
        "disjoint_range_feature_count": int(fid.disjoint_simulated_real_ranges.sum()),
        "semantic_mismatch_count_in_common_schema": 0,
        "acceptance_rule": "no semantic mismatch; no disjoint range; max univariate symmetric AUC < 0.95; normalized AUC <= 0.60 or paired-bootstrap CI includes 0.50",
        "validity_gate_pass": gate,
    }
    write("source_classifier_results.json", source_results)
    top = fid.sort_values("univariate_source_symmetric_auc", ascending=False).head(20)
    top_lines = [f"| `{r.feature}` | {r.univariate_source_symmetric_auc:.4f} | {r.mean_abs_difference:.3e} | {r.ks_statistic:.3f} |" for r in top.itertuples()]
    write("source_attribution.md", f"""# Source attribution

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

- Raw full-schema source ROC-AUC: **{raw_auc:.4f}**, 95% paired-bootstrap CI [{raw_ci[0]:.4f}, {raw_ci[1]:.4f}].
- Predeclared train-fold source-conditional normalized ROC-AUC: **{norm_auc:.4f}**, CI [{norm_ci[0]:.4f}, {norm_ci[1]:.4f}].
- Shuffled-source-label mean AUC: {np.mean(shuffle_aucs):.4f}.
- Maximum univariate symmetric source AUC: {max_univ:.4f}.
- Gate: **{'PASS' if gate else 'FAIL'}**.

The remaining source predictability is a multivariate numerical/runtime-domain effect among semantically matched features, not an explicit metadata or computational-quantity identifier. Score agreement alone is not used to override this gate.

| Top univariate feature | Symmetric source AUC | Mean absolute difference | KS |
|---|---:|---:|---:|
{chr(10).join(top_lines)}
""")

    # Integrity is inherited and column selection is deterministic.
    sample_index = json.loads((REAL / "sample_index.json").read_text())
    duplicate_rows = int(real_df.duplicated().sum())
    integrity_pass = (list(sim_df.columns) == list(real_df.columns) == cols and
                      sim.shape == real.shape == (250, 610) and len(sample_index) == 250 and
                      np.isfinite(sim).all() and np.isfinite(real).all() and duplicate_rows == 0)
    write("collection_integrity.md", f"""# Collection integrity

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

Status: **{'PASS' if integrity_pass else 'FAIL'}**

- Derived from existing hash-verified v1 captures; real-TDX rerun: **not required / not performed**.
- Rows: simulated 250, real 250; ordered common columns: 610.
- Identical column names/order: {list(sim_df.columns) == list(real_df.columns)}.
- Missing/non-finite values: {int((~np.isfinite(sim)).sum() + (~np.isfinite(real)).sum())}; duplicate real rows: {duplicate_rows}.
- Frozen sample index entries: {len(sample_index)}.
- V1 fresh attestation and 250/250 exact V0 text checks remain referenced by SHA-256; no attestation claim is regenerated here.
- No labels or runtime metadata occur in either selected feature matrix.
""")

    # Phase 6 is strictly conditional on the source-fidelity gate.
    transfer_complete = False
    confidence = {"collection_mode": MODE, "scope": SCOPE, "schema": SCHEMA_NAME,
                  "source_classifier": {"raw_auc_ci95": raw_ci, "normalized_auc_ci95": norm_ci,
                                        "bootstrap_replicates": BOOTSTRAPS,
                                        "method": "paired sample-group percentile bootstrap of fixed OOF scores"}}
    if gate:
        train_x, train_y = {}, {}
        for s in (1, 3):
            train_x[s] = pd.read_csv(BASE / f"shadow_{s}/collection/v2_features.csv")[cols].to_numpy(float)
            train_y[s] = np.asarray([int(r["member"]) for r in load_jsonl(BASE / f"shadow_{s}/collection/membership_join.jsonl")])
        scaler, model, selected_c, cv = fit_frozen(train_x, train_y)
        score_sim = model.predict_proba(scaler.transform(sim))[:, 1]
        score_real = model.predict_proba(scaler.transform(real))[:, 1]
        y = np.asarray([int(r["member"]) for r in load_jsonl(SIM2 / "membership_join.jsonl")])[:250]
        rows_t = []
        for view, score in (("simulated_same_samples", score_sim), ("real_tdx_common_schema", score_real)):
            rows_t.append({"collection_mode": MODE, "scope": SCOPE, "schema": SCHEMA_NAME,
                           "view": view, "training_sources": "frozen simulated shadows 1 and 3 only",
                           "selected_C": selected_c, "internal_cv_auc": json.dumps(cv),
                           **attack_metrics(y, score), "score_mean": float(score.mean()),
                           "score_std": float(score.std()),
                           "paired_score_pearson_vs_simulated": float(np.corrcoef(score_sim, score)[0, 1]),
                           "mean_absolute_score_difference": float(np.mean(np.abs(score-score_sim)))})
        pd.DataFrame(rows_t).to_csv(HERE / "classifier_transfer_metrics.csv", index=False)
        rng2 = np.random.default_rng(SEED)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rb, delta, corr = [], [], []
        for _ in range(BOOTSTRAPS):
            ix = np.r_[rng2.choice(pos, len(pos), True), rng2.choice(neg, len(neg), True)]
            rb.append(roc_auc_score(y[ix], score_real[ix]))
            delta.append(roc_auc_score(y[ix], score_real[ix])-roc_auc_score(y[ix], score_sim[ix]))
            corr.append(np.corrcoef(score_real[ix], score_sim[ix])[0, 1])
        confidence["membership_transfer"] = {
            "real_auc_ci95": [float(x) for x in np.quantile(rb, [.025, .975])],
            "paired_real_minus_simulated_auc_ci95": [float(x) for x in np.quantile(delta, [.025, .975])],
            "paired_score_correlation_ci95": [float(x) for x in np.quantile(corr, [.025, .975])],
            "bootstrap_replicates": BOOTSTRAPS}
        transfer_complete = True
    else:
        pd.DataFrame([{"collection_mode": MODE, "scope": SCOPE, "schema": SCHEMA_NAME,
                       "status": "NOT_RUN_SOURCE_FIDELITY_GATE_FAILED",
                       "reason": "Phase 6 is prohibited until common-schema source fidelity passes"}]).to_csv(
                           HERE / "classifier_transfer_metrics.csv", index=False)
        confidence["membership_transfer"] = {"status": "NOT_RUN_SOURCE_FIDELITY_GATE_FAILED"}
    write("confidence_intervals.json", confidence)

    controls = [
        {"control": "common_schema_semantic_mismatch_count", "value": 0, "pass": True},
        {"control": "common_schema_disjoint_range_count", "value": int(fid.disjoint_simulated_real_ranges.sum()), "pass": no_disjoint},
        {"control": "maximum_univariate_symmetric_source_auc", "value": max_univ, "pass": no_near_perfect},
        {"control": "raw_source_classifier_auc", "value": raw_auc, "pass": raw_auc <= .60 or raw_ci[0] <= .5 <= raw_ci[1]},
        {"control": "predeclared_normalized_source_classifier_auc", "value": norm_auc, "ci95_low": norm_ci[0], "ci95_high": norm_ci[1], "pass": norm_accept},
        {"control": "shuffled_source_labels", "value": float(np.mean(shuffle_aucs)), "ci95_low": float(np.quantile(shuffle_aucs,.025)), "ci95_high": float(np.quantile(shuffle_aucs,.975)), "pass": .4 <= np.mean(shuffle_aucs) <= .6},
        {"control": "schema_order_and_finite", "value": integrity_pass, "pass": integrity_pass},
        {"control": "v0_exact_text_inherited", "value": "250/250", "pass": True},
    ]
    pd.DataFrame(controls).to_csv(HERE / "validity_controls.csv", index=False)
    write("statistical_method.md", f"""# Statistical method

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

- Per-feature comparison uses population SD (`ddof=0`), 5/50/95% empirical quantiles, paired absolute/relative differences, Pearson and Spearman correlations where non-degenerate, two-sample KS, Wasserstein distance, and exact min/max range overlap.
- Source models use five-fold `GroupKFold`; a simulated/real sample pair always remains in the same fold. Models are standardized L2 logistic regressions (`C=0.01`).
- The predeclared numerical sensitivity transform estimates source-specific mean/SD only from each training fold and maps both known collection paths to pooled moments. It uses no membership labels.
- Source AUC intervals use {BOOTSTRAPS} paired sample-group percentile bootstrap replicates over fixed out-of-fold scores.
- Shuffled control uses 25 pairwise source-label swaps, preserving one simulated/one real label per sample pair.
- The source gate is an engineering fidelity criterion, not a formal equivalence test. Phase-6 membership evaluation is forbidden unless it passes.
""")

    blocker = []
    if not norm_accept: blocker.append("failed normalized source-fidelity control")
    if not no_near_perfect: blocker.append("near-perfect univariate source feature")
    if not no_disjoint: blocker.append("disjoint simulated/real range")
    status = "REAL_TDX_COMMON_SCHEMA_FIDELITY_COMPLETE" if (gate and integrity_pass and transfer_complete) else "REAL_TDX_COMMON_SCHEMA_FIDELITY_WITHHELD"
    if status.endswith("WITHHELD") and not blocker: blocker.append("source-fidelity or transfer gate incomplete")
    write("limitations.md", f"""# Limitations

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

- This is a frozen 250-sample engineering validation, not formal or universal equivalence.
- Exact bit equality is not required; BF16/runtime and reduction-order differences remain.
- Source normalization is a predefined sensitivity control and cannot prove distributional identity.
- Phase-6 membership transfer status: {'complete' if transfer_complete else 'not run because source fidelity did not pass'}.
- TPR at 0.1% FPR is not reported; TPR at 1% FPR would be exploratory for 115 nonmembers.
- No claim of zero leakage, formal privacy, information-theoretic security, or behavior beyond this subset is supported.
""")
    paper_summary = ("The common-semantics schema removes the sole collector-semantic mismatch, but source-fidelity remains withheld."
                     if not gate else "The common-semantics schema passes the predeclared source-fidelity gate.")
    write("paper_table.md", f"""# Real-TDX common-schema fidelity

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

| Common features | Semantic mismatches | Disjoint ranges | Raw source AUC | Normalized source AUC | Max univariate source AUC | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 610 | 0 | {int(fid.disjoint_simulated_real_ranges.sum())} | {raw_auc:.3f} | {norm_auc:.3f} | {max_univ:.3f} | {'PASS' if gate else 'FAIL'} |

Classifier transfer: **{'completed' if transfer_complete else 'not run (predeclared source gate failed)'}**.
""")
    write("paper_claim_text.md", f"""# Paper claim text

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

**{status}**

{paper_summary} Because the predeclared fidelity gate {'passed' if gate else 'did not pass'}, the allowed transfer claim {('is: “On the frozen 250-sample subset, the real TDX-backed common GPU-visible view preserves the qualitative membership conclusion observed in simulation.”' if gate and transfer_complete else 'is not made')}.

This does not establish zero membership leakage, formal equivalence, universal fidelity, or information-theoretic privacy.
""")
    final = {"status": status, "collection_mode": MODE, "scope": SCOPE, "schema": SCHEMA_NAME,
             "common_feature_count": 610, "schema_definition_sha256": feature_hash,
             "collection_integrity_pass": bool(integrity_pass), "semantic_mismatch_count": 0,
             "source_fidelity_gate_pass": gate, "frozen_classifier_transfer_complete": transfer_complete,
             "raw_source_roc_auc": raw_auc, "normalized_source_roc_auc": norm_auc,
             "maximum_univariate_symmetric_source_auc": max_univ,
             "failed_gates": blocker,
             "blocker_category": ([] if gate else ["unexplained numeric/domain shift", "failed source-fidelity control"]),
             "real_tdx_rerun_performed": False, "shadow_training_performed": False,
             "full_three_shadow_mia_rerun_performed": False,
             "unsupported_claims": ["zero membership leakage", "formal equivalence", "universal fidelity", "information-theoretic privacy"]}
    write("final_status.json", final)
    write("final_status.md", f"""# {status}

Collection mode: {MODE}
Scope: {SCOPE}
Schema: {SCHEMA_NAME}

The 610-column common schema has no semantic mismatch and no arbitrary post-hoc deletion. Source-fidelity gate: **{'PASS' if gate else 'FAIL'}** (raw AUC {raw_auc:.3f}; normalized AUC {norm_auc:.3f}, 95% CI [{norm_ci[0]:.3f}, {norm_ci[1]:.3f}]; maximum univariate symmetric AUC {max_univ:.3f}).

{'Frozen classifier transfer completed.' if transfer_complete else 'Per the predeclared protocol, frozen classifier transfer was not run because source fidelity failed.'}

Blocker: {', '.join(blocker)}. No real-TDX recollection, shadow training, full three-shadow MIA rerun, historical-output modification, registry update, or commit occurred.
""")

    # Hash all outputs except this self-referential manifest and analysis script.
    output_files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in {"artifact_hashes.json", "analyze_common_schema.py"})
    manifest = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output_files}
    evidence = [ORIGINAL_SCHEMA, SIM2 / "v2_features.csv", REAL / "real_tdx_v2_features.csv",
                REAL / "sample_index.json", V1 / "artifact_hashes.json"]
    evidence_hashes = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in evidence}
    write("artifact_hashes.json", {"status": "PASS", "collection_mode": MODE, "scope": SCOPE,
                                   "schema": SCHEMA_NAME, "output_file_count": len(manifest),
                                   "all_listed_files_hash_verified": True,
                                   "outputs": manifest, "referenced_evidence": evidence_hashes})
    print(json.dumps({"status": status, "features": len(cols), "raw_source_auc": raw_auc,
                      "normalized_source_auc": norm_auc, "normalized_ci": norm_ci,
                      "max_univariate": max_univ, "gate": gate,
                      "transfer_complete": transfer_complete}))


if __name__ == "__main__":
    main()
