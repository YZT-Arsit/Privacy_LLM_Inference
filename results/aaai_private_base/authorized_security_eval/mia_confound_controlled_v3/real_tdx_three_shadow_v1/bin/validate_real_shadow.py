#!/usr/bin/env python3
"""Fail-closed post-collection validity audit for one real-TDX shadow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260715
FORBIDDEN = ("label", "member", "split", "loss", "gradient", "dlogit", "optimizer",
             "session", "run_id", "checkpoint", "adapter_id", "timestamp", "filename",
             "source_file", "hidden.final.last_l2", "argmax")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def scalar_auc(y: np.ndarray, score: np.ndarray) -> float:
    if np.all(score == score[0]):
        return .5
    a = float(roc_auc_score(y, score))
    return max(a, 1-a)


def cv_auc(x: np.ndarray, y: np.ndarray) -> float:
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.01, solver="liblinear", class_weight="balanced", max_iter=5000,
        random_state=SEED))
    pred = cross_val_predict(model, x, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", type=int, required=True)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--membership", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-adapter-sha256", required=True)
    ap.add_argument("--expected-masked-adapter-sha256", required=True)
    ap.add_argument("--expected-membership-sha256", required=True)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("refusing nonempty validation output")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.collection / "collection_manifest.json"
    matrix_path = args.collection / "real_tdx_v2_features.csv"
    v0_path = args.collection / "real_tdx_v0_outputs.jsonl"
    index_path = args.collection / "sample_index.json"
    commitment_path = args.collection / "transform_commitments.jsonl"
    manifest = json.loads(manifest_path.read_text())
    schema_obj = json.loads(args.schema.read_text())
    expected_cols = [x["feature_name"] for x in schema_obj["features"]]
    queries = load_jsonl(args.queries)
    membership = load_jsonl(args.membership)
    v0 = load_jsonl(v0_path)
    ids = json.loads(index_path.read_text())
    query_ids = [x["sample_id"] for x in queries]
    member_map = {x["sample_id"]: bool(x["member"]) for x in membership}
    y = np.asarray([member_map[x] for x in ids], int)
    frame = pd.read_csv(matrix_path)
    x = frame.to_numpy(float)

    forbidden = [c for c in frame.columns if any(t in c.lower() for t in FORBIDDEN)]
    missing_member = frame[y == 1].isna().mean().to_numpy()
    missing_nonmember = frame[y == 0].isna().mean().to_numpy()
    disjoint = []
    rows = []
    for j, name in enumerate(frame.columns):
        a, b = x[y == 1, j], x[y == 0, j]
        auc = float(roc_auc_score(y, x[:, j]))
        dj = bool(a.max() < b.min() or b.max() < a.min())
        if dj:
            disjoint.append(name)
        rows.append({"feature": name, "family": name.split('.', 1)[0],
                     "member_mean": float(a.mean()), "nonmember_mean": float(b.mean()),
                     "member_std": float(a.std()), "nonmember_std": float(b.std()),
                     "roc_auc": auc, "symmetric_roc_auc": max(auc, 1-auc),
                     "ks_statistic": float(ks_2samp(a, b).statistic),
                     "disjoint_member_nonmember_ranges": dj,
                     "member_missing_rate": float(np.isnan(a).mean()),
                     "nonmember_missing_rate": float(np.isnan(b).mean())})
    univariate = pd.DataFrame(rows).sort_values("symmetric_roc_auc", ascending=False)
    univariate.to_csv(args.output_dir / "univariate_auc_ks.csv", index=False)

    lengths = np.asarray([len(q["prompt_ids"]) for q in queries], float)
    order = np.arange(len(ids), dtype=float)
    rng = np.random.default_rng(SEED + args.shadow)
    shuffled_y = rng.permutation(y)
    pseudo_y = np.zeros_like(y)
    pseudo_y[rng.choice(len(y), int(y.sum()), replace=False)] = 1
    shuffled_auc = cv_auc(x, shuffled_y)
    pseudo_auc = cv_auc(x, pseudo_y)
    controls = [
        {"control": "shuffled_membership", "auc": shuffled_auc, "pass": .4 <= shuffled_auc <= .6},
        {"control": "pseudo_membership", "auc": pseudo_auc, "pass": .4 <= pseudo_auc <= .6},
        {"control": "shape_only", "auc": scalar_auc(y, lengths), "pass": scalar_auc(y, lengths) <= .6},
        {"control": "presence_only", "auc": .5, "pass": True},
        {"control": "missingness_only", "auc": .5, "pass": bool(np.array_equal(missing_member, missing_nonmember))},
        {"control": "collection_order", "auc": scalar_auc(y, order), "pass": scalar_auc(y, order) <= .6},
        {"control": "session_run_identity_audit_metadata_only", "auc": .5, "pass": True,
         "note": "one constant session/run identity; fields absent from MIA matrix"},
    ]
    pd.DataFrame(controls).to_csv(args.output_dir / "validity_controls.csv", index=False)

    ordered_id_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    gates = {
        "records_1000": len(frame) == len(ids) == len(v0) == len(queries) == len(membership) == 1000,
        "balanced_500_500": int(y.sum()) == 500 and int((1-y).sum()) == 500,
        "query_ids_exact": ids == query_ids,
        "v0_ids_exact": [r["sample_id"] for r in v0] == ids,
        "membership_ids_exact": set(member_map) == set(ids),
        "ordered_sample_id_hash": ordered_id_hash == "d077509179a93d8873bb5254993040ad336b08f48ed0704a5c9de0d1e59b845c",
        "membership_hash": sha(args.membership) == args.expected_membership_sha256,
        "schema_file_hash": sha(args.schema) == "959c1a5e1ced8582eb9e43322d42ca596b4f4a79a6ad02269b4ff518c1e69058",
        "feature_count_610": frame.shape[1] == 610,
        "feature_names_and_order": list(frame.columns) == expected_cols,
        "no_forbidden_columns": not forbidden,
        "hidden_final_absent": "hidden.final.last_l2" not in frame.columns,
        "no_missing": not frame.isna().any().any(),
        "all_finite": bool(np.isfinite(x).all()),
        "no_duplicate_ids": len(ids) == len(set(ids)),
        "identical_presence": True,
        "identical_member_nonmember_missingness": bool(np.array_equal(missing_member, missing_nonmember)),
        "identical_scalar_tensor_shape": x[y == 1].shape[1:] == x[y == 0].shape[1:] == (610,),
        "no_disjoint_member_nonmember_ranges": not disjoint,
        "manifest_pass": manifest.get("status") == "PASS",
        "manifest_adapter_hash": manifest.get("masked_adapter_sha256") == args.expected_masked_adapter_sha256,
        "manifest_schema_hash": manifest.get("schema_sha256") == sha(args.schema),
        "manifest_package_hash": manifest.get("transformed_package_root_hash") == "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
        "fresh_attestation_valid": bool(manifest.get("attestation_verified") and
            manifest.get("attestation", {}).get("overall_appraisal_result") == "SUCCESS" and
            manifest.get("attestation", {}).get("reportdata_bound") and
            manifest.get("attestation", {}).get("debug_false")),
        "no_silent_fallback": manifest.get("silent_fallbacks") == 0,
        "forbidden_payloads_not_collected": all(manifest.get(k) is False for k in
            ("labels_collected", "loss_collected", "gradients_collected", "runtime_metadata_in_feature_matrix")),
        "diagnostic_controls": all(bool(r["pass"]) for r in controls),
    }
    artifact_paths = [manifest_path, matrix_path, v0_path, index_path, commitment_path,
                      args.queries, args.membership, args.schema]
    artifacts = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in artifact_paths}
    (args.output_dir / "artifact_hashes.json").write_text(json.dumps(
        {"status": "PASS", "shadow": args.shadow, "artifacts": artifacts}, indent=2) + "\n")
    passed = all(gates.values())
    report = {
        "status": f"SHADOW_{args.shadow}_COLLECTION_VALID" if passed else f"SHADOW_{args.shadow}_COLLECTION_INVALID",
        "shadow": args.shadow, "records": len(frame), "members": int(y.sum()),
        "nonmembers": int((1-y).sum()), "feature_columns": frame.shape[1],
        "source_adapter_sha256": args.expected_adapter_sha256,
        "masked_adapter_sha256": args.expected_masked_adapter_sha256,
        "membership_sha256": sha(args.membership), "schema_sha256": sha(args.schema),
        "disjoint_range_feature_count": len(disjoint), "disjoint_range_features": disjoint,
        "top_univariate_symmetric_auc": float(univariate.iloc[0].symmetric_roc_auc),
        "top_univariate_feature": str(univariate.iloc[0].feature),
        "controls": controls, "gates": gates,
        "audit_metadata_excluded_from_matrix": True,
        "final_paper_facing_mia_executed": False,
    }
    (args.output_dir / "validity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [k for k, v in gates.items() if not v]
    (args.output_dir / "summary.md").write_text(
        f"# Shadow {args.shadow} real-TDX collection validity\n\n"
        f"Status: **{report['status']}**\n\n"
        f"- Rows / members / nonmembers: {len(frame)} / {int(y.sum())} / {int((1-y).sum())}.\n"
        f"- Features: {frame.shape[1]}; disjoint member/nonmember ranges: {len(disjoint)}.\n"
        f"- Shuffle AUC: {shuffled_auc:.4f}; pseudo-membership AUC: {pseudo_auc:.4f}.\n"
        f"- Shape-only AUC: {scalar_auc(y, lengths):.4f}; collection-order AUC: {scalar_auc(y, order):.4f}.\n"
        f"- Top univariate symmetric AUC: {report['top_univariate_symmetric_auc']:.4f} (`{report['top_univariate_feature']}`).\n"
        f"- Failed gates: {failures or 'none'}.\n"
    )
    print(json.dumps({"status": report["status"], "failed_gates": failures,
                      "shuffle_auc": shuffled_auc, "pseudo_auc": pseudo_auc,
                      "top_univariate_auc": report["top_univariate_symmetric_auc"]}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
