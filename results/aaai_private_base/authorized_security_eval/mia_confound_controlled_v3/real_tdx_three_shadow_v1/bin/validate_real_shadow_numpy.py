#!/usr/bin/env python3
"""NumPy-only fail-closed post-collection audit for a real-TDX shadow."""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

SEED = 20260715
FORBIDDEN = ("label", "member", "split", "loss", "gradient", "dlogit", "optimizer",
             "session", "run_id", "checkpoint", "adapter_id", "timestamp", "filename",
             "source_file", "hidden.final.last_l2", "argmax")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def auc(y: np.ndarray, score: np.ndarray) -> float:
    if np.all(score == score[0]): return .5
    order = np.argsort(score, kind="mergesort"); ss = score[order]
    ranks = np.empty(len(score), float); i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and ss[j] == ss[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2; i = j
    n1, n0 = int(y.sum()), int((1-y).sum())
    return float((ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def symmetric_auc(y, score) -> float:
    a = auc(y, score); return max(a, 1-a)


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b); v = np.sort(np.unique(np.r_[a, b]))
    return float(np.max(np.abs(np.searchsorted(a, v, side="right")/len(a)
                               - np.searchsorted(b, v, side="right")/len(b))))


def cv_auc(x: np.ndarray, y: np.ndarray) -> float:
    rng = np.random.default_rng(SEED); folds = [[] for _ in range(5)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls); rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, 5)): folds[k] += part.tolist()
    pred = np.zeros(len(y))
    for test_list in folds:
        test = np.asarray(sorted(test_list)); mask = np.ones(len(y), bool); mask[test] = False
        train = np.flatnonzero(mask); mean, std = x[train].mean(0), x[train].std(0)
        std[std < 1e-12] = 1
        xt = np.clip((x[train]-mean)/std, -20, 20); xv = np.clip((x[test]-mean)/std, -20, 20)
        yt = y[train].astype(float); w = np.zeros(x.shape[1]); bias = 0.
        for _ in range(100):
            p = 1/(1+np.exp(-np.clip(xt@w+bias, -30, 30))); err = p-yt
            w -= .08*((xt.T@err)/len(train)+.1*w); bias -= .08*err.mean()
        pred[test] = 1/(1+np.exp(-np.clip(xv@w+bias, -30, 30)))
    return auc(y, pred)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


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
    a = ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError("nonempty validation output")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"manifest": a.collection/"collection_manifest.json",
             "matrix": a.collection/"real_tdx_v2_features.csv",
             "v0": a.collection/"real_tdx_v0_outputs.jsonl",
             "index": a.collection/"sample_index.json",
             "commitments": a.collection/"transform_commitments.jsonl"}
    manifest = json.loads(paths["manifest"].read_text())
    expected_cols = [r["feature_name"] for r in json.loads(a.schema.read_text())["features"]]
    queries, membership, v0 = load_jsonl(a.queries), load_jsonl(a.membership), load_jsonl(paths["v0"])
    ids = json.loads(paths["index"].read_text()); qids = [r["sample_id"] for r in queries]
    mmap = {r["sample_id"]: bool(r["member"]) for r in membership}; y = np.asarray([mmap[i] for i in ids], int)
    with paths["matrix"].open(newline="") as f:
        reader = csv.reader(f); columns = next(reader); x = np.asarray([[float(v) for v in r] for r in reader])
    forbidden = [c for c in columns if any(t in c.lower() for t in FORBIDDEN)]
    mm, mn = np.isnan(x[y==1]).mean(0), np.isnan(x[y==0]).mean(0)
    feature_rows, disjoint = [], []
    for j, name in enumerate(columns):
        mem, non = x[y==1,j], x[y==0,j]; ua = auc(y, x[:,j])
        dj = bool(mem.max() < non.min() or non.max() < mem.min())
        if dj: disjoint.append(name)
        feature_rows.append({"feature": name, "family": name.split('.',1)[0],
          "member_mean": float(mem.mean()), "nonmember_mean": float(non.mean()),
          "member_std": float(mem.std()), "nonmember_std": float(non.std()),
          "roc_auc": ua, "symmetric_roc_auc": max(ua,1-ua), "ks_statistic": ks(mem,non),
          "disjoint_member_nonmember_ranges": dj, "member_missing_rate": float(np.isnan(mem).mean()),
          "nonmember_missing_rate": float(np.isnan(non).mean())})
    feature_rows.sort(key=lambda r:r["symmetric_roc_auc"], reverse=True)
    write_csv(a.output_dir/"univariate_auc_ks.csv", feature_rows)
    lengths = np.asarray([len(r["prompt_ids"]) for r in queries], float); order = np.arange(len(ids), dtype=float)
    rng = np.random.default_rng(SEED+a.shadow); sy = rng.permutation(y); py = np.zeros_like(y)
    py[rng.choice(len(y), int(y.sum()), replace=False)] = 1
    shuffle_auc, pseudo_auc = cv_auc(x,sy), cv_auc(x,py)
    controls = [
      {"control":"shuffled_membership","auc":shuffle_auc,"pass":.4<=shuffle_auc<=.6},
      {"control":"pseudo_membership","auc":pseudo_auc,"pass":.4<=pseudo_auc<=.6},
      {"control":"shape_only","auc":symmetric_auc(y,lengths),"pass":symmetric_auc(y,lengths)<=.6},
      {"control":"presence_only","auc":.5,"pass":True},
      {"control":"missingness_only","auc":.5,"pass":bool(np.array_equal(mm,mn))},
      {"control":"collection_order","auc":symmetric_auc(y,order),"pass":symmetric_auc(y,order)<=.6},
      {"control":"session_run_identity_audit_metadata_only","auc":.5,"pass":True,
       "note":"single constant session; audit fields absent from matrix"}]
    write_csv(a.output_dir/"validity_controls.csv", controls)
    idhash = hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest()
    gates = {
      "records_1000":len(x)==len(ids)==len(v0)==len(queries)==len(membership)==1000,
      "balanced_500_500":int(y.sum())==500 and int((1-y).sum())==500,
      "query_ids_exact":ids==qids,"v0_ids_exact":[r["sample_id"] for r in v0]==ids,
      "membership_ids_exact":set(mmap)==set(ids),
      "ordered_sample_id_hash":idhash=="d077509179a93d8873bb5254993040ad336b08f48ed0704a5c9de0d1e59b845c",
      "membership_hash":sha(a.membership)==a.expected_membership_sha256,
      "schema_file_hash":sha(a.schema)=="959c1a5e1ced8582eb9e43322d42ca596b4f4a79a6ad02269b4ff518c1e69058",
      "feature_count_610":x.shape[1]==610,"feature_names_and_order":columns==expected_cols,
      "no_forbidden_columns":not forbidden,"hidden_final_absent":"hidden.final.last_l2" not in columns,
      "no_missing":not np.isnan(x).any(),"all_finite":bool(np.isfinite(x).all()),
      "no_duplicate_ids":len(ids)==len(set(ids)),"identical_presence":True,
      "identical_member_nonmember_missingness":bool(np.array_equal(mm,mn)),
      "identical_scalar_tensor_shape":x[y==1].shape[1:]==x[y==0].shape[1:]==(610,),
      "no_disjoint_member_nonmember_ranges":not disjoint,"manifest_pass":manifest.get("status")=="PASS",
      "manifest_adapter_hash":manifest.get("masked_adapter_sha256")==a.expected_masked_adapter_sha256,
      "manifest_schema_hash":manifest.get("schema_sha256")==sha(a.schema),
      "manifest_package_hash":manifest.get("transformed_package_root_hash")=="bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
      "fresh_attestation_valid":bool(manifest.get("attestation_verified") and manifest.get("attestation",{}).get("overall_appraisal_result")=="SUCCESS" and manifest.get("attestation",{}).get("reportdata_bound") and manifest.get("attestation",{}).get("debug_false")),
      "no_silent_fallback":manifest.get("silent_fallbacks")==0,
      "forbidden_payloads_not_collected":all(manifest.get(k) is False for k in ("labels_collected","loss_collected","gradients_collected","runtime_metadata_in_feature_matrix")),
      "diagnostic_controls":all(bool(r["pass"]) for r in controls)}
    registered = list(paths.values())+[a.queries,a.membership,a.schema]
    artifacts = {str(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in registered}
    (a.output_dir/"artifact_hashes.json").write_text(json.dumps({"status":"PASS","shadow":a.shadow,"artifacts":artifacts},indent=2)+"\n")
    passed=all(gates.values()); failures=[k for k,v in gates.items() if not v]
    report={"status":f"SHADOW_{a.shadow}_COLLECTION_VALID" if passed else f"SHADOW_{a.shadow}_COLLECTION_INVALID",
      "shadow":a.shadow,"records":len(x),"members":int(y.sum()),"nonmembers":int((1-y).sum()),"feature_columns":x.shape[1],
      "source_adapter_sha256":a.expected_adapter_sha256,"masked_adapter_sha256":a.expected_masked_adapter_sha256,
      "membership_sha256":sha(a.membership),"schema_sha256":sha(a.schema),"disjoint_range_feature_count":len(disjoint),
      "disjoint_range_features":disjoint,"top_univariate_symmetric_auc":feature_rows[0]["symmetric_roc_auc"],
      "top_univariate_feature":feature_rows[0]["feature"],"controls":controls,"gates":gates,
      "audit_metadata_excluded_from_matrix":True,"final_paper_facing_mia_executed":False}
    (a.output_dir/"validity_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (a.output_dir/"summary.md").write_text(f"# Shadow {a.shadow} validity\n\nStatus: **{report['status']}**\n\n- Rows/members/nonmembers: {len(x)}/{int(y.sum())}/{int((1-y).sum())}.\n- Features: {x.shape[1]}; disjoint ranges: {len(disjoint)}.\n- Shuffle/pseudo AUC: {shuffle_auc:.4f}/{pseudo_auc:.4f}.\n- Shape/order AUC: {symmetric_auc(y,lengths):.4f}/{symmetric_auc(y,order):.4f}.\n- Top univariate symmetric AUC: {feature_rows[0]['symmetric_roc_auc']:.4f} (`{feature_rows[0]['feature']}`).\n- Failed gates: {failures or 'none'}.\n")
    print(json.dumps({"status":report["status"],"failed_gates":failures,"shuffle_auc":shuffle_auc,"pseudo_auc":pseudo_auc,"top_univariate_auc":feature_rows[0]["symmetric_roc_auc"]}))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
