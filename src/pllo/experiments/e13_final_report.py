"""E13: final paper-facing evaluation consolidation (pure parsing).

Consolidates every stage -- E1/E2 correctness, E3 remote scaling, E4 setup cost,
E5 no-LoRA comparison, E6-E8 private LoRA, E9 public-task utility, E10 LoRA
utility, E12 latency baselines, the security negative tests, the deployment-truth
inference, and the paper-claim validation -- into ten tables plus an explicit
limitations section. Honest by construction: every consumed report keeps its own
``dry_run`` / ``paper_ready`` label, deployment truth is re-inferred, and claims
are taken only from the claim validator (which refuses overclaims).

stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pllo.experiments.claim_validator import build_claim_report
from pllo.experiments.deployment_truth import deployment_truth_report

__all__ = ["build_e13_report", "render_e13_md", "LIMITATIONS"]

# Fixed limitations the paper MUST state (verbatim intent from the spec).
LIMITATIONS = [
    "HTTP/SSH tunnel transport is a research prototype, not production transport.",
    "The TDX-attested no-LoRA deployment has been validated; the LoRA attested "
    "real run must be separately validated after server restart.",
    "E7 is a minimal private LoRA update prototype, not full GPU-offloaded "
    "private LoRA training.",
    "Public benchmark subsets are used for cost-controlled utility validation; "
    "full benchmark scaling is future work / optional.",
    "The boundary embedding artifact contains trusted secrets and must remain "
    "inside the TDX guest.",
    "The folded package currently stores F32 folded operators for numerical "
    "fidelity.",
]


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _label(rep, fallback):
    return _g(rep, "stage") or fallback


def _correctness_rows(correctness, e5):
    rows = []
    for rep in list(correctness or []) + ([e5] if e5 else []):
        if not isinstance(rep, dict):
            continue
        rows.append({
            "source": _label(rep, "decode"),
            "dry_run": _g(rep, "dry_run"),
            "tokens_exact_match": _g(rep, "tokens_exact_match"),
            "token_match_rate": _g(rep, "token_match_rate"),
            "allclose": _g(rep, "allclose"),
            "audit_passed": _g(rep, "audit_passed"),
        })
    return rows


def _public_utility_rows(e9_list):
    rows = []
    for rep in e9_list or []:
        if not isinstance(rep, dict):
            continue
        rows.append({
            "dataset": _g(rep, "dataset"), "backend": _g(rep, "backend"),
            "task_type": _g(rep, "task_type"),
            "metric_name": _g(rep, "metric_name"),
            "metric_value": _g(rep, "metric_value"),
            "num_examples": _g(rep, "num_examples"),
            "dry_run": _g(rep, "dry_run"), "paper_ready": _g(rep, "paper_ready"),
        })
    return rows


def _lora_utility_row(e10):
    if not isinstance(e10, dict):
        return None
    return {k: e10.get(k) for k in (
        "dataset_name", "task_type", "metric_name", "base_metric",
        "plaintext_lora_metric", "folded_lora_metric",
        "tdx_attested_folded_lora_metric", "lora_gain_plaintext",
        "lora_gain_folded", "lora_gain_preserved_ratio",
        "folded_lora_preserves_gain", "utility_preserved", "security_ok",
        "dry_run", "paper_ready")}


def _security_matrix(sources):
    rows = []
    for rep in sources:
        if not isinstance(rep, dict):
            continue
        # only include reports that carry security posture fields
        if all(rep.get(k) is None for k in (
                "worker_has_mask_secrets", "gpu_visible_plaintext_fields",
                "audit_passed", "worker_has_raw_lora")):
            continue
        rows.append({
            "source": _label(rep, "report"),
            "worker_has_mask_secrets": rep.get("worker_has_mask_secrets"),
            "worker_has_raw_lora": rep.get("worker_has_raw_lora"),
            "tee_used_on_gpu": rep.get("tee_used_on_gpu"),
            "gpu_visible_plaintext_fields": rep.get(
                "gpu_visible_plaintext_fields"),
            "leaked_secret_fields": rep.get("leaked_secret_fields"),
            "audit_passed": rep.get("audit_passed"),
        })
    return rows


def _setup_cost_row(e4, e8, e10):
    row = {}
    if isinstance(e4, dict):
        for k in ("folded_package_size_gb", "setup_time_s", "build_time_s",
                  "embedding_artifact_size_gb", "load_time_s"):
            if e4.get(k) is not None:
                row[k] = e4.get(k)
    # folded-LoRA cost from E8 cost section if present
    cost = _g(e8, "cost") if isinstance(e8, dict) else None
    if isinstance(cost, dict):
        for k in ("folded_lora_package_size_gb", "folded_lora_setup_time_s",
                  "decode_latency_overhead_s", "memory_overhead_mb"):
            if cost.get(k) is not None:
                row[k] = cost.get(k)
    return row or None


def build_e13_report(inputs: dict) -> dict:
    # results feed deployment-truth + claim validation. Start from any explicit
    # results, then fold in every provided posture-bearing report so the truth /
    # claims tables reflect the evaluation even if --result-json was not passed.
    results = list(inputs.get("results") or [])
    _seen = {id(item.get("report")) for item in results
             if isinstance(item, dict)}

    def _fold(rep, label):
        if isinstance(rep, dict) and id(rep) not in _seen:
            _seen.add(id(rep))
            results.append({"file": _g(rep, "stage") or label, "report": rep})

    for rep in inputs.get("correctness") or []:
        _fold(rep, "correctness")
    for rep in inputs.get("e9") or []:
        _fold(rep, "e9")
    for key in ("e5", "e10", "security_negative"):
        _fold(inputs.get(key), key)

    truth_rows = []
    for item in results:
        rep = item.get("report")
        if isinstance(rep, dict):
            dt = deployment_truth_report(rep)
            truth_rows.append({
                "file": item.get("file"),
                "source_stage": dt.get("source_stage"),
                "gpu_real": _g(dt, "truth", "gpu_real"),
                "tee_real": _g(dt, "truth", "tee_real"),
                "tee_type": _g(dt, "truth", "tee_type"),
                "attestation_verified": _g(dt, "truth", "attestation_verified"),
                "runtime_hash_bound": _g(dt, "truth", "runtime_hash_bound"),
                "lora_enabled": _g(dt, "truth", "lora_enabled"),
                "boundary_mode": _g(dt, "truth", "boundary_mode"),
                "allowed_claims": dt.get("allowed_claims"),
                "forbidden_claims": dt.get("forbidden_claims"),
            })

    claims = inputs.get("claims")
    if claims is None:
        claims = build_claim_report(results,
                                    required_claims=inputs.get("required_claims"))

    sec_sources = (list(inputs.get("correctness") or [])
                   + list(inputs.get("e9") or [])
                   + ([inputs["e10"]] if inputs.get("e10") else [])
                   + ([inputs["e5"]] if inputs.get("e5") else []))

    return {
        "stage": "e13_final_evaluation_report",
        "correctness": _correctness_rows(inputs.get("correctness"),
                                         inputs.get("e5")),
        "public_task_utility": _public_utility_rows(inputs.get("e9")),
        "lora_utility": _lora_utility_row(inputs.get("e10")),
        "security_audit_matrix": _security_matrix(sec_sources),
        "security_negative_tests": inputs.get("security_negative"),
        "deployment_truth": truth_rows,
        "latency_overhead": _g(inputs.get("latency"), "rows")
        or (inputs.get("latency") if isinstance(inputs.get("latency"), list)
            else None),
        "setup_cost": _setup_cost_row(inputs.get("e4"), inputs.get("e8"),
                                      inputs.get("e10")),
        "paper_claims": {
            "supported_claims": claims.get("supported_claims"),
            "unsupported_claims": claims.get("unsupported_claims"),
            "overclaim_risks": claims.get("overclaim_risks"),
            "warnings": claims.get("warnings"),
        },
        "e3_remote_scaling": inputs.get("e3"),
        "limitations": LIMITATIONS,
        "note": "Consolidated, honestly-labeled evaluation. Each row keeps its "
                "source dry_run/paper_ready label; deployment truth is re-inferred "
                "and claims come only from the overclaim-refusing validator.",
    }


def _b(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(";", ":"))
    return str(v)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_b(c) for c in r) + " |")
    return out


def render_e13_md(rep: dict) -> str:
    L = ["# Final evaluation report (E13)", "", "_%s_" % rep["note"], ""]

    L += ["## 1. Correctness", ""]
    L += _table(["source", "dry_run", "tokens_exact_match", "token_match_rate",
                 "allclose", "audit_passed"],
                [[r["source"], r["dry_run"], r["tokens_exact_match"],
                  r["token_match_rate"], r["allclose"], r["audit_passed"]]
                 for r in rep["correctness"]] or [["(none)", "", "", "", "", ""]])

    L += ["", "## 2. Public task utility preservation", ""]
    L += _table(["dataset", "backend", "task_type", "metric", "value",
                 "n", "dry_run", "paper_ready"],
                [[r["dataset"], r["backend"], r["task_type"], r["metric_name"],
                  r["metric_value"], r["num_examples"], r["dry_run"],
                  r["paper_ready"]] for r in rep["public_task_utility"]]
                or [["(none)", "", "", "", "", "", "", ""]])

    L += ["", "## 3. LoRA utility preservation", ""]
    lu = rep["lora_utility"]
    if lu:
        L += _table(["dataset", "base", "plaintext_lora", "folded_lora",
                     "preserved_ratio", "preserves_gain", "utility_preserved",
                     "paper_ready"],
                    [[lu["dataset_name"], lu["base_metric"],
                      lu["plaintext_lora_metric"], lu["folded_lora_metric"],
                      lu["lora_gain_preserved_ratio"],
                      lu["folded_lora_preserves_gain"], lu["utility_preserved"],
                      lu["paper_ready"]]])
    else:
        L += ["_(not provided)_"]

    L += ["", "## 4. Security audit matrix", ""]
    L += _table(["source", "worker_has_mask_secrets", "worker_has_raw_lora",
                 "tee_used_on_gpu", "gpu_visible_plaintext", "leaked_secret",
                 "audit_passed"],
                [[r["source"], r["worker_has_mask_secrets"],
                  r["worker_has_raw_lora"], r["tee_used_on_gpu"],
                  r["gpu_visible_plaintext_fields"] or "[]",
                  r["leaked_secret_fields"] or "[]", r["audit_passed"]]
                 for r in rep["security_audit_matrix"]]
                or [["(none)", "", "", "", "", "", ""]])

    L += ["", "## 5. Security negative tests", ""]
    sn = rep["security_negative_tests"]
    if isinstance(sn, dict) and sn.get("cases"):
        L += ["_%d/%d caught, all_passed=%s_"
              % (sn.get("num_pass"), sn.get("num_cases"), sn.get("all_passed")),
              ""]
        L += _table(["negative_test_name", "expected_failure", "actually_failed",
                     "pass"],
                    [[c["negative_test_name"], c["expected_failure"],
                      c["actually_failed"], c["pass"]] for c in sn["cases"]])
    else:
        L += ["_(not provided)_"]

    L += ["", "## 6. Deployment truth", ""]
    L += _table(["file", "gpu_real", "tee_real", "tee_type",
                 "attestation_verified", "runtime_hash_bound", "lora_enabled",
                 "boundary_mode"],
                [[r["file"], r["gpu_real"], r["tee_real"], r["tee_type"],
                  r["attestation_verified"], r["runtime_hash_bound"],
                  r["lora_enabled"], r["boundary_mode"]]
                 for r in rep["deployment_truth"]]
                or [["(none)", "", "", "", "", "", "", ""]])

    L += ["", "## 7. Latency / overhead", ""]
    lat = rep["latency_overhead"]
    if isinstance(lat, list) and lat:
        L += _table(["backend", "total_latency_s", "latency_per_token_s",
                     "tokens_per_s", "overhead_vs_plaintext", "peak_gpu_mem_mb"],
                    [[r.get("backend"), r.get("total_latency_s"),
                      r.get("latency_per_token_s"), r.get("tokens_per_s"),
                      r.get("overhead_vs_plaintext_h800"),
                      r.get("peak_gpu_memory_mb")] for r in lat])
    else:
        L += ["_(not provided)_"]

    L += ["", "## 8. Setup / provisioning cost", ""]
    sc = rep["setup_cost"]
    if sc:
        L += _table(["metric", "value"], [[k, v] for k, v in sc.items()])
    else:
        L += ["_(not provided)_"]

    L += ["", "## 9. Supported paper claims", ""]
    pc = rep["paper_claims"]
    L += ["**Supported:**"] + (["- %s" % c for c in (pc["supported_claims"] or [])]
                               or ["- (none)"])
    L += ["", "**Unsupported:**"] + (
        ["- %s" % c for c in (pc["unsupported_claims"] or [])] or ["- (none)"])
    if pc.get("overclaim_risks"):
        L += ["", "**Overclaim risks:**"]
        L += ["- %s <- %s (%s)" % (o["claim"], o["file"], ",".join(o["reasons"]))
              for o in pc["overclaim_risks"]]

    L += ["", "## 10. Limitations", ""]
    L += ["%d. %s" % (i + 1, x) for i, x in enumerate(rep["limitations"])]
    L += [""]
    return "\n".join(L)
