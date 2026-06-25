"""Final paper-claim validator -- decide which claims the experiment JSONs support.

Given every experiment report, classify each via the deployment-truth inference
and a small set of report-kind heuristics, then decide which of the fixed claim
classes are *supported by real evidence*. The whole point is to PREVENT overclaim:

* dry-run / fixture reports never support a paper-facing real-deployment claim;
* no-LoRA runs never support LoRA claims;
* H800-remote-but-not-attested runs never support TDX-attested claims;
* synthetic-LoRA never supports a real-HF-adapter utility claim;
* ``production_ready_serving`` is unsupported unless a report explicitly carries
  ``production_transport=True``.

For each claim we record supporting ``evidence_files`` (or ``missing_evidence``)
and any ``overclaim_risks`` -- reports whose *shape* matches a real claim but
whose gate (real / lora / attested / paper_ready) fails.

stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pllo.experiments.deployment_truth import infer_deployment_truth

__all__ = ["CLAIM_CLASSES", "load_results", "build_claim_report",
           "render_claim_md"]

# The fixed claim classes (order is the report order).
CLAIM_CLASSES = [
    "no_lora_full_qwen_folded_package_built",
    "no_lora_h800_local_package_decode",
    "no_lora_h800_remote_package_decode",
    "no_lora_tdx_lite_remote_package_decode",
    "no_lora_tdx_attested_remote_package_decode",
    "public_benchmark_utility_preserved",
    "folded_lora_dry_run_validated",
    "folded_lora_h800_real_validated",
    "folded_lora_tdx_attested_validated",
    "private_lora_training_tiny_prototype",
    "security_negative_tests_passed",
    "no_gpu_visible_plaintext",
    "no_worker_mask_secrets",
    "real_tdx_attestation_bound_to_runtime_hash",
    "production_ready_serving",
]


def load_results(paths) -> list:
    out = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            out.append({"file": str(p), "report": None, "error": "missing"})
            continue
        try:
            out.append({"file": str(p),
                        "report": json.loads(fp.read_text(encoding="utf-8"))})
        except Exception as exc:                            # noqa: BLE001
            out.append({"file": str(p), "report": None, "error": str(exc)})
    return out


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _stage(r):
    return _g(r, "stage") or ""


def _is_decode(r):
    return (_g(r, "package_backed_decode") is not None
            or "decode" in _stage(r) or _g(r, "tokens_exact_match") is not None)


def _tokens_ok(r):
    # a decode is "validated" if tokens matched (None counts as not-validated)
    return _g(r, "tokens_exact_match") is True


# Each entry: name -> predicate(report, truth) -> bool (this report SUPPORTS it),
# plus a "shape" predicate used to flag overclaim risk when the gate fails.

def _supports(name, r, t):
    real = (t.get("dry_run") is not True)
    lora = bool(t.get("lora_enabled"))
    attested = (t.get("tee_real") and t.get("attestation_verified")
                and t.get("runtime_hash_bound") is True)

    if name == "no_lora_full_qwen_folded_package_built":
        return (real and not lora
                and (t.get("folded_package_loaded") is True
                     or t.get("folded_package_valid") is True
                     or _g(r, "lora_package_built") is None and _g(
                         r, "num_shards") is not None))
    if name == "no_lora_h800_local_package_decode":
        return (real and not lora and _g(r, "package_backed_decode") is True
                and t.get("gpu_real") and not t.get("gpu_worker_remote")
                and _tokens_ok(r))
    if name == "no_lora_h800_remote_package_decode":
        return (real and not lora and _g(r, "package_backed_decode") is True
                and t.get("gpu_worker_remote") and _tokens_ok(r))
    if name == "no_lora_tdx_lite_remote_package_decode":
        return (real and not lora and _g(r, "package_backed_decode") is True
                and t.get("gpu_worker_remote")
                and t.get("boundary_mode") == "lite" and _tokens_ok(r))
    if name == "no_lora_tdx_attested_remote_package_decode":
        return (real and not lora and attested and t.get("gpu_worker_remote")
                and _g(r, "package_backed_decode") is True and _tokens_ok(r))
    if name == "public_benchmark_utility_preserved":
        # ONLY a pairwise/aggregate preservation report (baseline vs candidate)
        # can support this -- a single E9 metric value never can.
        return (_stage(r) in ("e9_pairwise_utility_preservation",
                              "e9_aggregate_utility_preservation")
                and _g(r, "utility_preserved") is True
                and _g(r, "paper_ready") is True
                and _g(r, "dry_run") is not True)
    if name == "folded_lora_dry_run_validated":
        # dry-run is EXPECTED here; just needs a lora decode with tokens matched
        return (lora and _tokens_ok(r))
    if name == "folded_lora_h800_real_validated":
        return (real and lora and t.get("gpu_worker_remote") and _tokens_ok(r))
    if name == "folded_lora_tdx_attested_validated":
        return (real and lora and attested and _tokens_ok(r))
    if name == "private_lora_training_tiny_prototype":
        return (_stage(r) == "private_lora_training_probe"
                and _g(r, "loss_decreased") is True
                and _g(r, "audit_passed") is True)
    if name == "security_negative_tests_passed":
        return (_stage(r) == "security_negative_tests"
                and _g(r, "all_passed") is True)
    if name == "no_gpu_visible_plaintext":
        return (real and _is_decode(r) and not _g(r, "gpu_visible_plaintext_fields")
                and _g(r, "audit_passed") is True)
    if name == "no_worker_mask_secrets":
        return (real and _is_decode(r)
                and _g(r, "worker_has_mask_secrets") is False
                and _g(r, "audit_passed") is True)
    if name == "real_tdx_attestation_bound_to_runtime_hash":
        return (real and attested)
    if name == "production_ready_serving":
        return t.get("production_transport") is True
    return False


# "shape" matches the claim structurally even if a gate fails -> overclaim risk.
def _shape(name, r, t):
    lora = bool(t.get("lora_enabled"))
    if name.startswith("no_lora") and "decode" in name:
        return (not lora and _g(r, "package_backed_decode") is True)
    if name == "no_lora_full_qwen_folded_package_built":
        return _g(r, "folded_package_loaded") is True
    if name in ("folded_lora_h800_real_validated",
                "folded_lora_tdx_attested_validated"):
        return lora and _is_decode(r)
    if name == "public_benchmark_utility_preserved":
        # a single E9 metric / E10 report "looks like" utility evidence but does
        # NOT qualify -> flag as an overclaim risk.
        return _stage(r) in ("e9_task_utility_benchmark",
                             "e10_lora_utility_benchmark",
                             "e9_pairwise_utility_preservation",
                             "e9_aggregate_utility_preservation")
    if name == "real_tdx_attestation_bound_to_runtime_hash":
        return t.get("attestation_evidence_present") is True
    return False


def build_claim_report(results: list, required_claims=None) -> dict:
    enriched = []
    for item in results:
        rep = item.get("report")
        truth = infer_deployment_truth(rep) if isinstance(rep, dict) else {}
        enriched.append({"file": item.get("file"), "report": rep,
                         "truth": truth})

    supported = {}
    overclaim = []
    for claim in CLAIM_CLASSES:
        evidence = []
        risk_files = []
        for e in enriched:
            if e["report"] is None:
                continue
            if _supports(claim, e["report"], e["truth"]):
                evidence.append(e["file"])
            elif _shape(claim, e["report"], e["truth"]):
                # shape matches but gate failed -> potential overclaim source
                reasons = []
                if e["truth"].get("dry_run") is True:
                    reasons.append("dry_run")
                if claim.endswith("attested") or "attested" in claim:
                    if not (e["truth"].get("tee_real")
                            and e["truth"].get("attestation_verified")):
                        reasons.append("not_attested")
                if claim.startswith("folded_lora") and not e["truth"].get(
                        "lora_enabled"):
                    reasons.append("no_lora")
                if (claim == "public_benchmark_utility_preserved"
                        and _stage(e["report"]) not in (
                            "e9_pairwise_utility_preservation",
                            "e9_aggregate_utility_preservation")):
                    reasons.append("single_e9_metric_not_preservation")
                if reasons:
                    risk_files.append({"file": e["file"], "reasons": reasons})
        supported[claim] = evidence
        for rf in risk_files:
            overclaim.append({"claim": claim, **rf})

    supported_claims = [c for c in CLAIM_CLASSES if supported[c]]
    unsupported_claims = [c for c in CLAIM_CLASSES if not supported[c]]
    missing_evidence = {c: "no qualifying real evidence found"
                        for c in unsupported_claims}

    warnings = []
    # production claim must stay unsupported unless explicit production transport
    if "production_ready_serving" in supported_claims:
        warnings.append("production_ready_serving is marked supported -- ensure a "
                        "real production transport exists; default deployment is a "
                        "research-prototype HTTP/SSH tunnel.")
    if required_claims:
        for rc in required_claims:
            if rc not in supported_claims:
                warnings.append("REQUIRED claim not supported: %s" % rc)

    return {
        "stage": "paper_claim_validation",
        "num_results": len(results),
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "evidence_files": {c: supported[c] for c in supported_claims},
        "missing_evidence": missing_evidence,
        "overclaim_risks": overclaim,
        "warnings": warnings,
        "required_claims": list(required_claims) if required_claims else [],
        "all_required_supported": (
            all(rc in supported_claims for rc in required_claims)
            if required_claims else None),
    }


def render_claim_md(rep: dict) -> str:
    L = ["# Paper claim validation", "",
         "- results parsed: %d" % rep["num_results"],
         "- supported: %d / %d claim classes"
         % (len(rep["supported_claims"]), len(rep["supported_claims"])
            + len(rep["unsupported_claims"])), "",
         "## Supported claims", "",
         "| claim | evidence files |", "| --- | --- |"]
    for c in rep["supported_claims"]:
        L.append("| %s | %s |" % (c, "; ".join(rep["evidence_files"][c])))
    L += ["", "## Unsupported claims", "", "| claim | reason |", "| --- | --- |"]
    for c in rep["unsupported_claims"]:
        L.append("| %s | %s |" % (c, rep["missing_evidence"][c]))
    if rep["overclaim_risks"]:
        L += ["", "## Overclaim risks (shape matches but gate failed)", "",
              "| claim | file | reasons |", "| --- | --- | --- |"]
        for o in rep["overclaim_risks"]:
            L.append("| %s | %s | %s |"
                     % (o["claim"], o["file"], ",".join(o["reasons"])))
    if rep["warnings"]:
        L += ["", "## Warnings", ""]
        L += ["- %s" % w for w in rep["warnings"]]
    L.append("")
    return "\n".join(L)
