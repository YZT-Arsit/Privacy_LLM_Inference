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
    "latency_baseline_available",
    "no_gpu_visible_plaintext",
    "no_worker_mask_secrets",
    "real_tdx_attestation_bound_to_runtime_hash",
    "production_ready_serving",
]

# Claim families whose evidence is nonlinear-design specific: a claim tagged for
# one design cannot be supported by evidence produced under another design. The
# validator emits per-backend support for these and parses ``claim[backend]``
# required-claim syntax against them.
BACKEND_SENSITIVE_CLAIMS = [
    "no_lora_tdx_attested_remote_package_decode",
    "public_benchmark_utility_preserved",
    "folded_lora_tdx_attested_validated",
    "latency_baseline_available",
    "security_negative_tests_passed",
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
    if name == "latency_baseline_available":
        rows = _g(r, "rows") or []
        good = [row for row in rows if isinstance(row, dict)
                and row.get("paper_ready") is True and row.get("dry_run") is not True]
        return (_stage(r) == "latency_baselines" and bool(good))
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


def _report_nonlinear_backend(r):
    """Canonical nonlinear design recorded in a report (or None)."""
    nb = _g(r, "nonlinear_backend") or _g(r, "nonlinear_design_name")
    if not nb:
        return None
    try:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        return normalize_nonlinear_backend(nb)
    except Exception:
        return str(nb)


def _parse_tagged_claim(claim):
    """``"name[backend]"`` -> ``(name, backend)``; ``"name"`` -> ``(name, None)``.

    The backend is normalized to a canonical design name when possible."""
    claim = str(claim)
    if claim.endswith("]") and "[" in claim:
        name, _, rest = claim.partition("[")
        backend = rest[:-1].strip()
        try:
            from pllo.experiments.nonlinear_designs import (
                normalize_nonlinear_backend)
            backend = normalize_nonlinear_backend(backend)
        except Exception:
            pass
        return name.strip(), backend
    return claim.strip(), None


def build_claim_report(results: list, required_claims=None,
                       paper_facing: bool = False) -> dict:
    """Validate which claim classes are backed by real evidence.

    ``paper_facing=True`` additionally REJECTS legacy non-paper-facing nonlinear
    designs (``current`` / ``trusted_shortcut``) from backing any backend-
    sensitive claim (requirement: paper-facing runs use only A_rightmul /
    amulet_secure_R). The behavioral checks (tag-only, trusted_calls>0) always
    apply regardless of this flag."""
    from pllo.experiments.nonlinear_designs import (
        trusted_shortcut_tag_only, nonlinear_tag_only,
        report_nonlinear_trusted_calls_clean, NON_PAPER_FACING_DESIGNS,
        PAPER_FACING_DESIGNS)
    enriched = []
    for item in results:
        rep = item.get("report")
        truth = infer_deployment_truth(rep) if isinstance(rep, dict) else {}
        enriched.append({"file": item.get("file"), "report": rep,
                         "truth": truth,
                         "nonlinear_backend": _report_nonlinear_backend(rep)})
    # reports TAGGED trusted_shortcut but lacking real Amulet-lift execution
    # evidence: they actually ran the 'current' path, so they may NOT back a
    # [trusted_shortcut] claim (only the design tag would be a lie).
    tag_only_ts_files = {e["file"] for e in enriched
                         if e["report"] is not None
                         and trusted_shortcut_tag_only(e["report"])}
    # GENERIC tag-only across migrated designs (trusted_shortcut / A_rightmul /
    # amulet_secure_R): execution-bearing report tagged with the design but
    # lacking its measured execution evidence.
    tag_only_files = {e["file"] for e in enriched
                      if e["report"] is not None
                      and nonlinear_tag_only(e["report"])}
    # HARD REJECT: any execution-bearing nonlinear report with a trusted
    # nonlinear crossing (trusted_calls > 0) -- violates single-TEE-entry.
    trusted_calls_violation_files = {
        e["file"] for e in enriched
        if e["report"] is not None
        and not report_nonlinear_trusted_calls_clean(e["report"])}

    supported = {}
    supported_backends = {}          # claim -> {backend or "unspecified": [files]}
    overclaim = []
    for claim in CLAIM_CLASSES:
        evidence = []
        risk_files = []
        per_backend = {}
        for e in enriched:
            if e["report"] is None:
                continue
            if _supports(claim, e["report"], e["truth"]):
                evidence.append(e["file"])
                bk = e["nonlinear_backend"] or "unspecified"
                reasons_block = []
                # a tag-only (any migrated design) report cannot back a
                # per-backend claim -- the design never actually executed.
                if e["file"] in tag_only_files:
                    reasons_block.append("nonlinear_design_tag_only_not_executed")
                # a report with a trusted nonlinear crossing violates the
                # single-TEE-entry contract and is rejected outright.
                if e["file"] in trusted_calls_violation_files:
                    reasons_block.append("nonlinear_trusted_calls_gt_zero")
                # paper-facing mode: backend-sensitive claims may NOT be backed
                # by a legacy design (current / trusted_shortcut).
                if (paper_facing and claim in BACKEND_SENSITIVE_CLAIMS
                        and bk in NON_PAPER_FACING_DESIGNS):
                    reasons_block.append(
                        "non_paper_facing_design_%s_in_paper_claim" % bk)
                if not reasons_block:
                    per_backend.setdefault(bk, []).append(e["file"])
                else:
                    overclaim.append({"claim": claim, "file": e["file"],
                                      "reasons": reasons_block})
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
        supported_backends[claim] = per_backend
        for rf in risk_files:
            overclaim.append({"claim": claim, **rf})

    supported_claims = [c for c in CLAIM_CLASSES if supported[c]]
    unsupported_claims = [c for c in CLAIM_CLASSES if not supported[c]]
    missing_evidence = {c: "no qualifying real evidence found"
                        for c in unsupported_claims}

    # ---- nonlinear-design awareness -------------------------------------
    backends_seen = sorted({e["nonlinear_backend"] for e in enriched
                            if e["nonlinear_backend"]})
    # which designs have ANY paper-facing supporting evidence
    designs_evaluated = sorted({
        bk for c in BACKEND_SENSITIVE_CLAIMS
        for bk in supported_backends.get(c, {})
        if bk != "unspecified"})
    try:
        from pllo.experiments.nonlinear_designs import list_nonlinear_backends
        all_designs = list_nonlinear_backends()
    except Exception:                                       # pragma: no cover
        all_designs = ["current", "trusted_shortcut"]
    designs_not_evaluated = [d for d in all_designs if d not in designs_evaluated]

    # backend-tagged support: "claim[backend]" supported iff a report tagged
    # with that backend supports the claim (cross-backend evidence never counts).
    backend_tagged_supported = []
    supported_claims_by_backend = {d: [] for d in all_designs}
    for c in BACKEND_SENSITIVE_CLAIMS:
        for bk, files in supported_backends.get(c, {}).items():
            if bk == "unspecified" or not files:
                continue
            backend_tagged_supported.append("%s[%s]" % (c, bk))
            if bk in supported_claims_by_backend:
                supported_claims_by_backend[bk].append(c)

    def _is_supported(claim_str):
        name, backend = _parse_tagged_claim(claim_str)
        if backend is None:
            return name in supported_claims
        # tagged: require evidence tagged with that exact backend
        return bool(supported_backends.get(name, {}).get(backend))

    warnings = []
    # production claim must stay unsupported unless explicit production transport
    if "production_ready_serving" in supported_claims:
        warnings.append("production_ready_serving is marked supported -- ensure a "
                        "real production transport exists; default deployment is a "
                        "research-prototype HTTP/SSH tunnel.")
    # tag-only trusted_shortcut reports: loud, explicit refusal reason
    if tag_only_ts_files:
        warnings.append(
            "trusted_shortcut_not_executed_in_real_path: %d report(s) are TAGGED "
            "trusted_shortcut but carry no Amulet-lift execution evidence "
            "(amulet_lift_executed / lifted_nonlinear_ops_count / lift_k / "
            "lifted_gpu_bytes); they actually ran the 'current' path and cannot "
            "back any public_benchmark_utility_preserved[trusted_shortcut] / "
            "no_lora_tdx_attested_remote_package_decode[trusted_shortcut] / "
            "folded_lora_tdx_attested_validated[trusted_shortcut] claim. Files: %s"
            % (len(tag_only_ts_files), sorted(tag_only_ts_files)))
    # if only one design is evaluated, the report must say so explicitly
    if len(designs_evaluated) == 1 and designs_not_evaluated:
        warnings.append(
            "only nonlinear design(s) %s have paper-facing evidence; design(s) %s "
            "were NOT evaluated -- state this explicitly in the paper (do not "
            "imply both designs are supported)."
            % (designs_evaluated, designs_not_evaluated))

    required_norm = list(required_claims) if required_claims else []
    if required_claims:
        for rc in required_claims:
            if not _is_supported(rc):
                warnings.append("REQUIRED claim not supported: %s" % rc)
        # if any required claim is tagged for a design that has zero evidence,
        # surface the cross-backend gap clearly
        for rc in required_claims:
            name, backend = _parse_tagged_claim(rc)
            if backend is not None and backend not in designs_evaluated:
                warnings.append(
                    "REQUIRED claim %s targets nonlinear design %r which has NO "
                    "paper-facing evidence (cannot be backed by another design)."
                    % (rc, backend))

    return {
        "stage": "paper_claim_validation",
        "num_results": len(results),
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "evidence_files": {c: supported[c] for c in supported_claims},
        "evidence_backends": {c: supported_backends[c] for c in supported_claims},
        "missing_evidence": missing_evidence,
        "overclaim_risks": overclaim,
        "warnings": warnings,
        # nonlinear-design dimension
        "nonlinear_backends_seen": backends_seen,
        "nonlinear_designs_evaluated": designs_evaluated,
        "nonlinear_designs_not_evaluated": designs_not_evaluated,
        "backend_tagged_supported": sorted(backend_tagged_supported),
        "supported_claims_by_backend": supported_claims_by_backend,
        "trusted_shortcut_tag_only_files": sorted(tag_only_ts_files),
        "trusted_shortcut_executed_in_real_path": not tag_only_ts_files,
        # generic (all migrated designs) tag-only + trusted-call hard checks
        "nonlinear_tag_only_files": sorted(tag_only_files),
        "nonlinear_trusted_calls_violation_files": sorted(
            trusted_calls_violation_files),
        "nonlinear_trusted_calls_clean": not trusted_calls_violation_files,
        "paper_facing_designs": list(PAPER_FACING_DESIGNS),
        "non_paper_facing_designs_seen": sorted(
            b for b in backends_seen if b in NON_PAPER_FACING_DESIGNS),
        "both_nonlinear_designs_supported": (
            len(designs_evaluated) >= 2),
        "required_claims": required_norm,
        "all_required_supported": (
            all(_is_supported(rc) for rc in required_claims)
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
    if rep.get("nonlinear_backends_seen") is not None:
        L += ["", "## Nonlinear designs", "",
              "- designs seen in evidence: %s"
              % (", ".join(rep.get("nonlinear_backends_seen") or []) or "none"),
              "- designs evaluated (paper-facing): %s"
              % (", ".join(rep.get("nonlinear_designs_evaluated") or []) or "none"),
              "- designs NOT evaluated: %s"
              % (", ".join(rep.get("nonlinear_designs_not_evaluated") or [])
                 or "none"),
              "- both designs supported: %s"
              % rep.get("both_nonlinear_designs_supported"), "",
              "| backend-tagged claim |", "| --- |"]
        for c in rep.get("backend_tagged_supported") or []:
            L.append("| %s |" % c)
    if rep["warnings"]:
        L += ["", "## Warnings", ""]
        L += ["- %s" % w for w in rep["warnings"]]
    L.append("")
    return "\n".join(L)
