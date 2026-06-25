"""Final submission gate -- go/no-go decision for paper submission.

Takes ALL result JSONs and decides whether the paper's headline claims are backed
by real, paper-ready evidence. The gate is preflight-style: it FAILS unless every
required check holds. It reuses the claim validator (which itself reuses the
deployment-truth inference) so a single source of truth decides what each report
actually demonstrates -- dry-run / non-attested / no-LoRA reports never satisfy a
real-deployment claim.

Returns 0 iff the gate passed, else 1.

Example::

    python scripts/final_submission_gate.py \\
        --result-json outputs/e9_mmlu.json \\
        --result-json outputs/agg.json \\
        ... \\
        --final-artifact-tar dist/paper_artifacts.tar \\
        --output-json outputs/final_gate.json \\
        --output-md outputs/final_gate.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.claim_validator import (  # noqa: E402
    build_claim_report,
    load_results,
)
from pllo.experiments.deployment_truth import infer_deployment_truth  # noqa: E402
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    list_nonlinear_backends,
    nonlinear_backend_metadata,
    normalize_nonlinear_backend,
    parse_nonlinear_backends,
)

# Claim bases that assert the *nonlinear design itself* is formally secure. A
# design can back these ONLY if its registry ``security_claim_status`` is
# ``established`` (currently only ``current``); ``trusted_shortcut`` is
# ``under_discussion`` so it must FAIL such a claim until proofs/tests are added
# and the registry is updated.
FORMAL_SECURITY_CLAIM_BASES = {
    "formal_security", "design_security_proven", "design_formally_secure",
    "nonlinear_design_formally_secure",
}


def _parse_tagged_claim(claim):
    claim = str(claim)
    if claim.endswith("]") and "[" in claim:
        name, _, rest = claim.partition("[")
        backend = rest[:-1].strip()
        try:
            backend = normalize_nonlinear_backend(backend)
        except Exception:                                   # noqa: BLE001
            pass
        return name.strip(), backend
    return claim.strip(), None


def _security_claim_status(backend):
    try:
        return nonlinear_backend_metadata(backend).get("security_claim_status")
    except Exception:                                       # noqa: BLE001
        return None


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _stage(r):
    return _g(r, "stage") or ""


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def _transcript_passed(reports):
    """A security_transcript_scan report that did not fail/leak.

    Tolerant of key names: accept fail==False, leak_found==False, or passed==True.
    """
    for r in reports:
        if not isinstance(r, dict):
            continue
        if _stage(r) != "security_transcript_scan":
            continue
        if _g(r, "passed") is True:
            return True, r
        if _g(r, "fail") is False:
            return True, r
        if _g(r, "leak_found") is False:
            return True, r
        leaks = _g(r, "leaks")
        if isinstance(leaks, list) and not leaks:
            return True, r
    return False, None


def _count_public_benchmarks(reports):
    """Distinct paper_ready e9_task_utility_benchmark reports (by dataset, else
    by identity)."""
    datasets = set()
    countless = 0
    for r in reports:
        if not isinstance(r, dict):
            continue
        if _stage(r) != "e9_task_utility_benchmark":
            continue
        if _g(r, "paper_ready") is not True or _g(r, "dry_run") is True:
            continue
        ds = _g(r, "dataset")
        if ds:
            datasets.add(ds)
        else:
            countless += 1
    return len(datasets) + countless


def build_gate_report(results, *, required_claims=None, final_artifact_tar=None,
                      claim_tdx_attested_lora=False,
                      nonlinear_backends=None) -> dict:
    reports = [item.get("report") for item in results]
    reports = [r for r in reports if isinstance(r, dict)]

    # Split formal-security claims out: they are validated against the design
    # registry's security_claim_status, NOT the evidence-based claim validator
    # (which would spuriously reject them as unknown claim classes).
    required_claims = list(required_claims) if required_claims else []
    formal_security_required = []
    other_required = []
    for rc in required_claims:
        base, _bk = _parse_tagged_claim(rc)
        (formal_security_required if base in FORMAL_SECURITY_CLAIM_BASES
         else other_required).append(rc)

    claim_rep = build_claim_report(results,
                                   required_claims=other_required or None)
    supported = set(claim_rep["supported_claims"])
    backend_tagged = set(claim_rep.get("backend_tagged_supported") or [])

    checks = []
    warnings = list(claim_rep.get("warnings") or [])

    # 1. >=3 distinct paper_ready public-benchmark reports
    n_bench = _count_public_benchmarks(reports)
    checks.append(_check(
        "three_paper_ready_public_benchmarks", n_bench >= 3,
        "found %d distinct paper_ready e9_task_utility_benchmark report(s); "
        "need >=3" % n_bench))

    # 2. >=1 passing utility-preservation report
    pres_ok = False
    for r in reports:
        if _stage(r) in ("e9_pairwise_utility_preservation",
                         "e9_aggregate_utility_preservation"):
            if (_g(r, "utility_preserved") is True
                    and _g(r, "paper_ready") is True
                    and _g(r, "dry_run") is not True):
                pres_ok = True
                break
    checks.append(_check(
        "utility_preservation_passes", pres_ok,
        "need >=1 pairwise/aggregate report with utility_preserved & paper_ready "
        "& not dry_run"))

    # 3. real TDX-attested no-LoRA remote package decode
    checks.append(_check(
        "no_lora_tdx_attested_remote_package_decode",
        "no_lora_tdx_attested_remote_package_decode" in supported,
        "claim validator must support a real TDX-attested no-LoRA package decode"))

    # 4. real folded-LoRA H800 run
    checks.append(_check(
        "folded_lora_h800_real_validated",
        "folded_lora_h800_real_validated" in supported,
        "claim validator must support a real folded-LoRA H800 run"))

    # 5. (optional) TDX-attested folded-LoRA run
    if claim_tdx_attested_lora:
        checks.append(_check(
            "folded_lora_tdx_attested_validated",
            "folded_lora_tdx_attested_validated" in supported,
            "--claim-tdx-attested-lora set: need a real TDX-attested folded-LoRA "
            "run supported"))

    # 6. security negative tests passed
    checks.append(_check(
        "security_negative_tests_passed",
        "security_negative_tests_passed" in supported,
        "claim validator must support security_negative_tests_passed"))

    # 7. transcript scan passed
    tr_ok, _tr = _transcript_passed(reports)
    checks.append(_check(
        "security_transcript_scan_passed", tr_ok,
        "need a security_transcript_scan report with fail==False / leaks==[] / "
        "passed==True"))

    # 8. deployment truth: production_ready_serving must not be over-claimed
    prod_overclaim = False
    prod_detail = "no report claims production-ready serving without production_transport"
    for r in reports:
        truth = infer_deployment_truth(r)
        if (_g(r, "claims_production_ready_serving") is True
                and truth.get("production_transport") is not True):
            prod_overclaim = True
            prod_detail = ("a report claims production_ready_serving without "
                           "production_transport=True")
            break
    if "production_ready_serving" in supported:
        # claim validator only supports it with production_transport; still guard
        any_prod_transport = any(
            infer_deployment_truth(r).get("production_transport") is True
            for r in reports)
        if not any_prod_transport:
            prod_overclaim = True
            prod_detail = ("production_ready_serving supported without any "
                           "production_transport evidence")
    checks.append(_check(
        "deployment_truth_no_production_overclaim", not prod_overclaim,
        prod_detail))

    # 9. claim validator supports required (evidence-based) claims
    if other_required:
        all_req = claim_rep.get("all_required_supported") is True
        checks.append(_check(
            "required_claims_supported", all_req,
            "claim validator all_required_supported must be True for: %s"
            % ", ".join(other_required)))
    elif not required_claims:
        warnings.append("no --required-claims given; required-claims check skipped")

    # 10. latency baseline exists
    checks.append(_check(
        "latency_baseline_available",
        "latency_baseline_available" in supported,
        "claim validator must support latency_baseline_available"))

    # 11. final artifact tar exists
    tar_ok = bool(final_artifact_tar) and Path(final_artifact_tar).is_file()
    checks.append(_check(
        "final_artifact_tar_exists", tar_ok,
        "expected packaged tar at %s" % final_artifact_tar))

    # 12. per-backend (optional)
    per_backend = None
    if nonlinear_backends:
        per_backend = {}
        for bk in nonlinear_backends:
            need = [
                "no_lora_tdx_attested_remote_package_decode[%s]" % bk,
                "public_benchmark_utility_preserved[%s]" % bk,
            ]
            missing = [c for c in need if c not in backend_tagged]
            ok = not missing
            per_backend[bk] = {"ok": ok, "missing": missing}
            checks.append(_check(
                "nonlinear_backend_%s" % bk, ok,
                "backend %s must have backend-tagged support for: %s; missing: %s"
                % (bk, ", ".join(need), ", ".join(missing) or "none")))

    # 13. formal-security guard: a design with security_claim_status != established
    #     cannot back a FORMAL security claim, and we always flag design B.
    scope_designs = list(nonlinear_backends) if nonlinear_backends else \
        list(claim_rep.get("nonlinear_designs_evaluated") or [])
    for bk in scope_designs:
        status = _security_claim_status(bk)
        if status != "established":
            warnings.append(
                "design_B_security_not_formally_claimed: nonlinear design %r has "
                "security_claim_status=%s; correctness/performance/utility claims "
                "are allowed but FORMAL security claims are not." % (bk, status))

    for rc in formal_security_required:
        base, bk = _parse_tagged_claim(rc)
        backends = [bk] if bk else (scope_designs or list_nonlinear_backends())
        for b in backends:
            status = _security_claim_status(b)
            ok = (status == "established")
            checks.append(_check(
                "formal_security_claim[%s]" % b, ok,
                "formal security claim %r requires registry "
                "security_claim_status=established for design %r (is %s); add "
                "proofs/tests and update the registry first" % (rc, b, status)))
            if not ok:
                warnings.append(
                    "trusted_shortcut_cannot_support_formal_security_claim: "
                    "design %r security_claim_status=%s cannot back %r"
                    % (b, status, rc))

    gate_passed = all(c["ok"] for c in checks)
    blockers = [c["detail"] for c in checks if not c["ok"]]

    report = {
        "stage": "final_submission_gate",
        "gate_passed": gate_passed,
        "blockers": blockers,
        "warnings": warnings,
        "allowed_claims": claim_rep["supported_claims"],
        "checks": checks,
        "num_results": len(results),
        "required_claims": list(required_claims) if required_claims else [],
    }
    if per_backend is not None:
        report["per_backend"] = per_backend
        report["nonlinear_backends"] = list(nonlinear_backends)
    return report


def render_md(report: dict) -> str:
    L = ["# Final submission gate", "",
         "- gate_passed: **%s**" % report.get("gate_passed"),
         "- results parsed: %d" % report.get("num_results", 0), "",
         "## Checks", "", "| check | ok | detail |", "| --- | --- | --- |"]
    for c in report.get("checks", []):
        L.append("| %s | %s | %s |" % (c["name"], c["ok"], c["detail"]))
    if report.get("per_backend"):
        L += ["", "## Per-backend", "", "| backend | ok | missing |",
              "| --- | --- | --- |"]
        for bk, v in report["per_backend"].items():
            L.append("| %s | %s | %s |"
                     % (bk, v["ok"], ", ".join(v["missing"]) or "none"))
    if report.get("blockers"):
        L += ["", "## Blockers", ""]
        L += ["- %s" % b for b in report["blockers"]]
    if report.get("allowed_claims"):
        L += ["", "## Allowed (supported) claims", ""]
        L += ["- %s" % c for c in report["allowed_claims"]]
    if report.get("warnings"):
        L += ["", "## Warnings", ""]
        L += ["- %s" % w for w in report["warnings"]]
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-json", action="append", default=[])
    ap.add_argument("--required-claims", default=None)
    ap.add_argument("--final-artifact-tar", default=None)
    ap.add_argument("--claim-tdx-attested-lora", action="store_true",
                    default=False)
    ap.add_argument("--nonlinear-backends", default=None)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    results = load_results(args.result_json)
    required_claims = None
    if args.required_claims:
        required_claims = [c.strip() for c in args.required_claims.split(",")
                           if c.strip()]
    nonlinear_backends = None
    if args.nonlinear_backends:
        nonlinear_backends = parse_nonlinear_backends(args.nonlinear_backends)

    report = build_gate_report(
        results, required_claims=required_claims,
        final_artifact_tar=args.final_artifact_tar,
        claim_tdx_attested_lora=args.claim_tdx_attested_lora,
        nonlinear_backends=nonlinear_backends)

    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        om = Path(args.output_md)
        om.parent.mkdir(parents=True, exist_ok=True)
        om.write_text(render_md(report), encoding="utf-8")

    print("=== Final submission gate ===")
    print("gate_passed=%s" % report["gate_passed"])
    for c in report["checks"]:
        print("  [%s] %s" % ("OK" if c["ok"] else "XX", c["name"]))
    if report["blockers"]:
        print("BLOCKERS:")
        for b in report["blockers"]:
            print("  - %s" % b)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
