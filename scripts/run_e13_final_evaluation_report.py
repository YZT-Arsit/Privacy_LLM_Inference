"""E13 runner: consolidate every stage into the final paper-facing report.

Pure parsing of the prior experiment JSONs into ten tables + a limitations
section (JSON + Markdown). Deployment truth is re-inferred and paper claims come
only from the overclaim-refusing validator, so the consolidated report cannot
claim more than the evidence supports.

Example::

    python scripts/run_e13_final_evaluation_report.py \\
        --correctness-json outputs/qwen7b_folded_full_decode_probe.json \\
        --correctness-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \\
        --e3-json outputs/e3_remote_decode_scaling.json \\
        --e4-json outputs/e4_setup_cost.json \\
        --e5-json outputs/e5_final_comparison.json \\
        --e8-json outputs/e8_lora_final_report.json \\
        --e9-json outputs/e9_mmlu_tdx_attested.json \\
        --e10-json outputs/e10_lora_utility.json \\
        --latency-json outputs/e12_latency_baselines.json \\
        --security-negative-json outputs/security_negative_tests.json \\
        --output-json outputs/e13_final_evaluation.json \\
        --output-md   docs/paper_draft/evaluation_full.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e13_final_report import (  # noqa: E402
    build_e13_report,
    render_e13_md,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)


def _load(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correctness-json", action="append", default=[])
    ap.add_argument("--e3-json", default=None)
    ap.add_argument("--e4-json", default=None)
    ap.add_argument("--e5-json", default=None)
    ap.add_argument("--e8-json", default=None)
    ap.add_argument("--e9-json", action="append", default=[])
    ap.add_argument("--e10-json", default=None)
    ap.add_argument("--latency-json", default=None)
    ap.add_argument("--security-negative-json", default=None)
    ap.add_argument("--result-json", action="append", default=[],
                    help="extra reports for deployment-truth + claim validation")
    ap.add_argument("--claims-json", default=None,
                    help="precomputed paper_claim_validation.json (else computed)")
    ap.add_argument("--required-claims", default=None)
    ap.add_argument("--output-json", default="outputs/e13_final_evaluation.json")
    ap.add_argument("--output-md", default="docs/paper_draft/evaluation_full.md")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    correctness = [r for r in (_load(p) for p in args.correctness_json) if r]
    e9 = [r for r in (_load(p) for p in args.e9_json) if r]
    e5 = _load(args.e5_json)
    e10 = _load(args.e10_json)
    security_negative = _load(args.security_negative_json)

    # results feeding truth + claims: explicit --result-json plus everything we
    # already loaded that carries a deployment/utility posture.
    results = []
    seen = set()

    def _add_result(path, rep):
        if rep is None:
            return
        key = path or json.dumps(rep, sort_keys=True, default=str)[:64]
        if key in seen:
            return
        seen.add(key)
        results.append({"file": path or "(inline)", "report": rep})

    for p in args.result_json:
        _add_result(p, _load(p))
    for p, r in zip(args.correctness_json, correctness):
        _add_result(p, r)
    for p, r in zip(args.e9_json, e9):
        _add_result(p, r)
    _add_result(args.e5_json, e5)
    _add_result(args.e10_json, e10)
    _add_result(args.security_negative_json, security_negative)

    required = [c.strip() for c in args.required_claims.split(",")
                if c.strip()] if args.required_claims else None

    report = build_e13_report({
        "correctness": correctness, "e3": _load(args.e3_json),
        "e4": _load(args.e4_json), "e5": e5, "e8": _load(args.e8_json),
        "e9": e9, "e10": e10, "latency": _load(args.latency_json),
        "security_negative": security_negative, "results": results,
        "claims": _load(args.claims_json), "required_claims": required,
    })
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e13_md(report), encoding="utf-8")

    print("=== E13 final evaluation report ===")
    print("correctness rows=%d public_utility rows=%d security_matrix rows=%d"
          % (len(report["correctness"]), len(report["public_task_utility"]),
             len(report["security_audit_matrix"])))
    print("deployment_truth rows=%d supported_claims=%d"
          % (len(report["deployment_truth"]),
             len(report["paper_claims"]["supported_claims"] or [])))
    print("\nE13 REPORT WRITTEN: %s / %s" % (args.output_json, args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
