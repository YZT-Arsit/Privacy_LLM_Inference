"""Validate which paper claims the experiment JSONs actually support.

Feed every experiment report; the validator infers deployment truth per report
and decides which fixed claim classes are supported by REAL evidence, refusing to
let dry-run/fixtures, no-LoRA runs, non-attested runs, or synthetic-LoRA back a
stronger claim than they earn. ``production_ready_serving`` stays unsupported
unless a report carries an explicit production transport.

Example::

    python scripts/validate_paper_claims.py \\
        --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \\
        --result-json outputs/qwen7b_lora_folded_remote_decode_probe.json \\
        --result-json outputs/e10_lora_utility.json \\
        --result-json outputs/security_negative_tests.json \\
        --required-claims no_lora_tdx_attested_remote_package_decode,\\
folded_lora_dry_run_validated \\
        --output-json outputs/paper_claim_validation.json \\
        --output-md   outputs/paper_claim_validation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.claim_validator import (  # noqa: E402
    build_claim_report,
    load_results,
    render_claim_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-json", action="append", default=[], required=True)
    ap.add_argument("--required-claims", default=None,
                    help="comma-separated claim classes that MUST be supported")
    ap.add_argument("--fail-on-missing-required", default="true")
    ap.add_argument("--output-json", default="outputs/paper_claim_validation.json")
    ap.add_argument("--output-md", default="outputs/paper_claim_validation.md")
    args = ap.parse_args()

    required = [c.strip() for c in args.required_claims.split(",")
                if c.strip()] if args.required_claims else None
    results = load_results(args.result_json)
    rep = build_claim_report(results, required_claims=required)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_claim_md(rep), encoding="utf-8")

    print("=== paper claim validation ===")
    if "nonlinear_designs_evaluated" in rep:
        print("nonlinear_designs_evaluated: %s"
              % (", ".join(rep.get("nonlinear_designs_evaluated") or [])
                 or "(none)"))
        print("both_nonlinear_designs_supported: %s"
              % rep.get("both_nonlinear_designs_supported"))
        bts = rep.get("backend_tagged_supported")
        if bts:
            print("backend_tagged_supported: %s" % ", ".join(bts))
    print("supported (%d):" % len(rep["supported_claims"]))
    for c in rep["supported_claims"]:
        print("  + %s  [%s]" % (c, "; ".join(rep["evidence_files"][c])))
    print("unsupported (%d):" % len(rep["unsupported_claims"]))
    for c in rep["unsupported_claims"]:
        print("  - %s" % c)
    if rep["overclaim_risks"]:
        print("overclaim risks:")
        for o in rep["overclaim_risks"]:
            print("  ! %s <- %s (%s)" % (o["claim"], o["file"],
                                         ",".join(o["reasons"])))
    for w in rep["warnings"]:
        print("WARNING: %s" % w)

    fail_on = str(args.fail_on_missing_required).strip().lower() in {
        "1", "true", "yes"}
    if required and fail_on and not rep["all_required_supported"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
