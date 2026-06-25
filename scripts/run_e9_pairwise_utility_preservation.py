"""E9 utility preservation: pairwise (baseline vs candidate) or aggregate.

Pairwise: compares a plaintext baseline E9 report to a folded/TDX/LoRA candidate
E9 report and decides whether the metric drop is within thresholds.

Aggregate: combines several pairwise reports and reports overall preservation only
if every required dataset (MMLU, GSM8K, BoolQ, AG News/SST-2) passes.

``utility_preserved`` is True only when within threshold AND both inputs are real
(paper_ready, not dry_run). Pure parsing -- no model / GPU.

Pairwise example::

    python scripts/run_e9_pairwise_utility_preservation.py \\
        --baseline-json outputs/e9_mmlu_plaintext_local.json \\
        --candidate-json outputs/e9_mmlu_tdx_attested_remote.json \\
        --max-abs-drop 0.02 --max-rel-drop 0.05 \\
        --output-json outputs/e9_mmlu_pairwise.json \\
        --output-md outputs/e9_mmlu_pairwise.md

Aggregate example::

    python scripts/run_e9_pairwise_utility_preservation.py --aggregate \\
        --pairwise-json outputs/e9_mmlu_pairwise.json \\
        --pairwise-json outputs/e9_gsm8k_pairwise.json \\
        --pairwise-json outputs/e9_boolq_pairwise.json \\
        --pairwise-json outputs/e9_sst2_pairwise.json \\
        --required-datasets mmlu,gsm8k,boolq,sst2 \\
        --output-json outputs/e9_aggregate_utility.json \\
        --output-md outputs/e9_aggregate_utility.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.utility_preservation import (  # noqa: E402
    REQUIRED_DATASETS,
    aggregate_preservation,
    load_json,
    pairwise_preservation,
    render_aggregate_md,
    render_pairwise_md,
)


def _write(report, output_json, output_md, render):
    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if output_md:
        p = Path(output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render(report), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate", action="store_true",
                    help="aggregate mode over --pairwise-json reports")
    ap.add_argument("--baseline-json", default=None)
    ap.add_argument("--candidate-json", default=None)
    ap.add_argument("--max-abs-drop", type=float, default=0.02)
    ap.add_argument("--max-rel-drop", type=float, default=0.05)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--task-type", default=None)
    ap.add_argument("--pairwise-json", action="append", default=[])
    ap.add_argument("--required-datasets",
                    default=",".join(REQUIRED_DATASETS))
    ap.add_argument("--fail-on-not-preserved", action="store_true", default=False)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    if args.aggregate or args.pairwise_json:
        reports = [load_json(p) for p in args.pairwise_json]
        reports = [r for r in reports if r]
        required = [d.strip() for d in args.required_datasets.split(",")
                    if d.strip()]
        report = aggregate_preservation(reports, required_datasets=required)
        _write(report, args.output_json, args.output_md, render_aggregate_md)
        print("=== E9 aggregate utility preservation ===")
        print("covered=%s missing=%s utility_preserved=%s paper_ready=%s"
              % (report["covered_datasets"], report["missing_datasets"],
                 report["utility_preserved"], report["paper_ready"]))
        preserved = report["utility_preserved"]
    else:
        if not (args.baseline_json and args.candidate_json):
            ap.error("pairwise mode requires --baseline-json and --candidate-json")
        baseline = load_json(args.baseline_json)
        candidate = load_json(args.candidate_json)
        if baseline is None or candidate is None:
            print("ERROR: could not load baseline/candidate JSON", file=sys.stderr)
            return 2
        report = pairwise_preservation(
            baseline, candidate, max_abs_drop=args.max_abs_drop,
            max_rel_drop=args.max_rel_drop, dataset=args.dataset,
            task_type=args.task_type)
        _write(report, args.output_json, args.output_md, render_pairwise_md)
        print("=== E9 pairwise utility preservation ===")
        print("dataset=%s baseline=%s candidate=%s delta_abs=%s delta_rel=%s"
              % (report["dataset"], report["baseline_metric"],
                 report["candidate_metric"], report["delta_abs"],
                 report["delta_rel"]))
        print("within_threshold=%s utility_preserved=%s paper_ready=%s"
              % (report["within_threshold"], report["utility_preserved"],
                 report["paper_ready"]))
        preserved = report["utility_preserved"]

    print("\nE9 UTILITY PRESERVATION WRITTEN")
    if args.fail_on_not_preserved and not preserved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
