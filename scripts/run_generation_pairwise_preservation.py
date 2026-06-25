"""Generation pairwise preservation (plaintext_local vs folded_remote current).

Two modes:

1. PAIRWISE (default) -- given a plaintext baseline report and a folded candidate
   report (both from ``run_generation_utility_benchmark.py``) emit a pairwise
   preservation report: metric_abs_drop, metric_rel_drop, exact_output_match_rate,
   length_delta_mean, latency_ratio, audit_passed, utility_preserved. JSON/CSV/MD.

2. SUMMARY (``--summary-input`` repeated) -- combine several pairwise reports into
   one final summary table across datasets. JSON/CSV/MD.

CURRENT design only (a trusted_shortcut candidate is refused). No LLM judge, no
subjective quality scoring.

Examples::

    python scripts/run_generation_pairwise_preservation.py \\
        --baseline-json out/gsm8k128_plaintext.json \\
        --candidate-json out/gsm8k128_folded.json \\
        --output-json out/gsm8k128_pairwise.json --output-md out/gsm8k128_pairwise.md

    python scripts/run_generation_pairwise_preservation.py \\
        --summary-input out/gsm8k128_pairwise.json \\
        --summary-input out/cnndm_pairwise.json \\
        --output-json out/generation_summary.json --output-md out/generation_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_metrics import (  # noqa: E402
    load_json,
    pairwise_generation_preservation,
    render_pairwise_csv,
    render_pairwise_md,
    render_summary_csv,
    render_summary_md,
    summarize_pairwise,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-json", default=None,
                    help="plaintext_local utility report")
    ap.add_argument("--candidate-json", default=None,
                    help="folded_remote (current) utility report")
    ap.add_argument("--summary-input", action="append", default=[],
                    help="pairwise report(s) to combine into a final summary "
                         "table (repeatable); switches to SUMMARY mode")
    ap.add_argument("--max-abs-drop", type=float, default=0.05)
    ap.add_argument("--max-rel-drop", type=float, default=0.10)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    # ---- SUMMARY mode ---------------------------------------------------
    if args.summary_input:
        reports = []
        for sp in args.summary_input:
            r = load_json(sp)
            if r is None:
                print("ERROR: cannot read pairwise report %s" % sp,
                      file=sys.stderr)
                return 3
            reports.append(r)
        out = summarize_pairwise(reports)
        render_md, render_csv = render_summary_md, render_summary_csv
        print("=== generation preservation summary ===")
        print("datasets=%d all_utility_preserved=%s all_paper_ready=%s"
              % (out["num_datasets"], out["all_utility_preserved"],
                 out["all_paper_ready"]))
    # ---- PAIRWISE mode --------------------------------------------------
    else:
        if not (args.baseline_json and args.candidate_json):
            print("ERROR: pairwise mode needs --baseline-json and "
                  "--candidate-json (or use --summary-input)", file=sys.stderr)
            return 3
        baseline = load_json(args.baseline_json)
        candidate = load_json(args.candidate_json)
        if baseline is None:
            print("ERROR: cannot read baseline %s" % args.baseline_json,
                  file=sys.stderr)
            return 3
        if candidate is None:
            print("ERROR: cannot read candidate %s" % args.candidate_json,
                  file=sys.stderr)
            return 3
        try:
            out = pairwise_generation_preservation(
                baseline, candidate, max_abs_drop=args.max_abs_drop,
                max_rel_drop=args.max_rel_drop)
        except ValueError as exc:                            # current-only guard
            print("ERROR: %s" % exc, file=sys.stderr)
            return 3
        render_md, render_csv = render_pairwise_md, render_pairwise_csv
        print("=== generation pairwise preservation (%s vs %s) ==="
              % (out["candidate_backend"], out["baseline_backend"]))
        print("dataset=%s metric=%s abs_drop=%s rel_drop=%s exact_out=%s "
              "utility_preserved=%s paper_ready=%s"
              % (out["dataset_name"], out["metric_name"], out["metric_abs_drop"],
                 out["metric_rel_drop"], out["exact_output_match_rate"],
                 out["utility_preserved"], out["paper_ready"]))

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if args.output_csv:
        pc = Path(args.output_csv)
        pc.parent.mkdir(parents=True, exist_ok=True)
        pc.write_text(render_csv(out), encoding="utf-8")
    if args.output_md:
        pm = Path(args.output_md)
        pm.parent.mkdir(parents=True, exist_ok=True)
        pm.write_text(render_md(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
