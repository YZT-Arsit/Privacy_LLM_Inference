"""Stage 7.6 -- paper-claims consistency scanner CLI.

Scans project markdown / LaTeX summary files for unsafe phrases and
writes a *bounded* report (compact JSON / aggregate CSV / Markdown).

By default the report is compact: summary counts, top-offender files and
terms, and a capped set of examples per category. The full per-occurrence
list is **never** serialized unless ``--write-full-occurrences`` is given,
and even then it is capped at ``--max-full-occurrences``. A hard
``--max-report-mb`` guard prevents any multi-GB file from being written.

CPU-only. No network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.stage_7_6_claims_consistency import (  # noqa: E402
    ClaimsReportConfig,
    build_claims_consistency_report,
    write_reports,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(REPO_ROOT),
                    help="Repository root to scan (default: this repo).")
    ap.add_argument("--output-dir", default="outputs",
                    help="Directory for the report files.")
    ap.add_argument("--write-full-occurrences", action="store_true",
                    help="Also write a capped per-occurrence CSV/JSON dump "
                         "(off by default).")
    ap.add_argument("--max-full-occurrences", type=int, default=100_000,
                    help="Hard cap on full-occurrence rows (default 100000).")
    ap.add_argument("--max-examples-per-category", type=int, default=25)
    ap.add_argument("--max-examples-per-file", type=int, default=25)
    ap.add_argument("--max-top-files", type=int, default=50)
    ap.add_argument("--max-top-terms", type=int, default=50)
    ap.add_argument("--max-report-mb", type=float, default=100,
                    help="Hard size guard (MB). If a report would exceed "
                         "this, a summary-only report is written instead.")
    args = ap.parse_args(argv)

    cfg = ClaimsReportConfig(
        write_full_occurrences=args.write_full_occurrences,
        max_full_occurrences=args.max_full_occurrences,
        max_examples_per_category=args.max_examples_per_category,
        max_examples_per_file=args.max_examples_per_file,
        max_top_files=args.max_top_files,
        max_top_terms=args.max_top_terms,
        max_report_mb=args.max_report_mb,
    )

    report = build_claims_consistency_report(repo_root=args.repo_root)
    j, c, m = write_reports(report, outputs_dir=args.output_dir, config=cfg)
    print(f"Wrote: {j}")
    print(f"Wrote: {c}")
    print(f"Wrote: {m}")
    print(
        f"passes_consistency_check={report['passes_consistency_check']} "
        f"total_occurrences={len(report.get('occurrences', []))} "
        f"write_full_occurrences={cfg.write_full_occurrences} "
        f"max_report_mb={cfg.max_report_mb}"
    )


if __name__ == "__main__":
    main()
