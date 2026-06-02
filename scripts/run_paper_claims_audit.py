#!/usr/bin/env python
"""Stage 7.5 — paper claims audit runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_claims_audit import (  # noqa: E402
    PaperClaimsAuditConfig,
    run_paper_claims_audit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paper-results-dir", type=Path,
        default=PROJECT_ROOT / "paper_results",
    )
    p.add_argument(
        "--outputs-dir", type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PaperClaimsAuditConfig(
        paper_results_dir=str(args.paper_results_dir),
        outputs_dir=str(args.outputs_dir),
    )
    report = run_paper_claims_audit(cfg)
    print(
        f"counts_by_status={report['counts_by_status']}"
        f" paper_claims_audit_status={report['paper_claims_audit_status']}"
    )
    print(f"Wrote {args.paper_results_dir}/json/paper_claims_audit.json")
    print(f"Wrote {args.paper_results_dir}/markdown/paper_claims_audit.md")
    print(f"Wrote {args.paper_results_dir}/latex/paper_claims_audit.tex")


if __name__ == "__main__":
    main()
