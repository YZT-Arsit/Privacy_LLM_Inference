#!/usr/bin/env python
"""Stage 7.5 — paper artifact consolidation runner.

Aggregates ``outputs/*.json`` into ``paper_results/`` CSV / Markdown /
LaTeX tables. No new ops, no new attacks; pure aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_artifact_consolidation import (  # noqa: E402
    PaperArtifactConsolidationConfig,
    run_paper_artifact_consolidation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument(
        "--paper-results-dir", type=Path,
        default=PROJECT_ROOT / "paper_results",
    )
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument(
        "--exclude-missing-artifacts", dest="include_missing_artifacts",
        action="store_false", default=True,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PaperArtifactConsolidationConfig(
        outputs_dir=str(args.outputs_dir),
        paper_results_dir=str(args.paper_results_dir),
        strict=args.strict,
        include_missing_artifacts=args.include_missing_artifacts,
    )
    report = run_paper_artifact_consolidation(cfg)
    inv = report["artifact_inventory"]
    present = sum(1 for r in inv if r["status"] == "present")
    missing = len(report["missing_artifacts"])
    print(
        f"inventory={len(inv)} present={present} missing={missing}"
        f" correctness_rows={len(report['correctness_summary'])}"
        f" security_rows={len(report['security_proxy_summary'])}"
        f" workload_rows={len(report['workload_summary'])}"
        f" lora_training_rows={len(report['lora_training_summary'])}"
        f" limitations_rows={len(report['limitations_summary'])}"
    )
    print(f"Wrote {args.paper_results_dir}/json/artifact_inventory.json")
    print(f"Wrote {args.paper_results_dir}/csv/*.csv")
    print(f"Wrote {args.paper_results_dir}/latex/*.tex")
    print(f"Wrote {args.paper_results_dir}/markdown/*.md")


if __name__ == "__main__":
    main()
