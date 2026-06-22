"""Stage 7.1 -- generate paper artifacts from verified reports (no network)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.paper_artifact_generator import (  # noqa: E402
    PaperArtifactConfig,
    write_paper_artifacts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/paper_artifacts")
    ap.add_argument("--cost-report",
                    default="outputs/full_pipeline_cost_leakage.json")
    ap.add_argument("--skeleton-report",
                    default="outputs/masked_causal_lm_skeleton_probe.json")
    ap.add_argument("--boundary-report",
                    default="outputs/causal_lm_boundary_probe.json")
    ap.add_argument("--rope-report", default="outputs/rope_gqa_probe.json")
    args = ap.parse_args()

    cfg = PaperArtifactConfig(
        output_dir=args.output_dir, cost_report_path=args.cost_report,
        skeleton_report_path=args.skeleton_report,
        boundary_report_path=args.boundary_report,
        rope_report_path=args.rope_report,
    )
    report = write_paper_artifacts(cfg)

    for f in report["written_files"]:
        print(f"Wrote: {f}")
    src = report["metadata"]["source_reports"]
    print(f"source_reports={src}")
    missing = report["metadata"]["missing_inputs"]
    print(f"missing_inputs={missing if missing else 'none'}")


if __name__ == "__main__":
    main()
