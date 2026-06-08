"""Runner for Stage 7.7a lm-head scalability experiment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.lm_head_scalability import (  # noqa: E402
    LMHeadScalabilityConfig,
    run_lm_head_scalability,
    write_reports,
)


def main() -> None:
    cfg = LMHeadScalabilityConfig()
    report = run_lm_head_scalability(cfg=cfg)
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"status={report['status']} modes={report['modes_evaluated']}")


if __name__ == "__main__":
    main()
