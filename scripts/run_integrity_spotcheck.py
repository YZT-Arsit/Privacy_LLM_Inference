"""Runner for Stage 7.7e integrity spot-check prototype."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.integrity_spotcheck import (  # noqa: E402
    IntegritySpotCheckConfig,
    run_integrity_spotcheck,
    write_reports,
)


def main() -> None:
    cfg = IntegritySpotCheckConfig()
    rep = run_integrity_spotcheck(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']}")
    for mode in rep["modes_evaluated"]:
        info = rep["per_mode"][mode]
        last = info["corruption_present_curves"][-1]
        print(
            f"  [{mode:32s}] max_checked_fraction={last['checked_fraction']} "
            f"detection_rate={last['empirical_detection_rate']}"
        )


if __name__ == "__main__":
    main()
