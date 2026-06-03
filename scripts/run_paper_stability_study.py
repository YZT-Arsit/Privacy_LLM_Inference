"""Stage 7.5b - run the paper stability sweep (CPU only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_stability_study import (  # noqa: E402
    PaperStabilityStudyConfig,
    run_paper_stability_study,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--small", action="store_true", help="Run a tiny sweep for smoke testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.small:
        cfg = PaperStabilityStudyConfig(
            output_dir=str(args.output_dir),
            seeds=(2021, 2022),
            batch_sizes=(1, 2),
            seq_lens=(4,),
            hidden_sizes=(16,),
            true_ranks=(2,),
            padded_ranks=(8,),
        )
    else:
        cfg = PaperStabilityStudyConfig(output_dir=str(args.output_dir))
    report = run_paper_stability_study(cfg)
    print(f"Wrote {args.output_dir}/paper_stability_study.json")
    for s in report["summary_rows"]:
        print(
            f"  {s['experiment']:38s} "
            f"trials={s['trials']:4d} "
            f"allclose={s['allclose_rate']:.4f} "
            f"p95={s['max_error_p95']:.3e} "
            f"max={s['max_error_max']:.3e} "
            f"fail={s['failure_count']}"
        )


if __name__ == "__main__":
    main()
