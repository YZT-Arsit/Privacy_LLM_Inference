"""Stage 7.5b - run the paper mitigation ablation (CPU only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_ablation_study import (  # noqa: E402
    PaperAblationStudyConfig,
    run_paper_ablation_study,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--num-trials", type=int, default=32)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PaperAblationStudyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_trials=args.num_trials,
        true_rank=args.true_rank,
        padded_rank=args.padded_rank,
    )
    report = run_paper_ablation_study(cfg)
    print(f"Wrote {args.output_dir}/paper_ablation_study.json")
    for row in report["rows"]:
        print(
            f"  {row['component']:32s} {row['setting']:32s} "
            f"err={row['max_abs_error']:.3e} "
            f"risk={row['risk_level']:25s} "
            f"role={row['interpretation']}"
        )


if __name__ == "__main__":
    main()
