"""Stage 7.5b - run the paper baseline comparison (CPU only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_baseline_comparison import (  # noqa: E402
    PaperBaselineComparisonConfig,
    run_paper_baseline_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=8)
    p.add_argument("--num-repeats", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PaperBaselineComparisonConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        true_rank=args.true_rank,
        padded_rank=args.padded_rank,
        num_repeats=args.num_repeats,
    )
    report = run_paper_baseline_comparison(cfg)
    print(f"Wrote {args.output_dir}/paper_baseline_comparison.json")
    for row in report["rows"]:
        print(
            f"  {row['variant']:50s} "
            f"err={row['correctness_error']:.3e} "
            f"runtime_ms={row['local_runtime_ms']:.4f} "
            f"risk={row['proxy_risk_level']}"
        )


if __name__ == "__main__":
    main()
