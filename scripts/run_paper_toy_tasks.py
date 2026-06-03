"""Stage 7.5b - run the paper toy-task workload (CPU only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.paper_toy_tasks import (  # noqa: E402
    PaperToyTaskConfig,
    run_paper_toy_tasks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-train-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PaperToyTaskConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        true_rank=args.true_rank,
        padded_rank=args.padded_rank,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        lr=args.lr,
    )
    report = run_paper_toy_tasks(cfg)
    print(f"Wrote {args.output_dir}/paper_toy_tasks.json")
    for row in report["rows"]:
        print(
            f"  {row['task_name']:32s} "
            f"loss_diff={row['loss_diff']:.3e} "
            f"acc_diff={row['accuracy_diff']:.3e} "
            f"token_match={row['token_match_rate']:.4f} "
            f"allclose={row['allclose']}"
        )


if __name__ == "__main__":
    main()
