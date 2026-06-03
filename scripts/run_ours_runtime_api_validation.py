"""Stage 7.5c - run the deployable runtime API validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.ours_runtime_api_validation import (  # noqa: E402
    OursRuntimeAPIValidationConfig,
    run_ours_runtime_api_validation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--true-rank", type=int, default=2)
    p.add_argument("--padded-rank", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OursRuntimeAPIValidationConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        true_rank=args.true_rank,
        padded_rank=args.padded_rank,
    )
    report = run_ours_runtime_api_validation(cfg)
    print(f"Wrote {args.output_dir}/ours_runtime_api_validation.json")
    for row in report["rows"]:
        print(
            f"  {row['component']:36s} "
            f"err={row['correctness_error']:.3e} "
            f"allclose={row['allclose']} "
            f"backend={row['backend']} "
            f"sanitized={row['transcript_sanitized']}"
        )


if __name__ == "__main__":
    main()
