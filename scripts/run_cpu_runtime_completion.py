"""Stage 7.5b - run the extended CPU runtime benchmark (local emulation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.cpu_runtime_completion import (  # noqa: E402
    CPURuntimeCompletionConfig,
    run_cpu_runtime_completion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-repeats", type=int, default=20)
    p.add_argument("--small", action="store_true",
                   help="Tiny sweep for smoke testing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.small:
        cfg = CPURuntimeCompletionConfig(
            output_dir=str(args.output_dir),
            seed=args.seed,
            num_warmup=1, num_repeats=2,
            batch_sizes=(1,), seq_lens=(4,), hidden_sizes=(16,),
        )
    else:
        cfg = CPURuntimeCompletionConfig(
            output_dir=str(args.output_dir),
            seed=args.seed,
            num_warmup=args.num_warmup,
            num_repeats=args.num_repeats,
        )
    report = run_cpu_runtime_completion(cfg)
    print(f"Wrote {args.output_dir}/cpu_runtime_completion.json")
    for r in report["rows"]:
        print(
            f"  {r['component']:36s} {r['variant']:18s} "
            f"mean={r['mean_ms']:.4f} median={r['median_ms']:.4f} std={r['std_ms']:.4f}"
        )


if __name__ == "__main__":
    main()
