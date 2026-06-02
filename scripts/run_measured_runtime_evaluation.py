#!/usr/bin/env python
"""Stage 7.5 — measured runtime evaluation runner (local emulation).

Drives existing inference / LoRA primitives under a small synthetic
workload and records wall-clock latency. **This is local emulation
only — NOT real TEE wall-time, NO real sleep.**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.measured_runtime_evaluation import (  # noqa: E402
    MeasuredRuntimeEvaluationConfig,
    run_measured_runtime_evaluation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "paper_results",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-repeats", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--include-gpu-if-available", action="store_true", default=False,
    )
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument(
        "--include-modern-decoder-wrapper",
        action="store_true", default=False,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MeasuredRuntimeEvaluationConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_warmup=args.num_warmup,
        num_repeats=args.num_repeats,
        device=args.device,
        dtype=args.dtype,
        include_gpu_if_available=args.include_gpu_if_available,
        strict=args.strict,
        include_modern_decoder_wrapper=args.include_modern_decoder_wrapper,
    )
    report = run_measured_runtime_evaluation(cfg)
    rows = report["rows"]
    measured = sum(
        1 for r in rows if r.get("mean_ms") is not None
    )
    skipped = sum(
        1 for r in rows if r.get("skipped_with_reason") is not None
        and r.get("mean_ms") is None
    )
    print(
        f"measured={measured} skipped={skipped} total={len(rows)}"
        f" wall_time_source={report['wall_time_source']}"
        f" is_real_tee_wall_time={report['is_real_tee_wall_time']}"
    )
    print(f"Wrote {args.output_dir}/json/measured_runtime.json")
    print(f"Wrote {args.output_dir}/csv/measured_runtime.csv")
    print(f"Wrote {args.output_dir}/markdown/measured_runtime.md")
    print(f"Wrote {args.output_dir}/latex/measured_runtime.tex")


if __name__ == "__main__":
    main()
