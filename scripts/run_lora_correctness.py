#!/usr/bin/env python
"""Run correctness checks for LoRA linear obfuscated execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.evaluation import compute_correctness_metrics
from pllo.ops.lora_linear import lora_linear_obfuscated, lora_linear_plain
from pllo.utils.seed import set_seed


def parse_dtype(dtype_name: str) -> torch.dtype:
    """Parse a supported dtype name."""
    if dtype_name == "float64":
        return torch.float64
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype_name!r}; expected 'float64' or 'float32'")


def tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return correctness tolerances appropriate for the selected dtype."""
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--d-in", type=int, default=16)
    parser.add_argument("--d-out", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--pad-scale", type=float, default=1.0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--use-pad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/lora_correctness.json"))
    return parser.parse_args()


def main() -> None:
    """Generate random tensors, run LoRA paths, and write JSON metrics."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    if args.rank > min(args.d_in, args.d_out):
        raise ValueError(
            f"rank must be <= min(d_in, d_out), got rank={args.rank}, "
            f"d_in={args.d_in}, d_out={args.d_out}"
        )
    atol, rtol = tolerances_for_dtype(dtype)

    x = torch.randn(args.seq_len, args.d_in, dtype=dtype, device=device)
    w = torch.randn(args.d_in, args.d_out, dtype=dtype, device=device)
    a = torch.randn(args.d_in, args.rank, dtype=dtype, device=device)
    b = torch.randn(args.rank, args.d_out, dtype=dtype, device=device)
    bias = None if args.no_bias else torch.randn(args.d_out, dtype=dtype, device=device)

    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(
        x,
        w,
        a,
        b,
        bias,
        use_pad=args.use_pad,
        pad_scale=args.pad_scale,
    )

    result = {
        "config": {
            "seq_len": args.seq_len,
            "d_in": args.d_in,
            "d_out": args.d_out,
            "rank": args.rank,
            "seed": args.seed,
            "device": str(device),
            "dtype": str(dtype),
            "use_pad": args.use_pad,
            "pad_scale": args.pad_scale,
            "bias": bias is not None,
            "atol": atol,
            "rtol": rtol,
        },
        "metrics": compute_correctness_metrics(reference, candidate, atol=atol, rtol=rtol),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
