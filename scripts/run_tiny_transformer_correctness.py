#!/usr/bin/env python
"""Run correctness checks for the Stage 2 tiny Transformer."""

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
from pllo.models import (
    ObfuscatedTinyDecoderOnlyTransformer,
    PlainTinyDecoderOnlyTransformer,
    TinyTransformerConfig,
)
from pllo.utils.seed import set_seed


def parse_dtype(dtype_name: str) -> torch.dtype:
    """Parse supported dtype names."""
    if dtype_name == "float64":
        return torch.float64
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype_name!r}; expected 'float64' or 'float32'")


def tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float]:
    """Return correctness tolerances for a dtype."""
    if dtype is torch.float32:
        return 1e-4, 1e-4
    return 1e-8, 1e-6


def top1_match_rate(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute token-wise top-1 match rate for logits."""
    return float((reference.argmax(dim=-1) == candidate.argmax(dim=-1)).to(torch.float64).mean().item())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/tiny_transformer_correctness.json"))
    return parser.parse_args()


def main() -> None:
    """Run plain and obfuscated tiny Transformer forwards and write JSON metrics."""
    args = parse_args()
    set_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)
    config = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dtype=dtype,
        device=str(device),
    )
    atol, rtol = tolerances_for_dtype(dtype)

    plain = PlainTinyDecoderOnlyTransformer(config)
    obfuscated = ObfuscatedTinyDecoderOnlyTransformer.from_plain(plain, config)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    plain_logits = plain(input_ids)
    recovered_logits = obfuscated(input_ids)
    metrics = compute_correctness_metrics(plain_logits, recovered_logits, atol=atol, rtol=rtol)
    metrics["top1_match_rate"] = top1_match_rate(plain_logits, recovered_logits)

    result = {
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "atol": atol,
            "rtol": rtol,
        },
        "metrics": metrics,
        "stage2_simplifications": {
            "trusted_layernorm": True,
            "trusted_gelu": True,
            "kv_cache": False,
            "huggingface": False,
            "modelscope": False,
            "real_tee": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
