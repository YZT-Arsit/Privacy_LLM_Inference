#!/usr/bin/env python
"""Run Stage 3 KV-cache prefill/decode correctness checks."""

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

from pllo.cache import cache_invariant_metrics
from pllo.evaluation import compute_correctness_metrics, top1_match_rate
from pllo.models import ObfuscatedTinyDecoderOnlyTransformer, PlainTinyDecoderOnlyTransformer, TinyTransformerConfig
from pllo.utils.seed import set_seed


def parse_dtype(name: str) -> torch.dtype:
    """Parse supported dtype names."""
    return torch.float32 if name == "float32" else torch.float64


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return tolerances for dtype."""
    return (1e-4, 1e-4) if dtype is torch.float32 else (1e-8, 1e-6)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/kv_cache_correctness.json"))
    return parser.parse_args()


def main() -> None:
    """Run KV cache correctness and write JSON metrics."""
    args = parse_args()
    set_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    atol, rtol = tolerances(dtype)
    device = torch.device(args.device)
    config = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.prompt_len + args.decode_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dtype=dtype,
        device=str(device),
    )
    plain = PlainTinyDecoderOnlyTransformer(config)
    obf = ObfuscatedTinyDecoderOnlyTransformer.from_plain(plain, config)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.prompt_len), device=device)

    plain_logits, plain_cache = plain.prefill(input_ids)
    obf_logits, obf_cache = obf.prefill(input_ids)
    prefill_metrics = compute_correctness_metrics(plain_logits, obf_logits, atol=atol, rtol=rtol)
    prefill_metrics["top1_match_rate"] = top1_match_rate(plain_logits, obf_logits)
    cache_metrics = cache_invariant_metrics(plain_cache, obf_cache, atol=atol, rtol=rtol)

    decode_metrics = []
    next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(args.decode_steps):
        plain_logits, plain_cache = plain.decode_step(next_token, plain_cache)
        obf_logits, obf_cache = obf.decode_step(next_token, obf_cache)
        metrics = compute_correctness_metrics(plain_logits, obf_logits, atol=atol, rtol=rtol)
        metrics["top1_match_rate"] = top1_match_rate(plain_logits, obf_logits)
        decode_metrics.append(metrics)
        cache_metrics = cache_invariant_metrics(plain_cache, obf_cache, atol=atol, rtol=rtol)
        next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_step_max = {
        "max_abs_error": max(float(m["max_abs_error"]) for m in decode_metrics),
        "mean_abs_error": max(float(m["mean_abs_error"]) for m in decode_metrics),
        "relative_l2_error": max(float(m["relative_l2_error"]) for m in decode_metrics),
        "allclose": all(bool(m["allclose"]) for m in decode_metrics),
        "top1_match_rate": min(float(m["top1_match_rate"]) for m in decode_metrics),
    }

    result = {
        "config": vars(args) | {"device": str(device)},
        "logits_metrics": {"prefill": prefill_metrics, "decode_step_max": decode_step_max},
        "cache_invariant_metrics": cache_metrics,
        "stage3_scope": {
            "prefill_decode": True,
            "kv_cache": True,
            "greedy_generation": False,
            "huggingface": False,
            "modelscope": False,
            "real_tee": False,
        },
    }
    result["config"]["output"] = str(result["config"]["output"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
