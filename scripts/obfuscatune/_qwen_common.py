"""Shared helpers for the ObfuscaTune-Qwen scripts (local, CPU-friendly)."""

from __future__ import annotations

import argparse

from _common import REPO_ROOT, OUTPUT_ROOT, torch_dtype, write_json  # noqa: F401
import torch  # noqa: E402


def add_qwen_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name-or-path", default=None,
                   help="local path to a Qwen2/2.5 checkpoint (never downloads)")
    p.add_argument("--tiny-random-qwen", action="store_true",
                   help="build a tiny random Qwen2 (default if no path)")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--intermediate-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)      # GQA by default
    p.add_argument("--head-dim", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")


def build_qwen(args: argparse.Namespace):
    from pllo.baselines.obfuscatune.qwen_config import (
        extract_arch, load_qwen, make_tiny_qwen_config)
    dt = torch_dtype(args.dtype)
    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=dt)
        src = f"local:{args.model_name_or_path}"
    else:
        cfg = make_tiny_qwen_config(
            vocab_size=args.vocab_size, hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size, num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads, num_key_value_heads=args.num_kv_heads,
            head_dim=args.head_dim)
        model = load_qwen(tiny_random_qwen=cfg, seed=args.seed, dtype=dt)
        src = "tiny_random_qwen"
    return model, extract_arch(model), src


def qwen_input_ids(vocab: int, seq_len: int, seed: int, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq_len), generator=g)


def qwen_planned_config(args: argparse.Namespace, extra: dict | None = None) -> dict:
    cfg = {
        "model_name_or_path": args.model_name_or_path,
        "tiny_random_qwen": args.model_name_or_path is None,
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim, "intermediate_size": args.intermediate_size,
        "vocab_size": args.vocab_size, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "dtype": args.dtype, "seed": args.seed,
        "device": args.device,
    }
    if extra:
        cfg.update(extra)
    return cfg


def out_dir(args: argparse.Namespace, default_sub: str):
    from pathlib import Path
    if args.output_dir:
        return Path(args.output_dir)
    return OUTPUT_ROOT / default_sub
