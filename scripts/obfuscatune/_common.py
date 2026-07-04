"""Shared helpers for the ObfuscaTune baseline scripts (local, CPU-friendly)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "outputs" / "obfuscatune"


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Model source: local path or a tiny random config (never downloads)."""
    p.add_argument("--model-name-or-path", default=None,
                   help="local path to a GPT-2 checkpoint (local_files_only)")
    p.add_argument("--tiny-random-config", action="store_true",
                   help="build a tiny random GPT-2 (no download); default if no path")
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--n-positions", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned config and exit without running")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def build_model(args: argparse.Namespace):
    from pllo.baselines.obfuscatune.hf_gpt2_obfuscatune import (
        load_gpt2, make_tiny_gpt2_config)
    dt = torch_dtype(args.dtype)
    if args.model_name_or_path:
        model = load_gpt2(model_name_or_path=args.model_name_or_path,
                          seed=args.seed, dtype=dt)
        n_layer = len(model.transformer.h)
        vocab = model.config.vocab_size
        src = f"local:{args.model_name_or_path}"
    else:
        cfg = make_tiny_gpt2_config(
            vocab_size=args.vocab_size, n_positions=args.n_positions,
            n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head)
        model = load_gpt2(tiny_random_config=cfg, seed=args.seed, dtype=dt)
        n_layer = args.n_layer
        vocab = args.vocab_size
        src = "tiny_random_config"
    return model, {"model_source": src, "num_layers": n_layer, "vocab_size": vocab}


def fixed_input_ids(vocab: int, seq_len: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (seq_len,), generator=g)


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def planned_config(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {
        "model_name_or_path": args.model_name_or_path,
        "tiny_random_config": args.model_name_or_path is None,
        "n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
        "vocab_size": args.vocab_size, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "dtype": args.dtype,
        "seed": args.seed, "device": args.device,
    }
    if extra:
        cfg.update(extra)
    return cfg
