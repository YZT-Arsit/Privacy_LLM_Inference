"""Shared helpers for the attack-framework scripts (local, CPU, offline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "outputs" / "attacks"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target-methods", default="plaintext_gpu,stip_qwen,obfuscatune_qwen_orthogonal,ours_amulet_style")
    p.add_argument("--source", choices=["toy", "tiny_qwen", "stip"], default="toy")
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--num-restarts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")


def out_dir(args, sub: str) -> Path:
    return Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / sub


def rep_for(target_method, args):
    from pllo.attacks.defense_adapters import build_representation
    src = "stip" if target_method == "stip_qwen" and args.source != "toy" else args.source
    return build_representation(target_method, source=src if src != "tiny_qwen" else "toy",
                                hidden_size=args.hidden_size, vocab_size=args.vocab_size,
                                num_samples=args.max_samples, seed=args.seed,
                                model_name_or_path=args.model_name_or_path)
