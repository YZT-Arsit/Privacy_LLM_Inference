#!/usr/bin/env python
"""Attack-surface evaluation for one masked-inference variant.

Measures the three distinct public attack surfaces (kept separate):
  A stable-state    -- norm_corr, gram_corr on the offloaded stable state
  B folded-weights  -- left self-Gram recoverability of the left mask
  C transient       -- whether the true nonlinear channel is directly observable

Usage:
    python scripts/run_variant_attack_eval.py --variant kronecker_lifted_linear \
        --lift-k 2 --attack norm,gram,weight_alignment
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from pllo.experiments.lifted_attack_surface import run_variant_attack_surface  # noqa: E402
from pllo.experiments.lifted_variants import variant_report_fields  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "variant_comparison"


def run(args) -> dict:
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    surf = run_variant_attack_surface(
        args.variant, m=args.rows, d=args.hidden, p=args.out_hidden,
        k=args.lift_k, seed=args.seed, dtype=dtype, device=args.device,
        token_safe_rows=not args.cross_token_mix,
    )
    surf["variant_record"] = variant_report_fields(args.variant, lift_factor=args.lift_k)
    surf["attacks_requested"] = args.attack.split(",")
    surf["cross_token_mix"] = args.cross_token_mix
    surf["honest_note"] = (
        "token-safe (KV-preserving) lift keeps per-token norm/Gram proportional to "
        "plaintext -> norm_corr/gram_corr ~1.0, same as a plain orthogonal right "
        "mask. Only cross-token dense mixing reduces it, and that breaks KV/causal "
        "order (--cross-token-mix, KV-UNSAFE, for reference only).")
    return surf


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="kronecker_lifted_linear")
    p.add_argument("--lift-k", type=int, default=2)
    p.add_argument("--attack", default="norm,gram,weight_alignment")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--out-hidden", type=int, default=64)
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cross-token-mix", action="store_true",
                   help="use KV-UNSAFE cross-token mixing (reference only)")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = run(args)
    out_json = args.output_json or (
        DEFAULT_OUT / args.variant / f"attack_eval_k{args.lift_k}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
