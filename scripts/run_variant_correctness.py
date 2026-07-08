#!/usr/bin/env python
"""Correctness + cost runner for the masked-inference variants.

Variant C (``kronecker_lifted_linear``) is exercised end-to-end on:
  * a toy random MLP (fp64 exact), and
  * an optional real GPT-2 / tiny-GPT-2 block MLP (clean-skips if unavailable).

For variants A/B the correctness of the lifted-linear primitive is not applicable
(they keep the plain right-masked linear path); this runner reports their
stable-state form for the record and focuses the exact-error measurement on C.

Usage:
    python scripts/run_variant_correctness.py --variant kronecker_lifted_linear \
        --model tiny --lift-k 2
    python scripts/run_variant_correctness.py --variant kronecker_lifted_linear \
        --model gpt2 --layers 1 --seq-len 16 --lift-k 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from pllo.experiments.lifted_variants import variant_report_fields  # noqa: E402
from pllo.ops.kronecker_lifted_linear import (  # noqa: E402
    kron_lift,
    kron_unlift,
    lifted_mlp_report_fields,
    lifted_mlp_residual,
    lifted_mlp_squeeze,
    make_lift_params,
)
from pllo.ops.nonlinear_islands import gelu_reference, silu_reference  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "variant_comparison"


def _errs(y_hat, expected):
    max_abs = float((y_hat - expected).abs().max().item())
    rel = float((torch.linalg.norm(y_hat - expected)
                 / (torch.linalg.norm(expected) + 1e-30)).item())
    return max_abs, rel


def _cost_fields(m, d, f, k, materialized_kron_products, extra_matmuls):
    return {
        "lift_factor_k": int(k),
        "hidden_dim_before": int(d),
        "hidden_dim_after": int(d * k),
        "intermediate_dim_before": int(f),
        "intermediate_dim_after": int(f * k),
        # Linear matmul FLOPs scale ~k^2 (W_hat is d k x p k vs d x p).
        "theoretical_flop_multiplier": float(k * k),
        "theoretical_peak_memory_multiplier": float(k * k),
        "num_kronecker_products": int(materialized_kron_products),
        "num_extra_matmuls_vs_plain": int(extra_matmuls),
        "kron_materialized": True,
    }


def _run_mlp_case(x, w_up, b_up, w_down, b_down, k, *, swiglu, w_gate, b_gate,
                  seed, dtype, device, island_name):
    m, d = x.shape
    f = w_up.shape[1]
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    in_params = make_lift_params(m, d, k, generator=g, dtype=dtype, device=device)
    out_params = make_lift_params(m, d, k, pi=in_params.pi, generator=g,
                                  dtype=dtype, device=device)

    # Plain reference.
    up = x @ w_up + (b_up if b_up is not None else 0.0)
    if swiglu:
        gate = x @ w_gate + (b_gate if b_gate is not None else 0.0)
        act = silu_reference(gate) * up
    else:
        act = gelu_reference(up)
    y_plain = act @ w_down + (b_down if b_down is not None else 0.0)

    island = {"squeeze": lifted_mlp_squeeze, "residual": lifted_mlp_residual}[island_name]
    x_hat = kron_lift(x, in_params)

    # Warm + timed lifted run.
    for _ in range(2):
        res = island(x_hat, w_up, b_up, w_down, b_down, in_params, out_params,
                     activation="gelu", w_gate=w_gate, b_gate=b_gate, generator=g)
    t0 = time.perf_counter()
    res = island(x_hat, w_up, b_up, w_down, b_down, in_params, out_params,
                 activation="gelu", w_gate=w_gate, b_gate=b_gate, generator=g)
    lifted_ms = (time.perf_counter() - t0) * 1e3

    # Timed plain run.
    t0 = time.perf_counter()
    _up = x @ w_up + (b_up if b_up is not None else 0.0)
    if swiglu:
        _g = x @ w_gate + (b_gate if b_gate is not None else 0.0)
        _act = silu_reference(_g) * _up
    else:
        _act = gelu_reference(_up)
    _ = _act @ w_down + (b_down if b_down is not None else 0.0)
    plain_ms = (time.perf_counter() - t0) * 1e3

    y_hat = res["y_hat"]
    expected = kron_lift(y_plain, out_params)
    max_abs_lift, rel_lift = _errs(y_hat, expected)
    y_rec = kron_unlift(y_hat, out_params)
    max_abs_squeeze, rel_squeeze = _errs(y_rec, y_plain)

    n_kron = (3 if swiglu else 2) + 2  # up/(gate)/down weights + lift(x) + relift(act)
    return {
        "island": island_name,
        "activation": "swiglu" if swiglu else "gelu",
        "with_bias": b_up is not None,
        "max_abs_error_lifted_vs_lift_of_plain": max_abs_lift,
        "relative_l2_error_lifted": rel_lift,
        "max_abs_error_squeeze_vs_plain": max_abs_squeeze,
        "relative_l2_error_squeeze": rel_squeeze,
        "latency_ms_lifted": lifted_ms,
        "latency_ms_plain": plain_ms,
        "measured_latency_multiplier": (lifted_ms / plain_ms) if plain_ms > 0 else None,
        "cost": _cost_fields(m, d, f, k, materialized_kron_products=n_kron,
                             extra_matmuls=(2 if swiglu else 1)),
        "report": lifted_mlp_report_fields(
            out_params,
            variant_stage=("C2" if island_name == "residual" else "C1"),
            lifted_residual_stream=(island_name == "residual"),
            max_abs_error=max_abs_lift, relative_l2_error=rel_lift,
        ),
    }


def run_toy(k, seq_len, dtype, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    m, d, f = seq_len, 32, 64
    x = torch.randn(m, d, dtype=dtype, device=device, generator=g)
    w_up = torch.randn(d, f, dtype=dtype, device=device, generator=g)
    w_down = torch.randn(f, d, dtype=dtype, device=device, generator=g)
    w_gate = torch.randn(d, f, dtype=dtype, device=device, generator=g)
    cases = []
    for island in ("squeeze", "residual"):
        for swiglu in (False, True):
            cases.append(_run_mlp_case(
                x, w_up, None, w_down, None, k,
                swiglu=swiglu, w_gate=(w_gate if swiglu else None), b_gate=None,
                seed=seed, dtype=dtype, device=device, island_name=island))
    return {"model": "toy_random_mlp", "dims": {"m": m, "d": d, "f": f}, "cases": cases}


def run_gpt2(k, seq_len, layers, model_id, dtype, device, seed):
    try:
        from transformers import AutoModelForCausalLM
        from pllo.model_zoo.gpt2_conv1d_adapter import extract_conv1d_as_linear
    except Exception as exc:  # pragma: no cover - optional dep
        return {"model": model_id, "skipped": True, "skip_reason": f"import: {exc}"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - network/model absent
        return {"model": model_id, "skipped": True, "skip_reason": f"load: {exc}"}
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    blocks = model.transformer.h
    cases = []
    for li in range(min(layers, len(blocks))):
        mlp = blocks[li].mlp
        w_fc, b_fc = extract_conv1d_as_linear(mlp.c_fc)      # [d, 4d]
        w_proj, b_proj = extract_conv1d_as_linear(mlp.c_proj)  # [4d, d]
        w_fc = w_fc.to(dtype); w_proj = w_proj.to(dtype)
        b_fc = b_fc.to(dtype) if b_fc is not None else None
        b_proj = b_proj.to(dtype) if b_proj is not None else None
        d = w_fc.shape[0]
        x = torch.randn(seq_len, d, dtype=dtype, device=device, generator=g)
        for island in ("squeeze", "residual"):
            c = _run_mlp_case(
                x, w_fc, b_fc, w_proj, b_proj, k,
                swiglu=False, w_gate=None, b_gate=None,
                seed=seed + li, dtype=dtype, device=device, island_name=island)
            c["layer"] = li
            cases.append(c)
    return {
        "model": model_id, "skipped": False,
        "note": ("real GPT-2 MLP weights; activation = gelu_reference for BOTH "
                 "plain and lifted (tests lift exactness, not HF gelu_new match)"),
        "num_layers_tested": min(layers, len(blocks)),
        "cases": cases,
    }


def run(args) -> dict:
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    device = args.device
    out = {
        "variant": variant_report_fields(args.variant, lift_factor=args.lift_k),
        "lift_k": args.lift_k,
        "dtype": args.dtype,
        "device": device,
    }
    if args.variant != "kronecker_lifted_linear":
        out["note"] = ("lifted-linear correctness is only applicable to variant C; "
                       "A/B keep the plain right-masked linear path.")
        return out
    if args.model == "tiny":
        out["result"] = run_toy(args.lift_k, args.seq_len, dtype, device, args.seed)
    elif args.model in ("gpt2", "tiny-gpt2"):
        model_id = "sshleifer/tiny-gpt2" if args.model == "tiny-gpt2" else "gpt2"
        out["result"] = run_gpt2(args.lift_k, args.seq_len, args.layers, model_id,
                                 dtype, device, args.seed)
    else:
        raise ValueError(f"unknown model {args.model!r}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="kronecker_lifted_linear")
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "gpt2", "tiny-gpt2"])
    p.add_argument("--lift-k", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = run(args)
    out_json = args.output_json or (
        DEFAULT_OUT / args.variant / f"correctness_{args.model}_k{args.lift_k}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
