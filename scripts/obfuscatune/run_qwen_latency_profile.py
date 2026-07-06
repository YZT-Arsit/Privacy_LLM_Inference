"""ObfuscaTune-Qwen latency profile (prefill and decode, CPU simulator proxy).

Times unprotected / orthogonal / random for prefill and single-step decode and
reports mean/std/p50/p95. This is a CPU-simulator wall-time proxy, NOT a real
TEE measurement. Tiny random Qwen by default.

Example:
    python scripts/obfuscatune/run_qwen_latency_profile.py --tiny-random-qwen --num-runs 10
"""

from __future__ import annotations

import argparse
import statistics
import time

from _qwen_common import (
    add_qwen_model_args, build_qwen, out_dir, qwen_input_ids, qwen_planned_config,
    torch_dtype, write_json,
)


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_qwen_model_args(p)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--stages", nargs="+", choices=["prefill", "decode"],
                   default=["prefill", "decode"])
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(qwen_planned_config(args, {"num_runs": args.num_runs, "stages": args.stages,
                                         "modes": ["unprotected", "orthogonal", "random"]}))
        return

    import torch
    from pllo.baselines.obfuscatune.qwen_obfuscatune import run_prefill, run_decode_step

    model, arch, src = build_qwen(args)
    ids = qwen_input_ids(arch.vocab_size, args.seq_len, args.seed, args.batch_size)
    dt = torch_dtype(args.dtype)
    g = torch.Generator().manual_seed(args.seed + 1)
    next_id = torch.randint(0, arch.vocab_size, (args.batch_size, 1), generator=g)

    def time_call(fn):
        for _ in range(args.warmup):
            fn()
        ts = []
        for _ in range(args.num_runs):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return ts

    results = {}
    for stage in args.stages:
        base_mean = None
        rows = []
        for mode in ["unprotected", "orthogonal", "random"]:
            if stage == "prefill":
                fn = lambda m=mode: run_prefill(model, ids, m, dtype=dt, seed=args.seed)
            else:
                fn = lambda m=mode: run_decode_step(model, ids, next_id, m, dtype=dt, seed=args.seed)
            ts = time_call(fn)
            mean = statistics.mean(ts)
            if mode == "unprotected":
                base_mean = mean
            rows.append({"mode": mode, "wall_time_ms_mean": mean,
                         "wall_time_ms_std": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
                         "wall_time_ms_p50": _pct(ts, 0.5), "wall_time_ms_p95": _pct(ts, 0.95),
                         "slowdown_vs_unprotected": (mean / base_mean) if base_mean else None})
            print(f"[{stage}] {mode:12s} mean={mean:.2f}ms slowdown={rows[-1]['slowdown_vs_unprotected']:.2f}x")
        results[stage] = rows

    out = out_dir(args, "qwen_latency") / f"qwen_latency_{args.dtype}.json"
    write_json(out, {"config": qwen_planned_config(args, {"num_runs": args.num_runs}),
                     "note": "CPU simulator wall time; NOT a real-TEE measurement",
                     "model_name_or_path": src, "results": results})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
