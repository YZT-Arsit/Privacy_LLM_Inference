"""ObfuscaTune latency profile: unprotected vs orthogonal vs random.

Times a full-model forward under each mode and reports mean/std/p50/p95 wall
time plus the simulated boundary accounting (this is a CPU-simulator proxy, not
a real TEE deployment measurement). Defaults to a tiny random GPT-2.

Example:
    python scripts/obfuscatune/run_latency_profile.py --num-runs 20 --warmup 3
    python scripts/obfuscatune/run_latency_profile.py --dry-run
"""

from __future__ import annotations

import argparse
import statistics
import time

from _common import (
    OUTPUT_ROOT, add_model_args, build_model, fixed_input_ids, planned_config,
    write_json,
)


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--num-runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(planned_config(args, {"num_runs": args.num_runs, "warmup": args.warmup,
                                    "modes": ["unprotected", "obfuscatune_orthogonal",
                                              "obfuscatune_random"]}))
        return

    import torch
    from pllo.baselines.obfuscatune.hf_gpt2_obfuscatune import run_mode

    model, minfo = build_model(args)
    ids = fixed_input_ids(minfo["vocab_size"], args.seq_len, args.seed)
    dt = torch.float64 if args.dtype == "float64" else torch.float32

    results = []
    base_mean = None
    for mode in ["unprotected", "obfuscatune_orthogonal", "obfuscatune_random"]:
        for _ in range(args.warmup):
            run_mode(model, ids, mode, dtype=dt, seed=args.seed)
        times = []
        for _ in range(args.num_runs):
            t0 = time.perf_counter()
            run_mode(model, ids, mode, dtype=dt, seed=args.seed)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean = statistics.mean(times)
        if mode == "unprotected":
            base_mean = mean
        row = {
            "mode": mode,
            "wall_time_ms_mean": mean,
            "wall_time_ms_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "wall_time_ms_p50": _percentile(times, 0.5),
            "wall_time_ms_p95": _percentile(times, 0.95),
            "slowdown_vs_unprotected": (mean / base_mean) if base_mean else None,
            "num_runs": args.num_runs,
        }
        results.append(row)
        print(f"{mode:26s} mean={mean:.2f}ms p50={row['wall_time_ms_p50']:.2f} "
              f"p95={row['wall_time_ms_p95']:.2f} slowdown={row['slowdown_vs_unprotected']:.2f}x")

    out = OUTPUT_ROOT / "latency" / f"latency_{args.dtype}_L{minfo['num_layers']}.json"
    write_json(out, {"config": planned_config(args, {"num_runs": args.num_runs}),
                     "note": "CPU simulator wall time; NOT a real-TEE measurement",
                     "results": results})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
