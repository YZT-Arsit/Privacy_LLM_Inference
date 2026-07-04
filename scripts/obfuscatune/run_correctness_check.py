"""ObfuscaTune correctness check: unprotected vs orthogonal vs random.

Runs a fixed GPT-2 forward under three modes and reports the logit error of
each obfuscated mode against the unprotected reference, plus condition-number
stats and wall time. Writes JSON to ``outputs/obfuscatune/correctness/``.

Local + CPU-friendly. Never downloads: default builds a tiny random GPT-2.

Example (see docs/obfuscatune_baseline.md):
    python scripts/obfuscatune/run_correctness_check.py --tiny-random-config
    python scripts/obfuscatune/run_correctness_check.py --dry-run
"""

from __future__ import annotations

import argparse
import time

from _common import (  # noqa: E402  (path set in _common)
    OUTPUT_ROOT, add_model_args, build_model, fixed_input_ids, planned_config,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(planned_config(args, {"modes": ["unprotected", "obfuscatune_orthogonal",
                                              "obfuscatune_random"]}))
        return

    import torch
    from pllo.baselines.obfuscatune.hf_gpt2_obfuscatune import run_mode

    model, minfo = build_model(args)
    ids = fixed_input_ids(minfo["vocab_size"], args.seq_len, args.seed)
    dt = torch.float64 if args.dtype == "float64" else torch.float32

    results = []
    for mode in ["unprotected", "obfuscatune_orthogonal", "obfuscatune_random"]:
        t0 = time.perf_counter()
        r = run_mode(model, ids, mode, dtype=dt, seed=args.seed)
        ms = (time.perf_counter() - t0) * 1000.0
        pm = r["audit"].get("phase_metrics", {})
        row = {
            "model_name": minfo["model_source"],
            "dtype": args.dtype,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "num_layers": minfo["num_layers"],
            "mode": mode,
            "max_abs_error_vs_unprotected": r["correctness"]["max_abs_error"],
            "mean_abs_error_vs_unprotected": r["correctness"]["mean_abs_error"],
            "relative_l2_error": r["correctness"]["relative_l2_error"],
            "logits_argmax_match_rate": r["argmax_match_rate"],
            "nan_or_inf_count": r["nan_or_inf_count"],
            "condition_number_stats": {
                "mean": pm.get("condition_number_mean", 1.0),
                "max": pm.get("condition_number_max", 1.0),
            },
            "boundary_calls": pm.get("boundary_calls"),
            "tee_transfer_bytes": pm.get("tee_transfer_bytes"),
            "exposed_plaintext_tensors": r["audit"].get("exposed_plaintext_tensors", []),
            "wall_time_ms": ms,
        }
        results.append(row)
        print(f"{mode:26s} max_abs_err={row['max_abs_error_vs_unprotected']:.3e} "
              f"argmax_match={row['logits_argmax_match_rate']:.3f} "
              f"naninf={row['nan_or_inf_count']} wall={ms:.1f}ms")

    out = OUTPUT_ROOT / "correctness" / f"correctness_{args.dtype}_L{minfo['num_layers']}.json"
    write_json(out, {"config": planned_config(args), "results": results})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
