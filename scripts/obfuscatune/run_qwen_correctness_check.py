"""ObfuscaTune-Qwen prefill correctness (unified schema output).

Runs a Qwen2 prefill under one mode and reports logit error vs the unprotected
reference, plus the full unified-schema row (correctness/cache/security/cost/
numerical). Defaults to a tiny random Qwen (GQA) -- never downloads.

Example:
    python scripts/obfuscatune/run_qwen_correctness_check.py --tiny-random-qwen
    python scripts/obfuscatune/run_qwen_correctness_check.py --mode random --dry-run
"""

from __future__ import annotations

import argparse
import time

from _qwen_common import (
    add_qwen_model_args, build_qwen, out_dir, qwen_input_ids, qwen_planned_config,
    torch_dtype, write_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_qwen_model_args(p)
    p.add_argument("--mode", choices=["unprotected", "orthogonal", "random", "cond"],
                   default="orthogonal")
    p.add_argument("--condition-number", type=float, default=32.0)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(qwen_planned_config(args, {"mode": args.mode,
                                         "condition_number": args.condition_number}))
        return

    from pllo.baselines.obfuscatune.qwen_obfuscatune import run_prefill
    from pllo.baselines.obfuscatune.qwen_cache import cache_metrics, hf_cache_shapes
    from pllo.baselines.obfuscatune.qwen_metrics import (
        cost_block, estimate_param_ratios, method_id, numerical_block, qwen_result_row)
    from pllo.baselines.obfuscatune.qwen_obfuscatune import qwen_plain_logits

    model, arch, src = build_qwen(args)
    ids = qwen_input_ids(arch.vocab_size, args.seq_len, args.seed, args.batch_size)
    dt = torch_dtype(args.dtype)

    t0 = time.perf_counter()
    r = run_prefill(model, ids, args.mode, dtype=dt, seed=args.seed,
                    condition_number=args.condition_number)
    ms = (time.perf_counter() - t0) * 1000.0
    pm = r["audit"].get("phase_metrics", {})

    rmt = {"unprotected": "orthogonal", "orthogonal": "orthogonal",
           "random": "random", "cond": "cond"}[args.mode]
    method = "unprotected" if args.mode == "unprotected" else method_id(
        rmt, args.condition_number)
    tee_ratio, gpu_ratio = estimate_param_ratios(model)

    # cache shapes (obf prefill cache vs HF reference cache)
    ref_out = model(ids, use_cache=True)
    ref_shapes = hf_cache_shapes(ref_out.past_key_values)
    obf_shapes = r["cache"].shapes() if r["cache"] is not None else []
    cache = None if r["cache"] is None else cache_metrics(
        enabled=True, prefill_cache_len=r["cache"].length(), decode_cache_len=0,
        obf_shapes=obf_shapes, ref_shapes=ref_shapes)

    row = qwen_result_row(
        method=method, model_name_or_path=src, mode="prefill", dtype=args.dtype,
        batch_size=args.batch_size, seq_len=args.seq_len, arch=arch,
        correctness={**r["correctness"], "logits_argmax_match_rate": r["argmax_match_rate"]},
        cache=cache,
        cost=cost_block(wall_time_ms=ms, slowdown_vs_unprotected=None,
                        phase_metrics=pm, tee_param_ratio=tee_ratio, gpu_param_ratio=gpu_ratio),
        numerical=numerical_block(
            random_matrix_type=rmt,
            condition_number_target=(args.condition_number if args.mode == "cond" else
                                     (1.0 if args.mode in ("orthogonal", "unprotected") else None)),
            condition_number_mean=pm.get("condition_number_mean", 1.0),
            condition_number_max=pm.get("condition_number_max", 1.0),
            nan_or_inf_count=r["nan_or_inf_count"]),
    )
    print(f"mode={args.mode} method={method} max_abs={row['correctness']['max_abs_error']:.3e} "
          f"argmax={row['correctness']['logits_argmax_match_rate']:.3f} "
          f"cache_shape_match={cache['cache_shape_match'] if cache else 'n/a'}")
    out = out_dir(args, "qwen_correctness") / f"qwen_correctness_{args.mode}_{args.dtype}.json"
    write_json(out, row)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
