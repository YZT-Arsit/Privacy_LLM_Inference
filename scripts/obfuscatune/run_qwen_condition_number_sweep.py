"""ObfuscaTune-Qwen condition-number sweep (Table 2 numerical trend).

Sweeps κ ∈ {1, 8, 32, 128, 160} + naive random on a Qwen2 prefill and reports
how the logit error against the unprotected reference grows with κ. Defaults to
float32 so the growth is visible in finite precision. Tiny random Qwen (GQA).

Example:
    python scripts/obfuscatune/run_qwen_condition_number_sweep.py --tiny-random-qwen
"""

from __future__ import annotations

import argparse

from _qwen_common import (
    add_qwen_model_args, build_qwen, out_dir, qwen_input_ids, qwen_planned_config,
    torch_dtype, write_json,
)

DEFAULT_KAPPAS = [1.0, 8.0, 32.0, 128.0, 160.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_qwen_model_args(p)
    p.add_argument("--kappas", type=float, nargs="+", default=DEFAULT_KAPPAS)
    p.set_defaults(dtype="float32")
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(qwen_planned_config(args, {"kappas": args.kappas, "include_random": True}))
        return

    from pllo.baselines.obfuscatune.qwen_obfuscatune import run_prefill

    model, arch, src = build_qwen(args)
    ids = qwen_input_ids(arch.vocab_size, args.seq_len, args.seed, args.batch_size)
    dt = torch_dtype(args.dtype)

    rows = []
    for kappa in args.kappas:
        mode = "orthogonal" if kappa <= 1.0 else "cond"
        r = run_prefill(model, ids, mode, dtype=dt, seed=args.seed, condition_number=kappa)
        pm = r["audit"].get("phase_metrics", {})
        rows.append({"condition_number": kappa,
                     "matrix_type": "orthogonal" if kappa <= 1.0 else "conditioned",
                     "max_abs_error": r["correctness"]["max_abs_error"],
                     "relative_l2_error": r["correctness"]["relative_l2_error"],
                     "logits_argmax_match_rate": r["argmax_match_rate"],
                     "nan_or_inf_count": r["nan_or_inf_count"],
                     "condition_number_mean": pm.get("condition_number_mean")})
    r = run_prefill(model, ids, "random", dtype=dt, seed=args.seed)
    pm = r["audit"].get("phase_metrics", {})
    rows.append({"condition_number": None, "matrix_type": "gaussian",
                 "max_abs_error": r["correctness"]["max_abs_error"],
                 "relative_l2_error": r["correctness"]["relative_l2_error"],
                 "logits_argmax_match_rate": r["argmax_match_rate"],
                 "nan_or_inf_count": r["nan_or_inf_count"],
                 "condition_number_mean": pm.get("condition_number_mean")})

    for row in rows:
        cn = "random" if row["condition_number"] is None else f"{row['condition_number']:g}"
        print(f"kappa={cn:8s} rel_l2={row['relative_l2_error']:.3e} "
              f"max_abs={row['max_abs_error']:.3e} argmax={row['logits_argmax_match_rate']:.3f}")

    base = out_dir(args, "qwen_condition_sweep")
    stem = f"qwen_cond_sweep_{args.dtype}"
    write_json(base / f"{stem}.json", {"config": qwen_planned_config(args), "results": rows})
    cols = ["condition_number", "matrix_type", "max_abs_error", "relative_l2_error",
            "logits_argmax_match_rate", "nan_or_inf_count"]
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    (base / f"{stem}.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {base / stem}.json/.csv")


if __name__ == "__main__":
    main()
