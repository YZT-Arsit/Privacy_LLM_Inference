"""ObfuscaTune condition-number sweep (reproduces the paper's Table 2 trend).

Sweeps the obfuscation-matrix condition number kappa in {1, 8, 32, 128, 160}
plus the naive ``random`` matrix, on the SAME model + input, and reports how the
logit error against the unprotected reference grows with kappa. This is the
numerical-error trend of Table 2, NOT the downstream QA accuracy.

To make the error growth visible in finite precision the sweep defaults to
float32 (the paper's regime); use --dtype float64 to see the same ordering at a
smaller magnitude. Writes JSON + CSV + Markdown to
``outputs/obfuscatune/condition_sweep/``.

Example:
    python scripts/obfuscatune/run_condition_number_sweep.py --tiny-random-config
    python scripts/obfuscatune/run_condition_number_sweep.py --dry-run
"""

from __future__ import annotations

import argparse

from _common import (
    OUTPUT_ROOT, add_model_args, build_model, fixed_input_ids, planned_config,
    write_json,
)

DEFAULT_KAPPAS = [1.0, 8.0, 32.0, 128.0, 160.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--kappas", type=float, nargs="+", default=DEFAULT_KAPPAS)
    p.add_argument("--include-random", action="store_true", default=True)
    p.set_defaults(dtype="float32")
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(planned_config(args, {"kappas": args.kappas, "include_random": args.include_random}))
        return

    import torch
    from pllo.baselines.obfuscatune.hf_gpt2_obfuscatune import run_mode

    model, minfo = build_model(args)
    ids = fixed_input_ids(minfo["vocab_size"], args.seq_len, args.seed)
    dt = torch.float64 if args.dtype == "float64" else torch.float32

    rows = []
    for kappa in args.kappas:
        r = run_mode(model, ids, "obfuscatune_cond_sweep", dtype=dt, seed=args.seed,
                     condition_number=kappa)
        pm = r["audit"].get("phase_metrics", {})
        rows.append({
            "condition_number": kappa,
            "matrix_type": "cond" if kappa > 1.0 else "orthogonal",
            "max_abs_error": r["correctness"]["max_abs_error"],
            "mean_abs_error": r["correctness"]["mean_abs_error"],
            "relative_l2_error": r["correctness"]["relative_l2_error"],
            "logits_argmax_match_rate": r["argmax_match_rate"],
            "nan_or_inf_count": r["nan_or_inf_count"],
            "condition_number_measured_mean": pm.get("condition_number_mean"),
        })
    if args.include_random:
        r = run_mode(model, ids, "obfuscatune_random", dtype=dt, seed=args.seed)
        pm = r["audit"].get("phase_metrics", {})
        rows.append({
            "condition_number": None,
            "matrix_type": "random",
            "max_abs_error": r["correctness"]["max_abs_error"],
            "mean_abs_error": r["correctness"]["mean_abs_error"],
            "relative_l2_error": r["correctness"]["relative_l2_error"],
            "logits_argmax_match_rate": r["argmax_match_rate"],
            "nan_or_inf_count": r["nan_or_inf_count"],
            "condition_number_measured_mean": pm.get("condition_number_mean"),
        })

    for row in rows:
        cn = row["condition_number"]
        label = "random" if cn is None else f"kappa={cn:g}"
        print(f"{label:12s} rel_l2={row['relative_l2_error']:.3e} "
              f"max_abs={row['max_abs_error']:.3e} argmax={row['logits_argmax_match_rate']:.3f}")

    base = OUTPUT_ROOT / "condition_sweep"
    stem = f"cond_sweep_{args.dtype}_L{minfo['num_layers']}"
    write_json(base / f"{stem}.json", {"config": planned_config(args), "results": rows})

    # CSV
    cols = ["condition_number", "matrix_type", "max_abs_error", "mean_abs_error",
            "relative_l2_error", "logits_argmax_match_rate", "nan_or_inf_count"]
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    (base / f"{stem}.csv").parent.mkdir(parents=True, exist_ok=True)
    (base / f"{stem}.csv").write_text("\n".join(lines) + "\n")

    # Markdown
    md = ["# ObfuscaTune condition-number sweep",
          f"model={minfo['model_source']} layers={minfo['num_layers']} dtype={args.dtype}",
          "", "| kappa | matrix | rel_L2 error | max_abs error | argmax match |",
          "|---|---|---|---|---|"]
    for row in rows:
        cn = "random" if row["condition_number"] is None else f"{row['condition_number']:g}"
        md.append(f"| {cn} | {row['matrix_type']} | {row['relative_l2_error']:.3e} | "
                  f"{row['max_abs_error']:.3e} | {row['logits_argmax_match_rate']:.3f} |")
    (base / f"{stem}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {base / stem}.json/.csv/.md")


if __name__ == "__main__":
    main()
