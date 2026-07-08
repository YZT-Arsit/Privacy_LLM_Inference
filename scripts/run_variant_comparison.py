#!/usr/bin/env python
"""Compare all three masked-inference variants (correctness + attack surfaces).

Writes per-variant JSON under outputs/variant_comparison/<variant>/ and a summary
table under outputs/variant_comparison/summary/.

Usage:
    python scripts/run_variant_comparison.py \
        --variants right_pad_pure_right,right_pad_amulet_gelu,kronecker_lifted_linear \
        --model tiny --quick
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

import run_variant_correctness as vc  # noqa: E402  (sibling script)

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "variant_comparison"


def _correctness_for(variant, model, k, seq_len, layers, dtype_str, device, seed):
    ns = argparse.Namespace(
        variant=variant, model=model, lift_k=k, seq_len=seq_len, layers=layers,
        dtype=dtype_str, device=device, seed=seed, output_json=None)
    return vc.run(ns)


def _summary_rows(variant, attack, correctness):
    """Flatten into attack_surface x variant rows for the summary table."""
    rows = []
    a = attack["surface_A_stable_state"]
    b = attack["surface_B_folded_weights"]
    c = attack["surface_C_transient"]
    k = attack["config"]["lift_factor"]

    # Pull a representative correctness error (variant C only).
    corr_err = None
    if correctness.get("result", {}).get("cases"):
        corr_err = max(
            cs["max_abs_error_lifted_vs_lift_of_plain"]
            for cs in correctness["result"]["cases"])

    rows.append({
        "attack_surface": "A_stable_state", "variant": variant, "lift_factor": k,
        "norm_corr": a["norm_corr"], "gram_corr": a["gram_corr"],
        "token_recovery": None, "weight_alignment": None,
        "correctness_error": corr_err,
    })
    rows.append({
        "attack_surface": "B_folded_weights", "variant": variant, "lift_factor": k,
        "norm_corr": None, "gram_corr": None, "token_recovery": None,
        "weight_alignment": b["left_mask_diag_match_rel_err"],
        "weight_gram_diag_distinct_frac": b["folded_gram_diag_distinct_frac"],
        "correctness_error": corr_err,
    })
    rows.append({
        "attack_surface": "C_transient", "variant": variant, "lift_factor": k,
        "norm_corr": None, "gram_corr": None, "token_recovery": None,
        "weight_alignment": None,
        "true_channel_directly_observable": c["true_channel_directly_observable"],
        "correctness_error": corr_err,
    })
    return rows


def _markdown(all_rows, k):
    lines = [
        "# Variant comparison — attack surfaces & correctness",
        "",
        f"Lift factor k = {k}. Metrics: `norm_corr`/`gram_corr` in [-1,1] "
        "(1.0 = plaintext structure fully preserved = MORE leakage); "
        "`weight_alignment` = left-mask diagonal-match rel err "
        "(0.0 = left mask trivially recovered); `correctness_error` = max abs "
        "(variant C lifted MLP vs lift of plaintext MLP).",
        "",
        "| attack_surface | variant | k | norm_corr | gram_corr | weight_align | "
        "transient_observable | correctness_err |",
        "|---|---|---|---|---|---|---|---|",
    ]
    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x)
    for r in all_rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["attack_surface"], r["variant"], r["lift_factor"],
            fmt(r.get("norm_corr")), fmt(r.get("gram_corr")),
            fmt(r.get("weight_alignment")),
            fmt(r.get("true_channel_directly_observable")),
            fmt(r.get("correctness_error"))))
    lines += [
        "",
        "## Honest reading",
        "",
        "* **A and B are identical on surfaces A & B.** Amulet's `R_bar` lifts "
        "only the transient nonlinear tensor; the squeezed stable state and folded "
        "weights are the same plain right-masked objects, so norm/Gram/weight "
        "leakage is unchanged. A and B differ only on surface C (transient).",
        "* **The token-order-preserving Kronecker lift (variant C) does NOT reduce "
        "norm/Gram leakage on surface A** — `norm_corr`/`gram_corr` stay ≈1.0, "
        "because `||X⊗R|| = ||X||·||R||` and the token-block Gram is "
        "`(XXᵀ)·||R||²`. Only KV-UNSAFE cross-token mixing lowers it.",
        "* Variant C moves the folded LINEAR weights into a lifted `d·k`-dim space "
        "(surface B dims change; the naive same-dim diagonal match is N/A), but "
        "this is an obfuscation of block structure, **not** an information-theoretic "
        "removal of the Gram signal.",
        "* `formal_security_claim = False`, `experiment_only = True`, "
        "`production_qwen7b_integration = False`, attention/KV **not** lifted.",
    ]
    return "\n".join(lines)


def run(args) -> dict:
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    variants = args.variants.split(",")
    per_variant = {}
    all_rows = []
    for v in variants:
        attack = run_variant_attack_surface(
            v, m=args.rows, d=args.hidden, p=args.out_hidden, k=args.lift_k,
            seed=args.seed, dtype=dtype, device=args.device)
        correctness = _correctness_for(
            v, args.model, args.lift_k, args.seq_len, args.layers,
            args.dtype, args.device, args.seed)
        per_variant[v] = {
            "record": variant_report_fields(v, lift_factor=args.lift_k),
            "attack_surfaces": attack,
            "correctness": correctness,
        }
        all_rows.extend(_summary_rows(v, attack, correctness))
    return {"variants": variants, "lift_factor": args.lift_k,
            "per_variant": per_variant, "summary_rows": all_rows}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants",
                   default="right_pad_pure_right,right_pad_amulet_gelu,kronecker_lifted_linear")
    p.add_argument("--model", default="tiny", choices=["tiny", "gpt2", "tiny-gpt2"])
    p.add_argument("--lift-k", type=int, default=2)
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--out-hidden", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="smaller sizes (no-op default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = run(args)
    for v, data in rep["per_variant"].items():
        vd = DEFAULT_OUT / v
        vd.mkdir(parents=True, exist_ok=True)
        (vd / f"comparison_k{args.lift_k}.json").write_text(json.dumps(data, indent=2))
    sdir = DEFAULT_OUT / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"variant_comparison_k{args.lift_k}.json").write_text(
        json.dumps({"lift_factor": args.lift_k, "rows": rep["summary_rows"]}, indent=2))
    (sdir / f"variant_comparison_k{args.lift_k}.md").write_text(
        _markdown(rep["summary_rows"], args.lift_k))
    print(_markdown(rep["summary_rows"], args.lift_k))
    print(f"\nwrote {sdir}/variant_comparison_k{args.lift_k}.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
