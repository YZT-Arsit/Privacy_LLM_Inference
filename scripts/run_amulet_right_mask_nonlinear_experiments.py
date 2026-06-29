#!/usr/bin/env python
"""Amulet-style right-mask nonlinear island experiments (ReLU/GELU/SiLU/SwiGLU + Qwen MLP).

This is a *nonlinear-island experiment*, NOT the production Qwen7B benchmark
unless and until explicitly integrated. Every island maps ``U N -> phi(U) N``
(right-mask only); the sequence dimension is never left-masked in the stable
state, and no activation enters a TEE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.ops.amulet_right_mask_islands import (  # noqa: E402
    amulet_right_mask_activation,
    amulet_right_mask_island_report_fields,
    amulet_right_mask_swiglu,
    make_right_mask_amulet_params,
    run_amulet_right_mask_qwen_mlp,
)
from pllo.ops.nonlinear_islands import (  # noqa: E402
    gelu_reference,
    relu_reference,
    silu_reference,
)

_REFS = {"relu": relu_reference, "gelu": gelu_reference, "silu": silu_reference}


def _errors(out: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (out - expected).abs()
    max_abs = float(diff.max().item())
    rel_l2 = float(
        (torch.linalg.norm(out - expected) / (torch.linalg.norm(expected) + 1e-30)).item()
    )
    return max_abs, rel_l2


def _invertible(dim: int, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    M = torch.randn(dim, dim, dtype=dtype, generator=g)
    while abs(float(torch.linalg.det(M).item())) < 1e-3:
        M = torch.randn(dim, dim, dtype=dtype, generator=g)
    return M


def run(args: argparse.Namespace) -> dict:
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    g = torch.Generator()
    g.manual_seed(args.seed)
    m, d, f, k = args.batch_tokens, args.hidden_size, args.intermediate_size, args.kronecker_size

    probes: list[dict] = []
    last_rf = None

    # --- single-input activations: U N_ff -> phi(U) N_ff ---
    n_ff = _invertible(d, dtype, g)
    U = torch.randn(m, d, dtype=dtype, generator=g)
    params = make_right_mask_amulet_params(m, d, k, n_ff, generator=g)
    last_rf = params.r_factors
    for act in ("relu", "gelu", "silu"):
        out = amulet_right_mask_activation(U @ n_ff, params, act)
        expected = _REFS[act](U) @ n_ff
        max_abs, rel_l2 = _errors(out, expected)
        probes.append({
            "probe": f"{act}_right_mask_activation",
            "activation": act,
            "shape": [m, d], "k": k,
            "max_abs_error": max_abs, "relative_l2_error": rel_l2,
            "right_mask_output_verified": max_abs <= args.tol,
        })

    # --- SwiGLU: (G N, U N) -> [SiLU(G) * U] N ---
    G = torch.randn(m, d, dtype=dtype, generator=g)
    U2 = torch.randn(m, d, dtype=dtype, generator=g)
    sw_out = amulet_right_mask_swiglu(G @ n_ff, U2 @ n_ff, params)
    sw_exp = (silu_reference(G) * U2) @ n_ff
    max_abs, rel_l2 = _errors(sw_out, sw_exp)
    swiglu_ok = max_abs <= args.tol
    probes.append({
        "probe": "swiglu_right_mask_activation",
        "activation": "swiglu",
        "shape": [m, d], "k": k,
        "max_abs_error": max_abs, "relative_l2_error": rel_l2,
        "right_mask_output_verified": swiglu_ok,
    })

    # --- Qwen-style MLP: X N_in -> ... -> Y N_out ---
    wg = torch.randn(d, f, dtype=dtype, generator=g)
    wu = torch.randn(d, f, dtype=dtype, generator=g)
    wd = torch.randn(f, d, dtype=dtype, generator=g)
    bg = torch.randn(f, dtype=dtype, generator=g)
    bu = torch.randn(f, dtype=dtype, generator=g)
    bd = torch.randn(d, dtype=dtype, generator=g)
    X = torch.randn(m, d, dtype=dtype, generator=g)
    n_in = _invertible(d, dtype, g)
    n_in_inv = torch.linalg.inv(n_in)
    n_ff_mlp = _invertible(f, dtype, g)
    n_out = _invertible(d, dtype, g)
    mlp = run_amulet_right_mask_qwen_mlp(
        X, wg, bg, wu, bu, wd, bd, n_in, n_in_inv, n_ff_mlp, n_out,
        k=k, generator=g,
    )
    mlp_abs, mlp_rel = _errors(mlp["y_tilde"], mlp["expected_y_tilde"])
    rec_abs, _ = _errors(mlp["y_recovered"], mlp["y_plain"])
    last_rf = mlp["params"].r_factors
    probes.append({
        "probe": "qwen_style_mlp",
        "activation": "swiglu_mlp",
        "shape": [m, d], "intermediate": f, "k": k,
        "max_abs_error": mlp_abs, "relative_l2_error": mlp_rel,
        "recover_max_abs_error": rec_abs,
        "right_mask_output_verified": mlp_abs <= args.tol,
        "pad_enters_nonlinear_island": mlp["metadata"]["pad_enters_nonlinear_island"],
    })

    overall_max = max(p["max_abs_error"] for p in probes)
    audit = amulet_right_mask_island_report_fields(
        last_rf, max_abs_error=overall_max,
        relative_l2_error=max(p["relative_l2_error"] for p in probes),
        swiglu_verified=swiglu_ok,
    )
    return {
        "experiment": "amulet_right_mask_nonlinear_island",
        "note": "Nonlinear-island experiment only; NOT a full Qwen7B production "
                "benchmark unless later integrated.",
        "dtype": args.dtype,
        "tolerance": args.tol,
        "config": {
            "batch_tokens": m, "hidden_size": d,
            "intermediate_size": f, "kronecker_size": k, "seed": args.seed,
        },
        "all_probes_passed": all(p["right_mask_output_verified"] for p in probes),
        "max_abs_error_overall": overall_max,
        "audit": audit,
        "probes": probes,
    }


def _markdown(report: dict) -> str:
    a = report["audit"]
    lines = [
        "# Amulet right-mask nonlinear island experiments",
        "",
        report["note"],
        "",
        "## Construction",
        "",
        "For decoder-only generation we keep a right-mask stable invariant "
        "`H_tilde = H N`. Each nonlinear island instantiates an Amulet-style "
        "lift/shuffle/squeeze with `P = I`, `Q = N`:",
        "",
        "```",
        "Z = M1 (U_tilde (x) R2) M2 = pi3 ((pi1 U pi2) (x) R_bar) pi4",
        "S = phi(Z)",
        "out_tilde = M3 S M4 = phi(U) N",
        "M1 = pi3 (pi1 (x) R1)        M2 = (N^-1 pi2 (x) R3) pi4",
        "M3 = pi1^T E1 pi3^T          M4 = pi4^T E2 pi2^T N",
        "R_bar = R1 R2 R3, with exactly one secret entry R_bar[a,b] = 1.",
        "```",
        "",
        "The unit entry makes the squeeze `E1 . E2` select the true nonlinear "
        "value; for SwiGLU both branches share the schedule and the shared "
        "unit-copy is selected after `SiLU(gate) * up`.",
        "",
        "## Max error table",
        "",
        "| probe | activation | k | max_abs_error | relative_l2_error | verified |",
        "|---|---|---|---|---|---|",
    ]
    for p in report["probes"]:
        lines.append(
            f"| {p['probe']} | {p['activation']} | {p['k']} | "
            f"{p['max_abs_error']:.3e} | {p['relative_l2_error']:.3e} | "
            f"{'yes' if p['right_mask_output_verified'] else 'NO'} |"
        )
    lines += [
        "",
        "## R_bar unique-one audit",
        "",
        f"- `rbar_shape`: {a['rbar_shape']}",
        f"- `rbar_dense_single_one`: {a['rbar_dense_single_one']}",
        f"- `rbar_has_unique_one`: {a['rbar_has_unique_one']}",
        f"- `rbar_other_entries_not_one`: {a['rbar_other_entries_not_one']}",
        f"- `r_factor_product_verified`: {a['r_factor_product_verified']}",
        f"- `selected_coordinate_public`: {a['selected_coordinate_public']} "
        "(the secret coordinate (a,b) is never published)",
        "",
        "## No-TEE-boundary / right-mask audit",
        "",
        f"- `stable_state_invariant`: `{a['stable_state_invariant']}`",
        f"- `uses_left_sequence_mask`: {a['uses_left_sequence_mask']}",
        f"- `intermediate_tee_boundary_calls`: {a['intermediate_tee_boundary_calls']}",
        f"- `nonlinear_executed_on_gpu`: {a['nonlinear_executed_on_gpu']}",
        f"- `activation_supported`: {a['activation_supported']}",
        f"- `raw_rbar_visible_to_gpu`: {a['raw_rbar_visible_to_gpu']}",
        f"- `raw_n_visible_to_gpu`: {a['raw_n_visible_to_gpu']}",
        f"- `pad_enters_nonlinear_island`: {a['pad_enters_nonlinear_island']}",
        f"- `nonlinear_island_input_form`: `{a['nonlinear_island_input_form']}`",
        f"- `nonlinear_island_output_form`: `{a['nonlinear_island_output_form']}`",
        f"- `swiglu_verified`: {a['swiglu_verified']}",
        "",
        f"**All probes passed:** {report['all_probes_passed']} "
        f"(overall max abs error {report['max_abs_error_overall']:.3e}, "
        f"dtype {report['dtype']}).",
        "",
        "> Limitation: we do not claim arbitrary dense right masks commute with "
        "GELU/SiLU/SwiGLU. Correctness relies on the lift/shuffle/squeeze "
        "construction and the unique-one selected coordinate, which is assumed "
        "to be unidentifiable inside the shuffled Kronecker-expanded space.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--intermediate-size", type=int, default=32)
    p.add_argument("--batch-tokens", type=int, default=4)
    p.add_argument("--kronecker-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-7)
    p.add_argument(
        "--output-json", type=Path,
        default=PROJECT_ROOT / "outputs" / "amulet_right_mask_nonlinear_experiments.json",
    )
    p.add_argument(
        "--output-md", type=Path,
        default=PROJECT_ROOT / "outputs" / "amulet_right_mask_nonlinear_experiments.md",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_md.write_text(_markdown(report))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if not report["all_probes_passed"]:
        raise SystemExit("FAILED: not all right-mask island probes verified")


if __name__ == "__main__":
    main()
