#!/usr/bin/env python
"""Stage 7.2 — LoRA rank-padded correctness runner.

Drives :func:`pllo.experiments.lora_rank_padding_probe.run_lora_rank_padding_probe`
and writes ``outputs/lora_rank_padding_experiments.{json,csv,md}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.lora_rank_padding_probe import (  # noqa: E402
    LoRARankPaddingProbeConfig,
    VALID_OPTIMIZERS,
    rank_padding_csv_rows,
    run_lora_rank_padding_probe,
)
from pllo.ops.lora_rank_padding import VALID_DUMMY_STRATEGIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--optimizer", choices=list(VALID_OPTIMIZERS), default="sgd")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--fresh-u-per-step", action="store_true", default=True)
    p.add_argument("--no-fresh-u-per-step", dest="fresh_u_per_step", action="store_false")
    p.add_argument(
        "--dummy-strategy",
        choices=list(VALID_DUMMY_STRATEGIES),
        default="paired_cancellation_dummy",
    )
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument("--recover-grad-x", action="store_true", default=False)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    rp = report["rank_padding_correctness"]
    ds = report["dummy_rank_strategy"]
    oh = report["optimizer_handling"]
    sh = report["shape_level_rank_hiding"]

    lines: list[str] = []
    lines.append("# Stage 7.2 — LoRA Rank Padding / Hidden-Rank Prototype\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- Single LoRA-augmented linear with d_in={cfg['d_in']}, d_out={cfg['d_out']}."
    )
    lines.append(
        f"- true_rank={cfg['true_rank']}, padded_rank={cfg['padded_rank']}, alpha={cfg['alpha']},"
        f" batch_size={cfg['batch_size']}."
    )
    lines.append(
        f"- optimizer={cfg['optimizer']}, lr={cfg['lr']}, num_steps={cfg['num_steps']},"
        f" use_pad={cfg['use_pad']}, fresh_u_per_step={cfg['fresh_u_per_step']},"
        f" dummy_strategy={cfg['dummy_strategy']!r}, dtype={cfg['dtype']}."
    )
    lines.append(
        "- Synthetic private data; no network access; no PEFT integration.\n"
    )

    lines.append("## 2. Rank Padding Formula\n")
    lines.append("```")
    lines.append("A_pad ∈ R^{d_in × r_pad},     A_pad[:, :r] = A_real")
    lines.append("B_pad ∈ R^{r_pad × d_out},    B_pad[:r, :] = B_real")
    lines.append("Dummy slice:  A_pad[:, r:r_pad],  B_pad[r:r_pad, :]")
    lines.append("Invariant:    A_pad B_pad = A_real B_real")
    lines.append("LoRA scale:   alpha / true_rank   (NOT alpha / padded_rank)")
    lines.append("")
    lines.append("Masked forward (rank dim = r_pad):")
    lines.append("    Y_tilde = X_tilde W_tilde + (alpha / r) (X_tilde A_pad_tilde) B_pad_tilde")
    lines.append("              + bias_tilde + pad_compensation")
    lines.append("")
    lines.append("Masked backward (rank dim = r_pad):")
    lines.append("    grad_A_pad_tilde = (alpha/r) X_tilde^T (G_tilde B_pad_tilde^T)")
    lines.append("    grad_B_pad_tilde = (alpha/r) (X_tilde A_pad_tilde)^T G_tilde")
    lines.append("")
    lines.append("Trusted side recovery + real-slice extraction:")
    lines.append("    grad_A_pad = N_in^{-T} grad_A_pad_tilde U_pad^T (+ pad compensation)")
    lines.append("    grad_B_pad = U_pad^{-T} grad_B_pad_tilde N_out^T (+ pad compensation)")
    lines.append("    grad_A_real = grad_A_pad[:, :true_rank]")
    lines.append("    grad_B_real = grad_B_pad[:true_rank, :]")
    lines.append("```\n")

    lines.append("## 3. Dummy Rank Strategy\n")
    lines.append(f"- requested: `{ds['requested']}`")
    lines.append(f"- effective: `{ds['effective']}`")
    lines.append(f"- dummy_size: {ds['dummy_size']}")
    lines.append(f"- dummy_scale: {ds['dummy_scale']}")
    lines.append(f"- fresh_dummy_per_step: {ds['fresh_dummy_per_step']}")
    lines.append(
        f"- max dummy contribution norm across steps:"
        f" {ds['max_dummy_contribution_norm']:.3e}"
    )
    lines.append("")

    lines.append("## 4. Forward Correctness\n")
    lines.append(
        "| step | loss_plain | loss_padded | loss_diff | forward_err"
        " | dummy_contribution_norm |"
    )
    lines.append("|------|-----------|-------------|-----------|-------------|-------------------------|")
    for r in rp["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['loss_plain']:.6e}"
            f" | {r['loss_padded']:.6e}"
            f" | {r['loss_diff_abs']:.3e}"
            f" | {r['forward_max_abs_err']:.3e}"
            f" | {r['dummy_contribution_norm']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max loss diff: {rp['max_loss_diff']:.3e}")
    lines.append(
        f"- max dummy contribution norm: {rp['max_dummy_contribution_norm']:.3e}"
    )
    lines.append("")

    lines.append("## 5. Backward Correctness\n")
    lines.append(
        "| step | grad_A_real_err | grad_B_real_err"
        " | adapter_A_update_err | adapter_B_update_err |"
    )
    lines.append(
        "|------|-----------------|-----------------|----------------------|----------------------|"
    )
    for r in rp["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['grad_a_real_max_abs_err']:.3e}"
            f" | {r['grad_b_real_max_abs_err']:.3e}"
            f" | {r['adapter_a_update_max_abs_err']:.3e}"
            f" | {r['adapter_b_update_max_abs_err']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max grad_A real err: {rp['max_grad_a_real_err']:.3e}")
    lines.append(f"- max grad_B real err: {rp['max_grad_b_real_err']:.3e}")
    lines.append(
        f"- final adapter_A update err: {rp['final_adapter_a_update_err']:.3e}"
    )
    lines.append(
        f"- final adapter_B update err: {rp['final_adapter_b_update_err']:.3e}"
    )
    lines.append(f"- allclose: **{rp['allclose']}**\n")

    lines.append("## 6. Optimizer Handling\n")
    lines.append(f"- location: **{oh['location']}**")
    lines.append(f"- optimizer: {oh['optimizer']}")
    lines.append(f"- trainable_adapter_shape_a: {oh['trainable_adapter_shape_a']}")
    lines.append(f"- trainable_adapter_shape_b: {oh['trainable_adapter_shape_b']}")
    lines.append(f"- optimizer_state_shape_a: {oh['optimizer_state_shape_a']}")
    lines.append(f"- optimizer_state_shape_b: {oh['optimizer_state_shape_b']}")
    lines.append(
        f"- optimizer_state_contains_dummy: **{oh['optimizer_state_contains_dummy']}**"
    )
    lines.append(f"- dummy_update_applied: **{oh['dummy_update_applied']}**")
    lines.append(f"- note: {oh['note']}\n")

    lines.append("## 7. Shape-Level Rank Hiding\n")
    lines.append(f"- visible A_tilde_pad shape: {sh['a_tilde_pad_shape']}")
    lines.append(f"- visible B_tilde_pad shape: {sh['b_tilde_pad_shape']}")
    lines.append(
        f"- visible_rank_from_a_shape: **{sh['visible_rank_from_a_shape']}**"
    )
    lines.append(
        f"- visible_rank_from_b_shape: **{sh['visible_rank_from_b_shape']}**"
    )
    lines.append(
        f"- true_rank_hidden_from_shape: **{sh['true_rank_hidden_from_shape']}**"
    )
    lines.append(f"- padded_rank_visible: **{sh['padded_rank_visible']}**")
    lines.append(f"- note: {sh['note']}\n")

    lines.append("## 8. Limitations\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## 9. Next Stage Plan\n")
    for plan in report["next_stage_plan"]:
        lines.append(f"- {plan}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LoRARankPaddingProbeConfig(
        output_dir=str(args.output_dir),
        seed=args.seed, batch_size=args.batch_size,
        d_in=args.d_in, d_out=args.d_out,
        true_rank=args.true_rank, padded_rank=args.padded_rank,
        alpha=args.alpha, num_steps=args.num_steps, lr=args.lr,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        use_pad=args.use_pad, fresh_u_per_step=args.fresh_u_per_step,
        dummy_strategy=args.dummy_strategy, dummy_scale=args.dummy_scale,
        recover_grad_x=args.recover_grad_x,
        dtype=args.dtype, device=args.device,
    )
    report = run_lora_rank_padding_probe(cfg)
    json_path = args.output_dir / "lora_rank_padding_experiments.json"
    csv_path = args.output_dir / "lora_rank_padding_experiments.csv"
    md_path = args.output_dir / "lora_rank_padding_experiments.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = rank_padding_csv_rows(report)
    fieldnames = ["section", "step", "metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    rp = report["rank_padding_correctness"]
    sh = report["shape_level_rank_hiding"]
    oh = report["optimizer_handling"]
    print(
        f"allclose={rp['allclose']} max_loss_diff={rp['max_loss_diff']:.3e}"
        f" max_grad_a_err={rp['max_grad_a_real_err']:.3e}"
        f" max_grad_b_err={rp['max_grad_b_real_err']:.3e}"
        f" dummy_norm={rp['max_dummy_contribution_norm']:.3e}"
        f" visible_rank_from_shape={sh['visible_rank_from_a_shape']}"
        f" optimizer_state_contains_dummy={oh['optimizer_state_contains_dummy']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
