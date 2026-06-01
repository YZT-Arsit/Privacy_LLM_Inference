#!/usr/bin/env python
"""Stage 7.0 — LoRA private training prototype runner.

Drives :func:`pllo.experiments.lora_training_probe.run_lora_training_probe`
and emits ``outputs/lora_training_experiments.{json,csv,md}``.

The default config is small and fully synthetic; pytest never reads from
the network. Reports publish summary metrics + fingerprints only. Private
data, raw adapter tensors, optimizer state, and masks are never exported.
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

from pllo.experiments.lora_training_probe import (  # noqa: E402
    LoRATrainingProbeConfig,
    VALID_OPTIMIZERS,
    run_lora_training_probe,
    training_probe_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--optimizer", choices=list(VALID_OPTIMIZERS), default="sgd",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--fresh-u-per-step", action="store_true", default=True)
    p.add_argument(
        "--no-fresh-u-per-step", dest="fresh_u_per_step", action="store_false",
    )
    p.add_argument("--fresh-masks-per-step", action="store_true", default=True)
    p.add_argument(
        "--no-fresh-masks-per-step",
        dest="fresh_masks_per_step", action="store_false",
    )
    p.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    tc = report["training_step_correctness"]
    gr = report["gradient_and_optimizer_handling"]
    pc = report["pad_compensation"]

    lines: list[str] = []
    lines.append("# Stage 7.0 — LoRA Private Training Prototype\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- Tiny LoRA-augmented linear with d_in={cfg['d_in']}, d_out={cfg['d_out']},"
        f" rank={cfg['rank']}, alpha={cfg['alpha']}, batch_size={cfg['batch_size']}."
    )
    lines.append(
        f"- Optimizer = {cfg['optimizer']}, lr={cfg['lr']}, num_steps={cfg['num_steps']}."
    )
    lines.append(
        f"- use_pad={cfg['use_pad']}, fresh_u_per_step={cfg['fresh_u_per_step']},"
        f" fresh_masks_per_step={cfg['fresh_masks_per_step']}, dtype={cfg['dtype']}."
    )
    lines.append(
        "- Synthetic private data; no network access; no PEFT integration.\n"
    )

    lines.append("## 2. LoRA Linear Masking Formula\n")
    lines.append("```")
    lines.append("Plain:  Y = X W + (alpha / r) X A B + bias")
    lines.append("Masked: X_tilde     = (X - T_in) N_in    (or X N_in when use_pad=False)")
    lines.append("        W_tilde     = N_in^{-1} W N_out")
    lines.append("        A_tilde     = N_in^{-1} A U")
    lines.append("        B_tilde     = U^{-1}   B N_out")
    lines.append("        bias_tilde  = bias N_out")
    lines.append("        C_W         = T_in W N_out")
    lines.append("        C_LoRA      = (alpha / r) T_in A B N_out")
    lines.append("        Y_tilde     = X_tilde W_tilde")
    lines.append("                    + (alpha / r) (X_tilde A_tilde) B_tilde")
    lines.append("                    + bias_tilde + C_W + C_LoRA")
    lines.append("        Y_recovered = Y_tilde N_out^{-1}")
    lines.append("```\n")
    lines.append("LoRA adapter is NEVER merged into the public base weight W.\n")

    lines.append("## 3. Forward Correctness\n")
    lines.append("| step | loss_plain | loss_masked | loss_diff | forward_err |")
    lines.append("|------|-----------|-------------|-----------|-------------|")
    for r in tc["per_step"]:
        lines.append(
            f"| {r['step']} | {r['loss_plain']:.6e}"
            f" | {r['loss_masked']:.6e} | {r['loss_diff_abs']:.3e}"
            f" | {r['forward_max_abs_err']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max loss diff: {tc['max_loss_diff']:.3e}")
    lines.append(f"- final output err: {tc['final_output_err']:.3e}")
    lines.append(f"- allclose: **{tc['allclose']}**\n")

    lines.append("## 4. Training-Step Correctness\n")
    lines.append("| step | grad_A_err | grad_B_err | adapter_A_update_err | adapter_B_update_err |")
    lines.append("|------|-----------|-----------|---------------------|---------------------|")
    for r in tc["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['grad_a_max_abs_err']:.3e}"
            f" | {r['grad_b_max_abs_err']:.3e}"
            f" | {r['adapter_a_update_max_abs_err']:.3e}"
            f" | {r['adapter_b_update_max_abs_err']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max grad_A err: {tc['max_grad_a_err']:.3e}")
    lines.append(f"- max grad_B err: {tc['max_grad_b_err']:.3e}")
    lines.append(f"- final adapter_A update err: {tc['final_adapter_a_update_err']:.3e}")
    lines.append(f"- final adapter_B update err: {tc['final_adapter_b_update_err']:.3e}\n")

    lines.append("## 5. Gradient / Optimizer Handling\n")
    lines.append("| variable | visible_to_gpu |")
    lines.append("|----------|----------------|")
    for k, v in gr["gpu_visibility"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(f"- backward_location: **{gr['backward_location']}**")
    lines.append(f"- optimizer_state_location: **{gr['optimizer_state_location']}**")
    lines.append(f"- adapter_location: **{gr['adapter_location']}**")
    lines.append(f"- merge_adapter_into_w: **{gr['merge_adapter_into_w']}**")
    lines.append(f"- trusted_backward_status: {gr['trusted_backward_status']}")
    lines.append(f"- masked_backward_status: {gr['masked_backward_status']}\n")

    lines.append("## 6. Pad Compensation\n")
    lines.append(f"- use_pad: **{pc['use_pad']}**")
    lines.append(f"- pad_scale: {pc['pad_scale']}")
    lines.append(f"- compensation_formula: `{pc['compensation_formula']}`")
    lines.append(f"- trusted_only: {pc['is_trusted_only']}")
    lines.append(f"- forward_err_under_pad: {pc['forward_err_under_pad']:.3e}\n")

    lines.append("## 7. Limitations\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## 8. Next Stage Plan\n")
    for plan in report["next_stage_plan"]:
        lines.append(f"- {plan}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LoRATrainingProbeConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        d_in=args.d_in,
        d_out=args.d_out,
        rank=args.rank,
        alpha=args.alpha,
        num_steps=args.num_steps,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        use_pad=args.use_pad,
        fresh_u_per_step=args.fresh_u_per_step,
        fresh_masks_per_step=args.fresh_masks_per_step,
        dtype=args.dtype,
        device=args.device,
    )
    report = run_lora_training_probe(cfg)
    json_path = args.output_dir / "lora_training_experiments.json"
    csv_path = args.output_dir / "lora_training_experiments.csv"
    md_path = args.output_dir / "lora_training_experiments.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = training_probe_csv_rows(report)
    fieldnames = ["section", "step", "metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    tc = report["training_step_correctness"]
    print(
        f"allclose={tc['allclose']} max_loss_diff={tc['max_loss_diff']:.3e}"
        f" max_grad_a_err={tc['max_grad_a_err']:.3e}"
        f" max_grad_b_err={tc['max_grad_b_err']:.3e}"
        f" final_a_err={tc['final_adapter_a_update_err']:.3e}"
        f" final_b_err={tc['final_adapter_b_update_err']:.3e}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
