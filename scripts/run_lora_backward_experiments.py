#!/usr/bin/env python
"""Stage 7.1 — LoRA masked-backward correctness runner.

Drives :func:`pllo.experiments.lora_backward_probe.run_lora_backward_probe`
and writes ``outputs/lora_backward_experiments.{json,csv,md}``.
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

from pllo.experiments.lora_backward_probe import (  # noqa: E402
    LoRABackwardProbeConfig,
    VALID_OPTIMIZERS,
    backward_probe_csv_rows,
    run_lora_backward_probe,
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
    p.add_argument("--optimizer", choices=list(VALID_OPTIMIZERS), default="sgd")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--fresh-u-per-step", action="store_true", default=True)
    p.add_argument("--no-fresh-u-per-step", dest="fresh_u_per_step", action="store_false")
    p.add_argument("--fresh-masks-per-step", action="store_true", default=True)
    p.add_argument(
        "--no-fresh-masks-per-step", dest="fresh_masks_per_step",
        action="store_false",
    )
    p.add_argument("--recover-grad-x", action="store_true", default=False)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    mb = report["masked_backward_correctness"]
    fm = report["masked_backward_formula"]
    lh = report["loss_handling"]
    gh = report["gradient_handling"]
    oh = report["optimizer_handling"]
    pc = report["pad_compensation"]

    lines: list[str] = []
    lines.append("# Stage 7.1 — LoRA Masked Backward / Gradient-Side Obfuscation\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- Single LoRA-augmented linear, d_in={cfg['d_in']}, d_out={cfg['d_out']},"
        f" rank={cfg['rank']}, alpha={cfg['alpha']}, batch_size={cfg['batch_size']}."
    )
    lines.append(
        f"- optimizer={cfg['optimizer']}, lr={cfg['lr']}, num_steps={cfg['num_steps']},"
        f" use_pad={cfg['use_pad']}, fresh_u_per_step={cfg['fresh_u_per_step']},"
        f" recover_grad_x={cfg['recover_grad_x']}, dtype={cfg['dtype']}."
    )
    lines.append(
        "- Synthetic private data; no network access; no PEFT integration.\n"
    )

    lines.append("## 2. Masked LoRA Backward Formula\n")
    lines.append("```")
    lines.append(f"Upstream gradient mask: {fm['upstream_gradient_mask']}")
    lines.append(f"chain rule invariance:  {fm['chain_rule_invariance']}")
    lines.append(f"grad_A_tilde            = {fm['grad_a_tilde']}")
    lines.append(f"grad_B_tilde            = {fm['grad_b_tilde']}")
    lines.append(f"grad_X_tilde (optional) = {fm['grad_x_tilde']}")
    lines.append(f"grad_A recovery         = {fm['grad_a_recovery']}")
    lines.append(f"grad_B recovery         = {fm['grad_b_recovery']}")
    lines.append(f"grad_X recovery         = {fm['grad_x_recovery']}")
    lines.append("```\n")

    lines.append("## 3. Upstream Gradient Masking\n")
    lines.append(
        f"- max upstream-gradient invariance error |tr(G^T Y) - tr(G_tilde^T Y_tilde)|"
        f" = {mb['max_upstream_gradient_invariance_err']:.3e}"
    )
    cross = report["autograd_vs_analytic_step0"]
    lines.append(
        "- step-0 autograd vs analytic plain reference:"
        f" grad_A_err={cross['grad_a']:.3e},"
        f" grad_B_err={cross['grad_b']:.3e},"
        f" grad_X_err={cross['grad_x']:.3e}\n"
    )

    lines.append("## 4. Grad-A / Grad-B Recovery\n")
    lines.append(
        "| step | loss_diff | forward_err | grad_A_err | grad_B_err | grad_X_err |"
    )
    lines.append("|------|-----------|-------------|-----------|-----------|-----------|")
    for r in mb["per_step"]:
        gx = (
            f"{r['grad_x_max_abs_err']:.3e}"
            if r["grad_x_max_abs_err"] is not None else "—"
        )
        lines.append(
            f"| {r['step']} | {r['loss_diff_abs']:.3e}"
            f" | {r['forward_max_abs_err']:.3e}"
            f" | {r['grad_a_max_abs_err']:.3e}"
            f" | {r['grad_b_max_abs_err']:.3e} | {gx} |"
        )
    lines.append("")
    lines.append(f"- max grad_A err: {mb['max_grad_a_err']:.3e}")
    lines.append(f"- max grad_B err: {mb['max_grad_b_err']:.3e}")
    if cfg["recover_grad_x"]:
        lines.append(f"- max grad_X err: {mb['max_grad_x_err']:.3e}")
    lines.append(f"- masked_backward_allclose: **{mb['masked_backward_allclose']}**\n")

    lines.append("## 5. Training-Step Correctness\n")
    lines.append("| step | adapter_A_update_err | adapter_B_update_err |")
    lines.append("|------|---------------------|---------------------|")
    for r in mb["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['adapter_a_update_max_abs_err']:.3e}"
            f" | {r['adapter_b_update_max_abs_err']:.3e} |"
        )
    lines.append("")
    lines.append(f"- final adapter_A update err: {mb['final_adapter_a_update_err']:.3e}")
    lines.append(f"- final adapter_B update err: {mb['final_adapter_b_update_err']:.3e}")
    lines.append(f"- final output err: {mb['final_output_err']:.3e}")
    lines.append(f"- allclose: **{mb['allclose']}**\n")

    lines.append("## 6. Optimizer Handling\n")
    lines.append("| variable | visible_to_gpu |")
    lines.append("|----------|----------------|")
    for k, v in gh["gpu_visibility"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(
        f"- loss computation remains trusted ({lh['stage_7_1_status']}): {lh['note']}"
    )
    lines.append(
        f"- backward arithmetic on GPU: {gh['stage_7_1_status']}"
        f" (masked tensors only)."
    )
    lines.append(
        f"- optimizer remains trusted ({oh['stage_7_1_status']},"
        f" optimizer={oh['optimizer']}, lr={oh['lr']}): {oh['note']}"
    )
    lines.append(
        f"- pad compensation (trusted-only): {pc['grad_a_compensation_formula']};"
        f" {pc['grad_b_compensation_formula']}\n"
    )

    lines.append("## 7. Limitations\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## 8. Next Stage Plan\n")
    for plan in report["next_stage_plan"]:
        lines.append(f"- {plan}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LoRABackwardProbeConfig(
        output_dir=str(args.output_dir),
        seed=args.seed, batch_size=args.batch_size,
        d_in=args.d_in, d_out=args.d_out, rank=args.rank,
        alpha=args.alpha, num_steps=args.num_steps, lr=args.lr,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        use_pad=args.use_pad,
        fresh_u_per_step=args.fresh_u_per_step,
        fresh_masks_per_step=args.fresh_masks_per_step,
        recover_grad_x=args.recover_grad_x,
        dtype=args.dtype, device=args.device,
    )
    report = run_lora_backward_probe(cfg)
    json_path = args.output_dir / "lora_backward_experiments.json"
    csv_path = args.output_dir / "lora_backward_experiments.csv"
    md_path = args.output_dir / "lora_backward_experiments.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = backward_probe_csv_rows(report)
    fieldnames = ["section", "step", "metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    mb = report["masked_backward_correctness"]
    print(
        f"allclose={mb['allclose']} masked_backward_allclose={mb['masked_backward_allclose']}"
        f" max_loss_diff={mb['max_loss_diff']:.3e}"
        f" max_grad_a_err={mb['max_grad_a_err']:.3e}"
        f" max_grad_b_err={mb['max_grad_b_err']:.3e}"
        f" max_invariance_err={mb['max_upstream_gradient_invariance_err']:.3e}"
        f" final_a_err={mb['final_adapter_a_update_err']:.3e}"
        f" final_b_err={mb['final_adapter_b_update_err']:.3e}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
