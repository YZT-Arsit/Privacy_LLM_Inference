#!/usr/bin/env python
"""Stage 7.3 — multi-layer LoRA end-to-end training runner.

Drives
:func:`pllo.experiments.multilayer_lora_training.run_multilayer_lora_training`
and writes ``outputs/multilayer_lora_training_experiments.{json,csv,md}``.
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

from pllo.experiments.multilayer_lora_training import (  # noqa: E402
    MultiLayerLoRATrainingConfig,
    VALID_LORA_TARGETS,
    VALID_OPTIMIZERS,
    multilayer_lora_training_csv_rows,
    run_multilayer_lora_training,
)
from pllo.ops.lora_rank_padding import VALID_DUMMY_STRATEGIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--intermediate-size", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--optimizer", choices=list(VALID_OPTIMIZERS), default="sgd")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--fresh-u-per-step", action="store_true", default=True)
    p.add_argument(
        "--no-fresh-u-per-step", dest="fresh_u_per_step", action="store_false",
    )
    p.add_argument(
        "--dummy-strategy",
        choices=list(VALID_DUMMY_STRATEGIES),
        default="paired_cancellation_dummy",
    )
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument(
        "--lora-targets",
        nargs="+",
        default=list(VALID_LORA_TARGETS),
        choices=list(VALID_LORA_TARGETS),
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    spec = report["model_spec"]
    tc = report["training_correctness"]
    rp = report["rank_padding_summary"]
    op = report["optimizer_summary"]
    pl = report["per_layer_metrics"]

    lines: list[str] = []
    lines.append("# Stage 7.3 — Multi-Layer LoRA End-to-End Training\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- num_layers={cfg['num_layers']}, hidden_size={cfg['hidden_size']},"
        f" intermediate_size={cfg['intermediate_size']},"
        f" vocab_size={cfg['vocab_size']}."
    )
    lines.append(
        f"- batch_size={cfg['batch_size']}, seq_len={cfg['seq_len']},"
        f" alpha={cfg['alpha']}, true_rank={cfg['true_rank']},"
        f" padded_rank={cfg['padded_rank']}."
    )
    lines.append(
        f"- num_steps={cfg['num_steps']}, optimizer={cfg['optimizer']},"
        f" lr={cfg['lr']}, use_pad={cfg['use_pad']},"
        f" fresh_u_per_step={cfg['fresh_u_per_step']},"
        f" dummy_strategy={cfg['dummy_strategy']!r}, dtype={cfg['dtype']}."
    )
    lines.append(
        "- Synthetic private data + frozen public base weights;"
        " no network access; no PEFT integration.\n"
    )

    lines.append("## 2. Tiny Multi-Layer LoRA Model\n")
    lines.append(f"- num_layers: {spec['num_layers']}")
    lines.append(f"- hidden_size: {spec['hidden_size']}")
    lines.append(f"- intermediate_size: {spec['intermediate_size']}")
    lines.append(f"- modules_per_layer: {spec['modules_per_layer']}")
    lines.append(f"- total_lora_modules: {spec['total_lora_modules']}")
    lines.append(f"- lora_targets: {spec['lora_targets']}")
    lines.append(
        "- Each layer: q/k/v/o attention-like linears + SwiGLU MLP"
        " (gate, up, down). Base weights frozen; LoRA adapters trainable.\n"
    )

    lines.append("## 3. Multi-Layer Forward Correctness\n")
    lines.append("| step | loss_plain | loss_masked | loss_diff | logits_err | forward_err | dummy_norm |")
    lines.append("|------|------------|-------------|-----------|------------|-------------|------------|")
    for r in tc["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['loss_plain']:.6e}"
            f" | {r['loss_masked']:.6e}"
            f" | {r['loss_diff_abs']:.3e}"
            f" | {r['logits_max_abs_err']:.3e}"
            f" | {r['max_forward_err']:.3e}"
            f" | {r['max_dummy_contribution_norm']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max_loss_diff: {tc['max_loss_diff']:.3e}")
    lines.append(f"- max_forward_err: {tc['max_forward_err']:.3e}")
    lines.append(
        f"- max_dummy_contribution_norm: {tc['max_dummy_contribution_norm']:.3e}"
    )
    lines.append("")

    lines.append("## 4. Multi-Layer Masked Backward Correctness\n")
    lines.append(
        "| step | grad_A_real_err | grad_B_real_err"
        " | adapter_A_update_err | adapter_B_update_err |"
    )
    lines.append(
        "|------|------------------|------------------|----------------------|----------------------|"
    )
    for r in tc["per_step"]:
        lines.append(
            f"| {r['step']}"
            f" | {r['max_grad_a_real_err']:.3e}"
            f" | {r['max_grad_b_real_err']:.3e}"
            f" | {r['max_update_a_err']:.3e}"
            f" | {r['max_update_b_err']:.3e} |"
        )
    lines.append("")
    lines.append(f"- max grad_A real err: {tc['max_grad_a_real_err']:.3e}")
    lines.append(f"- max grad_B real err: {tc['max_grad_b_real_err']:.3e}")
    lines.append(f"- max update_A err: {tc['max_update_a_err']:.3e}")
    lines.append(f"- max update_B err: {tc['max_update_b_err']:.3e}")
    lines.append(f"- allclose: **{tc['allclose']}**\n")

    lines.append("## 5. Rank Padding Across Layers\n")
    lines.append(f"- dummy_strategy_requested: `{rp['dummy_strategy_requested']}`")
    lines.append(f"- true_rank: {rp['true_rank']}")
    lines.append(f"- padded_rank: {rp['padded_rank']}")
    lines.append(f"- lora_targets: {rp['lora_targets']}")
    lines.append(f"- num_lora_modules: {rp['num_lora_modules']}")
    lines.append(
        f"- true_rank_hidden_from_shape: **{rp['true_rank_hidden_from_shape']}**"
    )
    lines.append(f"- padded_rank_visible: **{rp['padded_rank_visible']}**\n")

    lines.append("## 6. Optimizer Handling\n")
    lines.append(f"- location: **{op['location']}**")
    lines.append(f"- optimizer: {op['optimizer']}")
    lines.append(f"- lr: {op['lr']}")
    lines.append(
        f"- any_optimizer_state_contains_dummy:"
        f" **{op['any_optimizer_state_contains_dummy']}**"
    )
    lines.append(
        f"- any_dummy_update_applied: **{op['any_dummy_update_applied']}**"
    )
    lines.append(f"- note: {op['note']}\n")

    lines.append("## 7. Per-Layer Metrics\n")
    lines.append(
        "| layer | module | true_r | pad_r | forward_err | grad_A_err"
        " | grad_B_err | update_A_err | update_B_err | visible_rank | hidden |"
    )
    lines.append(
        "|-------|--------|--------|-------|-------------|------------"
        "|------------|--------------|--------------|--------------|--------|"
    )
    for entry in pl:
        lines.append(
            f"| {entry['layer_index']}"
            f" | {entry['module_name']}"
            f" | {entry['true_rank']}"
            f" | {entry['padded_rank']}"
            f" | {entry['forward_max_abs_err']:.2e}"
            f" | {entry['grad_a_real_max_abs_err']:.2e}"
            f" | {entry['grad_b_real_max_abs_err']:.2e}"
            f" | {entry['update_a_max_abs_err']:.2e}"
            f" | {entry['update_b_max_abs_err']:.2e}"
            f" | {entry['visible_rank_from_a_shape']}"
            f" | {entry['true_rank_hidden_from_shape']} |"
        )
    lines.append("")

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
    cfg = MultiLayerLoRATrainingConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        true_rank=args.true_rank,
        padded_rank=args.padded_rank,
        alpha=args.alpha,
        num_steps=args.num_steps,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        use_pad=args.use_pad,
        fresh_u_per_step=args.fresh_u_per_step,
        dummy_strategy=args.dummy_strategy,
        dummy_scale=args.dummy_scale,
        lora_targets=tuple(args.lora_targets),
        dtype=args.dtype,
        device=args.device,
    )
    report = run_multilayer_lora_training(cfg)
    json_path = args.output_dir / "multilayer_lora_training_experiments.json"
    csv_path = args.output_dir / "multilayer_lora_training_experiments.csv"
    md_path = args.output_dir / "multilayer_lora_training_experiments.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = multilayer_lora_training_csv_rows(report)
    fieldnames = ["section", "scope", "layer", "module", "metric", "value", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    tc = report["training_correctness"]
    op = report["optimizer_summary"]
    rp = report["rank_padding_summary"]
    print(
        f"allclose={tc['allclose']} max_loss_diff={tc['max_loss_diff']:.3e}"
        f" max_grad_a={tc['max_grad_a_real_err']:.3e}"
        f" max_grad_b={tc['max_grad_b_real_err']:.3e}"
        f" max_update_a={tc['max_update_a_err']:.3e}"
        f" max_update_b={tc['max_update_b_err']:.3e}"
        f" num_lora_modules={rp['num_lora_modules']}"
        f" true_rank_hidden_from_shape={rp['true_rank_hidden_from_shape']}"
        f" any_opt_state_contains_dummy={op['any_optimizer_state_contains_dummy']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
