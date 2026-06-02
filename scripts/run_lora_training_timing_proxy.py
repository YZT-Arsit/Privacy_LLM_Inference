#!/usr/bin/env python
"""Stage 7.3 — LoRA training timing side-channel proxy runner.

Drives
:func:`pllo.experiments.lora_training_timing_proxy.run_lora_training_timing_proxy`
and writes ``outputs/lora_training_timing_proxy.{json,csv,md}``.
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

from pllo.experiments.lora_training_timing_proxy import (  # noqa: E402
    LoRATrainingTimingProxyConfig,
    VALID_CONSTANT_TIME_MODES,
    lora_training_timing_proxy_csv_rows,
    run_lora_training_timing_proxy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--true-ranks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--padded-ranks", type=int, nargs="+", default=[8, 16])
    p.add_argument(
        "--num-lora-modules", type=int, nargs="+", default=[2, 4, 7, 14],
    )
    p.add_argument(
        "--optimizers", nargs="+", default=["sgd", "adamw"],
        choices=["sgd", "adamw"],
    )
    p.add_argument("--timing-noise-std", type=float, default=0.05)
    p.add_argument(
        "--constant-time-training-mode",
        choices=list(VALID_CONSTANT_TIME_MODES),
        default="proxy_equalized",
    )
    p.add_argument("--samples-per-config", type=int, default=8)
    p.add_argument("--base-hidden", type=int, default=64)
    p.add_argument("--base-intermediate", type=int, default=128)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Stage 7.3 — LoRA Training Timing Side-Channel Proxy\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- batch_sizes={cfg['batch_sizes']}, seq_lens={cfg['seq_lens']},"
        f" true_ranks={cfg['true_ranks']}, padded_ranks={cfg['padded_ranks']}."
    )
    lines.append(
        f"- num_lora_modules={cfg['num_lora_modules']},"
        f" optimizers={cfg['optimizers']}."
    )
    lines.append(
        f"- timing_noise_std={cfg['timing_noise_std']},"
        f" samples_per_config={cfg['samples_per_config']},"
        f" base_hidden={cfg['base_hidden']},"
        f" base_intermediate={cfg['base_intermediate']}."
    )
    lines.append(
        f"- constant_time_training_mode="
        f"`{cfg['constant_time_training_mode']}`."
    )
    lines.append(f"- scope: {report['scope']}\n")

    lines.append("## 2. Training Timing Proxy Model\n")
    ttm = report["training_timing_model"]
    lines.append(
        "Total per-step latency proxy:"
        "  ```"
        "  latency = base_overhead_ms"
        "          + forward_ms"
        "          + backward_ms"
        "          + optimizer_ms"
        "          + mask_generation_ms"
        "          + boundary_ms"
        "          + rank_padding_dummy_ms"
        "          + Gaussian timing_noise(std=timing_noise_std) * total"
        "  ```"
    )
    lines.append("Cost-model constants:")
    for k, v in ttm["cost_model_constants"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"- num_samples_default: {ttm['num_samples_default']}")
    lines.append(
        f"- num_samples_rank_padding_off: {ttm['num_samples_rank_padding_off']}"
    )
    lines.append(f"- num_samples_zero_dummy: {ttm['num_samples_zero_dummy']}")
    lines.append(
        f"- num_samples_paired_dummy: {ttm['num_samples_paired_dummy']}"
    )
    lines.append(f"- note: {ttm['note']}\n")

    lines.append("## 3. Leakage Tasks\n")
    lines.append("Constant-time mode: **off**\n")
    lines.append(
        "| task | accuracy | chance | bucket_separation | risk |"
    )
    lines.append(
        "|------|----------|--------|--------------------|------|"
    )
    for task, payload in report["leakage_tasks_off"].items():
        lines.append(
            f"| {task}"
            f" | {payload['classification_accuracy']:.3f}"
            f" | {payload['random_chance_baseline']:.3f}"
            f" | {payload['bucket_separation']:.3f}"
            f" | {payload['risk_level']} |"
        )
    lines.append("")
    if report["leakage_tasks_proxy_equalized"]:
        lines.append("Constant-time mode: **proxy_equalized**\n")
        lines.append(
            "| task | accuracy | chance | bucket_separation | risk |"
        )
        lines.append(
            "|------|----------|--------|--------------------|------|"
        )
        for task, payload in report["leakage_tasks_proxy_equalized"].items():
            lines.append(
                f"| {task}"
                f" | {payload['classification_accuracy']:.3f}"
                f" | {payload['random_chance_baseline']:.3f}"
                f" | {payload['bucket_separation']:.3f}"
                f" | {payload['risk_level']} |"
            )
        lines.append("")

    lines.append("## 4. Constant-Time Training Proxy\n")
    cttp = report["constant_time_training_proxy"]
    lines.append(
        f"- constant_time_training_mode:"
        f" `{cttp['constant_time_training_mode']}`"
    )
    lines.append(f"- did_actually_sleep: **{cttp['did_actually_sleep']}**")
    for k, v in cttp["upper_bucket"].items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- note: {cttp['note']}\n")

    lines.append("## 5. Overhead Estimate\n")
    op = report["overhead_proxy"]
    lines.append(f"- mean_native_latency_ms: {op['mean_native_latency_ms']:.4f}")
    lines.append(f"- upper_latency_ms: {op['upper_latency_ms']:.4f}")
    lines.append(f"- overhead_ratio: {op['overhead_ratio']:.4f}")
    lines.append(f"- overhead_pct: {op['overhead_pct']:.2f}%")
    lines.append(f"- note: {op['note']}\n")

    lines.append("## 6. Limitations\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## 7. Next Stage Plan\n")
    for plan in report["next_stage_plan"]:
        lines.append(f"- {plan}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = LoRATrainingTimingProxyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_sizes=tuple(args.batch_sizes),
        seq_lens=tuple(args.seq_lens),
        true_ranks=tuple(args.true_ranks),
        padded_ranks=tuple(args.padded_ranks),
        num_lora_modules=tuple(args.num_lora_modules),
        optimizers=tuple(args.optimizers),
        timing_noise_std=args.timing_noise_std,
        constant_time_training_mode=args.constant_time_training_mode,
        samples_per_config=args.samples_per_config,
        base_hidden=args.base_hidden,
        base_intermediate=args.base_intermediate,
        dtype=args.dtype,
        device=args.device,
    )
    report = run_lora_training_timing_proxy(cfg)
    json_path = args.output_dir / "lora_training_timing_proxy.json"
    csv_path = args.output_dir / "lora_training_timing_proxy.csv"
    md_path = args.output_dir / "lora_training_timing_proxy.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = lora_training_timing_proxy_csv_rows(report)
    fieldnames = ["section", "attack", "strategy", "metric", "value", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    summary = report["summary"]
    print(
        f"constant_time_training_mode={cfg.constant_time_training_mode!r}"
        f" max_acc_off={summary['max_classification_accuracy_off']:.3f}"
        f" max_acc_eq={summary['max_classification_accuracy_proxy_equalized']}"
        f" leakage_reduction={summary['leakage_reduction_after_equalization']}"
        f" overhead_pct={report['overhead_proxy']['overhead_pct']:.2f}"
        f" lora_training_timing_proxy_status={report['lora_training_timing_proxy_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
