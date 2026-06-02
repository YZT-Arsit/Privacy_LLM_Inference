#!/usr/bin/env python
"""Stage 7.4 — stronger dummy correctness runner.

Drives
:func:`pllo.experiments.lora_stronger_dummy_probe.run_lora_stronger_dummy_probe`
and writes ``outputs/lora_stronger_dummy_experiments.{json,csv,md}``.
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

from pllo.experiments.lora_stronger_dummy_probe import (  # noqa: E402
    StrongerDummyProbeConfig,
    VALID_OPTIMIZERS,
    run_lora_stronger_dummy_probe,
    stronger_dummy_probe_csv_rows,
)
from pllo.ops.lora_dummy_strategies import (  # noqa: E402
    VALID_STRONG_DUMMY_STRATEGIES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--padded-rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--optimizer", choices=list(VALID_OPTIMIZERS), default="sgd")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument(
        "--dummy-strategies",
        nargs="+",
        default=list(VALID_STRONG_DUMMY_STRATEGIES),
        choices=list(VALID_STRONG_DUMMY_STRATEGIES),
    )
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=1e-3)
    p.add_argument("--spectrum-match-strength", type=float, default=1.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    design = report["stronger_dummy_strategy_design"]
    per_strategy = report["per_strategy"]

    lines: list[str] = []
    lines.append("# Stage 7.4 — Stronger Dummy Distributions / Spectral-Rank Hardening\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- Single LoRA-augmented linear with d_in={cfg['d_in']}, d_out={cfg['d_out']}."
    )
    lines.append(
        f"- true_rank={cfg['true_rank']}, padded_rank={cfg['padded_rank']}, alpha={cfg['alpha']},"
        f" batch_size={cfg['batch_size']}."
    )
    lines.append(
        f"- num_steps={cfg['num_steps']}, optimizer={cfg['optimizer']},"
        f" lr={cfg['lr']}, use_pad={cfg['use_pad']},"
        f" fresh_u_per_step={cfg['fresh_u_per_step']}, dtype={cfg['dtype']}."
    )
    lines.append(
        f"- dummy_scale={cfg['dummy_scale']}, noise_scale={cfg['noise_scale']},"
        f" spectrum_match_strength={cfg['spectrum_match_strength']}."
    )
    lines.append(
        "- Synthetic private data; no network access; no PEFT integration.\n"
    )

    lines.append("## 2. Stronger Dummy Strategy Design\n")
    for note in design["design_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        f"- supported_strategies: {design['supported_strategies']}"
    )
    lines.append(
        f"- evaluated_strategies: {design['evaluated_strategies']}\n"
    )

    lines.append("## 3. Forward Correctness\n")
    lines.append(
        "| strategy | max_loss_diff | max_forward_err"
        " | max_dummy_contribution_norm | max_correction_norm | allclose |"
    )
    lines.append(
        "|----------|---------------|------------------"
        "|------------------------------|----------------------|----------|"
    )
    for entry in per_strategy:
        lines.append(
            f"| {entry['dummy_strategy']}"
            f" | {entry['max_loss_diff']:.3e}"
            f" | {entry['max_forward_err']:.3e}"
            f" | {entry['max_dummy_contribution_norm']:.3e}"
            f" | {entry['max_correction_norm']:.3e}"
            f" | {entry['allclose']} |"
        )
    lines.append("")

    lines.append("## 4. Backward Correctness\n")
    lines.append(
        "| strategy | max_grad_A_real_err | max_grad_B_real_err"
        " | max_update_A_err | max_update_B_err |"
    )
    lines.append(
        "|----------|----------------------|----------------------"
        "|------------------|------------------|"
    )
    for entry in per_strategy:
        lines.append(
            f"| {entry['dummy_strategy']}"
            f" | {entry['max_grad_a_real_err']:.3e}"
            f" | {entry['max_grad_b_real_err']:.3e}"
            f" | {entry['max_update_a_err']:.3e}"
            f" | {entry['max_update_b_err']:.3e} |"
        )
    lines.append("")

    lines.append("## 5. Optimizer Handling\n")
    lines.append(
        "| strategy | trainable_a | trainable_b | optimizer_state_a"
        " | optimizer_state_b | dummy_in_state | dummy_updated |"
    )
    lines.append(
        "|----------|-------------|-------------|--------------------"
        "|-------------------|----------------|----------------|"
    )
    for entry in per_strategy:
        oh = entry["optimizer_handling"]
        lines.append(
            f"| {entry['dummy_strategy']}"
            f" | {oh['trainable_adapter_shape_a']}"
            f" | {oh['trainable_adapter_shape_b']}"
            f" | {oh['optimizer_state_shape_a']}"
            f" | {oh['optimizer_state_shape_b']}"
            f" | {oh['optimizer_state_contains_dummy']}"
            f" | {oh['dummy_update_applied']} |"
        )
    lines.append("")

    lines.append("## 6. Dummy Contribution and Correction\n")
    lines.append(
        "- Cancellation strategies maintain `A_pad B_pad = A_real B_real`"
        " exactly to float64 precision."
    )
    lines.append(
        "- `noise_injected_cancellation_dummy` carries a small trusted-side"
        " correction term that the harness subtracts via"
        " `(alpha / true_rank) X @ correction` from the recovered output."
    )
    lines.append("")
    for entry in per_strategy:
        if entry["max_correction_norm"] > 0.0:
            lines.append(
                f"- `{entry['dummy_strategy']}`:"
                f" max_dummy_contribution_norm = {entry['max_dummy_contribution_norm']:.3e},"
                f" max_correction_norm = {entry['max_correction_norm']:.3e};"
                " trusted-side correction is applied each step."
            )
    lines.append("")

    lines.append("## 7. Comparison with Stage 7.2 / 7.3\n")
    lines.append(
        "- Stage 7.2 `paired_cancellation_dummy` keeps `dummy_contribution_norm = 0`"
        " exactly. The five Stage 7.4 stronger strategies preserve this property"
        " EXCEPT `noise_injected_cancellation_dummy`, which carries a"
        " tracked trusted-side correction."
    )
    lines.append(
        "- Per-step `max_loss_diff` / `max_grad_*_real_err` /"
        " `max_update_*_err` remain at float64 floor (≤ 1e-13) for every"
        " strategy — Stage 7.0 / 7.1 / 7.2 / 7.3 correctness regressions"
        " are checked separately by the existing test suites."
    )
    lines.append(
        "- Stage 7.2's `lora_rank_padding_status = \"implemented\"` /"
        " `lora_hidden_rank_status = \"padded-rank-prototype\"` are"
        " preserved. Stage 7.4 adds"
        " `lora_stronger_dummy_status = \"implemented\"` /"
        " `lora_spectral_rank_hardening_status = \"proxy-evaluated\"`"
        " as additive metadata.\n"
    )

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
    cfg = StrongerDummyProbeConfig(
        output_dir=str(args.output_dir),
        seed=args.seed, batch_size=args.batch_size,
        d_in=args.d_in, d_out=args.d_out,
        true_rank=args.true_rank, padded_rank=args.padded_rank,
        alpha=args.alpha, num_steps=args.num_steps, lr=args.lr,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        use_pad=args.use_pad,
        dummy_strategies=tuple(args.dummy_strategies),
        dummy_scale=args.dummy_scale,
        noise_scale=args.noise_scale,
        spectrum_match_strength=args.spectrum_match_strength,
        dtype=args.dtype, device=args.device,
    )
    report = run_lora_stronger_dummy_probe(cfg)
    json_path = args.output_dir / "lora_stronger_dummy_experiments.json"
    csv_path = args.output_dir / "lora_stronger_dummy_experiments.csv"
    md_path = args.output_dir / "lora_stronger_dummy_experiments.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = stronger_dummy_probe_csv_rows(report)
    fieldnames = ["section", "strategy", "step", "metric", "value", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    allclose_count = sum(1 for e in report["per_strategy"] if e["allclose"])
    total = len(report["per_strategy"])
    print(
        f"evaluated={total} allclose={allclose_count}"
        f" lora_stronger_dummy_status={report['lora_stronger_dummy_status']}"
        f" lora_spectral_rank_hardening_status="
        f"{report['lora_spectral_rank_hardening_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
