#!/usr/bin/env python
"""Stage 7.2 — LoRA rank security proxy runner.

Drives
:func:`pllo.experiments.lora_rank_security_proxy.run_lora_rank_security_proxy`
and writes ``outputs/lora_rank_security_proxy.{json,csv,md}``.
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

from pllo.experiments.lora_rank_security_proxy import (  # noqa: E402
    LoRARankSecurityProxyConfig,
    rank_security_csv_rows,
    run_lora_rank_security_proxy,
)
from pllo.ops.lora_rank_padding import VALID_DUMMY_STRATEGIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--true-ranks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--padded-rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=32)
    p.add_argument("--pad-scale", type=float, default=1.0)
    p.add_argument("--membership-trials-per-sample", type=int, default=8)
    p.add_argument(
        "--dummy-strategy",
        choices=list(VALID_DUMMY_STRATEGIES),
        default="paired_cancellation_dummy",
    )
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Stage 7.2 — LoRA Rank Security Proxy\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- d_in={cfg['d_in']}, d_out={cfg['d_out']}, true_ranks={cfg['true_ranks']},"
        f" padded_rank={cfg['padded_rank']}, alpha={cfg['alpha']}."
    )
    lines.append(
        f"- num_trials={cfg['num_trials']}, membership_trials_per_sample="
        f"{cfg['membership_trials_per_sample']},"
        f" dummy_strategy={cfg['dummy_strategy']!r}, use_pad={cfg['use_pad']},"
        f" dtype={cfg['dtype']}."
    )
    lines.append(f"- scope: {report['scope']}\n")

    lines.append("## 2. Threat Model\n")
    lines.append(
        "- Passive GPU observer of the rank-padded transcript:"
        " (X_tilde, W_tilde, A_pad_tilde, B_pad_tilde, Y_tilde,"
        " G_tilde, grad_A_pad_tilde, grad_B_pad_tilde)."
    )
    lines.append(
        "- Knows model architecture, padded_rank dimension, and the masking"
        " scheme; does NOT know N_in, N_out, U_pad, T, plaintext"
        " A / B / A_pad / B_pad / G / grad_A / grad_B, optimizer state,"
        " true_rank, or the private (X, Y_target)."
    )
    lines.append(
        "- No hardware side-channel (cache / power / EM) and no active"
        " boundary attack."
    )
    lines.append(
        "- This is a *proxy*: ranks dummy strategies and padding levels"
        " under four sub-attacks. It does NOT prove security.\n"
    )

    lines.append("## 3. Shape-Level Rank Leakage\n")
    shape = report["shape_level_rank_leakage"]
    lines.append(
        "| strategy | true_rank | exposed_rank | true_rank_hidden_from_shape |"
    )
    lines.append(
        "|----------|-----------|--------------|-----------------------------|"
    )
    for sub in ("no_padding", "rank_padding"):
        for entry in shape[sub]:
            lines.append(
                f"| {sub} | {entry['true_rank']} | {entry['exposed_rank_value']}"
                f" | {entry['true_rank_hidden_from_shape']} |"
            )
    lines.append("")
    lines.append(f"- {shape['interpretation']}\n")

    lines.append("## 4. Spectral Rank Inference Proxy\n")
    lines.append(
        "| true_rank | inferred_no_padding | inferred_rank_padding_A_tilde"
        " | inferred_rank_padding_B_tilde | accuracy | risk |"
    )
    lines.append(
        "|-----------|----------------------|-------------------------------"
        "|-------------------------------|----------|------|"
    )
    for entry in report["spectral_rank_inference"]["rows"]:
        np_part = entry["no_padding"]
        pad_part = entry["rank_padding"]
        lines.append(
            f"| {entry['true_rank']} | {np_part['inferred_rank_mean']:.2f}"
            f" | {pad_part['inferred_rank_from_a_tilde_pad_mean']:.2f}"
            f" | {pad_part['inferred_rank_from_b_tilde_pad_mean']:.2f}"
            f" | {pad_part['rank_inference_accuracy']:.2f}"
            f" | {pad_part['risk_level']} |"
        )
    lines.append("")
    for entry in report["spectral_rank_inference"]["rows"]:
        lines.append(
            f"- **true_rank={entry['true_rank']}**:"
            f" {entry['rank_padding']['verdict']}"
        )
    lines.append("")
    lines.append(f"- {report['spectral_rank_inference']['interpretation']}\n")

    lines.append("## 5. Gradient Rank Inference Proxy\n")
    lines.append(
        "| true_rank | inferred_grad_A | inferred_grad_B | accuracy | risk |"
    )
    lines.append(
        "|-----------|-----------------|-----------------|----------|------|"
    )
    for entry in report["gradient_rank_inference"]["rows"]:
        lines.append(
            f"| {entry['true_rank']}"
            f" | {entry['inferred_rank_from_grad_a_tilde_pad_mean']:.2f}"
            f" | {entry['inferred_rank_from_grad_b_tilde_pad_mean']:.2f}"
            f" | {entry['rank_inference_accuracy']:.2f}"
            f" | {entry['risk_level']} |"
        )
    lines.append("")
    lines.append(f"- {report['gradient_rank_inference']['interpretation']}\n")

    lines.append("## 6. Membership / Linkability Proxy\n")
    lines.append(
        "| true_rank | same_sample_dist | different_sample_dist | AUC_proxy"
        " | linkability_rank | risk_level |"
    )
    lines.append(
        "|-----------|-------------------|------------------------|-----------"
        "|--------------------|------------|"
    )
    for entry in report["membership_style_linkability"]["rows"]:
        lines.append(
            f"| {entry['true_rank']}"
            f" | {entry['same_sample_distance_mean']:.3f}"
            f" | {entry['different_sample_distance_mean']:.3f}"
            f" | {entry['membership_auc_proxy']:.3f}"
            f" | {entry['linkability_rank']:.3f}"
            f" | {entry['risk_level']} |"
        )
    lines.append("")

    lines.append("## 7. Interpretation\n")
    interp = report["interpretation"]
    for k, v in interp.items():
        lines.append(f"- **{k}**: {v}")
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
    cfg = LoRARankSecurityProxyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed, d_in=args.d_in, d_out=args.d_out,
        true_ranks=tuple(args.true_ranks), padded_rank=args.padded_rank,
        alpha=args.alpha, num_trials=args.num_trials,
        pad_scale=args.pad_scale,
        membership_trials_per_sample=args.membership_trials_per_sample,
        dummy_strategy=args.dummy_strategy,
        dummy_scale=args.dummy_scale,
        use_pad=args.use_pad, dtype=args.dtype, device=args.device,
    )
    report = run_lora_rank_security_proxy(cfg)
    json_path = args.output_dir / "lora_rank_security_proxy.json"
    csv_path = args.output_dir / "lora_rank_security_proxy.csv"
    md_path = args.output_dir / "lora_rank_security_proxy.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = rank_security_csv_rows(report)
    fieldnames = ["section", "scope", "metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    print(
        f"true_ranks={list(cfg.true_ranks)} padded_rank={cfg.padded_rank}"
        f" dummy_strategy={cfg.dummy_strategy!r}"
        f" spectral_summary='{report['interpretation']['spectral_inference_summary']}'"
        f" gradient_summary='{report['interpretation']['gradient_inference_summary']}'"
        f" lora_rank_security_proxy_status={report['lora_rank_security_proxy_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
