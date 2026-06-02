#!/usr/bin/env python
"""Stage 7.4 — stronger-dummy LoRA security proxy runner.

Drives
:func:`pllo.experiments.lora_stronger_dummy_security_proxy.run_lora_stronger_dummy_security_proxy`
and writes ``outputs/lora_stronger_dummy_security_proxy.{json,csv,md}``.
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

from pllo.experiments.lora_stronger_dummy_security_proxy import (  # noqa: E402
    StrongerDummySecurityProxyConfig,
    run_lora_stronger_dummy_security_proxy,
    stronger_dummy_security_csv_rows,
)
from pllo.ops.lora_dummy_strategies import (  # noqa: E402
    VALID_STRONG_DUMMY_STRATEGIES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--true-ranks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--padded-rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=24)
    p.add_argument("--num-lora-modules-for-linkage", type=int, default=4)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--pad-scale", type=float, default=1.0)
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=1e-3)
    p.add_argument("--spectrum-match-strength", type=float, default=1.0)
    p.add_argument(
        "--dummy-strategies",
        nargs="+",
        default=list(VALID_STRONG_DUMMY_STRATEGIES),
        choices=list(VALID_STRONG_DUMMY_STRATEGIES),
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Stage 7.4 — Stronger Dummy LoRA Security Proxy\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- d_in={cfg['d_in']}, d_out={cfg['d_out']}, true_ranks={cfg['true_ranks']},"
        f" padded_rank={cfg['padded_rank']}, alpha={cfg['alpha']}."
    )
    lines.append(
        f"- num_trials={cfg['num_trials']},"
        f" num_lora_modules_for_linkage={cfg['num_lora_modules_for_linkage']},"
        f" use_pad={cfg['use_pad']}, dtype={cfg['dtype']}."
    )
    lines.append(
        f"- dummy_strategies={cfg['dummy_strategies']},"
        f" dummy_scale={cfg['dummy_scale']},"
        f" noise_scale={cfg['noise_scale']},"
        f" spectrum_match_strength={cfg['spectrum_match_strength']}."
    )
    lines.append(f"- scope: {report['scope']}\n")

    lines.append("## 2. Threat Model\n")
    lines.append(
        "- Passive GPU observer of the rank-padded transcript:"
        " (X_tilde, W_tilde, A_pad_tilde, B_pad_tilde, Y_tilde,"
        " G_tilde, grad_A_pad_tilde, grad_B_pad_tilde)."
    )
    lines.append(
        "- Knows model architecture, padded_rank, masking scheme; does"
        " NOT know N_in, N_out, U_pad, T, plaintext A / B, true_rank,"
        " dummy strategy choice, private (X, Y_target), optimizer state."
    )
    lines.append(
        "- No hardware side-channel and no active boundary attack."
    )
    lines.append(
        "- The dummy-strategy classifier is a generous attacker that sees"
        " per-bucket means; this is an upper bound, not a black-box"
        " attacker.\n"
    )

    lines.append("## 3. Spectral Rank Inference\n")
    lines.append(
        "| strategy | true_rank | cliff_acc | energy_acc | elbow_acc"
        " | ensemble_acc | risk |"
    )
    lines.append(
        "|----------|-----------|------------|-------------|------------"
        "|---------------|------|"
    )
    for r in report["spectral_rank_inference"]["rows"]:
        lines.append(
            f"| {r['dummy_strategy']}"
            f" | {r['true_rank']}"
            f" | {r['cliff_inference_accuracy']:.2f}"
            f" | {r['energy_inference_accuracy']:.2f}"
            f" | {r['elbow_inference_accuracy']:.2f}"
            f" | {r['ensemble_inference_accuracy']:.2f}"
            f" | {r['risk_level']} |"
        )
    lines.append("")
    lines.append(
        f"- {report['spectral_rank_inference']['interpretation']}\n"
    )

    lines.append("## 4. Gradient Rank Inference\n")
    lines.append(
        "| strategy | true_rank | grad_A_acc | grad_B_acc | ensemble | risk |"
    )
    lines.append(
        "|----------|-----------|-------------|-------------|----------|------|"
    )
    for r in report["gradient_rank_inference"]["rows"]:
        lines.append(
            f"| {r['dummy_strategy']}"
            f" | {r['true_rank']}"
            f" | {r['grad_a_cliff_accuracy']:.2f}"
            f" | {r['grad_b_cliff_accuracy']:.2f}"
            f" | {r['gradient_rank_inference_accuracy']:.2f}"
            f" | {r['risk_level']} |"
        )
    lines.append("")
    lines.append(
        f"- {report['gradient_rank_inference']['interpretation']}\n"
    )

    lines.append("## 5. Dummy Strategy Classification\n")
    cls = report["dummy_strategy_classification"]
    lines.append(
        f"- strategy_classification_accuracy:"
        f" {cls['strategy_classification_accuracy']:.3f}"
    )
    lines.append(
        f"- random_chance_baseline:"
        f" {cls['random_chance_baseline']:.3f}"
    )
    lines.append(f"- risk_level: **{cls['risk_level']}**")
    lines.append(
        f"- interpretation: {cls['interpretation']}\n"
    )

    lines.append("## 6. Cross-Layer Linkage\n")
    lines.append(
        "| strategy | layer_linkability_auc | retrieval_top1"
        " | same_module_sim | different_module_sim | risk |"
    )
    lines.append(
        "|----------|------------------------|-----------------"
        "|------------------|----------------------|------|"
    )
    for r in report["cross_layer_linkage"]["rows"]:
        lines.append(
            f"| {r['dummy_strategy']}"
            f" | {r['layer_linkability_auc']:.3f}"
            f" | {r['module_identity_retrieval_top1']:.3f}"
            f" | {r['same_module_similarity']:.3f}"
            f" | {r['different_module_similarity']:.3f}"
            f" | {r['risk_level']} |"
        )
    lines.append("")
    lines.append(
        f"- {report['cross_layer_linkage']['interpretation']}\n"
    )

    lines.append("## 7. Interpretation\n")
    for k, v in report["interpretation"].items():
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
    cfg = StrongerDummySecurityProxyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed, d_in=args.d_in, d_out=args.d_out,
        true_ranks=tuple(args.true_ranks),
        padded_rank=args.padded_rank,
        alpha=args.alpha,
        num_trials=args.num_trials,
        num_lora_modules_for_linkage=args.num_lora_modules_for_linkage,
        use_pad=args.use_pad,
        pad_scale=args.pad_scale,
        dummy_scale=args.dummy_scale,
        noise_scale=args.noise_scale,
        spectrum_match_strength=args.spectrum_match_strength,
        dummy_strategies=tuple(args.dummy_strategies),
        dtype=args.dtype,
        device=args.device,
    )
    report = run_lora_stronger_dummy_security_proxy(cfg)
    json_path = args.output_dir / "lora_stronger_dummy_security_proxy.json"
    csv_path = args.output_dir / "lora_stronger_dummy_security_proxy.csv"
    md_path = args.output_dir / "lora_stronger_dummy_security_proxy.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = stronger_dummy_security_csv_rows(report)
    fieldnames = ["section", "attack", "strategy", "metric", "value", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    summary = report["interpretation"]
    print(
        f"true_ranks={list(cfg.true_ranks)} padded_rank={cfg.padded_rank}"
        f" spectral_summary='{summary['spectral_summary']}'"
        f" gradient_summary='{summary['gradient_summary']}'"
        f" classification_summary='{summary['dummy_strategy_classification_summary']}'"
        f" linkage_summary='{summary['cross_layer_linkage_summary']}'"
        f" lora_stronger_dummy_security_status={report['lora_stronger_dummy_security_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
