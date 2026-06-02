#!/usr/bin/env python
"""Stage 7.3 — multi-layer LoRA cross-layer security proxy runner.

Drives
:func:`pllo.experiments.multilayer_lora_security_proxy.run_multilayer_lora_security_proxy`
and writes ``outputs/multilayer_lora_security_proxy.{json,csv,md}``.
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

from pllo.experiments.multilayer_lora_security_proxy import (  # noqa: E402
    MultiLayerLoRASecurityProxyConfig,
    multilayer_security_csv_rows,
    run_multilayer_lora_security_proxy,
)
from pllo.ops.lora_rank_padding import VALID_DUMMY_STRATEGIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--intermediate-size", type=int, default=64)
    p.add_argument("--true-ranks", type=int, nargs="+", default=[2, 4])
    p.add_argument("--padded-rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=32)
    p.add_argument("--membership-trials-per-sample", type=int, default=6)
    p.add_argument("--membership-num-steps", type=int, default=3)
    p.add_argument(
        "--dummy-strategy",
        choices=list(VALID_DUMMY_STRATEGIES),
        default="paired_cancellation_dummy",
    )
    p.add_argument("--dummy-scale", type=float, default=1.0)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--pad-scale", type=float, default=1.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Stage 7.3 — Multi-Layer LoRA Cross-Layer Security Proxy\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- num_layers={cfg['num_layers']}, hidden_size={cfg['hidden_size']},"
        f" intermediate_size={cfg['intermediate_size']}."
    )
    lines.append(
        f"- true_ranks={cfg['true_ranks']}, padded_rank={cfg['padded_rank']},"
        f" alpha={cfg['alpha']}, num_trials={cfg['num_trials']}."
    )
    lines.append(
        f"- dummy_strategy={cfg['dummy_strategy']!r}, use_pad={cfg['use_pad']},"
        f" dtype={cfg['dtype']}."
    )
    lines.append(f"- scope: {report['scope']}\n")

    lines.append("## 2. Threat Model\n")
    lines.append(
        "- Passive GPU observer across multiple LoRA-augmented linears in a"
        " multi-layer block stack."
    )
    lines.append(
        "- Sees per-module ``A_pad_tilde`` / ``B_pad_tilde`` /"
        " ``grad_A_pad_tilde`` / ``grad_B_pad_tilde``."
    )
    lines.append(
        "- Knows model architecture, padded_rank dimension, masking scheme;"
        " does NOT know per-module N_in / N_out / U_pad / T, plaintext"
        " A / B / A_pad / B_pad / G / grad_A / grad_B, optimizer state,"
        " true_rank, or private (X, Y_target)."
    )
    lines.append(
        "- No hardware side-channel and no active boundary attack."
    )
    lines.append(
        "- This is a *proxy*: ranks per-module strategies; no formal claim.\n"
    )

    lines.append("## 3. Cross-Layer Adapter Linkage\n")
    lines.append(
        "| strategy | layer_linkability_auc | retrieval_top1"
        " | same_module_sim | different_module_sim | risk |"
    )
    lines.append(
        "|----------|-----------------------|-----------------|------------------"
        "|----------------------|------|"
    )
    for entry in report["cross_layer_adapter_linkage"]["rows"]:
        lines.append(
            f"| {entry['strategy']}"
            f" | {entry['layer_linkability_auc']:.3f}"
            f" | {entry['module_identity_retrieval_top1']:.3f}"
            f" | {entry['same_module_similarity']:.3f}"
            f" | {entry['different_module_similarity']:.3f}"
            f" | {entry['risk_level']} |"
        )
    lines.append("")
    lines.append(
        f"- {report['cross_layer_adapter_linkage']['interpretation']}\n"
    )

    lines.append("## 4. Heterogeneous True Rank with Shared Padded Rank\n")
    lines.append(
        "| layer | module | true_rank | padded_rank | visible_rank"
        " | shape_hidden_rate | spectral_acc | gradient_acc | risk |"
    )
    lines.append(
        "|-------|--------|-----------|-------------|---------------"
        "|--------------------|--------------|--------------|------|"
    )
    for entry in report["heterogeneous_true_rank_with_shared_padded_rank"]["rows"]:
        lines.append(
            f"| {entry['layer_index']}"
            f" | {entry['module_name']}"
            f" | {entry['true_rank']}"
            f" | {entry['padded_rank']}"
            f" | {entry['visible_rank_from_shape']}"
            f" | {entry['true_rank_shape_hidden_rate']:.2f}"
            f" | {entry['spectral_rank_inference_accuracy']:.2f}"
            f" | {entry['gradient_rank_inference_accuracy']:.2f}"
            f" | {entry['risk_level']} |"
        )
    lines.append("")
    lines.append(
        f"- {report['heterogeneous_true_rank_with_shared_padded_rank']['interpretation']}\n"
    )

    lines.append("## 5. Multi-Step Membership Linkability\n")
    lines.append(
        "| module | same_sample_dist | different_sample_dist | AUC_proxy"
        " | linkability_rank | risk |"
    )
    lines.append(
        "|--------|-------------------|------------------------|-----------"
        "|--------------------|------|"
    )
    for entry in report["multi_step_membership_linkability"]["rows"]:
        lines.append(
            f"| {entry['module_name']}"
            f" | {entry['same_sample_distance_mean']:.3f}"
            f" | {entry['different_sample_distance_mean']:.3f}"
            f" | {entry['membership_auc_proxy']:.3f}"
            f" | {entry['linkability_rank']:.3f}"
            f" | {entry['risk_level']} |"
        )
    lines.append("")
    agg = report["multi_step_membership_linkability"]["aggregate"]
    lines.append(
        f"- mean_membership_auc_proxy: {agg['mean_membership_auc_proxy']:.3f}"
    )
    lines.append(
        f"- adapter_update_linkability: {agg['adapter_update_linkability']:.3f}"
    )
    lines.append(
        f"- {report['multi_step_membership_linkability']['interpretation']}\n"
    )

    lines.append("## 6. Interpretation\n")
    for k, v in report["interpretation"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

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
    cfg = MultiLayerLoRASecurityProxyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        true_ranks=tuple(args.true_ranks),
        padded_rank=args.padded_rank,
        alpha=args.alpha,
        num_trials=args.num_trials,
        membership_trials_per_sample=args.membership_trials_per_sample,
        membership_num_steps=args.membership_num_steps,
        dummy_strategy=args.dummy_strategy,
        dummy_scale=args.dummy_scale,
        use_pad=args.use_pad,
        pad_scale=args.pad_scale,
        dtype=args.dtype,
        device=args.device,
    )
    report = run_multilayer_lora_security_proxy(cfg)
    json_path = args.output_dir / "multilayer_lora_security_proxy.json"
    csv_path = args.output_dir / "multilayer_lora_security_proxy.csv"
    md_path = args.output_dir / "multilayer_lora_security_proxy.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = multilayer_security_csv_rows(report)
    fieldnames = ["section", "attack", "strategy", "metric", "value", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")
    print(
        f"true_ranks={list(cfg.true_ranks)} padded_rank={cfg.padded_rank}"
        f" dummy_strategy={cfg.dummy_strategy!r}"
        f" linkage_summary='{report['interpretation']['cross_layer_linkage_summary']}'"
        f" rank_summary='{report['interpretation']['heterogeneous_rank_summary']}'"
        f" shape_hidden_rate={report['interpretation']['true_rank_shape_hidden_rate']:.3f}"
        f" lora_multilayer_security_proxy_status={report['lora_multilayer_security_proxy_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
