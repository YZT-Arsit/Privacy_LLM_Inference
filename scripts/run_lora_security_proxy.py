#!/usr/bin/env python
"""Stage 7.0 — LoRA security proxy runner.

Drives :func:`pllo.experiments.lora_security_proxy.run_lora_security_proxy`
and emits ``outputs/lora_security_proxy.{json,csv,md}``.

Reports only summary statistics + fingerprints. Raw adapter tensors,
private data, masks, and pads are NEVER exported.
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

from pllo.experiments.lora_security_proxy import (  # noqa: E402
    LoRASecurityProxyConfig,
    VALID_STRATEGIES,
    run_lora_security_proxy,
    security_proxy_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--d-out", type=int, default=16)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=32)
    p.add_argument("--pad-scale", type=float, default=1.0)
    p.add_argument("--membership-trials-per-sample", type=int, default=8)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=list(VALID_STRATEGIES),
        default=list(VALID_STRATEGIES),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Stage 7.0 — LoRA Security Proxy\n")

    lines.append("## 1. Experiment Scope\n")
    lines.append(
        f"- d_in={cfg['d_in']}, d_out={cfg['d_out']}, rank={cfg['rank']},"
        f" alpha={cfg['alpha']}."
    )
    lines.append(
        f"- num_trials={cfg['num_trials']}, membership_trials_per_sample="
        f"{cfg['membership_trials_per_sample']}, dtype={cfg['dtype']}."
    )
    lines.append(f"- strategies: {report['strategies']}")
    lines.append(f"- scope: {report['scope']}\n")

    lines.append("## 2. Threat Model\n")
    lines.append(
        "- Passive GPU observer of the masked transcript "
        "(X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde) under one masked LoRA"
        " linear."
    )
    lines.append(
        "- Knows the model architecture, the LoRA rank dimension, and the"
        " masking scheme; does NOT know N_in, N_out, U, T, the plaintext"
        " A / B, or the private (X, Y_target)."
    )
    lines.append(
        "- No hardware side-channel (cache / power / EM) and no active"
        " boundary attack."
    )
    lines.append(
        "- This is a *proxy*: ranks strategies under three sub-attacks. It"
        " does NOT prove security.\n"
    )

    lines.append("## 3. Adapter Extraction Proxy\n")
    lines.append(
        "| strategy | dW_recovery_rel_l2 | adapter_A_rel_l2 | adapter_B_rel_l2"
        " | rank_signature_A | rank_visible | subspace_sim_A | sv_sim_A |"
    )
    lines.append(
        "|----------|---------------------|------------------|------------------"
        "|------------------|--------------|----------------|----------|"
    )
    for e in report["adapter_extraction_proxy"]:
        lines.append(
            f"| {e['strategy']}"
            f" | {e['delta_w_recovery_rel_l2_mean']:.3f}"
            f" | {e['adapter_a_recovery_rel_l2_mean']:.3f}"
            f" | {e['adapter_b_recovery_rel_l2_mean']:.3f}"
            f" | {e['rank_signature_a']}/{e['configured_rank']}"
            f" | {e['rank_visible_in_a_tilde_shape']}"
            f" | {e['subspace_similarity_a_mean']:.3f}"
            f" | {e['singular_value_similarity_a_mean']:.3f} |"
        )
    lines.append("")
    for e in report["adapter_extraction_proxy"]:
        lines.append(f"- **{e['strategy']}**: {e['interpretation']}")
    lines.append("")

    lines.append("## 4. Gradient Leakage Accounting\n")
    accounting = report["gradient_leakage_accounting"]
    representative_strategy = (
        "fresh_masks_fresh_u_with_pad"
        if "fresh_masks_fresh_u_with_pad" in accounting
        else next(iter(accounting), "")
    )
    representative = accounting.get(representative_strategy, [])
    lines.append(
        f"Representative strategy: **{representative_strategy}**.\n"
    )
    lines.append("| variable | visibility | plaintext | leakage_risk | mitigation | stage_7_0_status |")
    lines.append("|----------|------------|-----------|--------------|------------|------------------|")
    for row in representative:
        lines.append(
            f"| {row['name']} | {row['visibility']} | {row['contains_plaintext']}"
            f" | {row['leakage_risk']} | {row['mitigation']}"
            f" | {row['stage_7_0_status']} |"
        )
    lines.append("")
    lines.append(
        "Per-strategy variation lives in the JSON / CSV; the GPU-visibility"
        " contract is the same across strategies, only the *risk level*"
        " differs.\n"
    )

    lines.append("## 5. Membership-Style Linkability Proxy\n")
    lines.append(
        "| strategy | same_sample_dist | different_sample_dist | AUC_proxy"
        " | linkability_rank | risk_level |"
    )
    lines.append(
        "|----------|-------------------|------------------------|-----------"
        "|--------------------|------------|"
    )
    for m in report["membership_style_linkability_proxy"]:
        lines.append(
            f"| {m['strategy']}"
            f" | {m['same_sample_distance_mean']:.3f}"
            f" | {m['different_sample_distance_mean']:.3f}"
            f" | {m['membership_auc_proxy']:.3f}"
            f" | {m['linkability_rank']:.3f}"
            f" | {m['risk_level']} |"
        )
    lines.append("")

    lines.append("## 6. Interpretation\n")
    interp = report["interpretation"]
    for k, v in interp.items():
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
    cfg = LoRASecurityProxyConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        d_in=args.d_in,
        d_out=args.d_out,
        rank=args.rank,
        alpha=args.alpha,
        num_trials=args.num_trials,
        pad_scale=args.pad_scale,
        membership_trials_per_sample=args.membership_trials_per_sample,
        dtype=args.dtype,
        device=args.device,
        strategies=tuple(args.strategies),
    )
    report = run_lora_security_proxy(cfg)
    json_path = args.output_dir / "lora_security_proxy.json"
    csv_path = args.output_dir / "lora_security_proxy.csv"
    md_path = args.output_dir / "lora_security_proxy.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = security_proxy_csv_rows(report)
    fieldnames = ["section", "strategy", "metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md_path.write_text(_md(report), encoding="utf-8")

    auc_baseline = next(
        (m["membership_auc_proxy"] for m in report["membership_style_linkability_proxy"]
         if m["strategy"] == "fixed_masks_fixed_u"), None,
    )
    auc_fresh_pad = next(
        (m["membership_auc_proxy"] for m in report["membership_style_linkability_proxy"]
         if m["strategy"] == "fresh_masks_fresh_u_with_pad"), None,
    )
    print(
        f"strategies={len(report['strategies'])}"
        f" baseline_auc={auc_baseline}"
        f" fresh_with_pad_auc={auc_fresh_pad}"
        f" lora_security_proxy_status={report['lora_security_proxy_status']}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
