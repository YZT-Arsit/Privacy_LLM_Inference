#!/usr/bin/env python
"""Stage 5.4 — Adaptive permutation / linkability attacker experiments.

Runs three adaptive proxy attackers against six masking strategies and
writes structured JSON / CSV / Markdown reports to ``outputs/``:

* ``adaptive_island_attacks.json`` — full report.
* ``adaptive_island_attacks.csv``  — long-format
  ``section,attack,strategy,metric,value,notes`` rows.
* ``adaptive_island_attacks.md``   — human-readable report.

Default configuration is CPU-friendly: hidden=64, 512 train samples, 256
test samples, 200 MLP optimisation steps. The output never contains the
full permutation or mask tensors — only scalar metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.adaptive_island_attacker import (  # noqa: E402
    AdaptiveIslandAttackConfig,
    run_adaptive_island_attacks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-train-samples", type=int, default=512)
    parser.add_argument("--num-test-samples", type=int, default=256)
    parser.add_argument("--num-sessions", type=int, default=16)
    parser.add_argument("--samples-per-session", type=int, default=32)
    parser.add_argument("--permutation-pool-size", type=int, default=4)
    parser.add_argument("--attacker-steps", type=int, default=200)
    parser.add_argument("--attacker-lr", type=float, default=1e-2)
    parser.add_argument("--mlp-hidden-size", type=int, default=128)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_csv(path: Path, report: dict) -> None:
    rows: list[dict] = []
    for strategy, r in report["linear_inverter"]["strategies"].items():
        for metric in ("mse", "relative_l2_error", "cosine_similarity"):
            rows.append(
                {
                    "section": "linear_inverter",
                    "attack": "linear",
                    "strategy": strategy,
                    "metric": metric,
                    "value": r[metric],
                    "notes": "ridge least squares, V_test @ W vs X_test",
                }
            )
    for strategy, r in report["mlp_inverter"]["strategies"].items():
        for metric in (
            "mse",
            "relative_l2_error",
            "cosine_similarity",
            "final_train_loss",
        ):
            if r.get(metric) is None:
                continue
            rows.append(
                {
                    "section": "mlp_inverter",
                    "attack": "mlp",
                    "strategy": strategy,
                    "metric": metric,
                    "value": r[metric],
                    "notes": (
                        "Adam + ReLU two-layer MLP; "
                        f"steps={r.get('attacker_steps')}, "
                        f"mlp_hidden={r.get('mlp_hidden_size')}"
                    ),
                }
            )
    for strategy, r in report["permutation_recovery"]["signature_matching"].items():
        for metric in ("top1_recovery_rate", "top5_recovery_rate", "mean_correct_rank"):
            rows.append(
                {
                    "section": "permutation_recovery",
                    "attack": "signature_matching",
                    "strategy": strategy,
                    "metric": metric,
                    "value": r[metric],
                    "notes": "Stage 5.2b naive nearest-neighbour proxy on adapted data",
                }
            )
    for strategy, r in report["permutation_recovery"]["soft_assignment"].items():
        for metric in ("top1_recovery_rate", "top5_recovery_rate", "mean_correct_rank"):
            rows.append(
                {
                    "section": "permutation_recovery",
                    "attack": "soft_assignment",
                    "strategy": strategy,
                    "metric": metric,
                    "value": r[metric],
                    "notes": (
                        "Sinkhorn-style log-domain row/col normalisation; "
                        f"iters={r['iterations']} T={r['temperature']}"
                    ),
                }
            )
    for row in report["mitigation_summary"]["per_strategy"]:
        for metric in (
            "best_linear_relative_l2_error",
            "best_linear_cosine_similarity",
            "best_mlp_relative_l2_error",
            "best_mlp_cosine_similarity",
            "best_permutation_recovery_top1",
        ):
            value = row.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "section": "mitigation_decision",
                    "attack": "decision",
                    "strategy": row["strategy"],
                    "metric": metric,
                    "value": value,
                    "notes": (
                        f"risk_level={row['risk_level']},"
                        f" default_on_recommendation={row['default_on_recommendation']}"
                    ),
                }
            )
        rows.append(
            {
                "section": "mitigation_decision",
                "attack": "decision",
                "strategy": row["strategy"],
                "metric": "risk_level",
                "value": row["risk_level"],
                "notes": row["default_on_recommendation"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["section", "attack", "strategy", "metric", "value", "notes"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _render_md(report: dict) -> str:
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# Adaptive Permutation / Linkability Attacks — Stage 5.4")
    lines.append("")
    lines.append("## Experiment Scope")
    lines.append("")
    lines.append(
        f"- hidden_size: {cfg['hidden_size']}"
        f"; num_train_samples: {cfg['num_train_samples']}"
        f"; num_test_samples: {cfg['num_test_samples']}"
    )
    lines.append(
        f"- num_sessions: {cfg['num_sessions']}"
        f"; samples_per_session: {cfg['samples_per_session']}"
        f"; permutation_pool_size: {cfg['permutation_pool_size']}"
    )
    lines.append(
        f"- attacker_steps: {cfg['attacker_steps']}"
        f"; attacker_lr: {cfg['attacker_lr']}"
        f"; mlp_hidden_size: {cfg['mlp_hidden_size']}"
    )
    lines.append(
        f"- soft_assignment_iters: {cfg['soft_assignment_iters']}"
        f"; soft_assignment_temperature: {cfg['soft_assignment_temperature']}"
    )
    lines.append(f"- dtype: {cfg['dtype']}; device: {cfg['device']}; seed: {cfg['seed']}")
    lines.append("")
    lines.append("## Threat Model")
    lines.append("")
    lines.append(report["threat_model"])
    lines.append("")
    lines.append("## Structured Synthetic Activation Distribution")
    lines.append("")
    sd = report["structured_data"]
    lines.append(f"- channel_mean_range: {sd['channel_mean_range']}")
    lines.append(f"- channel_scale_range: {sd['channel_scale_range']}")
    lines.append(f"- channel_skew_profile: {sd['channel_skew_profile']}")
    lines.append(f"- distribution_summary: `{sd['distribution_summary']}`")
    lines.append("")
    lines.append("## Learned Linear Inverter")
    lines.append("")
    lines.append(
        "| strategy | relative_l2_error | cosine_similarity | mse |"
    )
    lines.append("|---|---|---|---|")
    for s, r in report["linear_inverter"]["strategies"].items():
        lines.append(
            f"| `{s}` | {r['relative_l2_error']:.4f} | "
            f"{r['cosine_similarity']:.4f} | {r['mse']:.4e} |"
        )
    lines.append(
        f"\nWeakest mitigation under linear inverter: "
        f"`{report['linear_inverter']['weakest_mitigation']}`."
    )
    lines.append("")
    lines.append("## Small MLP Inverter")
    lines.append("")
    lines.append(
        "| strategy | relative_l2_error | cosine_similarity | final_train_loss |"
        " mlp_improves_over_linear |"
    )
    lines.append("|---|---|---|---|---|")
    for s, r in report["mlp_inverter"]["strategies"].items():
        impr = report["mlp_inverter"]["improvement_over_linear"][s][
            "mlp_improves_over_linear"
        ]
        ftl = r.get("final_train_loss")
        ftl_s = "n/a" if ftl is None else f"{ftl:.4e}"
        lines.append(
            f"| `{s}` | {r['relative_l2_error']:.4f} | "
            f"{r['cosine_similarity']:.4f} | {ftl_s} | {impr} |"
        )
    lines.append("")
    lines.append("## Adaptive Permutation Recovery")
    lines.append("")
    lines.append("### Signature matching (Stage 5.2b naive nearest-neighbour proxy)")
    lines.append("")
    lines.append(
        "| strategy | top1_recovery_rate | top5_recovery_rate | mean_correct_rank |"
    )
    lines.append("|---|---|---|---|")
    for s, r in report["permutation_recovery"]["signature_matching"].items():
        lines.append(
            f"| `{s}` | {r['top1_recovery_rate']:.4f} | "
            f"{r['top5_recovery_rate']:.4f} | {r['mean_correct_rank']:.2f} |"
        )
    lines.append("")
    lines.append("### Soft assignment (Sinkhorn-style log-domain normalisation)")
    lines.append("")
    lines.append(
        "| strategy | top1_recovery_rate | top5_recovery_rate | mean_correct_rank |"
    )
    lines.append("|---|---|---|---|")
    for s, r in report["permutation_recovery"]["soft_assignment"].items():
        lines.append(
            f"| `{s}` | {r['top1_recovery_rate']:.4f} | "
            f"{r['top5_recovery_rate']:.4f} | {r['mean_correct_rank']:.2f} |"
        )
    lines.append(
        f"\nRandom chance top1 = "
        f"{report['permutation_recovery']['random_chance_top1']:.4f}"
        f" (1 / hidden_size)."
    )
    lines.append("")
    lines.append("## Mitigation Decision Table")
    lines.append("")
    lines.append(
        "| strategy | best_linear_rel_l2 | best_mlp_rel_l2 | best_perm_top1 | "
        "risk_level | default_on_recommendation |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in report["mitigation_summary"]["per_strategy"]:
        perm = row["best_permutation_recovery_top1"]
        perm_s = "n/a" if perm is None else f"{perm:.4f}"
        lines.append(
            f"| `{row['strategy']}` | {row['best_linear_relative_l2_error']:.4f} | "
            f"{row['best_mlp_relative_l2_error']:.4f} | {perm_s} | "
            f"{row['risk_level']} | `{row['default_on_recommendation']}` |"
        )
    lines.append("")
    lines.append(
        "Recommended default-on candidate: "
        f"`{report['mitigation_summary']['recommended_default_on_candidate']}`."
    )
    lines.append("")
    lines.append(report["mitigation_summary"]["default_on_caveat"])
    lines.append("")
    lines.append("### Required mitigations per strategy")
    lines.append("")
    for row in report["mitigation_summary"]["per_strategy"]:
        lines.append(f"- **`{row['strategy']}`**:")
        for m in row["required_mitigations"]:
            lines.append(f"  - {m}")
    lines.append("")
    lines.append("## Comparison with Stage 5.2b Naive Proxy")
    lines.append("")
    lines.append(
        "| strategy | naive_signature_matching_top1 | adaptive_soft_assignment_top1 |"
        " absolute_uplift |"
    )
    lines.append("|---|---|---|---|")
    for s, r in report["comparison_with_naive_proxy"]["per_strategy"].items():
        lines.append(
            f"| `{s}` | {r['naive_signature_matching_top1']:.4f} | "
            f"{r['adaptive_soft_assignment_top1']:.4f} | "
            f"{r['absolute_uplift']:+.4f} |"
        )
    lines.append("")
    lines.append(report["comparison_with_naive_proxy"]["note"])
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append(
        "- These are adaptive/proxy attacks, not formal security proofs."
    )
    lines.append(
        "- Dense sandwiching reduces tested recovery but does not imply"
        " semantic security."
    )
    lines.append(
        "- Default-on recommendations are conditional on the tested threat"
        " model only."
    )
    lines.append("")
    lines.append("## Next Stage Plan")
    lines.append("")
    lines.append(
        "- Stage 6.4 — Qwen / TinyLlama migration. Reuse the Stage 5.3a /"
        " 5.3b / 5.3c wrapper / probe pattern; behind a feature flag,"
        " default `trusted`."
    )
    lines.append(
        "- Stage 5.3d (deferred) — Full BERT / T5 obfuscated wrappers; only"
        " landed once an adaptive attacker against the chosen mitigation"
        " bundle (fresh permutation + dense sandwich + pad at Linear"
        " boundaries) is bounded below the agreed acceptance budget."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = AdaptiveIslandAttackConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        hidden_size=args.hidden_size,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        num_sessions=args.num_sessions,
        samples_per_session=args.samples_per_session,
        permutation_pool_size=args.permutation_pool_size,
        attacker_steps=args.attacker_steps,
        attacker_lr=args.attacker_lr,
        mlp_hidden_size=args.mlp_hidden_size,
        dtype=args.dtype,
        device=args.device,
    )
    report = run_adaptive_island_attacks(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "adaptive_island_attacks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "adaptive_island_attacks.csv", report)
    (args.output_dir / "adaptive_island_attacks.md").write_text(
        _render_md(report), encoding="utf-8"
    )
    # Headline summary on stdout.
    print("--- Stage 5.4 mitigation decisions ---")
    for row in report["mitigation_summary"]["per_strategy"]:
        print(
            f"  {row['strategy']:42s}: risk={row['risk_level']:6s}"
            f" reco={row['default_on_recommendation']}"
        )


if __name__ == "__main__":
    main()
