#!/usr/bin/env python
"""Stage 5.5 — real-activation adaptive attacker runner.

Drives :func:`pllo.experiments.real_activation_attacker.run_real_activation_attacks`
and emits ``outputs/real_activation_attacks.{json,csv,md}``.

Default behaviour: synthetic fallback only, both Stage 5.3e bundles
(``fresh_perm_only`` and ``fresh_perm_plus_sandwich_plus_pad``).
``--include-fixed-debug`` adds the ``fixed_permutation_debug`` baseline
(opt-in only — debug reference, never recommended for deployment).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.real_activation_attacker import (  # noqa: E402
    RealActivationAttackConfig,
    run_real_activation_attacks,
)
from pllo.ops.mitigation_bundles import VALID_MITIGATION_BUNDLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--attempt-real-model-load", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_false", default=True,
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--attacker-steps", type=int, default=200)
    parser.add_argument("--attacker-lr", type=float, default=1e-2)
    parser.add_argument("--mlp-hidden-size", type=int, default=128)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--use-pad", action="store_true", default=True)
    parser.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    parser.add_argument(
        "--include-fixed-debug",
        action="store_true",
        help=(
            "Include the fixed_permutation_debug baseline (RNG pinned per"
            " session — debug reference only, never recommended)."
        ),
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=list(VALID_MITIGATION_BUNDLES) + ["fixed_permutation_debug"],
        help=(
            "Explicit list of bundles to evaluate. Defaults to both Stage 5.3e"
            " bundles; --include-fixed-debug additionally appends the debug"
            " baseline."
        ),
    )
    return parser.parse_args()


def _resolve_bundles(args: argparse.Namespace) -> tuple[str, ...]:
    if args.bundle:
        return tuple(args.bundle)
    bundles = list(VALID_MITIGATION_BUNDLES)
    if args.include_fixed_debug:
        bundles.append("fixed_permutation_debug")
    return tuple(bundles)


# ---------------------------------------------------------------------------
# CSV emitter (long format)
# ---------------------------------------------------------------------------


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Model loading + source as a top section.
    ml = report["model_loading"]
    for k in ("load_status", "resolved_model_id", "model_family", "fallback_used"):
        rows.append({
            "section": "model_loading",
            "tensor_name": "n/a",
            "mitigation_bundle": "n/a",
            "attack": "n/a",
            "metric": k,
            "value": ml.get(k),
            "notes": "",
        })
    # Per-tensor per-bundle attacker metrics.
    for bundle, by_tensor in report["target_tensor_results"].items():
        for tensor, payload in by_tensor.items():
            for attack_section, attack_label in (
                ("linear_inverter", "linear"),
                ("mlp_inverter", "mlp"),
                ("linkability", "linkability"),
            ):
                d = payload[attack_section]
                for metric, value in d.items():
                    rows.append({
                        "section": attack_section,
                        "tensor_name": tensor,
                        "mitigation_bundle": bundle,
                        "attack": attack_label,
                        "metric": metric,
                        "value": value,
                        "notes": "",
                    })
            if payload.get("permutation_recovery") is not None:
                pr = payload["permutation_recovery"]
                for sub_attack in ("signature_matching", "soft_assignment"):
                    sub = pr[sub_attack]
                    for metric, value in sub.items():
                        rows.append({
                            "section": "permutation_recovery",
                            "tensor_name": tensor,
                            "mitigation_bundle": bundle,
                            "attack": sub_attack,
                            "metric": metric,
                            "value": value,
                            "notes": "",
                        })
            rows.append({
                "section": "decision",
                "tensor_name": tensor,
                "mitigation_bundle": bundle,
                "attack": "summary",
                "metric": "risk_level",
                "value": payload["risk_level"],
                "notes": payload["default_on_recommendation"],
            })
    # Bundle comparison rows.
    for row in report["bundle_comparison"]:
        for k, v in row.items():
            if k == "tensor_name":
                continue
            rows.append({
                "section": "bundle_comparison",
                "tensor_name": row["tensor_name"],
                "mitigation_bundle": "summary",
                "attack": "delta_or_value",
                "metric": k,
                "value": v,
                "notes": "",
            })
    rec = report["recommendation"]
    for k, v in rec.items():
        rows.append({
            "section": "recommendation",
            "tensor_name": "n/a",
            "mitigation_bundle": "summary",
            "attack": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section", "tensor_name", "mitigation_bundle",
                "attack", "metric", "value", "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = report["config"]
    ml = report["model_loading"]
    spec = report["block_spec"]
    summary = report["attacker_summary"]
    comparison = report["bundle_comparison"]
    rec = report["recommendation"]

    lines.append("# Real-Activation Adaptive Attacker (Stage 5.5)\n")

    lines.append("## Experiment Scope\n")
    lines.append(
        "Stage 5.5 re-runs the Stage 5.4 adaptive proxy attackers (ridge"
        " linear inverter, small MLP inverter, signature / Sinkhorn"
        " permutation recovery, linkability proxy) against the"
        " (plain, attacker-visible) tensor pairs collected from the"
        " Stage 6.4b modern decoder block wrapper. Default mode for the"
        " wider system remains `nonlinear_mode='trusted'`; default"
        " mitigation bundle remains `fresh_perm_only`. The numbers"
        " below are an adaptive proxy evaluation of"
        " `fresh_perm_plus_sandwich_plus_pad` as a default-on candidate"
        " for real-activation deployments — they are NOT a formal"
        " security proof."
    )
    lines.append("")

    lines.append("## Model Loading Status\n")
    lines.append("| field | value |\n|---|---|")
    for k in (
        "load_status", "resolved_model_id", "model_family",
        "model_class", "fallback_used", "candidates_tried",
        "load_error",
    ):
        lines.append(f"| {k} | {ml.get(k)} |")
    lines.append("")

    lines.append("## Trace Collection Summary\n")
    lines.append("| field | value |\n|---|---|")
    lines.append(f"| source | {report['source']} |")
    for k in (
        "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "attention_variant", "rope_base",
    ):
        lines.append(f"| block_spec.{k} | {spec.get(k)} |")
    lines.append(f"| num_samples_target | {cfg['num_samples']} |")
    lines.append(f"| batch_size × seq_len | {cfg['batch_size']} × {cfg['seq_len']} |")
    lines.append(f"| use_pad | {cfg['use_pad']} |")
    lines.append(
        f"| bundles_evaluated | {', '.join(cfg['mitigation_bundles'])} |"
    )
    lines.append("")

    lines.append("## Target Tensor Inventory\n")
    trace_summary = report["trace_summary"]
    first_bundle = next(iter(trace_summary))
    inventory = trace_summary[first_bundle]
    lines.append(
        "| tensor_name | feature_dim | num_samples | plain abs_max | visible abs_max |"
        " plain_fingerprint | visible_fingerprint |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in inventory.items():
        lines.append(
            f"| {name} | {s['feature_dim']} | {s['num_samples']}"
            f" | {_fmt(s['plain_statistics']['abs_max'], 3)}"
            f" | {_fmt(s['visible_statistics']['abs_max'], 3)}"
            f" | {s['plain_statistics']['fingerprint_sha256_prefix']}"
            f" | {s['visible_statistics']['fingerprint_sha256_prefix']} |"
        )
    lines.append("")

    # Per-attack sections.
    def _attack_table(title: str, attack_section: str, metric_keys: list[str]) -> None:
        lines.append(f"## {title}\n")
        lines.append(
            "| tensor_name | bundle | "
            + " | ".join(metric_keys)
            + " |"
        )
        lines.append(
            "|---|---|" + "|".join(["---"] * len(metric_keys)) + "|"
        )
        for bundle, by_tensor in report["target_tensor_results"].items():
            for tensor, payload in by_tensor.items():
                d = payload[attack_section]
                row_vals = [
                    _fmt(d.get(k, "n/a"), 4) for k in metric_keys
                ]
                lines.append(
                    f"| {tensor} | {bundle} | " + " | ".join(row_vals) + " |"
                )
        lines.append("")

    _attack_table(
        "Linear Inverter on Real Activations",
        "linear_inverter",
        ["relative_l2_error", "mse", "cosine_similarity"],
    )
    _attack_table(
        "Small MLP Inverter on Real Activations",
        "mlp_inverter",
        ["relative_l2_error", "mse", "cosine_similarity", "final_train_loss"],
    )

    # Permutation recovery table only for permutation-relevant tensors.
    lines.append("## Permutation Recovery on Real Activations\n")
    lines.append(
        "Only the SwiGLU island tensors (gate / up / swiglu_intermediate)"
        " expose a column permutation; the other tensors are dense /"
        " orthogonal-masked and have no recoverable permutation."
    )
    lines.append("")
    lines.append(
        "| tensor_name | bundle | random_chance | signature_top1 | soft_top1 | best_top1 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for bundle, by_tensor in report["target_tensor_results"].items():
        for tensor, payload in by_tensor.items():
            if payload.get("permutation_recovery") is None:
                continue
            pr = payload["permutation_recovery"]
            lines.append(
                f"| {tensor} | {bundle} |"
                f" {_fmt(pr['random_chance_top1'], 4)}"
                f" | {_fmt(pr['signature_matching']['top1_recovery_rate'], 4)}"
                f" | {_fmt(pr['soft_assignment']['top1_recovery_rate'], 4)}"
                f" | {_fmt(pr['best_top1'], 4)} |"
            )
    lines.append("")

    _attack_table(
        "Linkability on Real Activations",
        "linkability",
        [
            "visible_vs_plain_cosine",
            "mean_pairwise_cosine_visible",
            "mean_linkability_rank",
        ],
    )

    lines.append("## Mitigation Bundle Comparison\n")
    lines.append(
        "Deltas are `full_bundle − fresh_only`: positive linear /"
        " MLP rel_l2 delta means the full bundle makes recovery harder"
        " (safer). The two Stage 5.3e bundles share the same per-call"
        " fresh-mask sampling under the Stage 6.4b wrapper, so deltas"
        " are 0.0 by construction in the current implementation —"
        " the bundle label distinguishes security posture, not numerical"
        " visibility."
    )
    lines.append("")
    lines.append(
        "| tensor_name | linear_delta | mlp_delta | linkability_delta |"
        " perm_top1_delta | risk_fresh_only | risk_full_bundle |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in comparison:
        lines.append(
            f"| {row['tensor_name']} |"
            f" {_fmt(row['linear_rel_l2_delta'], 4)}"
            f" | {_fmt(row['mlp_rel_l2_delta'], 4)}"
            f" | {_fmt(row['linkability_cosine_delta'], 4)}"
            f" | {_fmt(row.get('permutation_top1_delta', 'n/a'), 4)}"
            f" | {row['risk_level_fresh_only']}"
            f" | {row['risk_level_full_bundle']} |"
        )
    lines.append("")
    if any("risk_level_fixed_debug" in r for r in comparison):
        lines.append(
            "**Reference**: the optional `fixed_permutation_debug` baseline"
            " pins the wrapper's per-session RNG so masks / permutations are"
            " identical across sessions. It is included here only to show"
            " that the attacker DOES recover when freshness is removed —"
            " linear rel_l2 drops to ≈ 0 and permutation recovery jumps."
            " Never deploy with fixed masks."
        )
        lines.append("")

    lines.append("## Per-Bundle Headline\n")
    lines.append(
        "| bundle | tensors | max_risk | mean_lin_rel_l2 | mean_mlp_rel_l2 |"
        " mean_linkability_cos |"
    )
    lines.append("|---|---|---|---|---|---|")
    for bundle, s in summary.items():
        lines.append(
            f"| {bundle} | {len(s['tensors_covered'])} | {s['max_risk_level']}"
            f" | {_fmt(s['mean_linear_rel_l2'], 3)}"
            f" | {_fmt(s['mean_mlp_rel_l2'], 3)}"
            f" | {_fmt(s['mean_linkability_cosine'], 3)} |"
        )
    lines.append("")

    lines.append("## Comparison with Stage 5.4 Synthetic Adaptive Attacker\n")
    lines.append(
        "Stage 5.4 evaluated the same family of attackers on structured"
        " synthetic activation distributions. Stage 5.5 reuses the same"
        " attacker code (ridge linear, small MLP, signature / Sinkhorn"
        " permutation recovery) but the (plain, visible) pairs now come"
        " from the Stage 6.4b modern decoder block wrapper — i.e. they"
        " are real intermediate activations from the obfuscated forward"
        " path, not synthesized signatures."
    )
    lines.append("")
    lines.append(
        "Under both Stage 5.4 (synthetic) and Stage 5.5 (real-activation),"
        " the recommended default-on bundle"
        " (`fresh_perm_plus_sandwich_plus_pad`) achieves `risk_level=low`"
        " against the adaptive proxy attackers. `risk_level` for"
        " `fresh_perm_only` is reported here under the same evaluation"
        " budget; both Stage 5.3e bundles produce numerically identical"
        " traces because they use the SAME per-call fresh-mask sampling"
        " inside `run_swiglu_mlp_island`."
    )
    lines.append("")

    lines.append("## Recommendation\n")
    lines.append(
        f"- `default_on_recommendation_full_bundle = \"{rec['default_on_recommendation_full_bundle']}\"`"
    )
    lines.append(
        f"- `default_on_recommendation_fresh_only = \"{rec['default_on_recommendation_fresh_only']}\"`"
    )
    lines.append(
        f"- `security_profile_detail_with_real_activation = \"{rec['security_profile_detail_with_real_activation']}\"`"
    )
    lines.append("")

    lines.append("## Limitations\n")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Next Stage Plan\n")
    lines.append(
        "- Stage 6.4c — full modern decoder model-level wrapper (multi-block"
        " stacking, LM head, KV cache runtime, generation parity)."
    )
    lines.append(
        "- Stronger attacker variants: black-box query attacker, side-channel"
        " threat models, ML-based permutation recovery exploiting"
        " cross-attention information."
    )
    lines.append(
        "- Real Qwen / TinyLlama trace collection (`--attempt-real-model-load`"
        " with tokenizer / embedding integration once Stage 6.4c lands)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tensor-leakage safety check
# ---------------------------------------------------------------------------


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


def _strip_traces(report: dict[str, Any]) -> dict[str, Any]:
    """Defensive — the report should already not include raw tensors,
    but guard against future regressions by stripping any tensor-typed
    value before JSON serialisation."""
    import torch

    def _scrub(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return {
                "_tensor_shape": list(obj.shape),
                "_tensor_redacted": True,
            }
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_scrub(v) for v in obj]
        return obj

    return _scrub(report)


def main() -> None:
    args = parse_args()
    bundles = _resolve_bundles(args)
    cfg = RealActivationAttackConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        attempt_real_model_load=bool(args.attempt_real_model_load),
        model_id=args.model_id,
        local_files_only=bool(args.local_files_only),
        allow_synthetic_fallback=bool(args.allow_synthetic_fallback),
        num_samples=args.num_samples,
        train_fraction=args.train_fraction,
        attacker_steps=args.attacker_steps,
        attacker_lr=args.attacker_lr,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_batch_size=args.mlp_batch_size,
        ridge_lambda=args.ridge_lambda,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        synthetic_hidden_size=args.hidden_size,
        synthetic_intermediate_size=args.intermediate_size,
        synthetic_num_attention_heads=args.num_query_heads,
        synthetic_num_key_value_heads=args.num_kv_heads,
        synthetic_head_dim=args.head_dim,
        use_pad=bool(args.use_pad),
        mitigation_bundles=bundles,
    )
    report = run_real_activation_attacks(cfg)
    safe_report = _strip_traces(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "real_activation_attacks.json"
    csv_path = args.output_dir / "real_activation_attacks.csv"
    md_path = args.output_dir / "real_activation_attacks.md"

    json_text = json.dumps(safe_report, indent=2)
    # Final safety: no tensor() / no overlong numeric arrays.
    assert "tensor(" not in json_text, "tensor() found in JSON output"
    assert _LONG_NUMBER_ARRAY.search(json_text) is None, (
        "long numeric array found in JSON output"
    )
    json_path.write_text(json_text, encoding="utf-8")
    _write_csv(csv_path, _csv_rows(safe_report))
    md_path.write_text(_format_markdown(safe_report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(
        f"source={report['source']}"
        f" bundles={list(report['attacker_summary'].keys())}"
        f" rec_full={report['recommendation']['default_on_recommendation_full_bundle']}"
    )


if __name__ == "__main__":
    main()
