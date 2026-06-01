#!/usr/bin/env python
"""Stage 5.5b — real-token-prompted real-activation attacker runner.

Drives :func:`pllo.experiments.real_token_activation_attacker.run_real_token_activation_attacks`
and emits ``outputs/real_token_activation_attacks.{json,csv,md}``.

Default behaviour: synthetic token + synthetic model fallback, both
Stage 5.3e bundles (``fresh_perm_only`` and
``fresh_perm_plus_sandwich_plus_pad``), max_new_tokens=3, greedy only.

Real tokenizer / real model loading is opt-in via
``--attempt-tokenizer-load`` and ``--attempt-real-model-load``. pytest
defaults stay synthetic so the test suite never reads from the network.
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

from pllo.experiments.real_token_activation_attacker import (  # noqa: E402
    RealTokenActivationAttackConfig,
    run_real_token_activation_attacks,
)
from pllo.ops.mitigation_bundles import VALID_MITIGATION_BUNDLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--attempt-real-model-load", action="store_true")
    p.add_argument("--attempt-tokenizer-load", action="store_true")
    p.add_argument("--model-id", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument(
        "--no-synthetic-fallback",
        dest="allow_synthetic_fallback",
        action="store_false", default=True,
    )
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--prompt-max-length", type=int, default=16)
    p.add_argument("--max-layers", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--attacker-steps", type=int, default=200)
    p.add_argument("--attacker-lr", type=float, default=1e-2)
    p.add_argument("--mlp-hidden-size", type=int, default=128)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument(
        "--nonlinear-mode",
        choices=("trusted", "compatible_islands"),
        default="compatible_islands",
    )
    p.add_argument(
        "--inter-block-mask-mode",
        choices=("plain_boundary", "masked_boundary_experimental"),
        default="plain_boundary",
        help=(
            "Stage 5.6 extension. Default plain_boundary preserves Stage"
            " 6.4c behaviour; masked_boundary_experimental routes through"
            " the masked-residual model-wrapper path."
        ),
    )
    p.add_argument(
        "--bundle", action="append",
        choices=list(VALID_MITIGATION_BUNDLES),
        help="Restrict bundle list. Defaults to both Stage 5.3e bundles.",
    )
    # Synthetic-fallback model shape.
    p.add_argument("--synthetic-vocab-size", type=int, default=256)
    p.add_argument("--synthetic-hidden-size", type=int, default=32)
    p.add_argument("--synthetic-intermediate-size", type=int, default=64)
    p.add_argument("--synthetic-num-query-heads", type=int, default=4)
    p.add_argument("--synthetic-num-kv-heads", type=int, default=2)
    p.add_argument("--synthetic-head-dim", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV emitter (long format)
# ---------------------------------------------------------------------------


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ml = report["model_loading"]
    for k in (
        "load_status", "resolved_model_id", "model_family", "fallback_used",
    ):
        rows.append({
            "section": "model_loading", "scope": "n/a", "tensor_name": "n/a",
            "mitigation_bundle": "n/a", "attack": "n/a",
            "metric": k, "value": ml.get(k), "notes": "",
        })
    tl = report.get("tokenizer_loading", {})
    for k in ("tokenizer_status", "tokenizer_id", "tokenizer_error"):
        rows.append({
            "section": "tokenizer_loading", "scope": "n/a", "tensor_name": "n/a",
            "mitigation_bundle": "n/a", "attack": "n/a",
            "metric": k, "value": tl.get(k), "notes": "",
        })
    ps = report.get("prompt_summary", {})
    for k in (
        "token_source", "tokenizer_status", "num_prompts",
        "prompt_max_length", "vocab_size_used",
    ):
        rows.append({
            "section": "prompt_summary", "scope": "n/a", "tensor_name": "n/a",
            "mitigation_bundle": "n/a", "attack": "n/a",
            "metric": k, "value": ps.get(k), "notes": "",
        })
    for bundle, by_scope in report["target_tensor_results"].items():
        for scope, by_tensor in by_scope.items():
            for tensor, payload in by_tensor.items():
                for attack_section, attack_label in (
                    ("linear_inverter", "linear"),
                    ("mlp_inverter", "mlp"),
                    ("linkability", "linkability"),
                ):
                    d = payload.get(attack_section) or {}
                    for metric, value in d.items():
                        rows.append({
                            "section": attack_section, "scope": scope,
                            "tensor_name": tensor,
                            "mitigation_bundle": bundle, "attack": attack_label,
                            "metric": metric, "value": value, "notes": "",
                        })
                if payload.get("permutation_recovery") is not None:
                    pr = payload["permutation_recovery"]
                    for sub_attack in ("signature_matching", "soft_assignment"):
                        sub = pr.get(sub_attack, {})
                        for metric, value in sub.items():
                            rows.append({
                                "section": "permutation_recovery",
                                "scope": scope, "tensor_name": tensor,
                                "mitigation_bundle": bundle,
                                "attack": sub_attack,
                                "metric": metric, "value": value, "notes": "",
                            })
                rows.append({
                    "section": "decision", "scope": scope, "tensor_name": tensor,
                    "mitigation_bundle": bundle, "attack": "summary",
                    "metric": "risk_level", "value": payload["risk_level"],
                    "notes": payload["default_on_recommendation"],
                })
    for row in report["bundle_comparison"]:
        for k, v in row.items():
            if k in ("tensor_name", "scope"):
                continue
            rows.append({
                "section": "bundle_comparison", "scope": row["scope"],
                "tensor_name": row["tensor_name"],
                "mitigation_bundle": "summary", "attack": "delta_or_value",
                "metric": k, "value": v, "notes": "",
            })
    rec = report["recommendation"]
    for k, v in rec.items():
        rows.append({
            "section": "recommendation", "scope": "n/a", "tensor_name": "n/a",
            "mitigation_bundle": "summary", "attack": "n/a",
            "metric": k, "value": v, "notes": "",
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "section", "scope", "tensor_name", "mitigation_bundle",
                "attack", "metric", "value", "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


_LIMITATION_BACKSTOP = (
    "Not formal security; not a real TEE measurement;"
    " synthetic token fallback when tokenizer is unavailable;"
    " Dense sandwiching reduces tested recovery but does not imply semantic security."
)


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = report["config"]
    ml = report["model_loading"]
    tl = report.get("tokenizer_loading", {})
    ps = report.get("prompt_summary", {})
    spec = report.get("block_spec_summary", {})
    meta = report.get("metadata", {})
    summary = report["attacker_summary"]
    comparison = report["bundle_comparison"]
    gen_summary = report.get("generation_summary", {})
    rec = report["recommendation"]

    lines.append("# Real-Token-Prompted Real-Activation Attacker (Stage 5.5b)\n")

    lines.append("## Experiment Scope\n")
    lines.append(
        "Stage 5.5b re-runs the Stage 5.5 adaptive proxy attackers (ridge"
        " linear inverter, small MLP inverter, signature / Sinkhorn"
        " permutation recovery, linkability proxy) but the (plain, visible)"
        " trace pairs now come from the Stage 6.4c modern decoder"
        " model-level wrapper driven by real (or deterministically synthetic)"
        " token IDs — embedding lookup → N blocks → final RMSNorm → optionally"
        " masked LM head — covering both PREFILL and DECODE_STEP paths."
        " Default mode for the wider system remains `nonlinear_mode='trusted'`;"
        " default mitigation bundle remains `fresh_perm_only`. The numbers"
        " below are a real-token-prompted adaptive proxy evaluation of"
        " `fresh_perm_plus_sandwich_plus_pad` — they are NOT a formal"
        " security proof."
    )
    lines.append("")

    lines.append("## Model and Tokenizer Loading Status\n")
    lines.append("| field | value |\n|---|---|")
    for k in ("load_status", "resolved_model_id", "model_family",
              "fallback_used", "load_error"):
        lines.append(f"| model_loading.{k} | {ml.get(k)} |")
    for k in ("tokenizer_status", "tokenizer_id", "tokenizer_error"):
        lines.append(f"| tokenizer_loading.{k} | {tl.get(k)} |")
    lines.append(f"| source | {report.get('source')} |")
    lines.append(f"| num_layers_used | {meta.get('num_layers_used')} |")
    lines.append("")

    lines.append("## Prompt Set Summary\n")
    lines.append("| field | value |\n|---|---|")
    for k in ("token_source", "tokenizer_status", "num_prompts",
              "prompt_max_length", "vocab_size_used"):
        lines.append(f"| {k} | {ps.get(k)} |")
    lines.append(f"| max_new_tokens | {meta.get('max_new_tokens')} |")
    lines.append("")

    lines.append("## Trace Collection Summary\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k in ("hidden_size", "intermediate_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "attention_variant",
              "rope_base"):
        lines.append(f"| block_spec.{k} | {spec.get(k)} |")
    lines.append(f"| all_prefill_allclose | {meta.get('all_prefill_allclose')} |")
    lines.append(f"| all_decode_allclose | {meta.get('all_decode_allclose')} |")
    lines.append(f"| rope_position_increment | {meta.get('rope_position_increment')} |")
    lines.append(
        f"| rope_positions_seen | "
        f"[{meta.get('rope_positions_seen_min')}, "
        f"{meta.get('rope_positions_seen_max')}] |"
    )
    lines.append("")

    # Per-scope per-tensor inventory.
    lines.append("## Target Tensor Inventory (Prefill)\n")
    inventory_prefill = (
        report["trace_summary"].get(
            "fresh_perm_plus_sandwich_plus_pad", {},
        ).get("prefill", {})
    )
    lines.append(
        "| tensor_name | feature_dim | num_samples | plain abs_max"
        " | visible abs_max | plain_fingerprint | visible_fingerprint |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in inventory_prefill.items():
        lines.append(
            f"| {name} | {s['feature_dim']} | {s['num_samples']}"
            f" | {_fmt(s['plain_statistics']['abs_max'], 3)}"
            f" | {_fmt(s['visible_statistics']['abs_max'], 3)}"
            f" | {s['plain_statistics']['fingerprint_sha256_prefix']}"
            f" | {s['visible_statistics']['fingerprint_sha256_prefix']} |"
        )
    lines.append("")

    lines.append("## Target Tensor Inventory (Decode)\n")
    inventory_decode = (
        report["trace_summary"].get(
            "fresh_perm_plus_sandwich_plus_pad", {},
        ).get("decode", {})
    )
    if inventory_decode:
        lines.append(
            "| tensor_name | feature_dim | num_samples | plain abs_max"
            " | visible abs_max |"
        )
        lines.append("|---|---|---|---|---|")
        for name, s in inventory_decode.items():
            lines.append(
                f"| {name} | {s['feature_dim']} | {s['num_samples']}"
                f" | {_fmt(s['plain_statistics']['abs_max'], 3)}"
                f" | {_fmt(s['visible_statistics']['abs_max'], 3)} |"
            )
    else:
        lines.append("_Decode trace collection produced no rows_ (`max_new_tokens` may have been 1).")
    lines.append("")

    def _attack_table(title: str, attack_section: str, metric_keys: list[str]) -> None:
        lines.append(f"## {title}\n")
        lines.append(
            "| scope | tensor_name | bundle | "
            + " | ".join(metric_keys) + " |"
        )
        lines.append(
            "|---|---|---|" + "|".join(["---"] * len(metric_keys)) + "|"
        )
        for bundle, by_scope in report["target_tensor_results"].items():
            for scope, by_tensor in by_scope.items():
                for tensor, payload in by_tensor.items():
                    d = payload.get(attack_section) or {}
                    row_vals = [_fmt(d.get(k, "n/a"), 4) for k in metric_keys]
                    lines.append(
                        f"| {scope} | {tensor} | {bundle} | "
                        + " | ".join(row_vals) + " |"
                    )
        lines.append("")

    lines.append("## Prefill Real-Token Activation Attacks\n")
    lines.append(
        "Per-tensor attacker results on real-token-driven prefill activations."
        " Tensor-by-tensor breakdowns follow in the Linear / MLP / Permutation"
        " sections."
    )
    lines.append("")

    lines.append("## Decode-Step Real-Token Activation Attacks\n")
    lines.append(
        "Per-tensor attacker results on the single-token decode_step activations"
        " under the masked KV-cache append surface."
    )
    lines.append("")

    _attack_table(
        "Linear Inverter Results",
        "linear_inverter",
        ["relative_l2_error", "mse", "cosine_similarity"],
    )
    _attack_table(
        "Small MLP Inverter Results",
        "mlp_inverter",
        ["relative_l2_error", "mse", "cosine_similarity", "final_train_loss"],
    )

    lines.append("## Permutation Recovery Results\n")
    lines.append(
        "Only the SwiGLU island tensors (gate / up / swiglu_intermediate)"
        " expose a column permutation; the other tensors are dense /"
        " orthogonal-masked or plain at the inter-block boundary."
    )
    lines.append("")
    lines.append(
        "| scope | tensor_name | bundle | random_chance | signature_top1"
        " | soft_top1 | best_top1 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for bundle, by_scope in report["target_tensor_results"].items():
        for scope, by_tensor in by_scope.items():
            for tensor, payload in by_tensor.items():
                pr = payload.get("permutation_recovery")
                if pr is None:
                    continue
                lines.append(
                    f"| {scope} | {tensor} | {bundle}"
                    f" | {_fmt(pr['random_chance_top1'], 4)}"
                    f" | {_fmt(pr['signature_matching']['top1_recovery_rate'], 4)}"
                    f" | {_fmt(pr['soft_assignment']['top1_recovery_rate'], 4)}"
                    f" | {_fmt(pr['best_top1'], 4)} |"
                )
    lines.append("")

    _attack_table(
        "Linkability Results",
        "linkability",
        [
            "visible_vs_plain_cosine", "mean_pairwise_cosine_visible",
            "mean_linkability_rank",
        ],
    )

    lines.append("## Bundle Comparison\n")
    lines.append(
        "Deltas are `full_bundle − fresh_only`: positive linear / MLP rel_l2"
        " delta means the full bundle makes recovery harder (safer). The two"
        " Stage 5.3e bundles share the same per-call fresh-mask sampling under"
        " the Stage 6.4c wrapper, so deltas are 0.0 by construction —"
        " the bundle label distinguishes security posture, not numerical"
        " visibility."
    )
    lines.append("")
    lines.append(
        "| scope | tensor_name | inter_block_plain | linear_delta | mlp_delta"
        " | linkability_delta | perm_top1_delta | risk_fresh_only"
        " | risk_full_bundle |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in comparison:
        lines.append(
            f"| {row['scope']} | {row['tensor_name']}"
            f" | {row['inter_block_plain']}"
            f" | {_fmt(row['linear_rel_l2_delta'], 4)}"
            f" | {_fmt(row['mlp_rel_l2_delta'], 4)}"
            f" | {_fmt(row['linkability_cosine_delta'], 4)}"
            f" | {_fmt(row.get('permutation_top1_delta', 'n/a'), 4)}"
            f" | {row['risk_level_fresh_only']}"
            f" | {row['risk_level_full_bundle']} |"
        )
    lines.append("")

    lines.append("## Per-Bundle Headline\n")
    lines.append(
        "Headline grades are reported twice: `masked_only` excludes the"
        " structurally-plain inter-block tensors (`boundary_input`, `final`);"
        " `overall` includes them so the structural limitation stays visible."
    )
    lines.append("")
    lines.append(
        "| bundle | tensors | max_risk (masked_only) | max_risk (overall)"
        " | mean_lin_rel_l2 (masked) | mean_mlp_rel_l2 (masked)"
        " | mean_linkability_cos (masked) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for bundle, s in summary.items():
        lines.append(
            f"| {bundle} | {len(s['tensors_covered'])}"
            f" | {s['max_risk_level_masked_only']}"
            f" | {s['max_risk_level_overall']}"
            f" | {_fmt(s['mean_linear_rel_l2_masked_only'], 3)}"
            f" | {_fmt(s['mean_mlp_rel_l2_masked_only'], 3)}"
            f" | {_fmt(s['mean_linkability_cosine_masked_only'], 3)} |"
        )
    lines.append("")

    lines.append("## Generation Token Match\n")
    lines.append("| bundle | mean_token_match_rate | all_exact_match |")
    lines.append("|---|---|---|")
    for bundle, g in gen_summary.items():
        lines.append(
            f"| {bundle} | {_fmt(g.get('mean_token_match_rate', 0.0), 3)}"
            f" | {g.get('all_sequences_exact_match')} |"
        )
    lines.append("")

    lines.append("## Comparison with Stage 5.5 Random-Hidden Real-Activation Attacker\n")
    cmp = report.get("comparison_with_stage_5_5", {})
    lines.append(
        f"- Stage 5.5 artifact: `{cmp.get('stage_5_5_random_hidden_artifact')}`"
    )
    lines.append(
        f"- Stage 5.5b artifact: `{cmp.get('stage_5_5b_real_token_artifact')}`"
    )
    lines.append("")
    for d in cmp.get("key_differences", []):
        lines.append(f"- {d}")
    lines.append("")

    lines.append("## Recommendation\n")
    for k in (
        "default_on_recommendation_full_bundle_masked_only",
        "default_on_recommendation_full_bundle_overall",
        "default_on_recommendation_fresh_only_masked_only",
        "default_on_recommendation_fresh_only_overall",
        "security_profile_detail_with_real_token_activation",
    ):
        lines.append(f"- `{k} = \"{rec.get(k)}\"`")
    lines.append(f"- _Note_: {rec.get('note')}")
    lines.append("")

    lines.append("## Limitations\n")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    # Add a single backstop sentence so the markdown reliably mentions
    # "not formal security" / "not a real TEE measurement" /
    # "synthetic token fallback" / dense-sandwich caveat.
    lines.append(f"- {_LIMITATION_BACKSTOP}")
    lines.append("")

    lines.append("## Next Stage Plan\n")
    lines.append(
        "- Stage 5.6 — stronger attacker variants under the real-token surface:"
        " black-box query attacker (no plaintext supervision), side-channel"
        " threat models (timing / cache), ML-based permutation recovery"
        " exploiting cross-attention information."
    )
    lines.append(
        "- Stage 7.0 (deferred) — LoRA private-training path under the same"
        " obfuscation envelope (mask scheduling under autograd, fresh-mask"
        " budget per step)."
    )
    lines.append(
        "- Real Qwen / TinyLlama prompts via `--attempt-tokenizer-load"
        " --attempt-real-model-load --model-id <id>`."
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

    scrubbed = _scrub(report)
    # The in-memory traces dict carries the raw torch tensors; drop them.
    scrubbed.pop("traces", None)
    return scrubbed


def main() -> None:
    args = parse_args()
    bundles = tuple(args.bundle) if args.bundle else tuple(VALID_MITIGATION_BUNDLES)
    cfg = RealTokenActivationAttackConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        model_id=args.model_id,
        attempt_real_model_load=bool(args.attempt_real_model_load),
        attempt_tokenizer_load=bool(args.attempt_tokenizer_load),
        local_files_only=bool(args.local_files_only),
        allow_synthetic_fallback=bool(args.allow_synthetic_fallback),
        num_prompts=args.num_prompts,
        prompt_max_length=args.prompt_max_length,
        max_layers=args.max_layers,
        max_new_tokens=args.max_new_tokens,
        attacker_steps=args.attacker_steps,
        attacker_lr=args.attacker_lr,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_batch_size=args.mlp_batch_size,
        ridge_lambda=args.ridge_lambda,
        train_fraction=args.train_fraction,
        mitigation_bundles=bundles,
        use_pad=bool(args.use_pad),
        nonlinear_mode=args.nonlinear_mode,
        inter_block_mask_mode=args.inter_block_mask_mode,
        synthetic_vocab_size=args.synthetic_vocab_size,
        synthetic_hidden_size=args.synthetic_hidden_size,
        synthetic_intermediate_size=args.synthetic_intermediate_size,
        synthetic_num_attention_heads=args.synthetic_num_query_heads,
        synthetic_num_key_value_heads=args.synthetic_num_kv_heads,
        synthetic_head_dim=args.synthetic_head_dim,
    )
    report = run_real_token_activation_attacks(cfg)
    safe_report = _strip_traces(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "real_token_activation_attacks.json"
    csv_path = args.output_dir / "real_token_activation_attacks.csv"
    md_path = args.output_dir / "real_token_activation_attacks.md"

    json_text = json.dumps(safe_report, indent=2)
    assert "tensor(" not in json_text, "tensor() found in JSON output"
    assert _LONG_NUMBER_ARRAY.search(json_text) is None, (
        "long numeric array found in JSON output"
    )
    json_path.write_text(json_text, encoding="utf-8")
    _write_csv(csv_path, _csv_rows(safe_report))
    md_text = _format_markdown(safe_report)
    assert "tensor(" not in md_text, "tensor() found in markdown output"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(
        f"source={report.get('source')}"
        f" token_source={report['prompt_summary'].get('token_source')}"
        f" rec_full_masked={report['recommendation']['default_on_recommendation_full_bundle_masked_only']}"
    )


if __name__ == "__main__":
    main()
