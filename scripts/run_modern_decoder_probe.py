#!/usr/bin/env python
"""Stage 6.4 — Modern decoder-only (Qwen / TinyLlama / LLaMA) probe runner.

Composes the RMSNorm + SwiGLU + RoPE + GQA probes and writes structured
JSON / CSV / Markdown reports to ``outputs/``.

Default behaviour: ``attempt_real_model_load=False`` (synthetic tensors
only). Pass ``--attempt-real-model-load`` to try the registered
``modern_decoder_only`` candidates; failures are recorded silently and
the synthetic probe runs regardless.
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

from pllo.experiments.modern_decoder_probe import (  # noqa: E402
    ModernDecoderProbeConfig,
    run_modern_decoder_probe,
)
from pllo.ops.mitigation_bundles import (  # noqa: E402
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--attempt-real-model-load",
        action="store_true",
        help=(
            "Try the registered modern_decoder_only candidates. Falls back"
            " to synthetic on any failure (silent skip)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mitigation-bundle",
        default=DEFAULT_MITIGATION_BUNDLE,
        choices=list(VALID_MITIGATION_BUNDLES),
        help="Mitigation bundle for the SwiGLU probe (default fresh_perm_only).",
    )
    parser.add_argument(
        "--both-bundles",
        action="store_true",
        help="Run BOTH mitigation bundles back-to-back; output keys both runs.",
    )
    return parser.parse_args()


def _csv_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    ml = report["model_loading"]
    rows.append(
        {
            "section": "model_loading",
            "probe": "model_loading",
            "use_pad": "n/a",
            "metric": "status",
            "value": ml["status"],
            "notes": ml.get("reason") or "",
        }
    )
    spec = report["architecture_spec"]
    for k in (
        "architecture_type",
        "model_family",
        "norm_type",
        "activation_type",
        "position_encoding_type",
        "attention_variant",
        "hidden_size",
        "intermediate_size",
        "num_query_heads",
        "num_kv_heads",
        "head_dim",
    ):
        rows.append(
            {
                "section": "architecture_spec",
                "probe": "architecture_spec",
                "use_pad": "n/a",
                "metric": k,
                "value": spec[k],
                "notes": "",
            }
        )
    for use_pad_key, r in report["rmsnorm_probe"]["per_use_pad"].items():
        for metric in (
            "rms_core_max_abs_error",
            "folded_output_max_abs_error",
            "relative_l2_error",
            "cosine_similarity",
            "allclose",
            "online_extra_matmul_count",
        ):
            rows.append(
                {
                    "section": "rmsnorm_probe",
                    "probe": "rmsnorm",
                    "use_pad": use_pad_key,
                    "metric": metric,
                    "value": r[metric],
                    "notes": r["use_pad_note"],
                }
            )
    for use_pad_key, r in report["swiglu_probe"]["per_use_pad"].items():
        for metric in (
            "max_abs_error",
            "relative_l2_error",
            "cosine_similarity",
            "allclose",
            "online_extra_matmul_count",
            "permutation_dim",
            "shared_permutation_for_up_gate",
            "pad_placement",
        ):
            rows.append(
                {
                    "section": "swiglu_probe",
                    "probe": "swiglu",
                    "use_pad": use_pad_key,
                    "metric": metric,
                    "value": r[metric],
                    "notes": "",
                }
            )
    a = report["rope_probe"]["probe_a_post_rope_masking_invariant"]
    for metric in ("max_abs_error", "relative_l2_error", "allclose"):
        rows.append(
            {
                "section": "rope_probe",
                "probe": "rope_post_mask",
                "use_pad": "n/a",
                "metric": metric,
                "value": a[metric],
                "notes": a["requirement"],
            }
        )
    for family, m in report["rope_probe"][
        "probe_b_pre_rope_mask_commutation"
    ]["per_family"].items():
        for metric in ("max_abs_error", "commutes", "expected_behavior"):
            rows.append(
                {
                    "section": "rope_probe",
                    "probe": f"rope_pre_mask:{family}",
                    "use_pad": "n/a",
                    "metric": metric,
                    "value": m[metric],
                    "notes": "feasibility / negative-result probe",
                }
            )
    g = report["gqa_probe"]
    if g.get("status") == "ok":
        for metric in (
            "attention_variant",
            "group_size",
            "mask_dimension",
            "qk_constraint_max_error_per_q_head",
        ):
            rows.append(
                {
                    "section": "gqa_probe",
                    "probe": "gqa",
                    "use_pad": "n/a",
                    "metric": metric,
                    "value": g[metric],
                    "notes": "",
                }
            )
        for path in ("score_path", "value_path"):
            for metric in ("max_abs_error", "relative_l2_error", "allclose"):
                rows.append(
                    {
                        "section": "gqa_probe",
                        "probe": f"gqa:{path}",
                        "use_pad": "n/a",
                        "metric": metric,
                        "value": g[path][metric],
                        "notes": "",
                    }
                )
    return rows


def _render_md(report: dict) -> str:
    cfg = report["config"]
    ml = report["model_loading"]
    spec = report["architecture_spec"]
    rms = report["rmsnorm_probe"]
    swi = report["swiglu_probe"]
    rope = report["rope_probe"]
    gqa = report["gqa_probe"]
    g = report["global_summary"]
    lines: list[str] = []
    lines.append("# Modern Decoder-Only (Qwen / TinyLlama) — Stage 6.4 Probe")
    lines.append("")
    lines.append("## Experiment Scope")
    lines.append("")
    lines.append(
        f"- batch_size: {cfg['batch_size']}; seq_len: {cfg['seq_len']};"
        f" hidden_size: {cfg['hidden_size']}; intermediate_size: {cfg['intermediate_size']}"
    )
    lines.append(
        f"- num_query_heads: {cfg['num_query_heads']}; num_kv_heads: {cfg['num_kv_heads']};"
        f" head_dim: {cfg['head_dim']}"
    )
    lines.append(
        f"- dtype: {cfg['dtype']}; device: {cfg['device']}; seed: {cfg['seed']}"
    )
    lines.append("")
    lines.append("## Model Loading Status")
    lines.append("")
    lines.append(f"- status: `{ml['status']}`")
    lines.append(f"- model_id: `{ml.get('model_id')}`")
    lines.append(f"- model_family: `{ml.get('model_family')}`")
    if ml.get("reason"):
        lines.append(f"- reason: {ml['reason']}")
    if ml.get("candidates_tried"):
        lines.append("- candidates_tried:")
        for c in ml["candidates_tried"]:
            lines.append(f"  - `{c}`")
    lines.append("")
    lines.append("## Modern Decoder Architecture Spec")
    lines.append("")
    for k in (
        "architecture_type",
        "model_family",
        "norm_type",
        "activation_type",
        "position_encoding_type",
        "attention_variant",
        "hidden_size",
        "intermediate_size",
        "num_query_heads",
        "num_kv_heads",
        "head_dim",
    ):
        lines.append(f"- `{k}`: `{spec[k]}`")
    lines.append("")
    lines.append("## RMSNorm Orthogonal Island Probe")
    lines.append("")
    lines.append(
        "| use_pad | rms_core_max_abs_error | folded_output_max_abs_error |"
        " allclose | online_extra_matmul_count |"
    )
    lines.append("|---|---|---|---|---|")
    for k, r in rms["per_use_pad"].items():
        lines.append(
            f"| {r['use_pad']} | {r['rms_core_max_abs_error']:.3e} |"
            f" {r['folded_output_max_abs_error']:.3e} | {r['allclose']} |"
            f" {r['online_extra_matmul_count']} |"
        )
    lines.append("")
    lines.append("## SwiGLU Paired-Permutation Island Probe")
    lines.append("")
    lines.append(
        "| use_pad | max_abs_error | permutation_dim | shared_permutation_for_up_gate |"
        " online_extra_matmul_count | pad_placement |"
    )
    lines.append("|---|---|---|---|---|---|")
    for k, r in swi["per_use_pad"].items():
        lines.append(
            f"| {r['use_pad']} | {r['max_abs_error']:.3e} | {r['permutation_dim']} |"
            f" {r['shared_permutation_for_up_gate']} |"
            f" {r['online_extra_matmul_count']} | `{r['pad_placement']}` |"
        )
    lines.append("")
    lines.append("## RoPE-Aware Attention Probe")
    lines.append("")
    a = rope["probe_a_post_rope_masking_invariant"]
    lines.append("**Probe A — post-RoPE masking invariant (REQUIRED).**")
    lines.append("")
    lines.append(
        f"- requirement: {a['requirement']}"
    )
    lines.append(
        f"- max_abs_error: {a['max_abs_error']:.3e};"
        f" relative_l2_error: {a['relative_l2_error']:.3e};"
        f" allclose: {a['allclose']}"
    )
    lines.append(
        f"- qk_constraint_error: {rope['qk_constraint_error']:.3e}"
    )
    lines.append("")
    lines.append(
        "**Probe B — pre-RoPE mask commutation (feasibility / negative result).**"
    )
    lines.append("")
    lines.append(
        "| mask_family | expected_behavior | commutes | max_abs_error |"
    )
    lines.append("|---|---|---|---|")
    for fam, m in rope["probe_b_pre_rope_mask_commutation"]["per_family"].items():
        lines.append(
            f"| `{fam}` | {m['expected_behavior']} | {m['commutes']} |"
            f" {m['max_abs_error']:.3e} |"
        )
    lines.append("")
    lines.append(
        rope["probe_b_pre_rope_mask_commutation"]["rope_mask_compatibility_notes"]
    )
    lines.append("")
    lines.append("## GQA / MQA KV Shape Probe")
    lines.append("")
    if gqa.get("status") == "ok":
        lines.append(f"- attention_variant: `{gqa['attention_variant']}`")
        lines.append(f"- group_size: {gqa['group_size']}")
        lines.append(f"- mask_dimension: {gqa['mask_dimension']} (= head_dim)")
        lines.append(
            "- mask is per-head, NOT hidden_size, NOT num_heads."
        )
        lines.append(
            f"- qk_constraint_max_error_per_q_head:"
            f" {gqa['qk_constraint_max_error_per_q_head']:.3e}"
        )
        lines.append(
            f"- score_path: max_abs_error={gqa['score_path']['max_abs_error']:.3e},"
            f" allclose={gqa['score_path']['allclose']}"
        )
        lines.append(
            f"- value_path: max_abs_error={gqa['value_path']['max_abs_error']:.3e},"
            f" allclose={gqa['value_path']['allclose']}"
        )
    else:
        lines.append(f"- status: `{gqa.get('status')}`")
        lines.append(f"- reason: {gqa.get('reason')}")
    lines.append("")
    lines.append("## Workload / Integration Status")
    lines.append("")
    lines.append(f"- architecture_type: `{g['architecture_type']}`")
    lines.append(f"- model_family: `{g['model_family']}`")
    lines.append(f"- norm_type: `{g['norm_type']}`")
    lines.append(f"- activation_type: `{g['activation_type']}`")
    lines.append(f"- position_encoding_type: `{g['position_encoding_type']}`")
    lines.append(f"- attention_variant: `{g['attention_variant']}`")
    lines.append(f"- integration_level: `{g['integration_level']}` (Stage 6.4 — probe-level migration)")
    lines.append(f"- all_required_probes_allclose: `{g['all_required_probes_allclose']}`")
    lines.append(f"- online_extra_matmul_count: `{g['online_extra_matmul_count']}`")
    lines.append(f"- default_nonlinear_mode: `{g['default_nonlinear_mode']}`")
    lines.append(
        "- workload_profiler integration: see"
        " `wrapper_integration_status.ours_compatible_nonlinear_islands.qwen_or_modern_decoder`"
        " and `cross_architecture_summary.compatible_island_integration_status`."
    )
    lines.append("")
    lines.append("## Security Caveats from Stage 5.4")
    lines.append("")
    lines.append(
        f"- security_profile: `{g['security_profile']}`"
    )
    lines.append(
        "- inherits Stage 5.4 mitigation table: fixed permutation is"
        " unsafe_default_on; fresh permutation alone needs_more_evaluation;"
        " dense sandwich + fresh permutation + pad at Linear boundaries is"
        " the recommended default-on candidate."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    lines.append("## Next Stage Plan")
    lines.append("")
    lines.append(
        "- Stage 6.4b — Real Qwen / TinyLlama small-model loading and"
        " block-level wrapper integration behind the same"
        " `nonlinear_mode` feature flag (default `trusted`)."
    )
    lines.append(
        "- Stage 5.3e — Dense-sandwich integration inside the existing"
        " wrapper / probe paths so the Stage 5.4 default-on mitigation"
        " bundle becomes selectable end-to-end."
    )
    lines.append(
        "- Stage 5.5 — Stronger adaptive attackers on real model activations."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bundles = (
        list(VALID_MITIGATION_BUNDLES) if args.both_bundles else [args.mitigation_bundle]
    )
    reports: dict[str, dict] = {}
    for bundle in bundles:
        config = ModernDecoderProbeConfig(
            output_dir=str(args.output_dir),
            model_id=args.model_id,
            attempt_real_model_load=args.attempt_real_model_load,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=args.dtype,
            device=args.device,
            seed=args.seed,
            mitigation_bundle=bundle,
        )
        reports[bundle] = run_modern_decoder_probe(config)
    # Primary report = the bundle the user explicitly chose (or
    # fresh_perm_only when --both-bundles was passed without preference).
    primary_bundle = (
        args.mitigation_bundle if not args.both_bundles else DEFAULT_MITIGATION_BUNDLE
    )
    report = reports[primary_bundle]
    if args.both_bundles:
        report = dict(report)
        report["both_bundles_summary"] = {
            b: r["global_summary"] for b, r in reports.items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "modern_decoder_probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows = _csv_rows(report)
    with (args.output_dir / "modern_decoder_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["section", "probe", "use_pad", "metric", "value", "notes"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (args.output_dir / "modern_decoder_probe.md").write_text(
        _render_md(report), encoding="utf-8"
    )
    g = report["global_summary"]
    print(
        f"model_loading={report['model_loading']['status']}"
        f" family={g['model_family']}"
        f" attention_variant={g['attention_variant']}"
        f" mitigation_bundle={g.get('mitigation_bundle')}"
        f" all_allclose={g['all_required_probes_allclose']}"
        f" online_extra_matmul_count={g['online_extra_matmul_count']}"
    )
    if args.both_bundles:
        for b, r in reports.items():
            print(
                f"  bundle={b}: all_allclose={r['global_summary']['all_required_probes_allclose']}"
            )


if __name__ == "__main__":
    main()
