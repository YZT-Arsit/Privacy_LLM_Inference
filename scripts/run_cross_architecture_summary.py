#!/usr/bin/env python
"""Stage 6.3 — Cross-architecture coverage + correctness + workload summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    CrossArchitectureSummaryConfig,
    run_cross_architecture_summary,
)
from pllo.experiments.report_utils import (
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


UPSTREAM_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("attention_experiments.json", "scripts/run_attention_experiments.py"),
    ("encoder_attention_experiments.json", "scripts/run_encoder_attention_experiments.py"),
    ("cross_attention_experiments.json", "scripts/run_cross_attention_experiments.py"),
    ("architecture_coverage.json", "scripts/run_architecture_coverage.py"),
    ("workload_profile.json", "scripts/run_workload_profile.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument(
        "--require-existing-outputs",
        action="store_true",
        help="Fail if any upstream JSON is missing (default: record missing).",
    )
    parser.add_argument(
        "--rerun-upstream",
        action="store_true",
        help="Re-execute upstream scripts before aggregating (default: off).",
    )
    return parser.parse_args()


def _rerun_upstream(output_dir: Path) -> None:
    for _, script in UPSTREAM_SCRIPTS:
        cmd = [sys.executable, str(PROJECT_ROOT / script), "--output-dir", str(output_dir)]
        print(f"[rerun] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


CSV_FIELDS = (
    "architecture_type",
    "status",
    "model_id",
    "model_class",
    "attention_kind",
    "cache_type",
    "num_cells",
    "num_rows",
    "all_loaded_allclose",
    "max_output_error",
    "max_score_error",
    "max_prob_error",
    "max_cache_error",
    "use_pad_supported",
    "padding_mask_supported",
    "bias_q",
    "bias_k",
    "bias_v",
    "bias_o",
    "has_relative_attention_bias",
    "hidden_size",
    "num_layers",
    "num_heads",
    "trusted_shortcuts",
)


def _row(arch: dict) -> dict:
    bias = arch.get("bias_present") or {}
    spec = arch.get("coverage_spec") or {}
    return {
        "architecture_type": arch["architecture_type"],
        "status": arch["status"],
        "model_id": arch["model_id"],
        "model_class": arch["model_class"],
        "attention_kind": arch["attention_kind"],
        "cache_type": arch["cache_type"],
        "num_cells": arch["num_cells"],
        "num_rows": arch["num_rows"],
        "all_loaded_allclose": arch["all_loaded_allclose"],
        "max_output_error": arch["max_output_error"],
        "max_score_error": arch["max_score_error"],
        "max_prob_error": arch["max_prob_error"],
        "max_cache_error": arch["max_cache_error"],
        "use_pad_supported": "/".join(str(v) for v in arch["use_pad_supported"]),
        "padding_mask_supported": arch["padding_mask_supported"],
        "bias_q": bias.get("q"),
        "bias_k": bias.get("k"),
        "bias_v": bias.get("v"),
        "bias_o": bias.get("o"),
        "has_relative_attention_bias": arch["has_relative_attention_bias"],
        "hidden_size": spec.get("hidden_size"),
        "num_layers": spec.get("num_layers"),
        "num_heads": spec.get("num_heads"),
        "trusted_shortcuts": "/".join(arch["trusted_shortcuts"]),
    }


def _build_markdown(summary: dict) -> str:
    out: list[str] = []
    out.append(
        "# Privacy LLM Obfuscation — Cross-Architecture Summary (Stage 6.3)"
    )
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(summary["stage_note"])
    out.append("")

    # 1. Coverage table
    out.append("## Cross-architecture coverage table")
    out.append("")
    headers = [
        "architecture",
        "status",
        "model_id",
        "model_class",
        "attention_kind",
        "cache_type",
        "cells",
        "rows",
    ]
    rows = [
        [
            a["architecture_type"],
            a["status"],
            a["model_id"],
            a["model_class"],
            a["attention_kind"],
            a["cache_type"],
            a["num_cells"],
            a["num_rows"],
        ]
        for a in summary["architectures"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 2. Attention invariant summary
    out.append("## Attention invariant summary")
    out.append("")
    headers = [
        "architecture",
        "all allclose",
        "max output err",
        "max score err",
        "max prob err",
        "max cache err",
    ]
    rows = [
        [
            a["architecture_type"],
            a["all_loaded_allclose"],
            a["max_output_error"],
            a["max_score_error"],
            a["max_prob_error"],
            a["max_cache_error"],
        ]
        for a in summary["architectures"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 3. Cache support
    out.append("## Cache support summary")
    out.append("")
    headers = ["architecture", "cache_type", "max cache err"]
    rows = [
        [a["architecture_type"], a["cache_type"], a["max_cache_error"]]
        for a in summary["architectures"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # 4. Pad support
    out.append("## Pad support summary")
    out.append("")
    headers = [
        "architecture",
        "use_pad values seen",
        "padding mask supported",
        "bias (q/k/v/o)",
        "relative position bias",
    ]
    rows = []
    for a in summary["architectures"]:
        bias = a.get("bias_present") or {}
        bias_str = (
            f"{bias.get('q')}/{bias.get('k')}/{bias.get('v')}/{bias.get('o')}"
            if bias
            else "—"
        )
        rows.append(
            [
                a["architecture_type"],
                "/".join(str(v) for v in a["use_pad_supported"]) or "—",
                a["padding_mask_supported"],
                bias_str,
                a["has_relative_attention_bias"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    # 5. Workload summary
    out.append("## Workload summary (from Stage 5.0.1 profiler)")
    out.append("")
    workload = summary.get("workload", {})
    if workload.get("status") != "loaded":
        out.append("_workload_profile.json missing or unreadable._")
    else:
        headers = [
            "method",
            "implemented",
            "boundary calls",
            "boundary calls formula",
            "trusted compute ops",
            "gpu ops",
            "measured wall-time (ms)",
            "source",
        ]
        rows = []
        for m in workload["methods"]:
            rows.append(
                [
                    m["method"],
                    m["implemented"],
                    m["online_boundary_calls"],
                    m["boundary_calls_formula"],
                    m["online_trusted_compute_ops"],
                    m["online_gpu_ops"],
                    m["measured_wall_time_ms"],
                    m["wall_time_source"],
                ]
            )
        out.append(markdown_table(headers, rows))
    out.append("")

    # 6. Compatible Nonlinear Island Workload Projection (Stage 5.2c)
    out.append("## Compatible Nonlinear Island Workload Projection")
    out.append("")
    proj = summary.get("compatible_island_projection", {})
    if proj.get("status") != "available":
        out.append(
            "_Compatible nonlinear island workload projection is unavailable —"
            " upstream workload_profile.json does not yet contain the"
            " `ours_compatible_nonlinear_islands` method._"
        )
    else:
        record = proj.get("method_record", {})
        out.append(
            "ours_compatible_nonlinear_islands is a projected method based on"
            " Stage 5.2a correctness probes (28 cells, all_allclose=True,"
            " `online_extra_matmul_count = 0`) and Stage 5.2b security"
            " proxies. It is not yet integrated into GPT-2 / BERT / T5"
            " wrappers — Stage 5.3 is the integration step. Per-architecture"
            " status is `projected_from_probe`."
        )
        out.append("")
        out.append(
            f"- Boundary formula: `{record.get('boundary_calls_formula', 'n/a')}`"
        )
        out.append(
            f"- `online_extra_matmul_count` = {record.get('online_extra_matmul_count', 0)}"
        )
        out.append(
            f"- `security_profile` = `{record.get('security_profile', 'n/a')}`"
        )
        out.append("")
        headers = [
            "architecture",
            "model_id",
            "attention_kind",
            "current method",
            "current formula",
            "compatible formula",
            "boundary reduction",
            "trusted compute reduction",
            "online extra matmul",
            "status",
            "security_proxy_status",
        ]
        rows = []
        for entry in proj.get("per_architecture", []):
            rows.append(
                [
                    entry["architecture_type"],
                    entry["model_id"],
                    entry["attention_kind"],
                    entry["current_method"],
                    entry["current_boundary_formula"],
                    entry["compatible_boundary_formula"],
                    f"{entry['boundary_call_reduction']:.2%}"
                    if entry["boundary_call_reduction"] is not None
                    else "—",
                    f"{entry['trusted_compute_reduction']:.2%}"
                    if entry["trusted_compute_reduction"] is not None
                    else "—",
                    entry["online_extra_matmul_count"],
                    entry["status"],
                    entry["security_proxy_status"],
                ]
            )
        out.append(markdown_table(headers, rows))
        out.append("")
        out.append(
            "Security proxy caveats (from Stage 5.2b, applied to every"
            " architecture row above):"
        )
        out.append(
            "- Compatible mask families are weaker than unrestricted dense"
            " masks inside nonlinear islands."
        )
        out.append(
            "- Permutation islands hide channel identity but do not hide"
            " coordinate-value multisets."
        )
        out.append(
            "- Fresh permutation, dense sandwiching, and pad at Linear"
            " boundaries are required mitigations."
        )
        out.append(
            "- Not yet integrated into the GPT-2 / BERT / T5 wrappers"
            " (`projected_from_probe`, not measured). No real TEE isolation."
        )
        out.append("")

    # 6.5 Stage 5.3c — Compatible Island Integration Status (per architecture)
    integ = summary.get("compatible_island_integration_status", {})
    if integ.get("status") == "available":
        out.append("## Compatible Island Integration Status")
        out.append("")
        out.append(
            "Stage 5.3c — per-architecture status of the operator-compatible"
            " nonlinear-island integration. Default mode remains `trusted`"
            " for every wrapper; `compatible_islands` is gated behind a"
            " `nonlinear_mode` feature flag."
        )
        out.append("")
        rows: list[list[str]] = []
        headers = [
            "architecture_type",
            "model_id",
            "integration_level",
            "nonlinear_mode_available",
            "use_pad_supported",
            "online_extra_matmul_count",
            "security_proxy_status",
        ]
        for entry in integ.get("per_architecture", []):
            rows.append(
                [
                    entry["architecture_type"],
                    entry["model_id"],
                    entry["integration_level"],
                    "/".join(entry["nonlinear_mode_available"]),
                    str(entry["use_pad_supported"]),
                    str(entry["online_extra_matmul_count"]),
                    str(entry["security_proxy_status"]),
                ]
            )
        out.append(markdown_table(headers, rows))
        out.append("")
        scope = integ.get("measured_integration_scope")
        if scope:
            out.append(
                f"- `measured_integration_scope = \"{scope}\"`."
            )
        if "full_runtime_integrated" in integ:
            out.append(
                f"- `full_runtime_integrated = {integ['full_runtime_integrated']}`."
            )
        if "all_architecture_probe_level_implemented" in integ:
            out.append(
                "- `all_architecture_probe_level_implemented = "
                f"{integ['all_architecture_probe_level_implemented']}`."
            )
        out.append(
            "- GPT-2 model-level integration is available."
        )
        out.append(
            "- BERT/T5 are probe-level integrations, not full wrappers."
        )
        out.append(
            "- default mode remains `trusted`."
        )
        out.append(
            "- LayerNorm remains trusted unless explicitly stated otherwise."
        )
        out.append(
            "- no generation changes for BERT/T5."
        )
        out.append(
            "- security follows Stage 5.2b caveats (fresh permutation per"
            " session, dense sandwich at Linear boundaries, pad at Linear"
            " boundaries only)."
        )
        out.append(
            "- `security_profile` remains `proxy-evaluated, not formal`."
        )
        out.append(
            "- not a real TEE measurement."
        )
        out.append(
            "- not full BERT/T5 wrapper integration."
        )
        out.append("")
        out.append("### Per-architecture limitations")
        out.append("")
        for entry in integ.get("per_architecture", []):
            out.append(f"- **{entry['architecture_type']}**:")
            for lim in entry["limitations"]:
                out.append(f"  - {lim}")
        out.append("")

        # Stage 5.3e — Mitigation Bundle Support table.
        bundle_support = integ.get("mitigation_bundle_support", [])
        if bundle_support:
            out.append("## Mitigation Bundle Support")
            out.append("")
            out.append(
                "Stage 5.3e — per-architecture support for the two"
                " mitigation bundles."
                f" `default_mitigation_bundle = {integ['default_mitigation_bundle']!r}`"
                " (preserves backward compatibility);"
                f" `recommended_default_on_bundle = {integ['recommended_default_on_bundle']!r}`"
                f" → `{integ['recommended_default_on_status']}` per Stage 5.4"
                " adaptive proxy attackers. `compatible_islands` remains"
                " feature-flagged behind `nonlinear_mode`; default mode"
                " stays `\"trusted\"`."
            )
            out.append("")
            headers = [
                "architecture",
                "integration_level",
                "fresh_perm_only",
                "fresh_perm_plus_sandwich_plus_pad",
                "use_pad_supported",
                "dense_sandwich_enabled",
                "online_extra_matmul_count",
                "default_on_candidate",
                "security_profile",
            ]
            rows = [
                [
                    str(entry["architecture"]),
                    str(entry["integration_level"]),
                    str(entry["fresh_perm_only"]),
                    str(entry["fresh_perm_plus_sandwich_plus_pad"]),
                    str(entry["use_pad_supported"]),
                    str(entry["dense_sandwich_enabled"]),
                    str(entry["online_extra_matmul_count"]),
                    str(entry["default_on_candidate"]),
                    str(entry["security_profile"]),
                ]
                for entry in bundle_support
            ]
            out.append(markdown_table(headers, rows))
            out.append("")
            out.append(
                "- Bundle support is probe-level / wrapper-level: enabling"
                " the full bundle does NOT change the wrapper's default"
                " `nonlinear_mode` and does NOT promote"
                " `ours_compatible_nonlinear_islands.implemented` to"
                " `True`."
            )
            out.append(
                "- security is `adaptive-proxy-mitigated, not formal` when"
                " the full bundle is enabled; this is not a real TEE"
                " measurement."
            )
            out.append("")

    # 7. Trusted shortcuts per architecture
    out.append("## Trusted shortcuts still in place per architecture")
    out.append("")
    for a in summary["architectures"]:
        out.append(f"- **{a['architecture_type']}**:")
        for s in a["trusted_shortcuts"]:
            out.append(f"  - `{s}`")
    out.append("")

    # 7. Limitations
    out.append("## Limitations")
    out.append("")
    for a in summary["architectures"]:
        out.append(f"- **{a['architecture_type']}**:")
        for lim in a["limitations"]:
            out.append(f"  - {lim}")
    out.append("- This summary aggregates existing JSON; it does not re-run probes.")
    out.append("- It does not claim real TEE security; security claims are deferred to the security proxy report.")
    out.append("")

    # 8. Next stage plan
    out.append("## Next stage plan")
    out.append("")
    out.append(
        "- **Stage 5.1** — GPU-side LayerNorm primitive (replaces the trusted"
        " LayerNorm shortcut shared by all three architectures)."
    )
    out.append(
        "- **Stage 5.2** — GELU / activation primitive feasibility (replaces the"
        " trusted activation shortcut)."
    )
    out.append(
        "- **Stage 6.4** — Qwen / ModelScope migration on top of Stage 6.0+'s"
        " architecture scaffold once a non-trusted nonlinear primitive is ready."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.rerun_upstream:
        _rerun_upstream(args.output_dir)
    config = CrossArchitectureSummaryConfig(
        output_dir=str(args.output_dir),
        require_existing_outputs=args.require_existing_outputs,
    )
    summary = run_cross_architecture_summary(config)

    out_dir = args.output_dir
    write_json(out_dir / "cross_architecture_summary.json", summary)
    write_csv(
        out_dir / "cross_architecture_summary.csv",
        [_row(a) for a in summary["architectures"]],
        CSV_FIELDS,
    )
    write_text(
        out_dir / "cross_architecture_summary.md",
        _build_markdown(summary),
    )

    g = summary["global_summary"]
    print(
        f"architectures={g['num_architectures']}"
        f" aggregated={g['num_aggregated']} missing={g['num_missing']}"
        f" all_allclose={g['all_architectures_allclose']}"
        f" output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
