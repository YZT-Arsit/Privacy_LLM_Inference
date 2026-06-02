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

        # Stage 6.4b / 6.4c — Modern Decoder integration callout.
        modern_row = integ.get("modern_decoder_row") or {}
        if (
            modern_row.get("modern_decoder_model_wrapper_status") == "implemented"
        ):
            out.append("## Modern Decoder Model-Level Integration (Stage 6.4c)")
            out.append("")
            out.append(
                "Stage 6.4c stacks the Stage 6.4b block wrapper into a"
                " multi-layer model-level obfuscated decoder with"
                " embedding lookup, final RMSNorm, an optionally-masked"
                " LM head, KV-cache-aware prefill / decode_step, and a"
                " hand-written greedy generation loop. Real Qwen /"
                " TinyLlama loading is opt-in; pytest stays synthetic."
            )
            out.append("")
        elif modern_row.get("modern_decoder_block_wrapper_status") == "implemented":
            out.append("## Modern Decoder Block-Level Integration (Stage 6.4b)")
            out.append("")
            out.append(
                "Stage 6.4b lands a block-level obfuscated forward for"
                " modern decoder-only architectures (LLaMA / TinyLlama /"
                " Qwen / Qwen2). The wrapper loads a real HF model (best"
                " effort; falls back to synthetic on failure or when"
                " `attempt_real_model_load=False`), extracts one transformer"
                " block, and verifies the recovered obfuscated output"
                " matches the plain reference for both mitigation bundles"
                " and both `use_pad` values."
            )
            out.append("")
        if (
            modern_row.get("modern_decoder_model_wrapper_status") == "implemented"
            or modern_row.get("modern_decoder_block_wrapper_status") == "implemented"
        ):
            out.append("| field | value |")
            out.append("|---|---|")
            out.append(
                f"| integration_level | {modern_row.get('integration_level')} |"
            )
            out.append(
                f"| modern_decoder_block_wrapper_status |"
                f" {modern_row.get('modern_decoder_block_wrapper_status')} |"
            )
            out.append(
                "| norm_type / activation_type / position_encoding |"
                f" {modern_row.get('norm_type')} /"
                f" {modern_row.get('activation_type')} /"
                f" {modern_row.get('position_encoding_type')} |"
            )
            out.append(
                "| attention_variant |"
                f" {modern_row.get('attention_variant')} |"
            )
            out.append(
                "| online_extra_matmul_count |"
                f" {modern_row.get('online_extra_matmul_count')} |"
            )
            out.append(
                "| security_proxy_status |"
                f" {modern_row.get('security_proxy_status')} |"
            )
            out.append(
                "| block_level_correctness_artifact |"
                f" `{modern_row.get('block_level_correctness_artifact')}` |"
            )
            if modern_row.get("modern_decoder_model_wrapper_status") == "implemented":
                out.append(
                    "| modern_decoder_model_wrapper_status |"
                    f" {modern_row.get('modern_decoder_model_wrapper_status')} |"
                )
                out.append(
                    "| modern_decoder_generation_status |"
                    f" {modern_row.get('modern_decoder_generation_status')} |"
                )
                out.append(
                    "| modern_decoder_kv_cache_status |"
                    f" {modern_row.get('modern_decoder_kv_cache_status')} |"
                )
                out.append(
                    "| model_level_correctness_artifact |"
                    f" `{modern_row.get('model_level_correctness_artifact')}` |"
                )
            ra_status = modern_row.get("real_activation_attacker_status")
            if ra_status and ra_status != "not_yet":
                out.append(
                    "| real_activation_attacker_status |"
                    f" {ra_status} |"
                )
                out.append(
                    "| real_activation_attacker_scope |"
                    f" {modern_row.get('real_activation_attacker_scope')} |"
                )
                out.append(
                    "| real_activation_attacker_artifact |"
                    f" `{modern_row.get('real_activation_attacker_artifact')}` |"
                )
            rt_status = modern_row.get(
                "real_token_activation_attacker_status"
            )
            if rt_status and rt_status != "not_yet":
                out.append(
                    "| real_token_activation_attacker_status |"
                    f" {rt_status} |"
                )
                out.append(
                    "| real_token_activation_attacker_scope |"
                    f" {modern_row.get('real_token_activation_attacker_scope')} |"
                )
                out.append(
                    "| real_token_activation_attacker_artifact |"
                    " `"
                    f"{modern_row.get('real_token_activation_attacker_artifact')}"
                    "` |"
                )
                detail = modern_row.get(
                    "security_profile_detail_with_real_token_activation"
                )
                if detail:
                    out.append(
                        "| security_profile_detail_with_real_token_activation |"
                        f" {detail} |"
                    )
            ext_status = modern_row.get("extended_proxy_status")
            if ext_status and ext_status != "not_yet":
                out.append(
                    f"| extended_proxy_status | {ext_status} |"
                )
                out.append(
                    "| extended_proxy_artifact |"
                    f" `{modern_row.get('extended_proxy_artifact')}` |"
                )
                out.append(
                    "| inter_block_mask_mode_supported |"
                    f" {modern_row.get('inter_block_mask_mode_supported')} |"
                )
                out.append(
                    "| masked_boundary_experimental_status |"
                    f" {modern_row.get('masked_boundary_experimental_status')} |"
                )
                out.append(
                    "| constant_time_decode_proxy_status |"
                    f" {modern_row.get('constant_time_decode_proxy_status')} |"
                )
                ext_detail = modern_row.get(
                    "security_profile_detail_with_extended_proxy"
                )
                if ext_detail:
                    out.append(
                        "| security_profile_detail_with_extended_proxy |"
                        f" {ext_detail} |"
                    )
            sa_status = modern_row.get("stronger_attackers_status")
            if sa_status and sa_status != "not_yet":
                out.append(
                    f"| stronger_attackers_status | {sa_status} |"
                )
                out.append(
                    "| stronger_attackers_artifact |"
                    f" `{modern_row.get('stronger_attackers_artifact')}` |"
                )
                out.append(
                    "| blackbox_proxy_status |"
                    f" {modern_row.get('blackbox_proxy_status')} |"
                )
                out.append(
                    "| timing_sidechannel_proxy_status |"
                    f" {modern_row.get('timing_sidechannel_proxy_status')} |"
                )
                out.append(
                    "| inter_block_masking_gap_status |"
                    f" {modern_row.get('inter_block_masking_gap_status')} |"
                )
                out.append(
                    "| inter_block_masking_experimental_status |"
                    f" {modern_row.get('inter_block_masking_experimental_status')} |"
                )
                sa_detail = modern_row.get(
                    "security_profile_detail_with_stronger_attackers"
                )
                if sa_detail:
                    out.append(
                        "| security_profile_detail_with_stronger_attackers |"
                        f" {sa_detail} |"
                    )
            out.append("")
            if rt_status and rt_status != "not_yet":
                out.append(
                    "### Stage 5.5b Real-Token-Prompted Real-Activation Attacker"
                )
                out.append("")
                out.append(
                    "Stage 5.5b drives the Stage 6.4c model-level wrapper"
                    " (embedding + prefill + decode_step + greedy generation)"
                    " with real (or deterministic synthetic) input_ids and"
                    " replays the Stage 5.5 adaptive attacker family"
                    " (linear / MLP / Sinkhorn permutation / linkability)"
                    " against the resulting (plain, visible) trace pairs"
                    " across PREFILL and DECODE_STEP. Real tokenizer / real"
                    " model loading is opt-in; pytest stays synthetic. The"
                    " masked-tensor risk classification stays `low`; the"
                    " inter-block hidden states (`boundary_input` / `final`)"
                    " are plain at the model-wrapper boundary by construction"
                    " — this is a structural model-wrapper limitation, not a"
                    " Stage 5.5b attacker finding. Not formal security; not a"
                    " real TEE measurement."
                )
                out.append("")
            if ext_status and ext_status != "not_yet":
                out.append(
                    "### Stage 5.6 Extension — Inter-Block Masked Boundary + Constant-Time Decode Proxy"
                )
                out.append("")
                out.append(
                    "Stage 5.6 extension wires `masked_boundary_experimental`"
                    " through ObfuscatedModernDecoderModelWrapper so the"
                    " inter-block residual stays in a fresh orthogonal"
                    " `n_inter` mask space across all layers; the LM head"
                    " absorbs `n_inter` and `boundary_input` / `final` join"
                    " the masked tensor set. A `constant_time_decode_mode ="
                    " \"proxy_equalized\"` lever in the timing proxy"
                    " equalises per-step simulated latency to a per-method"
                    " upper bound (PROXY only — no sleep, no real wall-time"
                    " change). Defaults stay `plain_boundary` and `off`."
                    " Both opt-in via CLI flags. Promotion eligibility:"
                    " `security_profile_detail_with_extended_proxy ="
                    " \"inter-block-and-constant-time-proxy-evaluated, not"
                    " formal\"` when both modes are on and the envelope"
                    " stays low-risk. Not formal security; not a real TEE"
                    " measurement."
                )
                out.append("")
            lora_status = modern_row.get("lora_private_training_status")
            lora_backward_status = modern_row.get("lora_backward_status")
            lora_rank_padding_status = modern_row.get("lora_rank_padding_status")
            lora_multilayer_status = modern_row.get(
                "lora_multilayer_training_status"
            )
            lora_multilayer_security_status = modern_row.get(
                "lora_multilayer_security_proxy_status"
            )
            lora_training_timing_status = modern_row.get(
                "lora_training_timing_proxy_status"
            )
            lora_stronger_dummy_status = modern_row.get(
                "lora_stronger_dummy_status"
            )
            lora_stronger_dummy_security_status = modern_row.get(
                "lora_stronger_dummy_security_status"
            )
            lora_spectral_rank_hardening_status = modern_row.get(
                "lora_spectral_rank_hardening_status"
            )
            paper_artifact_consolidation_status = modern_row.get(
                "paper_artifact_consolidation_status"
            )
            measured_runtime_evaluation_status = modern_row.get(
                "measured_runtime_evaluation_status"
            )
            paper_claims_audit_status = modern_row.get(
                "paper_claims_audit_status"
            )
            if lora_status and lora_status != "not_yet":
                out.append("| lora_private_training_status |"
                           f" {lora_status} |")
                out.append("| lora_forward_masking_status |"
                           f" {modern_row.get('lora_forward_masking_status')} |")
                out.append("| lora_training_step_status |"
                           f" {modern_row.get('lora_training_step_status')} |")
                out.append("| lora_security_proxy_status |"
                           f" {modern_row.get('lora_security_proxy_status')} |")
                out.append("| lora_training_artifact |"
                           f" `{modern_row.get('lora_training_artifact')}` |")
                out.append("| lora_security_artifact |"
                           f" `{modern_row.get('lora_security_artifact')}` |")
                out.append("| lora_merge_adapter_into_w |"
                           f" {modern_row.get('lora_merge_adapter_into_w')} |")
                lora_detail = modern_row.get(
                    "security_profile_detail_with_lora"
                )
                if lora_detail:
                    out.append(
                        "| security_profile_detail_with_lora |"
                        f" {lora_detail} |"
                    )
            if lora_backward_status and lora_backward_status != "not_yet":
                out.append("| lora_backward_status |"
                           f" {lora_backward_status} |")
                out.append("| lora_loss_status |"
                           f" {modern_row.get('lora_loss_status')} |")
                out.append("| lora_optimizer_status |"
                           f" {modern_row.get('lora_optimizer_status')} |")
                out.append("| lora_gradient_security_proxy_status |"
                           f" {modern_row.get('lora_gradient_security_proxy_status')} |")
                out.append("| lora_backward_artifact |"
                           f" `{modern_row.get('lora_backward_artifact')}` |")
                out.append("| lora_gradient_security_artifact |"
                           f" `{modern_row.get('lora_gradient_security_artifact')}` |")
                lb_detail = modern_row.get(
                    "security_profile_detail_with_lora_backward"
                )
                if lb_detail:
                    out.append(
                        "| security_profile_detail_with_lora_backward |"
                        f" {lb_detail} |"
                    )
            if lora_rank_padding_status and lora_rank_padding_status != "not_yet":
                out.append("| lora_rank_padding_status |"
                           f" {lora_rank_padding_status} |")
                out.append("| lora_hidden_rank_status |"
                           f" {modern_row.get('lora_hidden_rank_status')} |")
                out.append("| lora_true_rank_hidden_from_shape |"
                           f" {modern_row.get('lora_true_rank_hidden_from_shape')} |")
                out.append("| lora_padded_rank_visible |"
                           f" {modern_row.get('lora_padded_rank_visible')} |")
                out.append("| lora_rank_padding_artifact |"
                           f" `{modern_row.get('lora_rank_padding_artifact')}` |")
                out.append("| lora_rank_security_artifact |"
                           f" `{modern_row.get('lora_rank_security_artifact')}` |")
                lrp_detail = modern_row.get(
                    "security_profile_detail_with_lora_rank_padding"
                )
                if lrp_detail:
                    out.append(
                        "| security_profile_detail_with_lora_rank_padding |"
                        f" {lrp_detail} |"
                    )
            if (
                (lora_multilayer_status and lora_multilayer_status != "not_yet")
                or (
                    lora_multilayer_security_status
                    and lora_multilayer_security_status != "not_yet"
                )
                or (
                    lora_training_timing_status
                    and lora_training_timing_status != "not_yet"
                )
            ):
                out.append("| lora_multilayer_training_status |"
                           f" {lora_multilayer_status} |")
                out.append("| lora_multilayer_training_artifact |"
                           f" `{modern_row.get('lora_multilayer_training_artifact')}` |")
                out.append("| lora_multilayer_security_proxy_status |"
                           f" {lora_multilayer_security_status} |")
                out.append("| lora_multilayer_security_artifact |"
                           f" `{modern_row.get('lora_multilayer_security_artifact')}` |")
                out.append("| lora_training_timing_proxy_status |"
                           f" {lora_training_timing_status} |")
                out.append("| lora_training_timing_artifact |"
                           f" `{modern_row.get('lora_training_timing_artifact')}` |")
                ml_detail = modern_row.get(
                    "security_profile_detail_with_lora_multilayer"
                )
                if ml_detail:
                    out.append(
                        "| security_profile_detail_with_lora_multilayer |"
                        f" {ml_detail} |"
                    )
            if (
                (
                    paper_artifact_consolidation_status
                    and paper_artifact_consolidation_status != "not_yet"
                )
                or (
                    measured_runtime_evaluation_status
                    and measured_runtime_evaluation_status != "not_yet"
                )
                or (
                    paper_claims_audit_status
                    and paper_claims_audit_status != "not_yet"
                )
            ):
                out.append("| paper_artifact_consolidation_status |"
                           f" {paper_artifact_consolidation_status} |")
                out.append("| paper_artifact_consolidation_artifact |"
                           f" `{modern_row.get('paper_artifact_consolidation_artifact')}` |")
                out.append("| measured_runtime_evaluation_status |"
                           f" {measured_runtime_evaluation_status} |")
                out.append("| measured_runtime_artifact |"
                           f" `{modern_row.get('measured_runtime_artifact')}` |")
                out.append("| paper_claims_audit_status |"
                           f" {paper_claims_audit_status} |")
                out.append("| paper_claims_audit_artifact |"
                           f" `{modern_row.get('paper_claims_audit_artifact')}` |")
            if (
                (
                    lora_stronger_dummy_status
                    and lora_stronger_dummy_status != "not_yet"
                )
                or (
                    lora_stronger_dummy_security_status
                    and lora_stronger_dummy_security_status != "not_yet"
                )
            ):
                out.append("| lora_stronger_dummy_status |"
                           f" {lora_stronger_dummy_status} |")
                out.append("| lora_stronger_dummy_artifact |"
                           f" `{modern_row.get('lora_stronger_dummy_artifact')}` |")
                out.append("| lora_stronger_dummy_security_status |"
                           f" {lora_stronger_dummy_security_status} |")
                out.append("| lora_stronger_dummy_security_artifact |"
                           f" `{modern_row.get('lora_stronger_dummy_security_artifact')}` |")
                out.append("| lora_spectral_rank_hardening_status |"
                           f" {lora_spectral_rank_hardening_status} |")
                sdh_detail = modern_row.get(
                    "security_profile_detail_with_lora_dummy_hardening"
                )
                if sdh_detail:
                    out.append(
                        "| security_profile_detail_with_lora_dummy_hardening |"
                        f" {sdh_detail} |"
                    )
            if sa_status and sa_status != "not_yet":
                out.append(
                    "### Stage 5.6 Stronger Attackers (Black-box + Timing + Inter-block Gap)"
                )
                out.append("")
                out.append(
                    "Stage 5.6 ships three proxy attackers that do NOT"
                    " require paired plaintext/visible internal supervision."
                    " (1) Black-box query attacker uses only generated"
                    " tokens + per-step logits summaries; mode / bundle /"
                    " use_pad distinguishability sits at random chance under"
                    " Stage 6.4c's exact-token-match guarantee. (2) Timing"
                    " side-channel proxy uses the Stage 5.2c op-count cost"
                    " model + Gaussian noise; decode_step and prompt-length"
                    " latency leakage is `high` (structural — any latency"
                    " observer can count decode steps), mitigation-bundle"
                    " distinguishability is `low`. (3) Inter-block residual"
                    " masking gap analysis confirms the Stage 5.5b finding"
                    " that `boundary_input` / `final` are plain at the"
                    " model-wrapper boundary; a single-transition math probe"
                    " verifies the orthogonal-mask fix is numerically correct,"
                    " but the full `masked_boundary_experimental` mode is"
                    " `not_implemented_in_stage_5_6` (deferred to Stage 5.6"
                    " extension / Stage 7.0). Envelope-integrity risk:"
                    " `low`. Structural-leakage risk: `high`. Not formal"
                    " security; not a real TEE measurement."
                )
                out.append("")
            if (
                (
                    paper_artifact_consolidation_status
                    and paper_artifact_consolidation_status != "not_yet"
                )
                or (
                    measured_runtime_evaluation_status
                    and measured_runtime_evaluation_status != "not_yet"
                )
                or (
                    paper_claims_audit_status
                    and paper_claims_audit_status != "not_yet"
                )
            ):
                out.append(
                    "### Stage 7.5 — Paper Artifact Consolidation +"
                    " Measured Runtime + Claims Audit"
                )
                out.append("")
                out.append(
                    "Stage 7.5 aggregates every existing"
                    " `outputs/*.json` produced by Stage 1 through Stage"
                    " 7.4 into paper-ready CSV / Markdown / LaTeX"
                    " tables, runs local-emulation wall-clock"
                    " measurements on the plain / masked / rank-padded /"
                    " multi-layer LoRA primitives, and classifies every"
                    " paper claim into `supported / proxy_supported /"
                    " unsupported`. **This is local emulation only —"
                    " NOT a real TEE wall-time measurement and NOT a"
                    " formal / cryptographic / semantic security claim.**"
                    " No new obfuscation primitives, no new attackers,"
                    " no inference-side default behaviour changes."
                    " The aggregator emits artifact_inventory,"
                    " correctness_summary, security_proxy_summary,"
                    " workload_summary, lora_training_summary,"
                    " limitations_summary, paper_claims_audit, and the"
                    " consolidated `paper_results/summary.md`. The"
                    " measured-runtime evaluator records mean / median /"
                    " std / min / max wall-time per component without"
                    " calling `time.sleep` and without loading any"
                    " network model. The claims audit ensures that"
                    " unsupported items (formal / cryptographic /"
                    " semantic security, real TEE wall-time, hardware"
                    " side-channel security, full Qwen / TinyLlama"
                    " fine-tune, PEFT integration, padded_rank hidden,"
                    " loss / optimizer outsourced, compromised-TEE"
                    " protection) carry explicit `paper_safe_wording`"
                    " + `unsafe_wording_to_avoid` flags so the paper"
                    " writer never accidentally overclaims."
                    " `security_profile` itself remains"
                    " `\"proxy-evaluated, not formal\"`."
                )
                out.append("")
            if (
                (
                    lora_stronger_dummy_status
                    and lora_stronger_dummy_status != "not_yet"
                )
                or (
                    lora_stronger_dummy_security_status
                    and lora_stronger_dummy_security_status != "not_yet"
                )
            ):
                out.append(
                    "### Stage 7.4 — Stronger Dummy Distributions /"
                    " Spectral-Rank Hardening"
                )
                out.append("")
                out.append(
                    "Stage 7.4 adds five stronger dummy strategies on top of"
                    " Stage 7.2's `zero_dummy / paired_cancellation_dummy`"
                    " baseline:"
                    " (1) `gaussian_matched_dummy` — paired cancellation with"
                    " R / S drawn from a Gaussian matched to per-column"
                    " statistics of `A_real` / `B_real`;"
                    " (2) `spectrum_matched_dummy` — paired cancellation"
                    " where R / S are scaled by singular values cycled from"
                    " the empirical `A_real` / `B_real` spectrum;"
                    " (3) `noise_injected_cancellation_dummy` — paired"
                    " cancellation + small noise on the dummy slice, with a"
                    " tracked trusted-side correction `correction ="
                    " A_pad[:, r:] @ B_pad[r:, :]` that the harness"
                    " subtracts via `(α / true_rank) X @ correction` from"
                    " the recovered output;"
                    " (4) `orthogonalized_cancellation_dummy` — paired"
                    " cancellation with R / S projected orthogonal to the"
                    " column / row span of `A_real` / `B_real`;"
                    " (5) `mixed_dummy_ensemble` — per-pair random"
                    " selection from the four cancellation strategies above."
                    " All five preserve forward / backward / SGD / AdamW"
                    " update correctness to float64 precision; the"
                    " stronger-dummy probe verifies `loss_diff` /"
                    " `max_grad_*_real_err` / `max_update_*_err` ≤ 1e-9"
                    " across every supported strategy."
                    " `lora_stronger_dummy_status = \"implemented\"`,"
                    " `lora_spectral_rank_hardening_status ="
                    " \"proxy-evaluated\"`."
                    " The security proxy reports four sub-attacks:"
                    " ensemble spectral-cliff / 99%-energy / log-elbow"
                    " inference, gradient-side spectral inference,"
                    " dummy-strategy classification via nearest-bucket-mean"
                    " on top-k normalised singular values, and the Stage"
                    " 7.3 cross-layer linkage proxy parametrised by dummy"
                    " strategy. Conservative verdicts per requirement 12 —"
                    " every paired-cancellation-derived strategy is reported"
                    " as `needs_more_evaluation` when accuracy ≤ 0.2;"
                    " `zero_dummy` stays at `high`. The dummy-strategy"
                    " classifier itself is reported honestly — Stage 7.4"
                    " does NOT claim cryptographic hiding."
                    " `lora_stronger_dummy_security_status ="
                    " \"implemented\"`,"
                    " `security_profile_detail_with_lora_dummy_hardening ="
                    " \"spectral-rank-hardening-proxy-evaluated, not"
                    " formal\"` (additive label only — top-level"
                    " `security_profile` stays `\"proxy-evaluated, not"
                    " formal\"`)."
                    " NOT full Qwen / TinyLlama / LLaMA LoRA fine-tuning,"
                    " NOT PEFT integration, NOT distributed training,"
                    " NOT real TEE training, NOT a real hardware"
                    " side-channel evaluation, NOT a heterogeneous"
                    " `padded_rank` scheme — `padded_rank` itself remains"
                    " visible from tensor shape."
                )
                out.append("")
            if (
                (lora_multilayer_status and lora_multilayer_status != "not_yet")
                or (
                    lora_multilayer_security_status
                    and lora_multilayer_security_status != "not_yet"
                )
                or (
                    lora_training_timing_status
                    and lora_training_timing_status != "not_yet"
                )
            ):
                out.append(
                    "### Stage 7.3 — Multi-Layer LoRA Training +"
                    " Cross-Layer Proxy + Training Timing Proxy"
                )
                out.append("")
                out.append(
                    "Stage 7.3 stacks the Stage 7.0 forward / Stage 7.1"
                    " masked backward / Stage 7.2 rank padding"
                    " primitives across multiple LoRA-augmented linears"
                    " (`q_proj / k_proj / v_proj / o_proj / gate_proj /"
                    " up_proj / down_proj`) in a tiny synthetic"
                    " Transformer-style block stack and verifies that"
                    " every per-module recovered output, every per-module"
                    " gradient (real slice), and every per-module"
                    " SGD / AdamW update matches the plain rank-`r`"
                    " reference to float64 precision. The optimizer state"
                    " is sized to `true_rank` for every LoRA module; the"
                    " dummy slice is never updated."
                    " `lora_multilayer_training_status = \"prototype\"`."
                    " The cross-layer security proxy reports linkage AUC"
                    " across four strategies"
                    " (`fixed_masks_shared_u / independent_u_per_layer /"
                    " fresh_masks_independent_u /"
                    " rank_padding_full_bundle`), inference accuracy under"
                    " heterogeneous `true_rank` with shared `padded_rank`"
                    " (shape-level rank hidden rate stays at 1.0 — only"
                    " `true_rank` is hidden, `padded_rank` remains"
                    " visible), and per-module multi-step membership"
                    " linkability."
                    " `lora_multilayer_security_proxy_status ="
                    " \"implemented\"`."
                    " The training timing proxy is a cost-model latency"
                    " simulator: per-step latency is composed from"
                    " `forward / backward / optimizer / mask_generation /"
                    " boundary / rank_padding_dummy` slices plus Gaussian"
                    " noise; we evaluate eight leakage tasks"
                    " (batch_size, seq_len, true_rank, padded_rank,"
                    " num_modules, optimizer, rank_padding_on,"
                    " dummy_strategy) under"
                    " `constant_time_training_mode ∈ {\"off\","
                    " \"proxy_equalized\"}`, with `proxy_equalized`"
                    " padding every step to the upper-bucket latency."
                    " **No real sleep, no real TEE wall-time, no"
                    " hardware side-channel.**"
                    " `lora_training_timing_proxy_status = \"implemented\"`."
                    " `security_profile_detail_with_lora_multilayer ="
                    " \"multi-layer-lora-proxy-evaluated, not formal\"`."
                    " `security_profile` itself remains"
                    " `\"proxy-evaluated, not formal\"`."
                    " NOT full Qwen / TinyLlama / LLaMA LoRA fine-tuning,"
                    " NOT PEFT integration, NOT distributed training,"
                    " NOT real TEE training, NOT a real hardware"
                    " side-channel evaluation."
                )
                out.append("")
            if lora_rank_padding_status and lora_rank_padding_status != "not_yet":
                out.append(
                    "### Stage 7.2 — LoRA Rank Padding / Hidden-Rank Prototype"
                )
                out.append("")
                out.append(
                    "Stage 7.2 stacks rank padding on top of the Stage 7.0"
                    " forward + Stage 7.1 backward path. The trusted side"
                    " constructs `A_pad ∈ R^{d_in × r_pad}`, `B_pad ∈"
                    " R^{r_pad × d_out}` with `A_pad B_pad = A_real B_real`"
                    " exactly, so the function value (and the LoRA scaling"
                    " `α / r`) are unchanged. The GPU only ever sees"
                    " `A_pad_tilde / B_pad_tilde / grad_A_pad_tilde /"
                    " grad_B_pad_tilde` whose rank dimension is `r_pad`;"
                    " **true rank `r` is hidden from tensor shape**. After"
                    " masked backward recovery, the trusted side slices"
                    " `grad_A_pad[:, :true_rank]` / `grad_B_pad[:true_rank, :]`"
                    " and feeds those into the SGD / AdamW step. The"
                    " optimizer state is sized to `true_rank`, never"
                    " `padded_rank`; the dummy slice is re-sampled fresh and"
                    " never persists into the optimizer. Two dummy strategies"
                    " are supported: `zero_dummy` (baseline; spectral"
                    " attacker reads `true_rank` back from `SVD(B_pad_tilde)`"
                    " exactly — proxy `risk_level = high`) and"
                    " `paired_cancellation_dummy` (pair dummies as"
                    " `[R, R], [S, -S]` so the spectral cliff sits at"
                    " `true_rank + ⌊(r_pad - r) / 2⌋`, an upper bound only —"
                    " proxy `risk_level = needs_more_evaluation`)."
                    " `padded_rank` itself remains visible to the GPU (Stage"
                    " 7.2 does not hide `r_pad`)."
                    " `security_profile_detail_with_lora_rank_padding ="
                    " \"rank-padding-proxy-evaluated, not formal\"`."
                    " `security_profile` itself stays `\"proxy-evaluated, not"
                    " formal\"`. NOT full Qwen / TinyLlama LoRA fine-tuning,"
                    " NOT PEFT integration, NOT distributed training, NOT"
                    " real TEE training."
                )
                out.append("")
            if lora_backward_status and lora_backward_status != "not_yet":
                out.append(
                    "### Stage 7.1 — LoRA Masked Backward / Gradient-Side Obfuscation"
                )
                out.append("")
                out.append(
                    "Stage 7.1 extends Stage 7.0 by sending masked gradient"
                    " tensors through the GPU boundary as well. The trusted"
                    " side still computes G = dL/dY (loss stays trusted) and"
                    " applies the optimizer update (SGD / AdamW state stays"
                    " trusted); the GPU runs the gradient matmuls on masked"
                    " tensors (G_tilde, X_tilde, A_tilde, B_tilde,"
                    " grad_A_tilde, grad_B_tilde, optional grad_X_tilde) and"
                    " returns masked gradients. Trusted side recovers via"
                    " grad_A = N_in^{-T} grad_A_tilde U^T (+ pad compensation"
                    " when use_pad=True) and grad_B = U^{-T} grad_B_tilde"
                    " N_out^T (+ pad compensation). Per-step grad_A / grad_B"
                    " match plain autograd to float64 precision across SGD /"
                    " AdamW × use_pad ∈ {True, False} × fresh_u_per_step ∈"
                    " {True, False}. Gradient leakage proxy ranks five"
                    " strategies; fresh masks drive gradient-side"
                    " membership-style AUC from ≈ 1.0 (fixed) to ≈ 0.5"
                    " (random) — though LoRA rank `r` is still visible from"
                    " the shape of grad_A_tilde / grad_B_tilde (rank padding"
                    " is Stage 7.2). `security_profile_detail_with_lora_backward"
                    " = \"masked-gradient-proxy-evaluated, not formal\"`."
                    " `security_profile` itself stays `\"proxy-evaluated, not"
                    " formal\"`. This is NOT full Qwen / TinyLlama LoRA"
                    " fine-tuning, NOT PEFT integration, NOT distributed"
                    " training, NOT real TEE training."
                )
                out.append("")
            if lora_status and lora_status != "not_yet":
                out.append(
                    "### Stage 7.0 — LoRA Private Training Prototype"
                )
                out.append("")
                out.append(
                    "Stage 7.0 lands a LoRA primitive (`src/pllo/ops/lora.py`)"
                    " + training-step correctness probe + leakage proxy:"
                    " GPU sees only masked transcript"
                    " `(X_tilde, W_tilde, A_tilde, B_tilde, Y_tilde)`;"
                    " backward / optimizer remain trusted; the adapter is"
                    " NEVER merged into the public base weight."
                    " `lora_private_training_status = \"prototype\"`."
                    " Forward correctness allclose under both"
                    " `use_pad=True` / `use_pad=False`; per-step loss /"
                    " gradient / update-error match the plain reference to"
                    " float64 precision. Leakage proxy ranks five strategies"
                    " (unmasked / fixed / fresh-U / fresh-masks / fresh+pad)"
                    " under three sub-attacks (adapter extraction, gradient"
                    " visibility accounting, membership-style linkability)."
                    " Rank `r` is visible from the shape of A_tilde / B_tilde"
                    " — rank padding is NOT implemented in Stage 7.0."
                    " `security_profile_detail_with_lora ="
                    " \"private-adapter-trusted-backward, not formal\"`."
                    " `security_profile` itself is unchanged"
                    " (`\"proxy-evaluated, not formal\"`). This is NOT full"
                    " Qwen / TinyLlama LoRA fine-tuning, NOT PEFT integration,"
                    " NOT distributed training, NOT real TEE training."
                )
                out.append("")
            out.append(
                "- Default mode for the wider system remains `\"trusted\"`;"
                " default mitigation bundle remains `\"fresh_perm_only\"`."
            )
            out.append(
                "- This is block-level integration, not a full model-level"
                " wrapper; `full_runtime_integrated` stays False."
            )
            out.append(
                "- No generation / decode_step / KV cache runtime is"
                " implemented at the wrapper level."
            )
            out.append(
                "- Not a real TEE measurement; not formal security."
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
