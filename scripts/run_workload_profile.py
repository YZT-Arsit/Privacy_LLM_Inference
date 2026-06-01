#!/usr/bin/env python
"""Stage 5.0.1 workload profile — calibrated TEE/GPU cost-model comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    DEFAULT_COST_MODEL,
    INTERACTION_CATEGORIES,
    MODULE_CATEGORIES,
    WORKLOAD_METHODS,
    WorkloadProfileConfig,
    run_workload_profile,
)
from pllo.experiments.report_utils import (
    fmt,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


CSV_FIELDS = (
    "method",
    "title",
    "implemented",
    "wall_time_source",
    "measured_wall_time_ms",
    "projected_wall_time_ms",
    "online_boundary_calls",
    "online_trusted_compute_ops",
    "online_trusted_transfer_bytes",
    "online_gpu_ops",
    "preprocessing_trusted_ops",
    "preprocessing_transfer_bytes",
    "boundary_calls_formula",
    "online_extra_matmul_count",
    "uses_compatible_nonlinear_islands",
    "security_profile",
)


def _method_rows(profile: dict) -> list[dict]:
    rows = []
    for name, payload in profile["methods"].items():
        rows.append(
            {
                "method": name,
                "title": payload["title"],
                "implemented": payload["implemented"],
                "wall_time_source": payload["wall_time_source"],
                "measured_wall_time_ms": payload["measured_wall_time_ms"],
                "projected_wall_time_ms": payload["projected_wall_time_ms"],
                "online_boundary_calls": payload["online_boundary_calls"],
                "online_trusted_compute_ops": payload["online_trusted_compute_ops"],
                "online_trusted_transfer_bytes": payload["online_trusted_transfer_bytes"],
                "online_gpu_ops": payload["online_gpu_ops"],
                "preprocessing_trusted_ops": payload["preprocessing_trusted_ops"],
                "preprocessing_transfer_bytes": payload["preprocessing_transfer_bytes"],
                "boundary_calls_formula": payload["boundary_calls_formula"],
                "online_extra_matmul_count": payload.get(
                    "online_extra_matmul_count", 0
                ),
                "uses_compatible_nonlinear_islands": payload.get(
                    "uses_compatible_nonlinear_islands", False
                ),
                "security_profile": payload.get("security_profile", "n/a"),
            }
        )
    return rows


def _build_markdown(profile: dict) -> str:
    out: list[str] = []
    cfg = profile["config"]
    out.append("# Privacy LLM Obfuscation — Calibrated Workload Profile (Stage 5.0.1)")
    out.append("")
    out.append(
        "Cost model splits every method into four explicit slices: **preprocessing"
        " trusted cost** (amortised), **online boundary crossings** (true"
        " trusted↔untrusted round trips), **online trusted compute**"
        " (LayerNorm / GELU / sampling / recovery FLOPs in the TEE), and"
        " **online GPU obfuscated compute** (linear matmuls, attention,"
        " LM head). Internal Python bookkeeping such as mask-state creation"
        " is **not** counted as a boundary call."
    )
    out.append("")
    out.append(
        f"`model_id={cfg['model_id']}`, `batch_size={cfg['batch_size']}`,"
        f" `prompt_len={cfg['prompt_len']}`, `max_new_tokens={cfg['max_new_tokens']}`,"
        f" `device={cfg['device']}`, `dtype={cfg['dtype']}`, `use_pad={cfg['use_pad']}`,"
        f" `warmup={cfg['warmup']}`, `repeat={cfg['repeat']}`."
    )
    out.append("")
    cal = profile["calibration"]
    out.append(
        f"GPU-FLOPs/ms calibration constant: `{cal['gpu_flops_per_ms']:.3e}`"
        f" (derived from measured `{cal['calibrated_from']}` wall time)."
    )
    out.append("")
    out.append(f"> **Warning:** {profile['interpretation']['cost_model_warning']}.")
    out.append("")

    out.append("## Method comparison")
    headers = [
        "method",
        "impl?",
        "wall_time_ms (measured/proj.)",
        "boundary calls",
        "boundary formula",
        "trusted compute (ops)",
        "trusted transfer (bytes)",
        "gpu (ops)",
    ]
    rows = []
    for method in WORKLOAD_METHODS:
        m = profile["methods"][method.name]
        wt = (
            f"{m['measured_wall_time_ms']:.3f}"
            if m["wall_time_source"] == "measured"
            else f"{m['projected_wall_time_ms']:.3f} (proj.)"
        )
        rows.append(
            [
                method.name,
                m["implemented"],
                wt,
                m["online_boundary_calls"],
                m["boundary_calls_formula"],
                m["online_trusted_compute_ops"],
                m["online_trusted_transfer_bytes"],
                m["online_gpu_ops"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Preprocessing (amortised; excluded from online latency)")
    rows = [
        [
            method.name,
            profile["methods"][method.name]["preprocessing_trusted_ops"],
            profile["methods"][method.name]["preprocessing_transfer_bytes"],
        ]
        for method in WORKLOAD_METHODS
    ]
    out.append(
        markdown_table(["method", "preprocessing_trusted_ops", "preprocessing_transfer_bytes"], rows)
    )
    out.append("")

    out.append("## Interaction breakdown (online slice by interaction type)")
    headers = ["interaction"] + [m.name for m in WORKLOAD_METHODS]
    for column_name, key in (
        ("boundary calls", "online_boundary_calls"),
        ("transfer bytes", "online_trusted_transfer_bytes"),
        ("trusted compute (ops)", "online_trusted_compute_ops"),
    ):
        out.append(f"### {column_name} per interaction")
        rows = []
        for category in INTERACTION_CATEGORIES:
            row = [category]
            for method in WORKLOAD_METHODS:
                row.append(profile["interaction_breakdown"][category][method.name][key])
            rows.append(row)
        out.append(markdown_table(headers, rows))
        out.append("")

    out.append("## Module breakdown (online slice by module category)")
    out.append("")
    headers = ["module"] + [m.name for m in WORKLOAD_METHODS]
    for column_name, key in (
        ("boundary calls", "online_boundary_calls"),
        ("trusted compute (ops)", "online_trusted_compute_ops"),
        ("trusted transfer (bytes)", "online_trusted_transfer_bytes"),
        ("gpu ops", "online_gpu_ops"),
    ):
        out.append(f"### {column_name} per module")
        rows = []
        for category in MODULE_CATEGORIES:
            row = [category]
            for method in WORKLOAD_METHODS:
                row.append(profile["module_breakdown"][category][method.name][key])
            rows.append(row)
        out.append(markdown_table(headers, rows))
        out.append("")

    # ---- Stage 5.2c: Compatible Nonlinear Islands Method ----
    islands_pm = profile["paper_metrics"].get("ours_compatible_nonlinear_islands")
    islands_record = profile["methods"].get("ours_compatible_nonlinear_islands")
    if islands_pm is not None and islands_record is not None:
        out.append("## Compatible Nonlinear Islands Method")
        out.append("")
        out.append(
            "ours_compatible_nonlinear_islands is a projected method based on"
            " Stage 5.2a correctness probes and Stage 5.2b security proxies."
            " It is not yet integrated into GPT-2 / BERT / T5 wrappers — Stage"
            " 5.3 is the integration step."
        )
        out.append("")
        out.append("### Boundary Call Formulas")
        out.append("")
        out.append(
            f"- `ours_current`: {profile['methods']['ours_current']['boundary_calls_formula']}"
        )
        out.append(
            f"- `ours_compatible_nonlinear_islands`: {islands_record['boundary_calls_formula']}"
        )
        out.append(
            f"- `ours_ideal_gpu_nonlinear`: {profile['methods']['ours_ideal_gpu_nonlinear']['boundary_calls_formula']}"
        )
        out.append("")
        out.append("### Trusted Compute Reduction")
        out.append("")
        out.append(
            f"- vs `ours_current`: {islands_pm['trusted_compute_reduction_vs_ours_current']:.2%}"
        )
        out.append(
            f"- vs `tslp_trusted_nonlinear_baseline`: {islands_pm['trusted_compute_reduction_vs_tslp']:.2%}"
        )
        out.append(
            f"- vs `ours_current` boundary call count: "
            f"{islands_pm['boundary_call_reduction_vs_ours_current']:.2%}"
        )
        out.append("")
        out.append("### Preprocessing Cost Increase")
        out.append("")
        pre_increase = islands_pm.get("preprocessing_cost_increase_vs_ours_current")
        if pre_increase is None:
            out.append(
                "- Preprocessing increase vs ours_current: not computable"
                " (ours_current preprocessing is zero)."
            )
        else:
            out.append(
                f"- Preprocessing increase vs `ours_current`: {pre_increase:.2%}"
                " (affine folding + permutation absorption + compatible mask"
                " generation, all amortised over many sessions)."
            )
        pb = islands_record.get("preprocessing_breakdown", {})
        if pb:
            out.append("")
            out.append("- Preprocessing breakdown (ops):")
            out.append(
                f"  - base weight obfuscation: {pb.get('base_weight_obfuscation_ops', 0)}"
            )
            out.append(
                f"  - affine folding: {pb.get('affine_folding_ops', 0)}"
            )
            out.append(
                f"  - permutation absorption: {pb.get('permutation_absorption_ops', 0)}"
            )
            out.append(
                f"  - compatible mask generation: {pb.get('compatible_mask_generation_ops', 0)}"
            )
        out.append("")
        out.append("### Online Extra Matmul Count")
        out.append("")
        out.append(
            f"- `online_extra_matmul_count = {islands_pm['online_extra_matmul_count']}`."
            " Stage 5.2a verified this across every MLP island cell —"
            " operator-compatible mask transitions are folded into adjacent"
            " Linear weights offline and add zero online matmuls."
        )
        out.append("")
        out.append("### Security Proxy Caveats")
        out.append("")
        for caveat in islands_pm.get("security_proxy_caveats", []):
            out.append(f"- {caveat}")
        out.append(
            "- This is not a real TEE measurement."
        )
        out.append("")

        # ---- Stage 5.3a: Wrapper Integration Status -----------------------
        wrapper_status = islands_record.get("wrapper_integration_status")
        if wrapper_status is not None:
            out.append("### Stage 5.3a Wrapper Integration Status")
            out.append("")
            out.append(
                "- `partial_implementation = "
                f"{islands_record.get('partial_implementation', False)}` — "
                "the GPT-2 single-block wrapper now exposes a "
                "`nonlinear_mode=\"compatible_islands\"` feature flag, but "
                "the GPT-2 model-level wrapper, BERT, and T5 paths are not "
                "yet wired up."
            )
            for target in ("gpt2_single_block", "gpt2_model_level", "bert", "t5"):
                out.append(f"- `{target}`: `{wrapper_status[target]}`")
            out.append(
                "- Default mode remains `trusted`; compatible_islands must"
                " not be enabled by default."
            )
            bert_status = wrapper_status.get("bert", "not_yet")
            t5_status = wrapper_status.get("t5", "not_yet")
            if (
                wrapper_status.get("gpt2_model_level") == "implemented"
                and bert_status == "implemented_probe_level"
                and t5_status == "implemented_probe_level"
            ):
                out.append(
                    "- GPT-2 model-level integration is available."
                )
                out.append(
                    "- BERT/T5 are probe-level integrations, not full"
                    " wrappers."
                )
                qwen_status = wrapper_status.get("qwen_or_modern_decoder")
                if qwen_status == "implemented_probe_level":
                    out.append(
                        "- Qwen / TinyLlama / modern decoder-only is a"
                        " probe-level migration (Stage 6.4): RMSNorm +"
                        " SwiGLU + RoPE-post-mask + GQA/MQA tensor probes."
                    )
                    out.append(
                        "- modern_decoder_probe: `implemented` (default"
                        " synthetic; HF model load is opt-in)."
                    )
                elif qwen_status == "implemented_block_level":
                    out.append(
                        "- Qwen / TinyLlama / modern decoder-only is a"
                        " block-level migration (Stage 6.4b): real HF"
                        " model loading (best-effort) + per-block"
                        " obfuscated forward over RMSNorm + RoPE attention"
                        " + GQA/MQA + SwiGLU MLP, both mitigation bundles"
                        " supported."
                    )
                    out.append(
                        "- modern_decoder_probe: `implemented` (Stage 6.4"
                        " tensor-level probes)."
                    )
                    out.append(
                        "- modern_decoder_block_wrapper: `implemented`"
                        " (Stage 6.4b block-level wrapper; synthetic"
                        " fallback for pytest, real HF load opt-in)."
                    )
                elif qwen_status == "implemented":
                    out.append(
                        "- Qwen / TinyLlama / modern decoder-only is a"
                        " model-level wrapper (Stage 6.4c): multi-block"
                        " stacking + embedding lookup + final RMSNorm +"
                        " optionally-masked LM head + KV-cache-aware"
                        " prefill / decode_step + hand-written greedy"
                        " generation over RMSNorm + RoPE attention +"
                        " GQA/MQA + SwiGLU MLP, both mitigation bundles"
                        " supported. This is not full BERT/T5 wrapper"
                        " integration."
                    )
                    out.append(
                        "- modern_decoder_probe: `implemented` (Stage 6.4"
                        " tensor-level probes)."
                    )
                    out.append(
                        "- modern_decoder_block_wrapper: `implemented`"
                        " (Stage 6.4b block-level wrapper)."
                    )
                    out.append(
                        "- modern_decoder_model_wrapper: `implemented`"
                        " (Stage 6.4c model-level wrapper; synthetic"
                        " fallback for pytest, real HF load opt-in)."
                    )
                    gen_status = wrapper_status.get(
                        "modern_decoder_generation_status"
                    )
                    if gen_status:
                        out.append(
                            f"- modern_decoder_generation_status: `{gen_status}`"
                            " (hand-written greedy loop compared against plain"
                            " reference token-for-token)."
                        )
                    kv_status = wrapper_status.get(
                        "modern_decoder_kv_cache_status"
                    )
                    if kv_status:
                        out.append(
                            f"- modern_decoder_kv_cache_status: `{kv_status}`"
                            " (per-layer masked K_tilde / V_tilde with append"
                            " invariant; per-kv-head N_K / N_V constant across"
                            " one generation session)."
                        )
                if islands_record.get("real_activation_attacker_status") == "implemented":
                    out.append("")
                    out.append("### Stage 5.5 Real-Activation Adaptive Attacker")
                    out.append("")
                    out.append(
                        "- `real_activation_attacker_status = "
                        f"\"{islands_record['real_activation_attacker_status']}\"`."
                    )
                    out.append(
                        "- `real_activation_attacker_scope = "
                        f"\"{islands_record['real_activation_attacker_scope']}\"`"
                        " (Stage 6.4b block-level activations)."
                    )
                    out.append(
                        "- `real_activation_attacker_artifact = "
                        f"\"{islands_record['real_activation_attacker_artifact']}\"`."
                    )
                    out.append(
                        "- `security_profile_detail_with_real_activation = "
                        f"\"{islands_record['security_profile_detail_with_real_activation']}\"`"
                        " — additive label only; `security_profile` itself"
                        " remains `\"proxy-evaluated, not formal\"`."
                    )
                    out.append(
                        "- This is NOT a real TEE measurement, NOT formal"
                        " security, and NOT a black-box query attack."
                        " `implemented` / `full_runtime_integrated` /"
                        " `wall_time_source` are unchanged."
                    )
                if islands_record.get("mitigation_bundle_selectable"):
                    out.append("")
                    out.append("### Stage 5.3e Dense-Sandwich Mitigation Integration")
                    out.append("")
                    out.append(
                        "- `mitigation_bundle_selectable = True`."
                    )
                    out.append(
                        "- `default_mitigation_bundle = "
                        f"\"{islands_record['default_mitigation_bundle']}\"`"
                        " (preserves backward compatibility for every Stage"
                        " 5.3a / 5.3b / 5.3c / 6.4 caller that omits the"
                        " bundle argument)."
                    )
                    out.append(
                        "- `recommended_default_on_bundle = "
                        f"\"{islands_record['recommended_default_on_bundle']}\"`."
                    )
                    out.append(
                        "- `recommended_default_on_status = "
                        f"\"{islands_record['recommended_default_on_status']}\"`"
                        " (per Stage 5.4 adaptive proxy attackers — NOT a"
                        " formal security claim)."
                    )
                    out.append(
                        "- `dense_sandwich_supported = "
                        f"{islands_record['dense_sandwich_supported']}`,"
                        " `boundary_pad_required = "
                        f"{islands_record['boundary_pad_required']}`,"
                        " `fresh_permutation_required = "
                        f"{islands_record['fresh_permutation_required']}`."
                    )
                    out.append(
                        "- `compatible_islands` remains feature-flagged"
                        " behind `nonlinear_mode`; default mode is still"
                        " `\"trusted\"`."
                    )
                    out.append(
                        "- security is `adaptive-proxy-mitigated, not"
                        " formal` when the full bundle is enabled; this is"
                        " not a real TEE measurement."
                    )
                scope = islands_record.get("measured_integration_scope")
                if scope:
                    out.append(
                        f"- `measured_integration_scope = \"{scope}\"`."
                    )
                if "full_runtime_integrated" in islands_record:
                    out.append(
                        "- `full_runtime_integrated = "
                        f"{islands_record['full_runtime_integrated']}`."
                    )
                if "all_architecture_probe_level_implemented" in islands_record:
                    out.append(
                        "- `all_architecture_probe_level_implemented = "
                        f"{islands_record['all_architecture_probe_level_implemented']}`."
                    )
                out.append(
                    "- `security_profile` remains `proxy-evaluated, not"
                    " formal`."
                )
            elif wrapper_status.get("gpt2_model_level") == "implemented":
                out.append(
                    "- GPT-2 model-level compatible island integration"
                    " available (Stage 5.3b); BERT/T5 integration pending"
                    " Stage 5.3c."
                )
                out.append(
                    "- measured GPT-2 model-level smoke, not full"
                    " cross-architecture measurement."
                )
                scope = islands_record.get("measured_integration_scope")
                if scope:
                    out.append(
                        f"- `measured_integration_scope = \"{scope}\"`."
                    )
            else:
                out.append(
                    "- GPT-2 single-block integration available; full-model"
                    " measured runtime pending Stage 5.3b."
                )
            out.append("")

    pm = profile["paper_metrics"]
    out.append("## Paper metrics")
    out.append("")
    out.append(
        f"- `boundary_call_reduction_vs_tslp` = {pm['boundary_call_reduction_vs_tslp']:.2%}"
        f" (ours_current vs tslp)"
    )
    out.append(
        f"- `trusted_transfer_reduction_vs_tslp` = {pm['trusted_transfer_reduction_vs_tslp']:.2%}"
    )
    out.append(
        f"- `online_trusted_compute_reduction_vs_tslp` = {pm['online_trusted_compute_reduction_vs_tslp']:.2%}"
    )
    out.append(f"- `gpu_offload_ratio` (ours_current) = {pm['gpu_offload_ratio']:.2%}")
    out.append(f"- `preprocessing_amortized` = `{pm['preprocessing_amortized']}`")
    out.append("- `boundary_calls_per_forward` =")
    for name, value in pm["boundary_calls_per_forward"].items():
        out.append(f"  - `{name}`: {value}")
    out.append("")

    interp = profile["interpretation"]
    out.append("## Interpretation")
    out.append("")
    out.append(f"- **Main online bottleneck (ours_current):** `{interp['main_online_bottleneck']}`")
    out.append(
        f"- **Next primitive to obfuscate on GPU:** `{interp['next_primitive_to_replace']}`"
    )
    out.append("")
    out.append(
        "*Note on ours_current vs TSLP boundary calls:* ours_current crosses"
        " the boundary once per obfuscated linear (4 per layer) while TSLP"
        " crosses once per non-linear (3 per layer plus ln_f). This is an"
        " **architectural** difference, not a bookkeeping artefact. Each"
        " ours_current crossing moves a smaller activation than a TSLP"
        " LayerNorm crossing, and the measured wall time is consistent with"
        " that tradeoff."
    )
    out.append("")

    out.append("## Method semantics & citation caveats")
    for method in WORKLOAD_METHODS:
        out.append(f"### `{method.name}` — {method.title}")
        out.append("")
        out.append(method.summary)
        out.append("")
        out.append(f"- Implemented: **{method.implemented}**")
        out.append(f"- Implementation note: {method.implementation_note}")
        out.append(f"- Caveat: {method.citation_caveat}")
        out.append("")

    out.append("## Limitations")
    for line in profile["limitations"]:
        out.append(f"- {line}")
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append(
        f"python scripts/run_workload_profile.py --batch-size {cfg['batch_size']}"
        f" --prompt-len {cfg['prompt_len']} --max-new-tokens {cfg['max_new_tokens']}"
        f" --warmup {cfg['warmup']} --repeat {cfg['repeat']} --use-pad {cfg['use_pad']}"
    )
    out.append("```")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = WorkloadProfileConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        use_pad=args.use_pad,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        cost_model=DEFAULT_COST_MODEL,
    )
    profile = run_workload_profile(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "workload_profile.json", profile)
    write_csv(out_dir / "workload_profile.csv", _method_rows(profile), CSV_FIELDS)
    write_text(out_dir / "workload_profile.md", _build_markdown(profile))

    headline = {
        name: {
            "boundary": m["online_boundary_calls"],
            "trusted_compute_ops": m["online_trusted_compute_ops"],
            "trusted_transfer_bytes": m["online_trusted_transfer_bytes"],
            "gpu_ops": m["online_gpu_ops"],
            "wall_time_ms": m["measured_wall_time_ms"]
            if m["wall_time_source"] == "measured"
            else m["projected_wall_time_ms"],
            "wall_time_source": m["wall_time_source"],
        }
        for name, m in profile["methods"].items()
    }
    print(
        f"main_online_bottleneck={profile['interpretation']['main_online_bottleneck']}, "
        f"next_primitive={profile['interpretation']['next_primitive_to_replace']}, "
        f"gpu_offload_ratio={profile['paper_metrics']['gpu_offload_ratio']:.3f}, "
        f"output_dir={out_dir}\n"
        f"headline={headline}"
    )


if __name__ == "__main__":
    main()
