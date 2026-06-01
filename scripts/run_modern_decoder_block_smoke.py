#!/usr/bin/env python
"""Stage 6.4b — Modern decoder-only block-level smoke runner.

Runs ``run_modern_decoder_block_probe`` with the requested
(use_pad × mitigation_bundle × nonlinear_mode) sweep and emits
``modern_decoder_block_wrapper_smoke.{json,md}``.

Default behaviour:

* ``--attempt-real-model-load`` OFF — synthetic fallback only, no network.
* ``--both-bundles`` ON via default config (``mitigation_bundles`` covers
  both bundles).
* ``--use-pad`` ``both`` — runs ``False`` and ``True``.
* ``--nonlinear-mode`` ``compatible_islands`` — the obfuscated path is
  what this smoke exists to validate.

The wider system default (``nonlinear_mode='trusted'``,
``mitigation_bundle='fresh_perm_only'``) is unchanged; that's the
upstream default for production wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.modern_decoder_block_probe import (  # noqa: E402
    ModernDecoderBlockProbeConfig,
    ModernDecoderLoadConfig,
    run_modern_decoder_block_probe,
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
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass local_files_only=True to AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--use-synthetic-fallback",
        dest="use_synthetic_fallback",
        action="store_true",
        default=True,
        help="(Default ON) Use synthetic block when real loading fails.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        dest="use_synthetic_fallback",
        action="store_false",
        help="Disable synthetic fallback; report skipped on failure.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--use-pad",
        default="both",
        choices=["true", "false", "both"],
        help="use_pad sweep (default 'both').",
    )
    parser.add_argument(
        "--mitigation-bundle",
        default=None,
        choices=list(VALID_MITIGATION_BUNDLES),
        help=(
            "Restrict to a single bundle. Default: sweep both bundles."
            f" Bundle default for the wider system remains"
            f" {DEFAULT_MITIGATION_BUNDLE!r}."
        ),
    )
    parser.add_argument(
        "--both-bundles",
        action="store_true",
        help="(Default already runs both bundles; this flag is documented for parity.)",
    )
    parser.add_argument(
        "--nonlinear-mode",
        default="compatible_islands",
        choices=["trusted", "compatible_islands"],
    )
    return parser.parse_args()


def _resolve_use_pad(arg: str) -> tuple[bool, ...]:
    if arg == "true":
        return (True,)
    if arg == "false":
        return (False,)
    return (False, True)


def _resolve_bundles(arg: str | None) -> tuple[str, ...]:
    if arg is None:
        return VALID_MITIGATION_BUNDLES
    return (arg,)


_LIMITATIONS = [
    "Block-level integration; not a full Qwen / TinyLlama model-level wrapper.",
    "No generation / decode_step / KV cache runtime is implemented.",
    "RoPE is handled using post-RoPE per-head masking; mask-before-RoPE"
    " dense commutation is not assumed.",
    "If synthetic fallback is used, results are not from real Qwen / TinyLlama weights.",
    "RMSNorm γ is folded into adjacent projection weights; the norm core"
    " runs in an orthogonal residual mask space.",
    "Residual alignment uses the same orthogonal N_res on both branches.",
    "Inherits Stage 5.4 mitigation requirements (fresh permutation +"
    " dense sandwich + boundary pad).",
    "This is not a real TEE measurement.",
    "This is not formal security.",
]

_NEXT_STAGE_PLAN = [
    "Stage 5.5 — adaptive attacker on real modern-decoder activations once"
    " block-level extraction is stable.",
    "Stage 6.4c — full modern-decoder model-level wrapper (multi-block,"
    " LM head, generation).",
    "Stage 5.3d — full BERT / T5 wrapper (MLM head, encoder-decoder generation)"
    " remains scheduled but is engineering-heavy.",
]


def _bool_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_markdown(report: dict) -> str:
    cfg = report["config"]
    load = report["model_loading"]
    spec = report["block_spec"]
    summary = report["summary"]
    runs = report["per_run"]

    lines: list[str] = []
    lines.append("# Modern Decoder Block-Level Wrapper Smoke (Stage 6.4b)\n")

    lines.append("## Experiment Scope\n")
    lines.append(
        "Stage 6.4b adds a block-level obfuscated forward for modern"
        " decoder-only architectures (LLaMA / TinyLlama / Qwen / Qwen2)."
        " The wrapper runs both a plain reference and a Stage 5.2a /"
        " 5.3e compatible-islands path on a single block, recovers the"
        " masked output, and reports allclose for each"
        " (use_pad × mitigation_bundle × nonlinear_mode) combination."
        " Default mode for the wider system remains 'trusted' and the"
        " default mitigation bundle remains 'fresh_perm_only'.\n"
    )

    lines.append("## Model Loading Status\n")
    lines.append("| field | value |\n|---|---|")
    for k in (
        "load_status",
        "resolved_model_id",
        "model_family",
        "model_class",
        "fallback_used",
        "candidates_tried",
        "load_error",
    ):
        lines.append(f"| {k} | {load.get(k)} |")
    lines.append("")

    lines.append("## Modern Decoder Block Spec\n")
    lines.append("| field | value |\n|---|---|")
    for k in (
        "model_family",
        "model_class",
        "block_path",
        "block_index",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "norm_type",
        "activation_type",
        "position_encoding_type",
        "attention_variant",
        "rope_base",
        "rope_scaling_kind",
    ):
        lines.append(f"| {k} | {spec.get(k)} |")
    lines.append("")

    lines.append("## Plain Reference vs HF Block Status\n")
    lines.append(
        "Plain reference is constructed from the **extracted weights**"
        " (no HF block forward call required); residual / RMSNorm / RoPE"
        " / GQA / SwiGLU paths are computed in row-vector convention."
        " The obfuscated path is compared against this plain reference."
        " For synthetic fallback, the plain reference is also synthetic.\n"
    )

    lines.append("## RMSNorm Handling\n")
    lines.append(
        "- Mode: **orthogonal_island_with_gamma_folded_into_qkv**.\n"
        "- N_res is orthogonal so rmsnorm_core(X @ N_res) = rmsnorm_core(X) @ N_res.\n"
        "- γ (input RMSNorm and post-attention RMSNorm) is folded into"
        " the adjacent q/k/v and gate/up projection weights.\n"
    )

    lines.append("## RoPE-Aware Attention Handling\n")
    lines.append(
        "- Mode: **rope_post_mask_only**.\n"
        "- RoPE is applied to plain q/k first; per-head Q/K masks N_Q,"
        " N_K with N_Q N_K^T = I are applied AFTER RoPE.\n"
        "- Pre-RoPE dense-mask commutation is not assumed.\n"
    )

    lines.append("## GQA / MQA Handling\n")
    lines.append(
        f"- attention_variant: {spec.get('attention_variant')}\n"
        f"- num_attention_heads={spec.get('num_attention_heads')},"
        f" num_key_value_heads={spec.get('num_key_value_heads')},"
        f" head_dim={spec.get('head_dim')}\n"
        "- One N_K / N_V per kv-head; per-q-head N_Q is derived from the"
        " corresponding kv-head's N_K via N_Q = N_K^{-T}.\n"
        "- repeat_kv is applied AFTER masking, matching HF semantics.\n"
    )

    lines.append("## SwiGLU Compatible Island Handling\n")
    lines.append(
        "- Mode: **compatible_island_paired_permutation**.\n"
        "- run_swiglu_mlp_island with shared permutation P on the up- and"
        " gate-branches.\n"
        "- pad_placement is linear_boundary_only; pad is never pushed"
        " through SwiGLU.\n"
        "- online_extra_matmul_count = 0.\n"
    )

    lines.append("## Mitigation Bundle Results\n")
    lines.append(
        "| bundle | use_pad | nonlinear_mode | max_abs_error | rel_l2_error |"
        " allclose | dense_sandwich_enabled | boundary_pad_enabled |"
        " default_on_candidate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append(
            "| {bundle} | {pad} | {mode} | {max_err:.3e} | {rel_l2:.3e} |"
            " {allclose} | {ds} | {bp} | {doc} |".format(
                bundle=r["mitigation_bundle"],
                pad=_bool_str(r["use_pad"]),
                mode=r["nonlinear_mode"],
                max_err=r["max_abs_error"],
                rel_l2=r["relative_l2_error"],
                allclose=_bool_str(r["allclose"]),
                ds=_bool_str(r["dense_sandwich_enabled"]),
                bp=_bool_str(r["boundary_pad_enabled"]),
                doc=_bool_str(r["default_on_candidate_under_stage_5_4"]),
            )
        )
    lines.append("")
    lines.append(
        f"- all_runs_allclose: **{_bool_str(summary['all_runs_allclose'])}**\n"
        f"- online_extra_matmul_count: {summary['online_extra_matmul_count']}\n"
        f"- implemented_block_level: {_bool_str(summary['implemented_block_level'])}\n"
        f"- full_runtime_integrated: {_bool_str(summary['full_runtime_integrated'])}\n"
        f"- mitigation_bundles_evaluated: {summary['mitigation_bundles_evaluated']}\n"
    )

    lines.append("## Limitations\n")
    for item in _LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Next Stage Plan\n")
    for item in _NEXT_STAGE_PLAN:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    use_pad_values = _resolve_use_pad(args.use_pad)
    bundles = _resolve_bundles(args.mitigation_bundle)
    config = ModernDecoderBlockProbeConfig(
        output_dir=str(args.output_dir),
        load=ModernDecoderLoadConfig(
            model_id=args.model_id,
            attempt_real_model_load=bool(args.attempt_real_model_load),
            allow_synthetic_fallback=bool(args.use_synthetic_fallback),
            device=args.device,
            dtype=args.dtype,
            local_files_only=bool(args.local_files_only),
        ),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        synthetic_hidden_size=args.hidden_size,
        synthetic_intermediate_size=args.intermediate_size,
        synthetic_num_attention_heads=args.num_query_heads,
        synthetic_num_key_value_heads=args.num_kv_heads,
        synthetic_head_dim=args.head_dim,
        synthetic_rope_base=args.rope_base,
        use_pad_values=use_pad_values,
        mitigation_bundles=bundles,
        nonlinear_mode=args.nonlinear_mode,
        seed=args.seed,
    )
    report = run_modern_decoder_block_probe(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "modern_decoder_block_wrapper_smoke.json"
    md_path = args.output_dir / "modern_decoder_block_wrapper_smoke.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_format_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"all_runs_allclose={report['summary']['all_runs_allclose']}"
        f" source={report['source']}"
        f" bundles={report['summary']['mitigation_bundles_evaluated']}"
    )


if __name__ == "__main__":
    main()
