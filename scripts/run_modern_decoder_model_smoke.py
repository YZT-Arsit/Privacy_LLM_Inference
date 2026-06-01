#!/usr/bin/env python
"""Stage 6.4c — Modern decoder model-level wrapper smoke runner.

Runs ``run_modern_decoder_model_probe`` and writes
``outputs/modern_decoder_model_wrapper_smoke.{json,md}``.

Default behaviour:

* ``--attempt-real-model-load`` OFF — synthetic fallback only, no network.
* ``--both-bundles`` — both Stage 5.3e bundles.
* ``--use-pad`` ``both`` — runs both ``False`` and ``True``.
* ``--max-new-tokens 3``.
* ``--max-layers 2`` (the wider system supports more; smoke keeps it tiny).
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

from pllo.experiments.modern_decoder_model_probe import (  # noqa: E402
    ModernDecoderModelWrapperConfig,
    run_modern_decoder_model_probe,
)
from pllo.ops.mitigation_bundles import (  # noqa: E402
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--model-id", default=None)
    p.add_argument("--attempt-real-model-load", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument(
        "--use-synthetic-fallback", dest="use_synthetic_fallback",
        action="store_true", default=True,
    )
    p.add_argument(
        "--no-synthetic-fallback", dest="use_synthetic_fallback",
        action="store_false",
    )
    p.add_argument("--max-layers", type=int, default=2)
    p.add_argument(
        "--use-pad", default="both", choices=["true", "false", "both"],
    )
    p.add_argument(
        "--mitigation-bundle", default=None,
        choices=list(VALID_MITIGATION_BUNDLES),
        help=(
            "Restrict to a single bundle. Default: sweep both bundles."
            f" Wider-system default remains {DEFAULT_MITIGATION_BUNDLE!r}."
        ),
    )
    p.add_argument(
        "--both-bundles", action="store_true",
        help="(Default already runs both bundles; flag documented for parity.)",
    )
    p.add_argument(
        "--nonlinear-mode", default="compatible_islands",
        choices=["trusted", "compatible_islands"],
    )
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--collect-traces", action="store_true")
    # Synthetic shape.
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--intermediate-size", type=int, default=64)
    p.add_argument("--num-query-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt-length", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


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
    "This is model-level wrapper smoke, not a real TEE deployment.",
    "Real wall-time is not measured.",
    "Only greedy generation is implemented.",
    "Beam search / top-k / top-p are not implemented.",
    "RoPE scaling variants are not fully implemented unless explicitly supported.",
    "Qwen/TinyLlama real loading is opt-in and may be skipped.",
    "No LoRA training path is implemented.",
    "Security remains proxy-evaluated, not formal.",
    "Not formal security; not a real TEE measurement.",
    "Inter-layer hidden states are recovered to plain space between blocks.",
]

_NEXT_STAGE_PLAN = [
    "Stage 5.5b — Real-token-prompted real-activation attacker, now that"
    " tokenizer / embedding path is wired and decode-step traces are"
    " collectable.",
    "Stage 5.6 — Stronger attacker variants (black-box query, side-channel,"
    " ML-based permutation recovery).",
    "Stage 5.3d (deferred) — Full BERT and T5 obfuscated wrappers (not just probes).",
]


def _bool_str(v: Any) -> str:
    return "true" if v else "false"


def _format_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    load = report["model_loading"]
    spec = report["block_spec"]
    summary = report["summary"]
    runs = report["per_run"]
    lines: list[str] = []
    lines.append("# Modern Decoder Model-Level Wrapper Smoke (Stage 6.4c)\n")

    lines.append("## Experiment Scope\n")
    lines.append(
        "Stage 6.4c stacks Stage 6.4b's per-block obfuscated forward"
        " into a multi-layer model-level wrapper with embedding lookup,"
        " final RMSNorm, an optionally-masked LM head, KV-cache-aware"
        " prefill / decode_step, and a hand-written greedy generation"
        " loop. Default mode for the wider system remains"
        " `nonlinear_mode='trusted'` and the default mitigation bundle"
        " remains `'fresh_perm_only'`."
    )
    lines.append("")

    lines.append("## Model Loading Status\n")
    lines.append("| field | value |\n|---|---|")
    for k in (
        "load_status", "resolved_model_id", "model_family", "model_class",
        "fallback_used", "candidates_tried", "load_error",
    ):
        lines.append(f"| {k} | {load.get(k)} |")
    lines.append("")

    lines.append("## Model-Level Wrapper Configuration\n")
    lines.append("| field | value |\n|---|---|")
    for k, v in [
        ("source", summary["source"]),
        ("model_family", summary["model_family"]),
        ("num_layers_used", summary["num_layers_used"]),
        ("hidden_size", summary["hidden_size"]),
        ("intermediate_size", summary["intermediate_size"]),
        ("num_attention_heads", summary["num_attention_heads"]),
        ("num_key_value_heads", summary["num_key_value_heads"]),
        ("head_dim", summary["head_dim"]),
        ("attention_variant", summary["attention_variant"]),
        ("vocab_size", summary["vocab_size"]),
        ("nonlinear_mode", summary["nonlinear_mode"]),
        ("mitigation_bundles_evaluated",
         ", ".join(summary["mitigation_bundles_evaluated"])),
        ("use_pad_values", summary["use_pad_values"]),
        ("max_new_tokens", summary["max_new_tokens"]),
    ]:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    def _runs_table(title: str, stage_key: str, metric_keys: list[str]) -> None:
        lines.append(f"## {title}\n")
        lines.append(
            "| bundle | use_pad | "
            + " | ".join(metric_keys) + " |"
        )
        lines.append("|---|---|" + "|".join(["---"] * len(metric_keys)) + "|")
        for r in runs:
            stage = r[stage_key]
            row: list[str] = []
            for k in metric_keys:
                if k == "allclose":
                    row.append(_bool_str(stage["logits_metrics"]["allclose"]))
                elif k == "max_abs_error":
                    row.append(f"{stage['logits_metrics']['max_abs_error']:.3e}")
                elif k == "top1_match_rate":
                    row.append(f"{stage['logits_metrics']['top1_match_rate']:.3f}")
                elif k == "cache_seq_len":
                    row.append(str(stage["cache_summary"]["total_seq_len"]))
                elif k == "num_layers":
                    row.append(str(stage["cache_summary"]["num_layers"]))
                elif k == "sequence_exact_match":
                    row.append(_bool_str(stage["sequence_exact_match"]))
                elif k == "token_match_rate":
                    row.append(f"{stage['token_match_rate']:.3f}")
                elif k == "lm_head_status":
                    row.append(stage.get("lm_head_status", "n/a"))
                else:
                    row.append(str(stage.get(k, "n/a")))
            lines.append(
                f"| {r['mitigation_bundle']} | {_bool_str(r['use_pad'])} | "
                + " | ".join(row) + " |"
            )
        lines.append("")

    _runs_table(
        "Full Forward Correctness",
        "full_forward",
        ["allclose", "max_abs_error", "top1_match_rate", "lm_head_status"],
    )
    _runs_table(
        "Prefill / Decode-Step Correctness",
        "prefill",
        ["allclose", "max_abs_error", "top1_match_rate",
         "cache_seq_len", "num_layers"],
    )
    lines.append("### Decode-Step (one token after prefill)\n")
    lines.append("| bundle | use_pad | allclose | top1 | new_seq_len | position |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        ds = r["decode_step"]
        lines.append(
            f"| {r['mitigation_bundle']} | {_bool_str(r['use_pad'])}"
            f" | {_bool_str(ds['logits_metrics']['allclose'])}"
            f" | {ds['logits_metrics']['top1_match_rate']:.3f}"
            f" | {ds['cache_summary']['total_seq_len']}"
            f" | {ds['position']} |"
        )
    lines.append("")

    _runs_table(
        "Greedy Generation Correctness",
        "greedy_generate",
        ["sequence_exact_match", "token_match_rate"],
    )

    lines.append("## KV Cache Invariants\n")
    lines.append(
        "- Per-layer cache holds `K_tilde = K @ N_K` and"
        " `V_tilde = V @ N_V` with one `N_K`/`N_V` per kv-head."
    )
    lines.append(
        "- Decode appends `k_new @ N_K` and `v_new @ N_V` along the seq"
        " axis using the cached mask material so the append invariant"
        " holds for the lifetime of one generation session."
    )
    lines.append(
        "- GQA: `repeat_kv` is applied on the masked cache *after* the"
        " append; per-q-head `N_Q = N_K[group]^{-T}` makes"
        " `q_tilde @ k_tilde_rep^T = q_rope @ k_rep^T`."
    )
    lines.append("")

    lines.append("## RoPE / GQA Handling\n")
    lines.append(
        "- RoPE uses post-RoPE per-head masking; mask-before-RoPE"
        " commutation is not assumed (Stage 6.4 Probe A invariant)."
    )
    lines.append(
        "- `decode_step` advances the RoPE absolute position via an"
        " explicit `position` argument; `_apply_rope_at` recomputes"
        " `cos`/`sin` at `[position, position+1)` for the new token."
    )
    lines.append(
        "- `rope_scaling` (linear / ntk / yarn) is recorded as a"
        " spec-level note; the default LLaMA-style base is used."
    )
    lines.append("")

    lines.append("## Mitigation Bundle Results\n")
    lines.append(
        "| bundle | use_pad | dense_sandwich | boundary_pad | default_on_candidate |"
    )
    lines.append("|---|---|---|---|---|")
    for r in runs:
        meta = r["full_forward"]["mitigation_bundle_metadata"]
        lines.append(
            f"| {r['mitigation_bundle']} | {_bool_str(r['use_pad'])}"
            f" | {_bool_str(meta['dense_sandwich_enabled'])}"
            f" | {_bool_str(meta['boundary_pad_enabled'])}"
            f" | {_bool_str(meta['default_on_candidate_under_stage_5_4'])} |"
        )
    lines.append("")

    lines.append("## Trace Hook Status\n")
    lines.append(
        f"- `collect_traces = {_bool_str(cfg.get('collect_traces', False))}` (default off)."
    )
    lines.append(
        "- The model wrapper re-exposes the Stage 5.5 block-level trace"
        " hook; setting `collect_traces=True` lets downstream stages"
        " (e.g. Stage 5.5b real-token-prompted attacker) capture per-layer"
        " intermediates without leaking raw tensors into JSON."
    )
    lines.append("")

    lines.append("## Limitations\n")
    for item in _LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Next Stage Plan\n")
    for item in _NEXT_STAGE_PLAN:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _strip_tensors(obj: Any) -> Any:
    """Defensive — never publish raw tensors to JSON."""
    import torch
    if isinstance(obj, torch.Tensor):
        return {"_tensor_shape": list(obj.shape), "_tensor_redacted": True}
    if isinstance(obj, dict):
        return {k: _strip_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_tensors(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    use_pad_values = _resolve_use_pad(args.use_pad)
    bundles = _resolve_bundles(args.mitigation_bundle)
    config = ModernDecoderModelWrapperConfig(
        model_id=args.model_id,
        attempt_real_model_load=bool(args.attempt_real_model_load),
        allow_synthetic_fallback=bool(args.use_synthetic_fallback),
        local_files_only=bool(args.local_files_only),
        nonlinear_mode=args.nonlinear_mode,
        mitigation_bundle=DEFAULT_MITIGATION_BUNDLE,
        max_layers=args.max_layers,
        seed=args.seed,
        collect_traces=bool(args.collect_traces),
        synthetic_vocab_size=args.vocab_size,
        synthetic_hidden_size=args.hidden_size,
        synthetic_intermediate_size=args.intermediate_size,
        synthetic_num_attention_heads=args.num_query_heads,
        synthetic_num_key_value_heads=args.num_kv_heads,
        synthetic_head_dim=args.head_dim,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        mitigation_bundles=bundles,
        use_pad_values=use_pad_values,
    )
    report = run_modern_decoder_model_probe(config)
    safe_report = _strip_tensors(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "modern_decoder_model_wrapper_smoke.json"
    md_path = args.output_dir / "modern_decoder_model_wrapper_smoke.md"
    text = json.dumps(safe_report, indent=2)
    assert "tensor(" not in text, "tensor() found in JSON"
    json_path.write_text(text, encoding="utf-8")
    md_path.write_text(_format_markdown(safe_report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"source={report['source']}"
        f" all_full_forward_allclose={report['summary']['all_full_forward_allclose']}"
        f" all_generation_exact_match={report['summary']['all_generation_exact_match']}"
    )


if __name__ == "__main__":
    main()
