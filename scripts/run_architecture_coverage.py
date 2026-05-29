#!/usr/bin/env python
"""Stage 6.0 architecture coverage — emit JSON / CSV / Markdown across three families."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.architectures import (
    ATTENTION_TAXONOMY,
    DEFAULT_ARCHITECTURE_MODELS,
    ArchitectureType,
    attention_kinds_for,
    inspect_architecture,
    load_for_architecture,
    spec_to_dict,
)
from pllo.architectures.architecture_inspector import _module_paths
from pllo.experiments.report_utils import (
    fmt,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-model-id",
        default=None,
        help="Override the decoder-only model id (default: registry).",
    )
    parser.add_argument(
        "--encoder-model-id",
        default=None,
        help="Override the encoder-only model id (default: registry).",
    )
    parser.add_argument(
        "--encdec-model-id",
        default=None,
        help="Override the encoder-decoder model id (default: registry).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


CSV_FIELDS = (
    "architecture_key",
    "status",
    "model_id",
    "architecture_type",
    "model_class",
    "has_encoder",
    "has_decoder",
    "has_cross_attention",
    "has_causal_self_attention",
    "has_bidirectional_self_attention",
    "supports_past_key_values",
    "has_lm_head",
    "has_mlm_head",
    "has_classification_head",
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_heads",
    "skip_reason",
)


def _candidates_for(architecture_key: str, override: str | None) -> tuple[str, ...]:
    if override is not None:
        return (override,)
    return DEFAULT_ARCHITECTURE_MODELS[architecture_key]


def _try_inspect(architecture_key: str, override: str | None) -> dict:
    candidates = _candidates_for(architecture_key, override)
    try:
        model_id, model = load_for_architecture(architecture_key, candidates=candidates)
    except Exception as exc:  # noqa: BLE001 — skip gracefully on any load error.
        return {
            "architecture_key": architecture_key,
            "status": "skipped",
            "model_id": candidates[0] if candidates else None,
            "candidates_tried": list(candidates),
            "skip_reason": f"{type(exc).__name__}: {exc}",
        }
    spec = inspect_architecture(model, model_id=model_id)
    module_paths = _module_paths(model, spec.architecture_type)
    return {
        "architecture_key": architecture_key,
        "status": "loaded",
        "model_id": model_id,
        "candidates_tried": list(candidates),
        "spec": spec_to_dict(spec),
        "module_paths": module_paths,
    }


def _csv_rows(coverage: list[dict]) -> list[dict]:
    rows = []
    for entry in coverage:
        if entry["status"] == "loaded":
            spec = entry["spec"]
            rows.append(
                {
                    "architecture_key": entry["architecture_key"],
                    "status": "loaded",
                    "model_id": entry["model_id"],
                    "architecture_type": spec["architecture_type"],
                    "model_class": spec["model_class"],
                    "has_encoder": spec["has_encoder"],
                    "has_decoder": spec["has_decoder"],
                    "has_cross_attention": spec["has_cross_attention"],
                    "has_causal_self_attention": spec["has_causal_self_attention"],
                    "has_bidirectional_self_attention": spec["has_bidirectional_self_attention"],
                    "supports_past_key_values": spec["supports_past_key_values"],
                    "has_lm_head": spec["has_lm_head"],
                    "has_mlm_head": spec["has_mlm_head"],
                    "has_classification_head": spec["has_classification_head"],
                    "vocab_size": spec["vocab_size"],
                    "hidden_size": spec["hidden_size"],
                    "num_layers": spec["num_layers"],
                    "num_heads": spec["num_heads"],
                    "skip_reason": None,
                }
            )
        else:
            rows.append(
                {
                    "architecture_key": entry["architecture_key"],
                    "status": "skipped",
                    "model_id": entry.get("model_id"),
                    "architecture_type": None,
                    "model_class": None,
                    "has_encoder": None,
                    "has_decoder": None,
                    "has_cross_attention": None,
                    "has_causal_self_attention": None,
                    "has_bidirectional_self_attention": None,
                    "supports_past_key_values": None,
                    "has_lm_head": None,
                    "has_mlm_head": None,
                    "has_classification_head": None,
                    "vocab_size": None,
                    "hidden_size": None,
                    "num_layers": None,
                    "num_heads": None,
                    "skip_reason": entry["skip_reason"],
                }
            )
    return rows


def _build_markdown(coverage: list[dict]) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Architecture Coverage (Stage 6.0)")
    out.append("")
    out.append(
        "Scaffolding for paper-grade multi-architecture experiments. The"
        " coverage report identifies whether each Transformer family"
        " (decoder-only, encoder-only, encoder-decoder) can be loaded and"
        " classified by the inspector. Stage 6.0 does **not** implement"
        " obfuscated wrappers for non-GPT-2 architectures — those are"
        " deferred to Stages 6.1 / 6.2."
    )
    out.append("")

    out.append("## Model coverage")
    headers = ["architecture", "status", "model_id", "model_class", "layers", "heads", "hidden", "skip reason"]
    rows = []
    for entry in coverage:
        if entry["status"] == "loaded":
            spec = entry["spec"]
            rows.append(
                [
                    entry["architecture_key"],
                    "loaded",
                    entry["model_id"],
                    spec["model_class"],
                    spec["num_layers"],
                    spec["num_heads"],
                    spec["hidden_size"],
                    "—",
                ]
            )
        else:
            rows.append(
                [
                    entry["architecture_key"],
                    "skipped",
                    entry.get("model_id"),
                    "—",
                    "—",
                    "—",
                    "—",
                    entry["skip_reason"],
                ]
            )
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Architecture type matrix")
    headers = [
        "architecture",
        "type",
        "encoder",
        "decoder",
        "cross-attn",
        "causal self-attn",
        "bidir self-attn",
        "past_key_values",
    ]
    rows = []
    for entry in coverage:
        if entry["status"] != "loaded":
            continue
        spec = entry["spec"]
        rows.append(
            [
                entry["architecture_key"],
                spec["architecture_type"],
                spec["has_encoder"],
                spec["has_decoder"],
                spec["has_cross_attention"],
                spec["has_causal_self_attention"],
                spec["has_bidirectional_self_attention"],
                spec["supports_past_key_values"],
            ]
        )
    out.append(markdown_table(headers, rows) if rows else "_No models loaded._")
    out.append("")

    out.append("## Output heads")
    headers = ["architecture", "lm_head", "mlm_head", "classification_head", "embedding path", "self-attn path", "cross-attn path", "output head path"]
    rows = []
    for entry in coverage:
        if entry["status"] != "loaded":
            continue
        spec = entry["spec"]
        paths = entry["module_paths"]
        rows.append(
            [
                entry["architecture_key"],
                spec["has_lm_head"],
                spec["has_mlm_head"],
                spec["has_classification_head"],
                paths["embedding"],
                paths["self_attention"],
                paths["cross_attention"],
                paths["output_head"],
            ]
        )
    out.append(markdown_table(headers, rows) if rows else "_No models loaded._")
    out.append("")

    out.append("## Attention taxonomy")
    headers = [
        "kind",
        "architecture",
        "Q source",
        "K source",
        "V source",
        "mask type",
        "cache type",
    ]
    rows = []
    for kind in ATTENTION_TAXONOMY:
        rows.append(
            [
                kind.name,
                kind.architecture_type.value,
                kind.q_source,
                kind.k_source,
                kind.v_source,
                kind.mask_type,
                kind.cache_type,
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Required invariants (per attention kind)")
    headers = ["kind", "architecture", "required invariant"]
    rows = []
    for kind in ATTENTION_TAXONOMY:
        rows.append([kind.name, kind.architecture_type.value, kind.required_invariant])
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Attention kinds expected per loaded model")
    headers = ["architecture", "expected attention kinds"]
    rows = []
    for entry in coverage:
        if entry["status"] != "loaded":
            continue
        arch_type = ArchitectureType(entry["spec"]["architecture_type"])
        kinds = ", ".join(k.name for k in attention_kinds_for(arch_type))
        rows.append([entry["architecture_key"], kinds])
    out.append(markdown_table(headers, rows) if rows else "_No models loaded._")
    out.append("")

    out.append("## Next-stage implementation plan")
    out.append("")
    out.append(
        "- **Stage 6.1** — Bidirectional self-attention probe for BERT-style"
        " encoder-only models (no autoregressive cache, padding masks instead"
        " of causal masks)."
    )
    out.append(
        "- **Stage 6.2** — Cross-attention probe for T5 / BART encoder-decoder"
        " models (K / V come from the encoder memory and stay constant across"
        " decoder steps, so the cache layout differs from Stage 4.8)."
    )
    out.append(
        "- **Stage 6.3** — Cross-architecture workload + security experiments"
        " (re-run the Stage 5.0.1 profiler + attention experiments over each"
        " architecture; verify the Q/K constraint and the cache invariant"
        " generalise)."
    )
    out.append(
        "- **Stage 6.4** — Qwen / ModelScope migration (delayed until the"
        " three baseline architectures are covered)."
    )
    out.append("")

    out.append("## Limitations")
    out.append("")
    out.append(
        "- Only the architecture inspector + attention taxonomy are populated"
        " in Stage 6.0. No obfuscated forward / cache / generation path"
        " exists for BERT / T5 / BART yet."
    )
    out.append(
        "- Model coverage depends on HuggingFace Hub access at run time;"
        " missing models are skipped, not failed."
    )
    out.append(
        "- `prajjwal1/bert-tiny` does not load via `AutoConfig` because the"
        " checkpoint config predates the modern `model_type` key. The"
        " registry falls back to `hf-internal-testing/tiny-bert` first."
    )
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append("python scripts/run_architecture_coverage.py")
    out.append("```")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    coverage: list[dict] = []
    for key, override in (
        ("decoder_only", args.decoder_model_id),
        ("encoder_only", args.encoder_model_id),
        ("encoder_decoder", args.encdec_model_id),
    ):
        coverage.append(_try_inspect(key, override))

    payload = {
        "coverage": coverage,
        "attention_taxonomy": [
            {
                "name": k.name,
                "architecture_type": k.architecture_type.value,
                "q_source": k.q_source,
                "k_source": k.k_source,
                "v_source": k.v_source,
                "mask_type": k.mask_type,
                "cache_type": k.cache_type,
                "required_invariant": k.required_invariant,
            }
            for k in ATTENTION_TAXONOMY
        ],
        "registry_defaults": {k: list(v) for k, v in DEFAULT_ARCHITECTURE_MODELS.items()},
    }

    out_dir: Path = args.output_dir
    write_json(out_dir / "architecture_coverage.json", payload)
    write_csv(out_dir / "architecture_coverage.csv", _csv_rows(coverage), CSV_FIELDS)
    write_text(out_dir / "architecture_coverage.md", _build_markdown(coverage))

    summary = {
        entry["architecture_key"]: {
            "status": entry["status"],
            "model_id": entry.get("model_id"),
            "architecture_type": (
                entry["spec"]["architecture_type"]
                if entry["status"] == "loaded"
                else None
            ),
        }
        for entry in coverage
    }
    print(f"output_dir={out_dir}, summary={summary}")


if __name__ == "__main__":
    main()
