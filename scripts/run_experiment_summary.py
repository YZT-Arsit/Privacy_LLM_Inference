#!/usr/bin/env python
"""Stage 4.10: aggregate every stage's correctness JSON into one report.

Reads all stage-level correctness JSON files produced by the other scripts in
this directory and emits three artifacts under ``outputs/``:

* ``experiment_summary.json`` — machine-readable aggregation of every stage.
* ``experiment_summary.csv``  — one row per (stage, use_pad-variant) pair.
* ``experiment_summary.md``   — paper / report ready Markdown with scope,
  trusted-shortcut limitations, and per-stage metric tables.

The script does *not* re-run any models by default. Pass ``--rerun`` to first
execute each upstream correctness script (with both ``use_pad`` variants where
applicable) into dedicated cache paths under ``outputs/_summary_runs/`` so the
``use_pad=true`` / ``use_pad=false`` comparison columns are populated from
fresh runs rather than whatever snapshot happens to live in ``outputs/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    title: str
    summary: str
    trusted_shortcuts: tuple[str, ...]
    script: str | None
    has_pad_variants: bool
    # snapshot paths used when --rerun is not passed
    snapshot_true: Path | None
    snapshot_false: Path | None = None
    # cache paths used when --rerun is passed (always pair-specific)
    rerun_true: Path | None = None
    rerun_false: Path | None = None
    # extra CLI args (besides --use-pad / --output) for rerun
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    # how to pass the pad flag for rerun
    pad_flag_style: str = "explicit"  # "explicit" => --use-pad true|false ; "boolean" => --use-pad / --no-use-pad


SUMMARY_CACHE_DIR = PROJECT_ROOT / "outputs" / "_summary_runs"


def _p(rel: str) -> Path:
    return PROJECT_ROOT / rel


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        stage_id="1",
        title="Static Linear (mask + pad)",
        summary=(
            "Right-multiply mask + one-time pad correctness for a standalone"
            " linear layer."
        ),
        trusted_shortcuts=("Mask & pad generation lives in SimulatedTEE.",),
        script="scripts/run_static_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/static_correctness.json"),
        snapshot_false=_p("outputs/static_correctness_no_pad_float32.json"),
        rerun_true=SUMMARY_CACHE_DIR / "static_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "static_pad_false.json",
        pad_flag_style="boolean",
    ),
    StageSpec(
        stage_id="1-lora",
        title="LoRA Linear (independent low-rank branch)",
        summary=(
            "Mask + pad correctness for a LoRA-adapted linear where base"
            " weight and adapter are obfuscated separately."
        ),
        trusted_shortcuts=("Mask & pad generation lives in SimulatedTEE.",),
        script="scripts/run_lora_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/lora_correctness.json"),
        snapshot_false=_p("outputs/lora_correctness_no_pad_float32.json"),
        rerun_true=SUMMARY_CACHE_DIR / "lora_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "lora_pad_false.json",
        pad_flag_style="boolean",
    ),
    StageSpec(
        stage_id="2",
        title="Tiny decoder-only Transformer (full sequence)",
        summary=(
            "End-to-end obfuscated forward through a hand-written tiny"
            " Transformer; logits compared against the plain reference."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm (Stage 2 shortcut).",
            "Trusted GELU (MLP activation evaluated in plaintext).",
        ),
        script="scripts/run_tiny_transformer_correctness.py",
        has_pad_variants=False,
        snapshot_true=_p("outputs/tiny_transformer_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "tiny_transformer.json",
    ),
    StageSpec(
        stage_id="3-cache",
        title="Tiny Transformer prefill / decode / KV cache",
        summary=(
            "Prefill + decode correctness with persistent obfuscated K/V"
            " cache; per-layer K_tilde / V_tilde invariants validated."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm carried forward from Stage 2.",
            "Trusted GELU carried forward from Stage 2.",
        ),
        script="scripts/run_kv_cache_correctness.py",
        has_pad_variants=False,
        snapshot_true=_p("outputs/kv_cache_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "kv_cache.json",
    ),
    StageSpec(
        stage_id="3-gen",
        title="Tiny Transformer greedy generation",
        summary=(
            "Greedy generation correctness for the tiny Transformer."
            " Compares plain vs obfuscated token sequences."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm carried forward from Stage 2.",
            "Trusted GELU carried forward from Stage 2.",
            "No sampling / beam search / EOS early-stop.",
        ),
        script="scripts/run_generation_correctness.py",
        has_pad_variants=False,
        snapshot_true=_p("outputs/generation_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "tiny_generation.json",
    ),
    StageSpec(
        stage_id="4.6",
        title="GPT-2 single-block obfuscated wrapper",
        summary=(
            "Single HuggingFace GPT-2 block obfuscated via fused c_attn"
            " block-diagonal Q/K/V masks; hidden states compared against the"
            " plain HF block."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm (ln_1 / ln_2 on plaintext).",
            "Trusted GELU.",
            "HF GPT-2 model is not modified (Conv1D-as-linear extraction).",
        ),
        script="scripts/run_gpt2_block_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/gpt2_block_correctness.json"),
        snapshot_false=_p("outputs/gpt2_block_correctness_no_pad.json"),
        rerun_true=SUMMARY_CACHE_DIR / "gpt2_block_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "gpt2_block_pad_false.json",
    ),
    StageSpec(
        stage_id="4.7",
        title="GPT-2 multi-block full forward logits",
        summary=(
            "Full GPT-2 forward through chained ObfuscatedGPT2BlockWrapper"
            " instances; diagonal vocab output mask on the LM head."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm (ln_1 / ln_2 / ln_f on plaintext).",
            "Trusted GELU.",
            "LM head: diagonal vocab output mask only — no pad.",
        ),
        script="scripts/run_gpt2_model_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/gpt2_model_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "gpt2_model_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "gpt2_model_pad_false.json",
    ),
    StageSpec(
        stage_id="4.8",
        title="GPT-2 prefill / decode / KV cache",
        summary=(
            "Internal ObfuscatedGPT2KVCache, prefill + decode_step,"
            " per-head K/V mask reuse, K_tilde / V_tilde invariants."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm.",
            "Trusted GELU.",
            "LM head: vocab output mask only.",
            "HF past_key_values used only as plaintext reference.",
        ),
        script="scripts/run_gpt2_cache_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/gpt2_cache_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "gpt2_cache_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "gpt2_cache_pad_false.json",
    ),
    StageSpec(
        stage_id="4.9",
        title="GPT-2 greedy generation",
        summary=(
            "generate_greedy() built directly on prefill + decode_step."
            " HF generate() is not called; no sampling / beam / EOS."
        ),
        trusted_shortcuts=(
            "Trusted LayerNorm.",
            "Trusted GELU.",
            "LM head: vocab output mask only.",
            "Greedy only — no sampling / beam / EOS early-stop.",
        ),
        script="scripts/run_gpt2_generation_correctness.py",
        has_pad_variants=True,
        snapshot_true=_p("outputs/gpt2_generation_correctness.json"),
        rerun_true=SUMMARY_CACHE_DIR / "gpt2_generation_pad_true.json",
        rerun_false=SUMMARY_CACHE_DIR / "gpt2_generation_pad_false.json",
    ),
)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(stage: StageSpec, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalized metric record from a stage JSON payload.

    All stages report a different schema; this function picks the relevant
    headline numbers used in the summary table.
    """
    metrics = payload.get("metrics", {}) or {}
    logits_metrics = payload.get("logits_metrics", {}) or {}
    cache_metrics = payload.get("cache_invariant_metrics", {}) or {}
    generation_metrics = payload.get("generation_metrics", {}) or {}

    # Stages 1 / 2: a single ``metrics`` block.
    headline = {
        "max_abs_error": metrics.get("max_abs_error"),
        "mean_abs_error": metrics.get("mean_abs_error"),
        "relative_l2_error": metrics.get("relative_l2_error"),
        "cosine_similarity": metrics.get("cosine_similarity"),
        "allclose": metrics.get("allclose"),
        "top1_match_rate": metrics.get("top1_match_rate"),
    }

    # Stage 3 KV cache / Stage 4.8 cache: prefill + decode_step_max + cache invariant.
    if "prefill" in logits_metrics or "decode_step_max" in logits_metrics:
        prefill = logits_metrics.get("prefill", {}) or {}
        decode_step_max = logits_metrics.get("decode_step_max", {}) or {}
        headline.update(
            {
                "prefill_max_abs_error": prefill.get("max_abs_error"),
                "prefill_allclose": prefill.get("allclose"),
                "prefill_top1": prefill.get("top1_match_rate"),
                "decode_step_max_abs_error": decode_step_max.get("max_abs_error"),
                "decode_step_max_allclose": decode_step_max.get("allclose"),
                "decode_step_max_top1": decode_step_max.get("top1_match_rate"),
            }
        )

    # Stage 4.8 cache wrapper: prefill_logits_metrics + decode_logits_metrics.
    if "prefill_logits_metrics" in payload:
        pl = payload["prefill_logits_metrics"] or {}
        headline.update(
            {
                "prefill_max_abs_error": pl.get("max_abs_error"),
                "prefill_allclose": pl.get("allclose"),
                "prefill_top1": pl.get("top1_match_rate"),
            }
        )
    if "decode_logits_metrics" in payload:
        dl = payload["decode_logits_metrics"] or {}
        headline.update(
            {
                "decode_max_abs_error_max": dl.get("max_abs_error_max"),
                "decode_top1_min": dl.get("top1_match_rate_min"),
                "decode_allclose_all": dl.get("allclose_all"),
            }
        )

    # Stage 3 / Stage 4.9 generation metrics.
    if generation_metrics:
        headline.update(
            {
                "token_match_rate": generation_metrics.get("token_match_rate"),
                "sequence_exact_match": generation_metrics.get("sequence_exact_match"),
            }
        )

    # Stage 4.9 logits_metrics has per_step + summary fields too.
    if "max_abs_error_max" in logits_metrics:
        headline.update(
            {
                "logits_max_abs_error_max": logits_metrics.get("max_abs_error_max"),
                "logits_allclose_all": logits_metrics.get("allclose_all"),
                "logits_top1_min": logits_metrics.get("top1_match_rate_min"),
            }
        )

    # Cache invariant block (Stage 3 + 4.8 + 4.9).
    if cache_metrics:
        headline.update(
            {
                "cache_max_key_error": cache_metrics.get("max_key_error"),
                "cache_max_value_error": cache_metrics.get("max_value_error"),
                "cache_allclose": cache_metrics.get("allclose"),
            }
        )
    return {k: v for k, v in headline.items() if v is not None}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def rerun_one(stage: StageSpec, variant: str, output_path: Path) -> None:
    """Execute the upstream script for one (stage, variant) pair."""
    script = stage.script
    if script is None:
        return
    cmd: list[str] = [sys.executable, str(PROJECT_ROOT / script)]
    cmd.extend(stage.extra_args)
    if stage.has_pad_variants:
        if stage.pad_flag_style == "boolean":
            cmd.append("--use-pad" if variant == "use_pad=true" else "--no-use-pad")
        else:
            cmd.extend(["--use-pad", "true" if variant == "use_pad=true" else "false"])
    cmd.extend(["--output", str(output_path)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def collect_stage(stage: StageSpec, use_rerun: bool) -> dict[str, Any]:
    """Collect both pad variants (when applicable) for one stage."""
    record: dict[str, Any] = {
        "stage": stage.stage_id,
        "title": stage.title,
        "summary": stage.summary,
        "trusted_shortcuts": list(stage.trusted_shortcuts),
        "script": stage.script,
        "has_pad_variants": stage.has_pad_variants,
        "variants": {},
    }
    variants = (
        ("use_pad=true", "use_pad=false")
        if stage.has_pad_variants
        else ("use_pad=true",)
    )
    for variant in variants:
        if use_rerun:
            target = stage.rerun_true if variant == "use_pad=true" else stage.rerun_false
            if target is None:
                continue
            rerun_one(stage, variant, target)
            source = target
        else:
            source = stage.snapshot_true if variant == "use_pad=true" else stage.snapshot_false
            if source is None:
                continue
        payload = read_json(source)
        variant_record: dict[str, Any] = {
            "source": _display_path(source),
            "payload_present": payload is not None,
        }
        if payload is not None:
            variant_record["config"] = payload.get("config", {})
            variant_record["metrics"] = extract_metrics(stage, payload)
        record["variants"][variant] = variant_record
    return record


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


CSV_FIELDS = (
    "stage",
    "title",
    "variant",
    "payload_present",
    "max_abs_error",
    "mean_abs_error",
    "cosine_similarity",
    "allclose",
    "top1_match_rate",
    "prefill_max_abs_error",
    "prefill_allclose",
    "prefill_top1",
    "decode_step_max_abs_error",
    "decode_step_max_allclose",
    "decode_step_max_top1",
    "decode_max_abs_error_max",
    "decode_top1_min",
    "decode_allclose_all",
    "token_match_rate",
    "sequence_exact_match",
    "logits_max_abs_error_max",
    "logits_allclose_all",
    "logits_top1_min",
    "cache_max_key_error",
    "cache_max_value_error",
    "cache_allclose",
)


def to_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for variant, payload in record["variants"].items():
            row = {
                "stage": record["stage"],
                "title": record["title"],
                "variant": variant,
                "payload_present": payload.get("payload_present", False),
            }
            metrics = payload.get("metrics", {})
            for key in CSV_FIELDS:
                if key in row:
                    continue
                row[key] = metrics.get(key)
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1.0 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("max_abs_error", "max_abs_error"),
    ("allclose", "allclose"),
    ("top1_match_rate", "top1"),
    ("prefill_max_abs_error", "prefill_max_err"),
    ("prefill_allclose", "prefill_allclose"),
    ("decode_step_max_abs_error", "decode_max_err"),
    ("decode_max_abs_error_max", "decode_max_err"),
    ("decode_allclose_all", "decode_allclose"),
    ("decode_step_max_allclose", "decode_allclose"),
    ("token_match_rate", "token_match"),
    ("sequence_exact_match", "seq_exact"),
    ("logits_max_abs_error_max", "logits_max_err"),
    ("logits_allclose_all", "logits_allclose"),
    ("cache_max_key_error", "cache_max_key_err"),
    ("cache_max_value_error", "cache_max_val_err"),
    ("cache_allclose", "cache_allclose"),
)


def _markdown_metric_table(record: dict[str, Any]) -> str:
    rows = []
    headers = ["metric"] + list(record["variants"].keys())
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    seen: set[str] = set()
    for key, label in METRIC_COLUMNS:
        # Skip metrics that no variant of this stage reports.
        if not any(key in v.get("metrics", {}) for v in record["variants"].values()):
            continue
        if label in seen:
            continue
        seen.add(label)
        cells = [label]
        for variant in record["variants"].values():
            cells.append(_fmt(variant.get("metrics", {}).get(key)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def to_markdown(records: list[dict[str, Any]], rerun: bool) -> str:
    lines: list[str] = []
    lines.append("# Privacy LLM Obfuscation — Experiment Summary (Stage 4.10)")
    lines.append("")
    lines.append(
        "This document aggregates the per-stage correctness JSON files into a"
        " single reproducibility report. Numbers below are read from"
        f" `outputs/_summary_runs/*.json` (fresh `--rerun`)." if rerun else
        "This document aggregates the per-stage correctness JSON files into a"
        " single reproducibility report. Numbers below are read from the"
        " current snapshot of `outputs/*.json`."
    )
    lines.append("")
    lines.append("## Stage Coverage")
    lines.append("")
    lines.append("| Stage | Title | Pad variants | Source script |")
    lines.append("|---|---|---|---|")
    for r in records:
        lines.append(
            "| {stage} | {title} | {pads} | `{script}` |".format(
                stage=r["stage"],
                title=r["title"],
                pads="yes" if r["has_pad_variants"] else "single",
                script=r["script"] or "—",
            )
        )
    lines.append("")
    lines.append("## Trusted-side Engineering Shortcuts (still active)")
    lines.append("")
    lines.append("| Stage | Trusted shortcuts |")
    lines.append("|---|---|")
    for r in records:
        bullets = "; ".join(r["trusted_shortcuts"]) if r["trusted_shortcuts"] else "—"
        lines.append(f"| {r['stage']} | {bullets} |")
    lines.append("")
    lines.append("## Per-Stage Metrics")
    lines.append("")
    for r in records:
        lines.append(f"### Stage {r['stage']} — {r['title']}")
        lines.append("")
        lines.append(r["summary"])
        lines.append("")
        if any(v["payload_present"] for v in r["variants"].values()):
            lines.append(_markdown_metric_table(r))
        else:
            sources = ", ".join(
                f"`{v['source']}`" for v in r["variants"].values() if v.get("source")
            )
            lines.append(f"_No payload found at {sources}._")
        lines.append("")
    lines.append("## Reproducibility — How to Regenerate Everything")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -e \".[dev,hf]\"")
    lines.append("pytest")
    lines.append("python scripts/run_experiment_summary.py --rerun")
    lines.append("```")
    lines.append("")
    lines.append(
        "The `--rerun` flag drives each upstream correctness script for both"
        " `use_pad=true` and `use_pad=false` (where applicable) and writes them"
        " to `outputs/_summary_runs/`. Without `--rerun` the aggregator reads"
        " the snapshot files already present in `outputs/`."
    )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run every upstream correctness script (both pad variants where applicable).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [collect_stage(stage, use_rerun=args.rerun) for stage in STAGES]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "rerun": args.rerun,
        "stages": records,
        "pad_variant_coverage": {
            r["stage"]: list(r["variants"].keys()) for r in records
        },
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    rows = to_csv_rows(records)
    with (output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (output_dir / "experiment_summary.md").write_text(
        to_markdown(records, rerun=args.rerun), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "stages_recorded": len(records),
                "rows_in_csv": len(rows),
                "outputs": [
                    _display_path(output_dir / "experiment_summary.json"),
                    _display_path(output_dir / "experiment_summary.csv"),
                    _display_path(output_dir / "experiment_summary.md"),
                ],
                "rerun": args.rerun,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
