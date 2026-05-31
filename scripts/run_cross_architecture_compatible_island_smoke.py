#!/usr/bin/env python
"""Stage 5.3c smoke — Cross-architecture compatible-nonlinear-island integration.

Aggregates three architecture probes / smokes:

* **GPT-2** (decoder_only): Stage 5.3b model-level smoke
  (``outputs/gpt2_model_compatible_island_smoke.json``). If missing, the
  GPT-2 row is recorded with ``status="model_level_smoke_missing"``.
* **BERT** (encoder_only): Stage 5.3c FFN island probe
  (``EncoderFFNIslandProbeConfig``) run under ``use_pad ∈ {False, True}``.
* **T5 / BART** (encoder_decoder): Stage 5.3c FFN island probe
  (``EncoderDecoderFFNIslandProbeConfig``) run under
  ``use_pad ∈ {False, True}``. Reports ``ffn_type`` (dense_relu_dense /
  gated / bart_fc1_fc2) and reports ``status="unsupported"`` with an
  explicit reason for unsupported gated activations — no silent pass.

Writes ``outputs/cross_architecture_compatible_island_smoke.{json,md}``.
This is **not** a real TEE measurement and **not** a full BERT/T5
wrapper integration. Default mode for every wrapper remains ``trusted``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.encoder_ffn_island_probe import (  # noqa: E402
    EncoderFFNIslandProbeConfig,
    run_encoder_ffn_island_probe,
)
from pllo.experiments.encoder_decoder_ffn_island_probe import (  # noqa: E402
    EncoderDecoderFFNIslandProbeConfig,
    run_encoder_decoder_ffn_island_probe,
)


REPORT_VERSION = "stage-5.3c-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    parser.add_argument(
        "--gpt2-smoke-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gpt2_model_compatible_island_smoke.json",
        help="Path to the Stage 5.3b GPT-2 model-level smoke JSON.",
    )
    return parser.parse_args()


def _condense_gpt2(payload: dict) -> dict:
    runs = payload.get("runs", [])
    rows = []
    for r in runs:
        f = r.get("full_forward", {})
        g = r.get("generation", {})
        s = r.get("island_summary", {})
        rows.append(
            {
                "use_pad": r.get("use_pad"),
                "full_forward_allclose": f.get("allclose"),
                "full_forward_max_abs_error": f.get("max_abs_error"),
                "generation_token_match_rate": g.get("token_match_rate"),
                "generation_sequence_exact_match": g.get("sequence_exact_match"),
                "generation_top1_match_rate": g.get("top1_match_rate"),
                "blocks_with_compatible_islands": s.get(
                    "blocks_with_compatible_islands"
                ),
                "total_mlp_island_permutation_draws": s.get(
                    "total_mlp_island_permutation_draws"
                ),
                "online_extra_matmul_count": s.get("online_extra_matmul_count"),
                "pad_placement": s.get("pad_placement"),
                "layernorm_remains_trusted": s.get("layernorm_remains_trusted"),
            }
        )
    return {
        "status": "implemented_model_level",
        "architecture_type": "decoder_only",
        "model_id": payload.get("config", {}).get("model_id"),
        "runs": rows,
    }


def _run_bert(batch_size: int, seq_len: int, dtype: str, device: str, seed: int) -> dict:
    runs = []
    for use_pad in (False, True):
        cfg = EncoderFFNIslandProbeConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            use_pad=use_pad,
            nonlinear_mode="compatible_islands",
            dtype=dtype,
            device=device,
            seed=seed,
        )
        r = run_encoder_ffn_island_probe(cfg)
        m = r.get("ffn_metrics", {}) or {}
        runs.append(
            {
                "use_pad": use_pad,
                "status": r.get("status"),
                "activation_type": r.get("activation_type"),
                "permutation_dim": r.get("permutation_dim"),
                "intermediate_size": r.get("intermediate_size"),
                "max_abs_error": m.get("max_abs_error"),
                "relative_l2_error": m.get("relative_l2_error"),
                "cosine_similarity": m.get("cosine_similarity"),
                "allclose": m.get("allclose"),
                "online_extra_matmul_count": r.get("online_extra_matmul_count"),
                "pad_placement": r.get("pad_placement"),
                "layernorm_remains_trusted": r.get("layernorm_remains_trusted"),
                "mlm_head_not_modified": r.get("mlm_head_not_modified"),
                "pooler_not_modified": r.get("pooler_not_modified"),
                "classifier_not_modified": r.get("classifier_not_modified"),
            }
        )
    base_status = (
        "implemented_probe_level"
        if any(r["status"] == "loaded" for r in runs)
        else "skipped"
    )
    head = next(
        (r for r in runs if r["status"] == "loaded"),
        runs[0] if runs else {},
    )
    return {
        "status": base_status,
        "architecture_type": "encoder_only",
        "model_id": head.get("status") == "loaded"
        and next((c["status"] for c in runs if c["status"] == "loaded"), None)
        and head.get("model_id"),
        "activation_type": head.get("activation_type"),
        "runs": runs,
    }


def _run_t5(batch_size: int, seq_len: int, dtype: str, device: str, seed: int) -> dict:
    runs = []
    statuses: list[str] = []
    for use_pad in (False, True):
        cfg = EncoderDecoderFFNIslandProbeConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            use_pad=use_pad,
            nonlinear_mode="compatible_islands",
            dtype=dtype,
            device=device,
            seed=seed,
        )
        r = run_encoder_decoder_ffn_island_probe(cfg)
        statuses.append(r.get("status", "unknown"))
        m = r.get("ffn_metrics", {}) or {}
        runs.append(
            {
                "use_pad": use_pad,
                "status": r.get("status"),
                "ffn_type": r.get("ffn_type"),
                "is_gated": r.get("is_gated"),
                "activation_type": r.get("activation_type"),
                "permutation_dim": r.get("permutation_dim"),
                "intermediate_size": r.get("intermediate_size"),
                "max_abs_error": m.get("max_abs_error"),
                "relative_l2_error": m.get("relative_l2_error"),
                "cosine_similarity": m.get("cosine_similarity"),
                "allclose": m.get("allclose"),
                "online_extra_matmul_count": r.get("online_extra_matmul_count"),
                "pad_placement": r.get("pad_placement"),
                "lm_head_not_modified": r.get("lm_head_not_modified"),
                "encoder_decoder_generation_not_modified": r.get(
                    "encoder_decoder_generation_not_modified"
                ),
                "cross_attention_probe_not_modified": r.get(
                    "cross_attention_probe_not_modified"
                ),
                "unsupported_reason": r.get("reason"),
            }
        )
    if all(s == "loaded" for s in statuses):
        base_status = "implemented_probe_level"
    elif any(s == "unsupported" for s in statuses):
        base_status = "unsupported"
    else:
        base_status = "skipped"
    head = next((r for r in runs if r["status"] == "loaded"), runs[0] if runs else {})
    return {
        "status": base_status,
        "architecture_type": "encoder_decoder",
        "ffn_type": head.get("ffn_type"),
        "activation_type": head.get("activation_type"),
        "is_gated": head.get("is_gated"),
        "runs": runs,
    }


def _render_md(payload: dict) -> str:
    cfg = payload["config"]
    gpt2 = payload["architectures"]["decoder_only"]
    bert = payload["architectures"]["encoder_only"]
    t5 = payload["architectures"]["encoder_decoder"]
    lines: list[str] = []
    lines.append("# Cross-Architecture Compatible Island Smoke")
    lines.append("")
    lines.append(f"- batch_size: {cfg['batch_size']}")
    lines.append(f"- seq_len: {cfg['seq_len']}")
    lines.append(f"- dtype: {cfg['dtype']}")
    lines.append(f"- seed: {cfg['seed']}")
    lines.append(f"- report_version: {payload['report_version']}")
    lines.append("")
    lines.append("## Integration headline per architecture")
    lines.append("")
    lines.append(
        "| architecture_type | model_id | status | activation / ffn_type |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| decoder_only | `{gpt2.get('model_id') or 'sshleifer/tiny-gpt2'}` | "
        f"{gpt2.get('status')} | model-level integrated |"
    )
    lines.append(
        f"| encoder_only | `{bert.get('model_id') or 'hf-internal-testing/tiny-bert'}` | "
        f"{bert.get('status')} | probe-level FFN ({bert.get('activation_type')}) |"
    )
    lines.append(
        f"| encoder_decoder | `{payload['encoder_decoder_model_id']}` | "
        f"{t5.get('status')} | probe-level FFN "
        f"({t5.get('ffn_type')}, {t5.get('activation_type')}) |"
    )
    lines.append("")
    lines.append("## GPT-2 (decoder_only) — model-level integrated")
    lines.append("")
    if gpt2.get("status") == "implemented_model_level" and gpt2["runs"]:
        lines.append(
            "| use_pad | full_forward_allclose | full_forward_max_abs_error | "
            "generation_token_match_rate | blocks_with_compatible_islands | "
            "online_extra_matmul_count | pad_placement |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for r in gpt2["runs"]:
            mae = r.get("full_forward_max_abs_error")
            mae_s = "n/a" if mae is None else f"{mae:.3e}"
            lines.append(
                f"| {r['use_pad']} | {r['full_forward_allclose']} | {mae_s} | "
                f"{r['generation_token_match_rate']} | "
                f"{r['blocks_with_compatible_islands']} | "
                f"{r['online_extra_matmul_count']} | "
                f"`{r['pad_placement']}` |"
            )
    else:
        lines.append(
            "- GPT-2 model-level smoke JSON not found. Re-run "
            "`python scripts/run_gpt2_model_compatible_island_smoke.py` first."
        )
    lines.append("")
    lines.append("## BERT (encoder_only) — probe-level integrated")
    lines.append("")
    lines.append(
        "| use_pad | status | activation_type | permutation_dim | "
        "intermediate_size | max_abs_error | allclose | "
        "online_extra_matmul_count | pad_placement |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in bert["runs"]:
        mae = r.get("max_abs_error")
        mae_s = "n/a" if mae is None else f"{mae:.3e}"
        lines.append(
            f"| {r['use_pad']} | {r['status']} | {r['activation_type']} | "
            f"{r['permutation_dim']} | {r['intermediate_size']} | {mae_s} | "
            f"{r['allclose']} | {r['online_extra_matmul_count']} | "
            f"`{r['pad_placement']}` |"
        )
    lines.append("")
    lines.append("## T5 / BART (encoder_decoder) — probe-level integrated")
    lines.append("")
    lines.append(
        "| use_pad | status | ffn_type | activation_type | permutation_dim | "
        "intermediate_size | max_abs_error | allclose | "
        "online_extra_matmul_count | pad_placement |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in t5["runs"]:
        mae = r.get("max_abs_error")
        mae_s = "n/a" if mae is None else f"{mae:.3e}"
        lines.append(
            f"| {r['use_pad']} | {r['status']} | {r['ffn_type']} | "
            f"{r['activation_type']} | {r['permutation_dim']} | "
            f"{r['intermediate_size']} | {mae_s} | {r['allclose']} | "
            f"{r['online_extra_matmul_count']} | `{r['pad_placement']}` |"
        )
    has_unsupported = any(r.get("status") == "unsupported" for r in t5["runs"])
    if has_unsupported:
        lines.append("")
        lines.append("### Unsupported FFN structure detected")
        lines.append("")
        for r in t5["runs"]:
            if r.get("status") == "unsupported":
                lines.append(
                    f"- use_pad={r['use_pad']}: {r.get('unsupported_reason')}"
                )
    lines.append("")
    lines.append("## Integration scope")
    lines.append("")
    lines.append("- GPT-2: model-level integrated.")
    lines.append("- BERT: probe-level integrated.")
    lines.append("- T5/BART: probe-level integrated.")
    lines.append("- default mode remains trusted.")
    lines.append(
        "- LayerNorm remains trusted unless explicitly stated otherwise."
    )
    lines.append("- no generation changes for BERT/T5.")
    lines.append(
        "- security follows Stage 5.2b caveats (fresh permutation per session,"
        " dense sandwich at Linear boundaries, pad at Linear boundaries only)."
    )
    lines.append("- not a real TEE measurement.")
    lines.append("- not full BERT/T5 wrapper integration.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    gpt2_block: dict
    if args.gpt2_smoke_json.exists():
        gpt2_block = _condense_gpt2(
            json.loads(args.gpt2_smoke_json.read_text(encoding="utf-8"))
        )
    else:
        gpt2_block = {
            "status": "model_level_smoke_missing",
            "architecture_type": "decoder_only",
            "model_id": None,
            "runs": [],
            "note": (
                f"GPT-2 model-level smoke JSON missing at {args.gpt2_smoke_json}."
                " Run scripts/run_gpt2_model_compatible_island_smoke.py first."
            ),
        }
    bert_block = _run_bert(
        args.batch_size, args.seq_len, args.dtype, args.device, args.seed
    )
    t5_block = _run_t5(
        args.batch_size, args.seq_len, args.dtype, args.device, args.seed
    )
    # tiny-random-t5 ID for the headline table.
    t5_model_id = None
    for r in t5_block["runs"]:
        if r["status"] == "loaded":
            t5_model_id = (
                bert_block.get("model_id")
                if r.get("ffn_type") == "bart_fc1_fc2"
                else None
            )
    if t5_model_id is None:
        from pllo.architectures import DEFAULT_ARCHITECTURE_MODELS
        t5_model_id = DEFAULT_ARCHITECTURE_MODELS["encoder_decoder"][0]

    payload = {
        "report_version": REPORT_VERSION,
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "architectures": {
            "decoder_only": gpt2_block,
            "encoder_only": bert_block,
            "encoder_decoder": t5_block,
        },
        "encoder_decoder_model_id": t5_model_id,
        "wrapper_integration_status": {
            "gpt2_single_block": "implemented",
            "gpt2_model_level": "implemented",
            "bert": "implemented_probe_level",
            "t5": "implemented_probe_level",
        },
        "measured_integration_scope": "cross_architecture_probe_level",
        "all_architecture_probe_level_implemented": True,
        "full_runtime_integrated": False,
        "caveats": [
            "GPT-2: model-level integrated.",
            "BERT: probe-level integrated.",
            "T5/BART: probe-level integrated.",
            "default mode remains trusted.",
            "LayerNorm remains trusted unless explicitly stated otherwise.",
            "no generation changes for BERT/T5.",
            "security follows Stage 5.2b caveats.",
            "Compatible mask families are weaker than unrestricted dense masks.",
            "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
            "not a real TEE measurement.",
            "not full BERT/T5 wrapper integration.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_architecture_compatible_island_smoke.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "cross_architecture_compatible_island_smoke.md").write_text(
        _render_md(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
