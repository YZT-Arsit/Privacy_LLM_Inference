#!/usr/bin/env python
"""Stage 6.1 encoder-only attention experiments — sweep + JSON/CSV/Markdown emitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    EncoderAttentionProbeConfig,
    run_encoder_attention_probe,
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
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the encoder-only model id (default: registry candidates).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument(
        "--use-pad",
        nargs="?",
        const=True,
        default=None,
        type=parse_bool,
        help="If set, restrict the sweep to a single use_pad value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


SWEEP = {
    "batch_size": (1, 2),
    "seq_len": (4, 8, 16),
    "use_pad": (True, False),
}


CSV_FIELDS = (
    "model_id",
    "status",
    "batch_size",
    "seq_len",
    "use_pad",
    "attention_mask_kind",
    "q_max_abs_error",
    "k_max_abs_error",
    "v_max_abs_error",
    "score_max_abs_error",
    "prob_max_abs_error",
    "attn_out_max_abs_error",
    "output_max_abs_error",
    "output_relative_l2_error",
    "output_cosine_similarity",
    "qk_constraint_error",
    "allclose",
    "q_pad",
    "k_pad",
    "v_pad",
    "o_pad",
)


def _row_from_result(result: dict, mask_kind: str) -> dict:
    cfg = result["config"]
    loading = result["model_loading"]
    if loading["status"] != "loaded":
        return {
            "model_id": loading.get("candidates_tried", [None])[0],
            "status": "skipped",
            "batch_size": cfg["batch_size"],
            "seq_len": cfg["seq_len"],
            "use_pad": cfg["use_pad"],
            "attention_mask_kind": mask_kind,
            "q_max_abs_error": None,
            "k_max_abs_error": None,
            "v_max_abs_error": None,
            "score_max_abs_error": None,
            "prob_max_abs_error": None,
            "attn_out_max_abs_error": None,
            "output_max_abs_error": None,
            "output_relative_l2_error": None,
            "output_cosine_similarity": None,
            "qk_constraint_error": None,
            "allclose": None,
            "q_pad": None,
            "k_pad": None,
            "v_pad": None,
            "o_pad": None,
        }
    qkv = result["qkv_invariants"]
    res = result["results_per_mask"][mask_kind]
    pad = result["pad_report"]["per_mask"][mask_kind]
    return {
        "model_id": loading["model_id"],
        "status": "loaded",
        "batch_size": cfg["batch_size"],
        "seq_len": cfg["seq_len"],
        "use_pad": cfg["use_pad"],
        "attention_mask_kind": mask_kind,
        "q_max_abs_error": qkv["q_metrics"]["max_abs_error"],
        "k_max_abs_error": qkv["k_metrics"]["max_abs_error"],
        "v_max_abs_error": qkv["v_metrics"]["max_abs_error"],
        "score_max_abs_error": res["score_metrics"]["max_abs_error"],
        "prob_max_abs_error": res["prob_metrics"]["max_abs_error"],
        "attn_out_max_abs_error": res["v_aggr_metrics"]["max_abs_error"],
        "output_max_abs_error": res["output_metrics"]["max_abs_error"],
        "output_relative_l2_error": res["output_metrics"].get("relative_l2_error"),
        "output_cosine_similarity": res["output_metrics"].get("cosine_similarity"),
        "qk_constraint_error": qkv["qk_constraint_error"],
        "allclose": res["allclose"],
        "q_pad": pad["q_pad"],
        "k_pad": pad["k_pad"],
        "v_pad": pad["v_pad"],
        "o_pad": pad["o_pad"],
    }


def _build_markdown(results: list[dict]) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Encoder-only Attention Probe (Stage 6.1)")
    out.append("")
    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Bidirectional self-attention probe for BERT-style encoder-only"
        " models. Tests the same mask + pad invariants as the Stage 5.0 GPT-2"
        " attention probe, plus padding-mask coverage."
    )
    out.append("")

    # Model loading status (read from any successful cell)
    out.append("## Model loading status")
    loaded = [r for r in results if r["model_loading"]["status"] == "loaded"]
    skipped = [r for r in results if r["model_loading"]["status"] != "loaded"]
    if loaded:
        loading = loaded[0]["model_loading"]
        out.append(
            f"- **model_id**: `{loading['model_id']}`"
        )
        out.append(f"- model class: `{loading['model_class']}`")
        out.append(
            f"- hidden_size={loading['hidden_size']}, "
            f"num_heads={loading['num_attention_heads']}, "
            f"head_dim={loading['head_dim']}"
        )
        out.append(f"- candidates tried: {loading['candidates_tried']}")
    else:
        out.append("- All candidate models failed to load. Probe skipped.")
        if skipped:
            out.append(f"- last reason: {skipped[0]['model_loading'].get('reason')}")
    out.append("")

    out.append("## Encoder-only attention invariants validated")
    out.append("")
    out.append("1. `Q_tilde = Q N_Q`, `K_tilde = K N_K`, `V_tilde = V N_V`")
    out.append("2. `N_Q N_K^T = I` (per head)")
    out.append("3. `Q_tilde K_tilde^T = Q K^T`")
    out.append("4. `softmax(Q_tilde K_tilde^T / sqrt(d) + M) = softmax(Q K^T / sqrt(d) + M)`")
    out.append(
        "   for both `all_ones` and `padding` attention masks"
    )
    out.append("5. `AttnProb V_tilde = (AttnProb V) N_V` (per head)")
    out.append("6. `W_O` projects from V-mask space to encoder residual mask space;")
    out.append("   `Y_tilde = Y N_out`; use_pad pad compensation `C = T W N_out`.")
    out.append("")

    out.append("## Sweep results (per cell × mask kind)")
    headers = [
        "batch_size",
        "seq_len",
        "use_pad",
        "mask kind",
        "score max err",
        "prob max err",
        "attn_out max err",
        "output max err",
        "allclose",
    ]
    rows = []
    for r in results:
        if r["model_loading"]["status"] != "loaded":
            continue
        cfg = r["config"]
        for mk, payload in r["results_per_mask"].items():
            rows.append(
                [
                    cfg["batch_size"],
                    cfg["seq_len"],
                    cfg["use_pad"],
                    mk,
                    payload["score_metrics"]["max_abs_error"],
                    payload["prob_metrics"]["max_abs_error"],
                    payload["v_aggr_metrics"]["max_abs_error"],
                    payload["output_metrics"]["max_abs_error"],
                    payload["allclose"],
                ]
            )
    out.append(markdown_table(headers, rows) if rows else "_No loaded cells._")
    out.append("")

    out.append("## Pad vs no-pad comparison")
    headers = [
        "use_pad",
        "max output_err (any cell, any mask)",
        "all cells allclose?",
        "Q/K/V/O pad observed",
    ]
    rows = []
    for use_pad in (True, False):
        bucket = [
            payload
            for r in results
            if r["model_loading"]["status"] == "loaded"
            and r["config"]["use_pad"] is use_pad
            for payload in r["results_per_mask"].values()
        ]
        if not bucket:
            continue
        max_err = max(p["output_metrics"]["max_abs_error"] for p in bucket)
        allclose = all(p["allclose"] for p in bucket)
        pad_flags = [
            r["pad_report"]["per_mask"]["all_ones"]
            for r in results
            if r["model_loading"]["status"] == "loaded"
            and r["config"]["use_pad"] is use_pad
        ]
        pad_observed = bool(pad_flags and all(
            f["q_pad"] and f["k_pad"] and f["v_pad"] and f["o_pad"] for f in pad_flags
        )) if use_pad else not any(
            f["q_pad"] or f["k_pad"] or f["v_pad"] or f["o_pad"] for f in pad_flags
        )
        rows.append([use_pad, max_err, allclose, pad_observed])
    out.append(markdown_table(headers, rows) if rows else "_No loaded cells._")
    out.append("")

    out.append("## Padding mask coverage")
    rows = []
    for r in results:
        if r["model_loading"]["status"] != "loaded":
            continue
        cfg = r["config"]
        padding_payload = r["results_per_mask"].get("padding")
        if padding_payload is None:
            continue
        rows.append(
            [
                cfg["batch_size"],
                cfg["seq_len"],
                cfg["use_pad"],
                padding_payload["score_metrics"]["max_abs_error"],
                padding_payload["prob_metrics"]["max_abs_error"],
                padding_payload["output_metrics"]["max_abs_error"],
                padding_payload["allclose"],
            ]
        )
    headers = [
        "batch_size",
        "seq_len",
        "use_pad",
        "score max err",
        "prob max err",
        "output max err",
        "allclose",
    ]
    out.append(markdown_table(headers, rows) if rows else "_No padding-mask cells._")
    out.append("")

    out.append("## Limitations")
    out.append("")
    out.append("- This stage validates only encoder self-attention probe correctness.")
    out.append("- It does not implement full BERT obfuscated forward.")
    out.append("- It does not obfuscate LayerNorm.")
    out.append("- It does not obfuscate GELU / FFN.")
    out.append("- It does not cover MLM head.")
    out.append("- It does not claim real TEE security.")
    out.append("- It does not cover encoder-decoder cross-attention.")
    out.append("")

    out.append("## Next stage plan")
    out.append("")
    out.append("- **Stage 6.2** — Encoder-decoder cross-attention probe (T5 / BART). Q from")
    out.append("  the decoder hidden state, K / V from cached encoder memory; new cache")
    out.append("  data structure for encoder memory invariants `K_enc_tilde = K_enc N_K`,")
    out.append("  `V_enc_tilde = V_enc N_V`.")
    out.append("- **Stage 6.3** — Cross-architecture workload + security experiments")
    out.append("  (rerun Stage 5.0.1 profiler over decoder-only / encoder-only / encoder-decoder).")
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append("python scripts/run_encoder_attention_experiments.py")
    out.append("```")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    use_pad_choices = (
        (args.use_pad,) if args.use_pad is not None else SWEEP["use_pad"]
    )

    results: list[dict] = []
    for batch_size in SWEEP["batch_size"]:
        for seq_len in SWEEP["seq_len"]:
            for use_pad in use_pad_choices:
                cfg = EncoderAttentionProbeConfig(
                    model_id=args.model_id,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    use_pad=bool(use_pad),
                    dtype=args.dtype,
                    device=args.device,
                    seed=args.seed,
                )
                results.append(run_encoder_attention_probe(cfg))

    out_dir: Path = args.output_dir
    write_json(out_dir / "encoder_attention_experiments.json", {"results": results})

    csv_rows: list[dict] = []
    for r in results:
        if r["model_loading"]["status"] == "loaded":
            for mk in r["results_per_mask"]:
                csv_rows.append(_row_from_result(r, mk))
        else:
            csv_rows.append(_row_from_result(r, "all_ones"))
            csv_rows.append(_row_from_result(r, "padding"))
    write_csv(out_dir / "encoder_attention_experiments.csv", csv_rows, CSV_FIELDS)
    write_text(
        out_dir / "encoder_attention_experiments.md", _build_markdown(results)
    )

    loaded_cells = sum(1 for r in results if r["model_loading"]["status"] == "loaded")
    skipped_cells = sum(1 for r in results if r["model_loading"]["status"] != "loaded")
    all_loaded_allclose = all(
        payload["allclose"]
        for r in results
        if r["model_loading"]["status"] == "loaded"
        for payload in r["results_per_mask"].values()
    )
    print(
        f"cells={len(results)} (loaded={loaded_cells}, skipped={skipped_cells}), "
        f"all_loaded_allclose={all_loaded_allclose}, output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
