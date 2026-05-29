#!/usr/bin/env python
"""Stage 6.2 encoder-decoder cross-attention experiments — sweep + JSON/CSV/Markdown emitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    CrossAttentionProbeConfig,
    run_cross_attention_probe,
)
from pllo.experiments.report_utils import (
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
        help="Override encoder-decoder model id (default: registry candidates).",
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
    "dec_seq_len": (1, 4),
    "enc_seq_len": (4, 8, 16),
    "use_pad": (True, False),
}


CSV_FIELDS = (
    "model_id",
    "family",
    "status",
    "batch_size",
    "dec_seq_len",
    "enc_seq_len",
    "use_pad",
    "encoder_mask_kind",
    "q_max_abs_error",
    "k_max_abs_error",
    "v_max_abs_error",
    "score_max_abs_error",
    "prob_max_abs_error",
    "v_aggr_max_abs_error",
    "output_max_abs_error",
    "output_relative_l2_error",
    "output_cosine_similarity",
    "qk_constraint_error",
    "cache_key_max_abs_error",
    "cache_value_max_abs_error",
    "cache_allclose",
    "allclose",
    "q_pad",
    "k_pad",
    "v_pad",
    "o_pad",
)


def _skipped_row(cfg: dict, candidate: str | None, mask_kind: str) -> dict:
    return {
        "model_id": candidate,
        "family": None,
        "status": "skipped",
        "batch_size": cfg["batch_size"],
        "dec_seq_len": cfg["dec_seq_len"],
        "enc_seq_len": cfg["enc_seq_len"],
        "use_pad": cfg["use_pad"],
        "encoder_mask_kind": mask_kind,
        "q_max_abs_error": None,
        "k_max_abs_error": None,
        "v_max_abs_error": None,
        "score_max_abs_error": None,
        "prob_max_abs_error": None,
        "v_aggr_max_abs_error": None,
        "output_max_abs_error": None,
        "output_relative_l2_error": None,
        "output_cosine_similarity": None,
        "qk_constraint_error": None,
        "cache_key_max_abs_error": None,
        "cache_value_max_abs_error": None,
        "cache_allclose": None,
        "allclose": None,
        "q_pad": None,
        "k_pad": None,
        "v_pad": None,
        "o_pad": None,
    }


def _row_from_result(result: dict, mask_kind: str) -> dict:
    cfg = result["config"]
    loading = result["model_loading"]
    if loading["status"] != "loaded":
        candidate = (loading.get("candidates_tried") or [None])[0]
        return _skipped_row(cfg, candidate, mask_kind)
    qkv = result["qkv_invariants"]
    res = result["results_per_mask"][mask_kind]
    pad = result["pad_report"]["per_mask"][mask_kind]
    cache = result["encoder_memory_cache"]
    return {
        "model_id": loading["model_id"],
        "family": loading["family"],
        "status": "loaded",
        "batch_size": cfg["batch_size"],
        "dec_seq_len": cfg["dec_seq_len"],
        "enc_seq_len": cfg["enc_seq_len"],
        "use_pad": cfg["use_pad"],
        "encoder_mask_kind": mask_kind,
        "q_max_abs_error": qkv["q_metrics"]["max_abs_error"],
        "k_max_abs_error": qkv["k_metrics"]["max_abs_error"],
        "v_max_abs_error": qkv["v_metrics"]["max_abs_error"],
        "score_max_abs_error": res["score_metrics"]["max_abs_error"],
        "prob_max_abs_error": res["prob_metrics"]["max_abs_error"],
        "v_aggr_max_abs_error": res["v_aggr_metrics"]["max_abs_error"],
        "output_max_abs_error": res["output_metrics"]["max_abs_error"],
        "output_relative_l2_error": res["output_metrics"].get("relative_l2_error"),
        "output_cosine_similarity": res["output_metrics"].get("cosine_similarity"),
        "qk_constraint_error": qkv["qk_constraint_error"],
        "cache_key_max_abs_error": cache["key_metrics"]["max_abs_error"],
        "cache_value_max_abs_error": cache["value_metrics"]["max_abs_error"],
        "cache_allclose": cache["allclose"],
        "allclose": res["allclose"],
        "q_pad": pad["q_pad"],
        "k_pad": pad["k_pad"],
        "v_pad": pad["v_pad"],
        "o_pad": pad["o_pad"],
    }


def _build_markdown(results: list[dict]) -> str:
    out: list[str] = []
    out.append(
        "# Privacy LLM Obfuscation — Encoder-decoder Cross-attention Probe (Stage 6.2)"
    )
    out.append("")
    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Encoder-decoder cross-attention probe: decoder hidden state supplies"
        " Q while encoder memory supplies K / V, so the input mask space for Q"
        " is independent of the input mask space for K / V. Validates the same"
        " mask + pad invariants as Stage 5.0 (decoder-only) and Stage 6.1"
        " (encoder-only), plus a probe-level ``EncoderMemoryCache`` carrying"
        " obfuscated K / V."
    )
    out.append("")

    # ---- Model loading status ----
    out.append("## Model loading status")
    loaded = [r for r in results if r["model_loading"]["status"] == "loaded"]
    skipped = [r for r in results if r["model_loading"]["status"] != "loaded"]
    if loaded:
        loading = loaded[0]["model_loading"]
        out.append(f"- **model_id**: `{loading['model_id']}`")
        out.append(f"- model class: `{loading['model_class']}`")
        out.append(f"- family: `{loading['family']}`")
        out.append(
            f"- hidden_size={loading['hidden_size']},"
            f" num_heads={loading['num_attention_heads']},"
            f" head_dim={loading['head_dim']},"
            f" inner_dim={loading['inner_dim']}"
        )
        out.append(
            f"- bias_present (q/k/v/o): {loading['bias_present']}"
        )
        out.append(
            f"- cross-attention has_relative_attention_bias:"
            f" {loading['cross_attention_has_relative_bias']}"
        )
        out.append(f"- candidates tried: {loading['candidates_tried']}")
    else:
        out.append("- All candidate models failed to load. Probe skipped.")
        if skipped:
            out.append(f"- last reason: {skipped[0]['model_loading'].get('reason')}")
    out.append("")

    # ---- Cross-attention invariants ----
    out.append("## Encoder-decoder cross-attention invariants validated")
    out.append("")
    out.append("1. `Q_dec_tilde = Q_dec N_Q_dec` (per head)")
    out.append("2. `K_enc_tilde = K_enc N_K_enc`")
    out.append("3. `V_enc_tilde = V_enc N_V_enc`")
    out.append("4. `N_Q_dec N_K_enc^T = I` (per head)")
    out.append("5. `Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T`")
    out.append(
        "6. `softmax(Q_dec_tilde K_enc_tilde^T / sqrt(d) + M_enc)"
        " = softmax(Q_dec K_enc^T / sqrt(d) + M_enc)` for both `all_ones`"
        " and `padding` encoder masks"
    )
    out.append("7. `AttnProb V_enc_tilde = (AttnProb V_enc) N_V_enc` (per head)")
    out.append(
        "8. `W_O` projects from V-mask space → decoder residual mask space"
        " `N_dec_out`; use_pad pad compensation `C = T W N_out`."
    )
    out.append("")

    # ---- EncoderMemoryCache invariants ----
    out.append("## Encoder memory cache invariants")
    out.append("")
    headers = [
        "batch_size",
        "dec_seq_len",
        "enc_seq_len",
        "use_pad",
        "K cache max err",
        "V cache max err",
        "cache allclose",
    ]
    rows = []
    for r in results:
        if r["model_loading"]["status"] != "loaded":
            continue
        cfg = r["config"]
        cache = r["encoder_memory_cache"]
        rows.append(
            [
                cfg["batch_size"],
                cfg["dec_seq_len"],
                cfg["enc_seq_len"],
                cfg["use_pad"],
                cache["key_metrics"]["max_abs_error"],
                cache["value_metrics"]["max_abs_error"],
                cache["allclose"],
            ]
        )
    out.append(markdown_table(headers, rows) if rows else "_No loaded cells._")
    out.append("")

    # ---- Sweep results ----
    out.append("## Sweep results (per cell × encoder mask kind)")
    headers = [
        "batch_size",
        "dec_seq_len",
        "enc_seq_len",
        "use_pad",
        "encoder mask",
        "score max err",
        "prob max err",
        "v_aggr max err",
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
                    cfg["dec_seq_len"],
                    cfg["enc_seq_len"],
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

    # ---- Pad vs no-pad comparison ----
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
        if use_pad:
            pad_observed = bool(pad_flags) and all(
                f["q_pad"] and f["k_pad"] and f["v_pad"] and f["o_pad"]
                for f in pad_flags
            )
        else:
            pad_observed = not any(
                f["q_pad"] or f["k_pad"] or f["v_pad"] or f["o_pad"]
                for f in pad_flags
            )
        rows.append([use_pad, max_err, allclose, pad_observed])
    out.append(markdown_table(headers, rows) if rows else "_No loaded cells._")
    out.append("")

    # ---- all_ones vs padding mask comparison ----
    out.append("## All-ones vs padding encoder-mask comparison")
    headers = [
        "encoder mask",
        "max output_err (any cell)",
        "all cells allclose?",
    ]
    rows = []
    for mk in ("all_ones", "padding"):
        bucket = [
            r["results_per_mask"].get(mk)
            for r in results
            if r["model_loading"]["status"] == "loaded"
        ]
        bucket = [b for b in bucket if b is not None]
        if not bucket:
            continue
        max_err = max(p["output_metrics"]["max_abs_error"] for p in bucket)
        allclose = all(p["allclose"] for p in bucket)
        rows.append([mk, max_err, allclose])
    out.append(markdown_table(headers, rows) if rows else "_No loaded cells._")
    out.append("")

    # ---- Limitations ----
    out.append("## Limitations")
    out.append("")
    out.append(
        "- This stage validates only encoder-decoder cross-attention probe correctness."
    )
    out.append("- It does not implement full T5/BART obfuscated forward.")
    out.append("- It does not implement encoder-decoder generation.")
    out.append("- It does not implement decoder self-attention cache.")
    out.append("- It does not obfuscate LayerNorm.")
    out.append("- It does not obfuscate FFN / activation.")
    out.append("- It does not cover LM head.")
    out.append("- It does not claim real TEE security.")
    out.append(
        "- Relative position bias is not handled unless explicitly added as a"
        " shared additive score bias in both plain and obfuscated paths."
    )
    out.append("")

    # ---- Next stage plan ----
    out.append("## Next stage plan")
    out.append("")
    out.append(
        "- **Stage 6.3** — Cross-architecture workload + security experiments."
        " Rerun the Stage 5.0.1 workload profiler over decoder-only /"
        " encoder-only / encoder-decoder to fill the 3×3 architecture × method"
        " matrix; document cost and leakage trade-offs per architecture using"
        " the probe data from Stages 5.0 / 6.1 / 6.2."
    )
    out.append("")

    # ---- Reproducibility ----
    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append("python scripts/run_cross_attention_experiments.py")
    out.append("```")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    use_pad_choices = (
        (args.use_pad,) if args.use_pad is not None else SWEEP["use_pad"]
    )

    results: list[dict] = []
    for batch_size in SWEEP["batch_size"]:
        for dec_seq_len in SWEEP["dec_seq_len"]:
            for enc_seq_len in SWEEP["enc_seq_len"]:
                for use_pad in use_pad_choices:
                    cfg = CrossAttentionProbeConfig(
                        model_id=args.model_id,
                        batch_size=batch_size,
                        dec_seq_len=dec_seq_len,
                        enc_seq_len=enc_seq_len,
                        use_pad=bool(use_pad),
                        dtype=args.dtype,
                        device=args.device,
                        seed=args.seed,
                    )
                    results.append(run_cross_attention_probe(cfg))

    out_dir: Path = args.output_dir
    write_json(out_dir / "cross_attention_experiments.json", {"results": results})

    csv_rows: list[dict] = []
    for r in results:
        if r["model_loading"]["status"] == "loaded":
            for mk in r["results_per_mask"]:
                csv_rows.append(_row_from_result(r, mk))
        else:
            csv_rows.append(_row_from_result(r, "all_ones"))
            csv_rows.append(_row_from_result(r, "padding"))
    write_csv(
        out_dir / "cross_attention_experiments.csv", csv_rows, CSV_FIELDS
    )
    write_text(
        out_dir / "cross_attention_experiments.md", _build_markdown(results)
    )

    loaded_cells = sum(
        1 for r in results if r["model_loading"]["status"] == "loaded"
    )
    skipped_cells = sum(
        1 for r in results if r["model_loading"]["status"] != "loaded"
    )
    all_loaded_allclose = all(
        payload["allclose"]
        for r in results
        if r["model_loading"]["status"] == "loaded"
        for payload in r["results_per_mask"].values()
    )
    cache_allclose = all(
        r["encoder_memory_cache"]["allclose"]
        for r in results
        if r["model_loading"]["status"] == "loaded"
    )
    print(
        f"cells={len(results)} (loaded={loaded_cells}, skipped={skipped_cells}),"
        f" all_loaded_allclose={all_loaded_allclose},"
        f" cache_allclose={cache_allclose}, output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
