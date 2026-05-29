#!/usr/bin/env python
"""Stage 5.0 attention experiments — sweep config + emit JSON / CSV / Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    ATTENTION_SWEEP,
    AttentionProbeConfig,
    run_attention_probe,
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


CSV_FIELDS = (
    "model_id",
    "batch_size",
    "seq_len",
    "decode_steps",
    "use_pad",
    "full_output_max_err",
    "score_max_err",
    "prob_max_err",
    "v_aggr_max_err",
    "qk_constraint_error",
    "prefill_output_max_err",
    "prefill_cache_key_err",
    "prefill_cache_value_err",
    "decode_output_max_err",
    "decode_cache_key_err",
    "decode_cache_value_err",
    "full_allclose",
    "prefill_cache_allclose",
    "decode_cache_append_allclose",
)


def _row_from_result(result: dict) -> dict:
    cfg = result["config"]
    full = result["full_attention"]
    prefill = result["prefill_attention"]
    decode = result["decode_attention"]
    return {
        "model_id": cfg["model_id"],
        "batch_size": cfg["batch_size"],
        "seq_len": cfg["seq_len"],
        "decode_steps": cfg["decode_steps"],
        "use_pad": cfg["use_pad"],
        "full_output_max_err": full["output_metrics"]["max_abs_error"],
        "score_max_err": full["score_metrics"]["max_abs_error"],
        "prob_max_err": full["prob_metrics"]["max_abs_error"],
        "v_aggr_max_err": full["v_aggr_metrics"]["max_abs_error"],
        "qk_constraint_error": full["qk_constraint_error"],
        "prefill_output_max_err": prefill["output_metrics"]["max_abs_error"],
        "prefill_cache_key_err": prefill["cache_key_metrics"]["max_abs_error"],
        "prefill_cache_value_err": prefill["cache_value_metrics"]["max_abs_error"],
        "decode_output_max_err": decode.get("decode_output_max_abs_error_max"),
        "decode_cache_key_err": decode["cache_append_key_metrics"]["max_abs_error"],
        "decode_cache_value_err": decode["cache_append_value_metrics"]["max_abs_error"],
        "full_allclose": full["allclose"],
        "prefill_cache_allclose": prefill["cache_invariant_allclose"],
        "decode_cache_append_allclose": decode["cache_append_invariant_allclose"],
    }


def _build_markdown(results: list[dict]) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Attention Experiments (Stage 5.0)")
    out.append("")
    out.append(
        "Six invariants validated per `(batch_size, seq_len, decode_steps, use_pad)` cell:"
    )
    out.append("")
    out.append("1. `Q_tilde K_tilde^T ≈ Q K^T`")
    out.append("2. `softmax(Q_tilde K_tilde^T / sqrt(d)) ≈ softmax(Q K^T / sqrt(d))`")
    out.append("3. `A V_tilde ≈ (A V) N_V`")
    out.append("4. `AttnOut_tilde ≈ AttnOut N_res`")
    out.append("5. `K_cache_tilde_new ≈ K_cache_new N_K` (prefill + decode append)")
    out.append("6. `V_cache_tilde_new ≈ V_cache_new N_V`")
    out.append("")
    out.append("All numbers are read from a fresh sweep over the registry in")
    out.append("`src.pllo.experiments.experiment_registry.ATTENTION_SWEEP`.")
    out.append("")

    out.append("## Sweep coverage")
    headers = ["batch_size", "seq_len", "decode_steps", "use_pad", "full_allclose", "cache_append_allclose"]
    rows = []
    for r in results:
        cfg = r["config"]
        rows.append([
            cfg["batch_size"],
            cfg["seq_len"],
            cfg["decode_steps"],
            cfg["use_pad"],
            r["full_attention"]["allclose"],
            r["decode_attention"]["cache_append_invariant_allclose"],
        ])
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Headline invariants — worst cell per dimension")
    headers = [
        "metric",
        "max over sweep",
        "all cells allclose?",
    ]
    rows = [
        [
            "Q K^T (score)",
            max(r["full_attention"]["score_metrics"]["max_abs_error"] for r in results),
            all(r["full_attention"]["score_metrics"]["allclose"] for r in results),
        ],
        [
            "softmax probs",
            max(r["full_attention"]["prob_metrics"]["max_abs_error"] for r in results),
            all(r["full_attention"]["prob_metrics"]["allclose"] for r in results),
        ],
        [
            "A V (per head)",
            max(r["full_attention"]["v_aggr_metrics"]["max_abs_error"] for r in results),
            all(r["full_attention"]["v_aggr_metrics"]["allclose"] for r in results),
        ],
        [
            "AttnOut (full path)",
            max(r["full_attention"]["output_metrics"]["max_abs_error"] for r in results),
            all(r["full_attention"]["output_metrics"]["allclose"] for r in results),
        ],
        [
            "Prefill K cache",
            max(r["prefill_attention"]["cache_key_metrics"]["max_abs_error"] for r in results),
            all(r["prefill_attention"]["cache_key_metrics"]["allclose"] for r in results),
        ],
        [
            "Prefill V cache",
            max(r["prefill_attention"]["cache_value_metrics"]["max_abs_error"] for r in results),
            all(r["prefill_attention"]["cache_value_metrics"]["allclose"] for r in results),
        ],
        [
            "Decode K cache append",
            max(r["decode_attention"]["cache_append_key_metrics"]["max_abs_error"] for r in results),
            all(r["decode_attention"]["cache_append_key_metrics"]["allclose"] for r in results),
        ],
        [
            "Decode V cache append",
            max(r["decode_attention"]["cache_append_value_metrics"]["max_abs_error"] for r in results),
            all(r["decode_attention"]["cache_append_value_metrics"]["allclose"] for r in results),
        ],
        [
            "N_Q N_K^T = I",
            max(r["full_attention"]["qk_constraint_error"] for r in results),
            "—",
        ],
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## use_pad = true vs false (worst cell of each)")
    by_pad = {True: [], False: []}
    for r in results:
        by_pad[r["config"]["use_pad"]].append(r)
    pad_rows = []
    for variant in (True, False):
        bucket = by_pad[variant]
        if not bucket:
            continue
        pad_rows.append(
            [
                variant,
                max(b["full_attention"]["output_metrics"]["max_abs_error"] for b in bucket),
                max(b["full_attention"]["score_metrics"]["max_abs_error"] for b in bucket),
                max(b["prefill_attention"]["cache_key_metrics"]["max_abs_error"] for b in bucket),
                max(b["decode_attention"]["cache_append_key_metrics"]["max_abs_error"] for b in bucket),
                all(b["full_attention"]["allclose"] for b in bucket)
                and all(b["decode_attention"]["cache_append_invariant_allclose"] for b in bucket),
            ]
        )
    out.append(
        markdown_table(
            [
                "use_pad",
                "max full_out_err",
                "max score_err",
                "max prefill_K_err",
                "max decode_K_err",
                "all_allclose",
            ],
            pad_rows,
        )
    )
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append("python scripts/run_attention_experiments.py")
    out.append("```")
    out.append("")
    out.append(
        "Sweep registry: `batch_size ∈ {1, 2}`, `seq_len ∈ {4, 8, 16}`,"
        " `decode_steps ∈ {1, 2, 4}`, `use_pad ∈ {true, false}` → 36 cells."
    )
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    use_pad_choices = (
        (args.use_pad,) if args.use_pad is not None else ATTENTION_SWEEP["use_pad"]
    )

    results: list[dict] = []
    for batch_size in ATTENTION_SWEEP["batch_size"]:
        for seq_len in ATTENTION_SWEEP["seq_len"]:
            for decode_steps in ATTENTION_SWEEP["decode_steps"]:
                for use_pad in use_pad_choices:
                    cfg = AttentionProbeConfig(
                        model_id=args.model_id,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        decode_steps=decode_steps,
                        use_pad=bool(use_pad),
                        dtype=args.dtype,
                        device=args.device,
                        seed=args.seed,
                    )
                    results.append(run_attention_probe(cfg))

    out_dir: Path = args.output_dir
    write_json(out_dir / "attention_experiments.json", {"results": results})
    write_csv(
        out_dir / "attention_experiments.csv",
        (_row_from_result(r) for r in results),
        CSV_FIELDS,
    )
    write_text(out_dir / "attention_experiments.md", _build_markdown(results))

    print(
        f"cells={len(results)}, all_full_allclose={all(r['full_attention']['allclose'] for r in results)}, "
        f"all_cache_allclose={all(r['decode_attention']['cache_append_invariant_allclose'] for r in results)}, "
        f"output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
