"""Runner for Stage 7.0 -- full-pipeline cost/leakage/ablation evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.full_pipeline_cost_leakage import (  # noqa: E402
    FullPipelineCostLeakageConfig,
    run_full_pipeline_cost_leakage,
)

_COST_COLS = [
    "variant", "implemented", "analytical_only", "gpu_flops_prefill",
    "gpu_flops_decode", "tee_flops_prefill", "tee_flops_decode",
    "transfer_bytes_prefill", "transfer_bytes_decode", "kv_cache_bytes",
    "boundary_calls", "handoff_gemm_flops", "lm_head_gpu_flops",
    "lm_head_tee_flops", "logits_recovery_flops",
]


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Stage 7.0 -- Full-Pipeline Cost / Leakage / Ablation Evaluation")
    w()
    s = report["summary"]
    w(f"- recommended_default: **{s['recommended_default']}**")
    w(f"- cheapest_secure_boundary: **{s['cheapest_secure_boundary']}**")
    w(f"- highest_tee_compute_variant: **{s['highest_tee_compute_variant']}**")
    w(f"- handoff_gemm_required_for_per_layer_masks: "
      f"**{s['handoff_gemm_required_for_per_layer_masks']}** | "
      f"no_intermediate_tee: **{s['no_intermediate_tee']}**")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Config")
    w()
    for kk, vv in report["config"].items():
        w(f"- {kk}: `{vv}`")
    w()
    w("## Cost comparison (FLOPs = 2*M*N*K)")
    w()
    w("| variant | impl | gpu_pf | gpu_dec | tee_pf | tee_dec | handoff | "
      "lm_gpu | lm_tee | recover | xfer_pf | xfer_dec | bndry |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in report["cost_breakdown"]:
        w(f"| {c['variant']} | {c['implemented']} | "
          f"{c['gpu_flops_prefill']:.2e} | {c['gpu_flops_decode']:.2e} | "
          f"{c['tee_flops_prefill']:.2e} | {c['tee_flops_decode']:.2e} | "
          f"{c['handoff_gemm_flops']:.2e} | {c['lm_head_gpu_flops']:.2e} | "
          f"{c['lm_head_tee_flops']:.2e} | {c['logits_recovery_flops']:.2e} | "
          f"{c['transfer_bytes_prefill']} | {c['transfer_bytes_decode']} | "
          f"{c['boundary_calls']} |")
    w()
    if report["timing_breakdown"]:
        w("## Timing (CPU synthetic proxy, ms)")
        w()
        w("| variant | total_mean | total_median | num_repeats | dtype |")
        w("|---|---|---|---|---|")
        for t in report["timing_breakdown"]:
            w(f"| {t['variant']} | {t['total_ms_mean']:.3f} | "
              f"{t['total_ms_median']:.3f} | {t['num_repeats']} | "
              f"{t['dtype']} |")
        w()
    w("## Leakage surfaces (what the GPU sees)")
    w()
    w("| variant | input_ids | plain_emb | masked_emb | plain_hidden | "
      "masked_hidden | attn_scores | plain_logits | masked_logits | "
      "sampled_ids | text_protected |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s2 in report["leakage_surfaces"]:
        w(f"| {s2['variant']} | {s2['input_ids_visible_to_gpu']} | "
          f"{s2['plaintext_embedding_visible_to_gpu']} | "
          f"{s2['masked_embedding_visible_to_gpu']} | "
          f"{s2['plaintext_hidden_visible_to_gpu']} | "
          f"{s2['masked_hidden_visible_to_gpu']} | "
          f"{s2['attention_scores_visible_to_gpu']} | "
          f"{s2['plaintext_logits_visible_to_gpu']} | "
          f"{s2['masked_logits_visible_to_gpu']} | "
          f"{s2['sampled_token_ids_visible_to_gpu']} | "
          f"{s2['final_output_text_semantics_protected']} |")
    w()
    lp = report.get("leakage_proxy") or {}
    if lp:
        w("## Leakage proxies (NOT security proofs)")
        w()
        if "vocab_mask" in lp:
            w("Vocab-mask token-index linkability (GPU argmax alignment / TEE "
              "recovery):")
            w()
            w("| mode | gpu_argmax_matches_plain | tee_recovered_top1 |")
            w("|---|---|---|")
            for k in ("no_mask", "permutation_only",
                      "permutation_plus_scaling"):
                m = lp["vocab_mask"][k]
                w(f"| `{k}` | "
                  f"{m['gpu_argmax_token_index_matches_plain']:.4f} | "
                  f"{m['tee_recovered_top1_matches_plain']:.4f} |")
            w()
            w(f"- {lp['vocab_mask']['note']}")
            w()
        if "rope_pair_norm" in lp:
            w("RoPE pair-norm linkability (cross-session / NN matching):")
            w()
            w("| mode | cross_session_corr | nn_matching_acc |")
            w("|---|---|---|")
            for k, mm in lp["rope_pair_norm"].items():
                w(f"| `{k}` | "
                  f"{mm['cross_session_pair_norm_correlation']:.4f} | "
                  f"{mm['nearest_neighbor_matching_accuracy_pair_norm']:.4f} |")
            w()
    w("## Paper claims")
    w()
    w("**Safe claims:**")
    w()
    for c in report["paper_claims"]["safe_claims"]:
        w(f"- {c}")
    w()
    w("**Unsafe claims (must NOT be made):**")
    w()
    for c in report["paper_claims"]["unsafe_claims"]:
        w(f"- {c}")
    w()
    w("**Required caveats:**")
    w()
    for c in report["paper_claims"]["required_caveats"]:
        w(f"- {c}")
    w()
    w("## Limitations")
    w()
    for lim in report["limitations"]:
        w(f"- {lim}")
    w()
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, report: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COST_COLS)
        writer.writeheader()
        for c in report["cost_breakdown"]:
            writer.writerow({k: c[k] for k in _COST_COLS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/full_pipeline_cost_leakage.json")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--prefill-seq-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--intermediate-size", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-key-value-heads", type=int, default=2)
    ap.add_argument("--num-repeats", type=int, default=5)
    args = ap.parse_args()

    cfg = FullPipelineCostLeakageConfig(
        num_layers=args.num_layers, prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps, vocab_size=args.vocab_size,
        hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
        num_heads=args.num_heads, num_key_value_heads=args.num_key_value_heads,
        num_repeats=args.num_repeats,
    )
    report = run_full_pipeline_cost_leakage(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    out_csv = out_json.with_suffix(".csv")
    _write_csv(out_csv, report)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")
    print(f"stage={report['stage']} variants={len(report['cost_breakdown'])} "
          f"recommended={report['summary']['recommended_default']}")


if __name__ == "__main__":
    main()
