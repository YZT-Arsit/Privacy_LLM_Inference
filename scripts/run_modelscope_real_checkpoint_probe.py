"""Stage 8.2 -- real ModelScope checkpoint GPU probe CLI.

Compact JSON + MD only (no tensor dumps). ModelScope downloads only; never
Hugging Face remote. Skips cleanly if modelscope/transformers/CUDA/checkpoint
are unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.modelscope_real_checkpoint_probe import (  # noqa: E402
    ModelScopeRealCheckpointProbeConfig,
    run_modelscope_real_checkpoint_probe,
)


def _render_markdown(r: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Stage 8.2 — Real ModelScope Checkpoint Probe")
    w()
    w(f"- model_id: **{r['config']['model_id']}**")
    w(f"- status: **{r['status']}**")
    rd = r.get("resolved_dtypes", {})
    w(f"- dtypes: model=**{rd.get('model', r.get('resolved_dtype'))}** "
      f"folding=**{rd.get('folding')}** "
      f"runtime=**{rd.get('folded_weight_runtime')}** "
      f"recovery=**{rd.get('recovery')}** | device: "
      f"**{r['config']['device']}**")
    env = r.get("environment", {})
    w(f"- cuda_available: **{env.get('cuda_available')}** | device_name: "
      f"**{env.get('device_name', 'n/a')}**")
    if r["status"] != "ok":
        if r.get("reason"):
            w(f"- reason: `{r['reason']}`")
        w()
        w(f"> {r['required_statement']}")
        w()
        w("## Caveats")
        for c in r.get("caveats", []):
            w(f"- {c}")
        w()
        return "\n".join(lines) + "\n"

    w(f"- model_type: **{r['model_type']}** | total_layers: "
      f"**{r['total_layers']}** | max_layers: **{r['max_layers']}** | "
      f"partial: **{r['partial_layer_diagnostic']}**")
    w(f"- hidden_size: **{r['hidden_size']}** | vocab_size: "
      f"**{r['vocab_size']}**")
    mask = r.get("mask", {})
    w(f"- mask_mode: **{mask.get('mask_mode')}** | strategy: "
      f"**{mask.get('residual_mask_strategy')}** | shared: "
      f"**{mask.get('shared_residual_mask')}**")
    w()
    w(f"> {r['required_statement']}")
    w()
    if "hf_baseline" in r:
        hb = r["hf_baseline"]
        w("## HF baseline (full model, greedy)")
        w()
        if hb.get("status") == "failed":
            w(f"- FAILED: `{hb.get('reason')}`")
        else:
            w(f"- latency_s: **{hb['latency_s']}** | tokens/s: "
              f"**{hb['tokens_per_s']}**")
            w(f"- new_token_ids: `{hb['new_token_ids']}`")
            w(f"- text_head: `{hb['generated_text_head']!r}`")
        w()
    if "extracted_plain" in r:
        ep = r["extracted_plain"]
        w("## Extracted-weight plaintext reference")
        w()
        w(f"- latency_s: **{ep['latency_s']}** | num_tokens: "
          f"**{ep['num_tokens']}**")
        w()
    if "masked_runtime" in r:
        mr = r["masked_runtime"]
        w("## Masked runtime (simulated TEE)")
        w()
        w(f"- token_match_rate_vs_extracted: "
          f"**{mr['token_match_rate_vs_extracted']}**")
        w(f"- recovered_logits_max_abs_error: "
          f"**{mr['recovered_logits_max_abs_error']:.3e}**")
        w(f"- masked_logits_max_abs_error: "
          f"**{mr['masked_logits_max_abs_error']:.3e}**")
        w(f"- latency_s (incl. reference): "
          f"**{mr['latency_s_with_reference']}**")
        w()
    if "bf16_diagnostics" in r and r["bf16_diagnostics"]:
        d = r["bf16_diagnostics"]
        w("## Mixed-precision diagnostics")
        w()
        w("| metric | value |")
        w("|---|---|")
        for k in ("embedding_boundary_max_abs_err",
                  "layer_0_input_invariant_max_abs_err",
                  "layer_0_output_invariant_max_abs_err",
                  "final_norm_core_max_abs_err", "masked_logits_max_abs_err",
                  "recovered_logits_max_abs_err",
                  "recovered_logits_mean_abs_err",
                  "recovered_logits_relative_l2_err",
                  "greedy_token_match_rate"):
            if k in d:
                v = d[k]
                w(f"| `{k}` | {v:.3e} |" if isinstance(v, float)
                  and 0 < abs(v) < 1e6 else f"| `{k}` | {v} |")
        m = d.get("top1_margin_stats", {})
        if m:
            w(f"| `top1_min_margin` | {m['min_margin']:.3e} |")
            w(f"| `top1_mean_margin` | {m['mean_margin']:.3e} |")
            w(f"| `positions_with_margin_below_error` | "
              f"{m['num_positions_with_margin_below_error']}/"
              f"{m['num_positions']} |")
        w()
    if "boundary" in r:
        b = r["boundary"]
        w("## Boundary accounting")
        w()
        w("| metric | value |")
        w("|---|---|")
        for k in ("boundary_calls", "tee_to_gpu_mb", "gpu_to_tee_mb",
                  "handoff_gemm_count", "handoff_gemm_flops",
                  "logits_recovery_flops"):
            w(f"| `{k}` | {b[k]} |")
        w()
    peak = (r.get("masked_runtime", {}) or {}).get("peak_cuda_memory")
    if peak:
        w("## Peak CUDA memory (masked phase)")
        w()
        for k, v in peak.items():
            w(f"- {k}: {v} MB")
        w()
    w("## Caveats")
    w()
    for c in r["caveats"]:
        w(f"- {c}")
    w()
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache-dir", default="/root/modelscope_cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    help="model load + HF baseline dtype")
    ap.add_argument("--folding-dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--folded-weight-runtime-dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--recovery-dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--compare-dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--prefill-seq-len", type=int, default=16)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--max-layers", default="1")
    ap.add_argument("--mask-mode", default="signed_permutation",
                    choices=["signed_permutation", "block_orthogonal",
                             "dense_orthogonal"])
    ap.add_argument("--residual-mask-strategy", default="shared",
                    choices=["shared", "per_layer"])
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--allow-dense-large-mask", action="store_true")
    ap.add_argument("--no-hf-baseline", action="store_true")
    ap.add_argument("--max-report-mb", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--output",
                    default="outputs/modelscope_real_checkpoint_probe.json")
    args = ap.parse_args()

    max_layers: int | str = ("all" if args.max_layers == "all"
                             else int(args.max_layers))
    cfg = ModelScopeRealCheckpointProbeConfig(
        model_id=args.model_id, cache_dir=args.cache_dir, device=args.device,
        dtype=args.dtype, folding_dtype=args.folding_dtype,
        folded_weight_runtime_dtype=args.folded_weight_runtime_dtype,
        recovery_dtype=args.recovery_dtype, compare_dtype=args.compare_dtype,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps, max_layers=max_layers,
        mask_mode=args.mask_mode,
        residual_mask_strategy=args.residual_mask_strategy,
        block_size=args.block_size,
        allow_dense_large_mask=args.allow_dense_large_mask,
        run_hf_baseline=not args.no_hf_baseline,
        max_report_mb=args.max_report_mb, seed=args.seed)

    report = run_modelscope_real_checkpoint_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str)
    size_mb = len(text.encode("utf-8")) / 2 ** 20
    if size_mb > args.max_report_mb:
        # Hard guard: drop verbose sub-dicts, keep summary.
        for k in ("environment",):
            report.pop(k, None)
        report["report_size_guard_triggered"] = True
        text = json.dumps(report, indent=2, default=str)
    out_json.write_text(text, encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json} ({round(size_mb, 4)} MB)")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']}")
    if report["status"] == "ok" and "masked_runtime" in report:
        mr = report["masked_runtime"]
        print(f"token_match_rate_vs_extracted="
              f"{mr['token_match_rate_vs_extracted']} "
              f"recovered_logits_err={mr['recovered_logits_max_abs_error']:.2e}")
    return 0 if report["status"] in ("ok",) or report["status"].startswith(
        "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
