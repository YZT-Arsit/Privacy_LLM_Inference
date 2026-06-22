"""Runner for Stage 6.8 -- full masked CausalLM skeleton probe (no network)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.masked_causal_lm_skeleton_probe import (  # noqa: E402
    MaskedCausalLMSkeletonProbeConfig,
    run_masked_causal_lm_skeleton_probe,
)


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    pre = report["prefill_metrics"]
    md = report["metadata"]
    w("# Stage 6.8 -- Full Masked CausalLM Skeleton Probe")
    w()
    w(f"- Stage: **{report['stage']}** | Status: **{report['status']}** | "
      f"all_allclose: **{report['all_allclose']}**")
    w(f"- token_match_rate: **{report['token_match_rate']}** | "
      f"num_layers: **{md['num_layers']}**")
    w(f"- security_status: **{md['security_status']}**")
    w(f"- residual_mask_handoff: **{md['residual_mask_handoff']}** | "
      f"handoff_transform: **{report['mask_metadata']['handoff_transform']}**")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Config")
    w()
    for kk, vv in report["config"].items():
        w(f"- {kk}: `{vv}`")
    w()
    w("## Prefill metrics")
    w()
    w("| metric | value |")
    w("|---|---|")
    for k in ("embedding_mask_max_abs_error", "final_hidden_max_abs_error",
              "masked_logits_max_abs_error", "recovered_logits_max_abs_error",
              "greedy_token_match_rate"):
        v = pre[k]
        w(f"| `{k}` | {v:.3e} |" if isinstance(v, float) and v < 1
          else f"| `{k}` | {v} |")
    w(f"| `prefill_allclose` | {pre['allclose']} |")
    w()
    w("Per-layer handoff input invariant `H_ell_tilde == H_ell_plain @ N_ell`:")
    w()
    w("| ell | handoff_max_abs_error |")
    w("|---|---|")
    for i, e in enumerate(pre["per_layer_handoff_max_abs_error"]):
        w(f"| {i} | {e:.3e} |")
    w()
    w("## Per-layer prefill")
    w()
    w("| layer | final_output | attn_score | mlp_output | cache_key | "
      "cache_value |")
    w("|---|---|---|---|---|---|")
    for pl in pre["per_layer"]:
        w(f"| {pl['layer']} | {pl['final_output_max_abs_error']:.3e} | "
          f"{pl['attention_score_max_abs_error']:.3e} | "
          f"{pl['mlp_output_max_abs_error']:.3e} | "
          f"{pl['cache_key_max_abs_error']:.3e} | "
          f"{pl['cache_value_max_abs_error']:.3e} |")
    w()
    w("## Decode steps")
    w()
    w("| step | pos | tok_match | final_hidden | masked_logits | "
      "recovered_logits | layer_out | cache_key | cache_value |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in report["decode_step_metrics"]:
        w(f"| {s['step']} | {s['position']} | {s['sampled_token_match']} | "
          f"{s['final_hidden_error']:.3e} | {s['masked_logits_error']:.3e} | "
          f"{s['recovered_logits_error']:.3e} | "
          f"{s['per_layer_output_error']:.3e} | "
          f"{s['per_layer_cache_append_key_error']:.3e} | "
          f"{s['per_layer_cache_append_value_error']:.3e} |")
    w()
    w("## Security metadata")
    w()
    for k in ("no_intermediate_tee", "input_ids_visible_to_gpu",
              "plaintext_embedding_visible_to_gpu",
              "plaintext_logits_visible_to_gpu", "masked_logits_visible_to_gpu",
              "logits_recovered_in_tee", "sampling_boundary",
              "decoder_runs_on_gpu_assumption", "semantic_security_claimed",
              "formal_security_claimed", "cryptographic_security_claimed"):
        w(f"- `{k}`: {md[k]}")
    w()
    w("## Limitations")
    w()
    for lim in report["limitations"]:
        w(f"- {lim}")
    w()
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="outputs/masked_causal_lm_skeleton_probe.json")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--prefill-seq-len", type=int, default=5)
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-key-value-heads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2031)
    args = ap.parse_args()

    cfg = MaskedCausalLMSkeletonProbeConfig(
        num_layers=args.num_layers, prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps, vocab_size=args.vocab_size,
        hidden_size=args.hidden_size, num_heads=args.num_heads,
        num_key_value_heads=args.num_key_value_heads, seed=args.seed,
    )
    report = run_masked_causal_lm_skeleton_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']} all_allclose={report['all_allclose']} "
          f"token_match_rate={report['token_match_rate']}")


if __name__ == "__main__":
    main()
