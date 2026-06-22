"""Runner for Stage 6.9 -- HF full-model masked CausalLM skeleton probe.

Compact reports only (summary metrics; no tensor dumps). CPU-only, no
network download, no HF generate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.hf_causal_lm_skeleton_probe import (  # noqa: E402
    HFCausalLMSkeletonProbeConfig,
    run_hf_causal_lm_skeleton_probe,
)


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    md = report.get("metadata", {})
    w("# Stage 6.9 -- HF Full-Model Masked CausalLM Skeleton Probe")
    w()
    w(f"- Status: **{report['status']}**")
    if report["status"] != "ok":
        if "reason" in report:
            w(f"- Reason: `{report['reason']}`")
        w()
        w(f"> {report.get('required_statement', '')}")
        w()
        w("## Limitations")
        w()
        for lim in report.get("limitations", []):
            w(f"- {lim}")
        w()
        return "\n".join(lines) + "\n"

    pre = report["prefill_metrics"]
    dec = report["decode_metrics"]
    w(f"- source: **{md['source']}** | model_type: **{md['model_type']}** | "
      f"family: **{md['model_family']}**")
    w(f"- layers_extracted: **{md['num_layers_extracted']}** | hidden: "
      f"**{md['hidden_size']}** | vocab: **{md['vocab_size']}**")
    w(f"- prefill_allclose: **{pre['allclose']}** | decode token_match_rate: "
      f"**{dec['token_match_rate']}**")
    w(f"- security_status: **{md['security_status']}**")
    w()
    w(f"> {report['required_statement']}")
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
    for k in ("embedding_mask_max_abs_error", "per_layer_max_abs_error",
              "per_layer_handoff_max_abs_error", "final_hidden_max_abs_error",
              "masked_logits_max_abs_error", "recovered_logits_max_abs_error",
              "greedy_token_match_rate"):
        v = pre[k]
        w(f"| `{k}` | {v:.3e} |" if isinstance(v, float) and 0 < v < 1
          else f"| `{k}` | {v} |")
    w(f"| `prefill_allclose` | {pre['allclose']} |")
    w()
    w("## Decode steps")
    w()
    w("| step | pos | tok_match | final_hidden | masked_logits | "
      "recovered_logits | layer_out |")
    w("|---|---|---|---|---|---|---|")
    for s in dec["per_step"]:
        w(f"| {s['step']} | {s['position']} | {s['sampled_token_match']} | "
          f"{s['final_hidden_error']:.3e} | {s['masked_logits_error']:.3e} | "
          f"{s['recovered_logits_error']:.3e} | "
          f"{s['per_layer_output_error']:.3e} |")
    w()
    w("## Boundary metadata")
    w()
    for k in ("input_ids_visible_to_gpu", "plaintext_embedding_visible_to_gpu",
              "plaintext_logits_visible_to_gpu", "masked_logits_visible_to_gpu",
              "logits_recovered_in_tee", "sampling_boundary",
              "local_files_only", "no_network_download", "no_gpu_required",
              "handoff_skip_term_needs_gemm", "semantic_security_claimed",
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
    ap.add_argument("--model-family", default="llama",
                    choices=["llama", "qwen2", "qwen"])
    ap.add_argument("--local-model-path", default=None)
    ap.add_argument("--output",
                    default="outputs/hf_causal_lm_skeleton_probe_llama.json")
    ap.add_argument("--max-layers", type=int, default=2)
    ap.add_argument("--prefill-seq-len", type=int, default=4)
    ap.add_argument("--decode-steps", type=int, default=2)
    ap.add_argument("--max-vocab-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2033)
    ap.add_argument("--use-input-pad", action="store_true")
    args = ap.parse_args()

    cfg = HFCausalLMSkeletonProbeConfig(
        model_family=args.model_family, local_model_path=args.local_model_path,
        prefill_seq_len=args.prefill_seq_len, decode_steps=args.decode_steps,
        max_layers=args.max_layers, max_vocab_size=args.max_vocab_size,
        seed=args.seed, use_input_pad=args.use_input_pad)
    report = run_hf_causal_lm_skeleton_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    status = report["status"]
    if status == "ok":
        print(f"status={status} "
              f"prefill_allclose={report['prefill_metrics']['allclose']} "
              f"token_match_rate={report['decode_metrics']['token_match_rate']}")
    else:
        print(f"status={status} reason={report.get('reason', '')}")


if __name__ == "__main__":
    main()
