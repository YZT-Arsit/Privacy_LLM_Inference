"""Runner for Stage 6.6 -- HF LLaMA/Qwen single-decoder-layer probe.

Local-only: never downloads. With no ``--local-model-path`` it uses a
randomly-initialised tiny HF layer (if transformers is available).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.hf_single_block_probe import (  # noqa: E402
    HFSingleBlockProbeConfig,
    run_hf_single_block_probe,
)

_PREFILL_KEYS = [
    "rms1_core_max_abs_error",
    "q_mask_max_abs_error", "k_mask_max_abs_error", "v_mask_max_abs_error",
    "attention_score_max_abs_error", "attention_prob_max_abs_error",
    "attention_output_max_abs_error", "residual1_max_abs_error",
    "rms2_core_max_abs_error", "gate_max_abs_error", "up_max_abs_error",
    "swiglu_hidden_max_abs_error", "mlp_output_max_abs_error",
    "final_output_max_abs_error", "cache_key_max_abs_error",
    "cache_value_max_abs_error",
]


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Stage 6.6 -- HF LLaMA/Qwen Single-Decoder-Layer Probe")
    w()
    w(f"- Stage: **{report['stage']}** | Status: **{report['status']}**")
    if report["status"] != "ok":
        w(f"- Reason: `{report.get('reason')}`")
        w()
        w(f"> {report['statement']}")
        w()
        return "\n".join(lines) + "\n"
    w(f"- model_family: **{report['model_family']}** | model_type: "
      f"**{report['model_type']}** | source: **{report['source']}**")
    w(f"- all_allclose: **{report['allclose']}** | "
      f"no_network_download: **{report['no_network_download']}** | "
      f"local_files_only: **{report['local_files_only']}**")
    md = report["metadata"]
    w(f"- security_status: **{md['security_status']}** | "
      f"semantic_security_claimed: **{md['semantic_security_claimed']}**")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Block config (inferred from layer)")
    w()
    for kk, vv in report["block_config"].items():
        w(f"- {kk}: `{vv}`")
    w()
    w("## Prefill metrics")
    w()
    w("| metric | max_abs_error |")
    w("|---|---|")
    for key in _PREFILL_KEYS:
        w(f"| `{key}` | {report['prefill_metrics'][key]:.3e} |")
    w(f"| **prefill_allclose** | **{report['prefill_allclose']}** |")
    w()
    w("## Decode steps")
    w()
    w("| step | position | output | cache key | cache value |")
    w("|---|---|---|---|---|")
    for s in report["decode_step_metrics"]:
        w(f"| {s['step']} | {s['position']} | "
          f"{s['output_max_abs_error']:.3e} | "
          f"{s['cache_append_key_max_abs_error']:.3e} | "
          f"{s['cache_append_value_max_abs_error']:.3e} |")
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
    ap.add_argument("--layer-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2029)
    ap.add_argument("--output", default="outputs/hf_single_block_probe.json")
    args = ap.parse_args()

    cfg = HFSingleBlockProbeConfig(
        model_family=args.model_family,
        local_model_path=args.local_model_path,
        layer_index=args.layer_index,
        seed=args.seed,
    )
    report = run_hf_single_block_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']}")
    if report["status"] == "ok":
        print(f"family={report['model_family']} source={report['source']} "
              f"allclose={report['allclose']} "
              f"final_out_err="
              f"{report['prefill_metrics']['final_output_max_abs_error']:.2e}")
    else:
        print(f"reason={report.get('reason')}")


if __name__ == "__main__":
    main()
