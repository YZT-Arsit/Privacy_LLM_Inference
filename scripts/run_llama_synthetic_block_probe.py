"""Runner for Stage 6.5 -- LLaMA/Qwen-like synthetic decoder block probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.llama_synthetic_block_probe import (  # noqa: E402
    LlamaSyntheticBlockProbeConfig,
    run_llama_synthetic_block_probe,
)

_PREFILL_KEYS = [
    "rms1_core_max_abs_error",
    "q_mask_max_abs_error", "k_mask_max_abs_error", "v_mask_max_abs_error",
    "attention_score_max_abs_error", "attention_prob_max_abs_error",
    "attention_av_max_abs_error", "attention_output_max_abs_error",
    "residual1_max_abs_error", "rms2_core_max_abs_error",
    "swiglu_gate_max_abs_error", "swiglu_up_max_abs_error",
    "swiglu_hidden_max_abs_error", "mlp_output_max_abs_error",
    "final_output_max_abs_error",
    "prefill_cache_key_max_abs_error", "prefill_cache_value_max_abs_error",
]


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# Stage 6.5 -- LLaMA/Qwen-like Synthetic Decoder Block Probe")
    w()
    w(f"- Stage: **{report['stage']}**")
    w(f"- Status: **{report['status']}** | all_allclose: "
      f"**{report['all_allclose']}**")
    md = report["metadata"]
    w(f"- Security status: **{md['security_status']}**")
    w(f"- no_intermediate_tee: **{md['no_intermediate_tee']}** | "
      f"no_hf_dependency: **{md['no_hf_dependency']}**")
    w("- Synthetic tensor-level block; CPU float64; no HF model loading.")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Config")
    w()
    for kk, vv in report["config"].items():
        w(f"- {kk}: `{vv}`")
    w()
    w("## Modes")
    w()
    for kk in ("residual_mask_family", "rmsnorm_mode", "attention_mode",
               "mlp_mode", "mask_family", "selector_lifted_swiglu_default"):
        w(f"- `{kk}`: {md[kk]}")
    w()

    def case(name: str, c: dict) -> None:
        w(f"## {name} (num_heads={c['num_heads']}, "
          f"num_key_value_heads={c['num_key_value_heads']}, "
          f"head_dim={c['head_dim']}, allclose={c['allclose']})")
        w()
        w("### Prefill")
        w()
        w("| metric | max_abs_error |")
        w("|---|---|")
        for key in _PREFILL_KEYS:
            w(f"| `{key}` | {c['prefill_metrics'][key]:.3e} |")
        w(f"| **prefill_allclose** | **{c['prefill_allclose']}** |")
        w()
        w("### Decode steps")
        w()
        w("| step | position | output | cache key | cache value |")
        w("|---|---|---|---|---|")
        for s in c["decode_step_metrics"]:
            w(f"| {s['step']} | {s['position']} | "
              f"{s['output_max_abs_error']:.3e} | "
              f"{s['cache_append_key_max_abs_error']:.3e} | "
              f"{s['cache_append_value_max_abs_error']:.3e} |")
        w()

    case("GQA case", report["gqa"])
    case("MHA case", report["mha"])

    w("## Caveats")
    w()
    for cav in md["caveats"]:
        w(f"- {cav}")
    w()
    w("## Limitations")
    w()
    for lim in report["limitations"]:
        w(f"- {lim}")
    w()
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/llama_synthetic_block_probe.json")
    ap.add_argument("--seed", type=int, default=2028)
    args = ap.parse_args()

    cfg = LlamaSyntheticBlockProbeConfig(seed=args.seed)
    report = run_llama_synthetic_block_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']} all_allclose={report['all_allclose']}")
    print(f"GQA final_output_err="
          f"{report['gqa']['prefill_metrics']['final_output_max_abs_error']:.2e} "
          f"MHA final_output_err="
          f"{report['mha']['prefill_metrics']['final_output_max_abs_error']:.2e}")


if __name__ == "__main__":
    main()
