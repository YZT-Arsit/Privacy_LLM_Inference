"""Runner for Stage 6.4 -- RoPE-compatible masked GQA/MHA attention probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.rope_gqa_probe import (  # noqa: E402
    RopeGQAProbeConfig,
    run_rope_gqa_probe,
)


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    cfg = report["config"]
    w("# Stage 6.4.1 -- RoPE-Compatible Masked GQA/MHA Attention Probe")
    w()
    w(f"- Stage: **{report['stage']}**")
    w(f"- Status: **{report['status']}** | all_allclose: "
      f"**{report['all_allclose']}**")
    w(f"- Security status: **{report['security_status']}**")
    w(f"- no_intermediate_tee: **{report['no_intermediate_tee']}**")
    w("- Synthetic tensor-level probe; CPU float64; no HF model loading.")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Config")
    w()
    for kk, vv in cfg.items():
        w(f"- {kk}: `{vv}`")
    w()

    def case(name: str, c: dict) -> None:
        w(f"### {name} (num_heads={c['num_heads']}, "
          f"num_key_value_heads={c['num_key_value_heads']}, "
          f"head_dim={c['head_dim']})")
        w()
        w("| metric | max_abs_error |")
        w("|---|---|")
        for key in (
            "rope_commutation_max_error", "score_max_abs_error",
            "prob_max_abs_error", "v_aggregation_max_abs_error",
            "output_max_abs_error", "prefill_cache_key_max_abs_error",
            "prefill_cache_value_max_abs_error", "rope_commutation_q_error",
            "cache_append_key_max_abs_error", "cache_append_value_max_abs_error",
        ):
            w(f"| `{key}` | {c[key]:.3e} |")
        w(f"| **allclose** | **{c['allclose']}** |")
        w()
        w("Decode steps:")
        w()
        w("| step | position | output | key | value |")
        w("|---|---|---|---|---|")
        for s in c["decode_steps"]:
            w(f"| {s['step']} | {s['position']} | "
              f"{s['output_max_abs_error']:.3e} | "
              f"{s['key_max_abs_error']:.3e} | "
              f"{s['value_max_abs_error']:.3e} |")
        w()

    family_titles = {
        "pairwise_rotation": "pairwise_rotation (correctness baseline)",
        "pairwise_complex_scaling":
            "pairwise_complex_scaling (preferred RoPE-compatible mask)",
    }
    w("## Correctness")
    w()
    for family, cases in report["correctness"].items():
        w(f"## {family_titles.get(family, family)}")
        w()
        case("MHA case", cases["mha"])
        case("GQA case", cases["gqa"])

    leakage = report.get("leakage_proxy")
    if leakage:
        w("## Leakage proxy (NOT a security proof)")
        w()
        w("| mode | same-session pair-norm corr | cross-session pair-norm "
          "corr | NN matching acc (pair-norm) |")
        w("|---|---|---|---|")
        for mode in ("no_mask", "pairwise_rotation",
                     "pairwise_complex_scaling"):
            m = leakage[mode]
            w(f"| `{mode}` | "
              f"{m['pair_norm_correlation_same_session']:.4f} | "
              f"{m['cross_session_pair_norm_correlation']:.4f} | "
              f"{m['nearest_neighbor_matching_accuracy_pair_norm']:.4f} |")
        w()
        w(f"- num_samples: `{leakage['num_samples']}`, "
          f"feature: `{leakage['feature']}`, "
          f"leakage_proxy_is_not_security_proof: "
          f"`{leakage['leakage_proxy_is_not_security_proof']}`")
        w("- preserved structure: "
          + "; ".join(leakage["preserved_structure"]))
        w()

    w("## Mask structure")
    w()
    for kk, vv in report["mask_structure"].items():
        w(f"- `{kk}`: {vv}")
    w()
    w("## Leakage caveats")
    w()
    for cav in report["metadata"]["leakage_caveats"]:
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
    ap.add_argument("--output", default="outputs/rope_gqa_probe.json")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    cfg = RopeGQAProbeConfig(seed=args.seed)
    report = run_rope_gqa_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']} all_allclose={report['all_allclose']}")
    print(f"MHA output_err={report['mha']['output_max_abs_error']:.2e} "
          f"GQA output_err={report['gqa']['output_max_abs_error']:.2e}")


if __name__ == "__main__":
    main()
