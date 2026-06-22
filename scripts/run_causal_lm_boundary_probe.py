"""Runner for Stage 6.7 -- masked CausalLM boundary probe (no network)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.causal_lm_boundary_probe import (  # noqa: E402
    CausalLMBoundaryProbeConfig,
    run_causal_lm_boundary_probe,
)


def _render_markdown(report: dict) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    m = report["metrics"]
    md = report["metadata"]
    w("# Stage 6.7 -- Masked CausalLM Boundary Probe")
    w()
    w(f"- Stage: **{report['stage']}** | Status: **{report['status']}** | "
      f"all_allclose: **{report['all_allclose']}**")
    w(f"- security_status: **{md['security_status']}**")
    w(f"- input_ids_visible_to_gpu: **{md['input_ids_visible_to_gpu']}** | "
      f"plaintext_logits_visible_to_gpu: "
      f"**{md['plaintext_logits_visible_to_gpu']}**")
    w(f"- sampling_boundary: **{md['sampling_boundary']}** | "
      f"vocab_mask_family: **{md['vocab_mask_family']}**")
    w()
    w(f"> {report['statement']}")
    w()
    w("## Config")
    w()
    for kk, vv in report["config"].items():
        w(f"- {kk}: `{vv}`")
    w()
    w("## Input boundary")
    w()
    w("| metric | value |")
    w("|---|---|")
    w(f"| `embedding_mask_max_abs_error` | "
      f"{m['embedding_mask_max_abs_error']:.3e} |")
    w(f"| `next_embedding_mask_max_abs_error` | "
      f"{m['next_embedding_mask_max_abs_error']:.3e} |")
    w()
    w("## Masked-logits output boundary + recovery")
    w()
    w("| metric | value |")
    w("|---|---|")
    for k in ("final_norm_core_max_abs_error", "masked_logits_max_abs_error",
              "recovered_logits_max_abs_error"):
        w(f"| `{k}` | {m[k]:.3e} |")
    w(f"| `logits_recovered_allclose` | {m['logits_recovered_allclose']} |")
    w()
    w("## Sampling")
    w()
    w("| metric | value |")
    w("|---|---|")
    w(f"| `greedy_token_match_rate` | {m['greedy_token_match_rate']} |")
    w(f"| `trusted_greedy_from_masked_match_rate` | "
      f"{m['trusted_greedy_from_masked_match_rate']} |")
    w(f"| `sampled_tokens_shape_ok` | {m['sampled_tokens_shape_ok']} |")
    w(f"| `seeded_sampling_deterministic` | "
      f"{m['seeded_sampling_deterministic']} |")
    w()
    w("## Security metadata")
    w()
    for k in ("no_intermediate_tee", "input_ids_visible_to_gpu",
              "plaintext_embedding_visible_to_gpu",
              "plaintext_logits_visible_to_gpu", "masked_logits_visible_to_gpu",
              "logits_recovered_in_tee", "next_token_ids_visible_to_gpu",
              "dense_vocab_mask_used", "semantic_security_claimed",
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
    ap.add_argument("--output", default="outputs/causal_lm_boundary_probe.json")
    ap.add_argument("--seed", type=int, default=2030)
    args = ap.parse_args()

    cfg = CausalLMBoundaryProbeConfig(seed=args.seed)
    report = run_causal_lm_boundary_probe(cfg)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"status={report['status']} all_allclose={report['all_allclose']}")
    m = report["metrics"]
    print(f"recovered_logits_err={m['recovered_logits_max_abs_error']:.2e} "
          f"greedy_match={m['greedy_token_match_rate']} "
          f"trusted_greedy_match={m['trusted_greedy_from_masked_match_rate']}")


if __name__ == "__main__":
    main()
