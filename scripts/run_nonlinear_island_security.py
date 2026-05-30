#!/usr/bin/env python
"""Stage 5.2 — Nonlinear island security proxy experiments + emitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    NonlinearIslandSecurityConfig,
    run_nonlinear_island_security_experiments,
)
from pllo.experiments.report_utils import (
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sessions", type=int, default=8)
    parser.add_argument("--num-samples-per-session", type=int, default=32)
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--pad-scale", type=float, default=1.0)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


CSV_FIELDS = ("section", "key", "metric", "value")


def _flatten(payload: dict) -> list[dict]:
    rows: list[dict] = []
    # Proxy 1
    for name, m in payload["permutation_recovery_proxy"]["per_strategy"].items():
        for metric, value in m.items():
            rows.append(
                {
                    "section": "permutation_recovery_proxy",
                    "key": name,
                    "metric": metric,
                    "value": value,
                }
            )
    # Proxy 2
    for name, m in payload["island_linkability_proxy"]["per_strategy"].items():
        for metric, value in m.items():
            rows.append(
                {
                    "section": "island_linkability_proxy",
                    "key": name,
                    "metric": metric,
                    "value": value,
                }
            )
    # Proxy 3
    for entry in payload["mask_family_accounting"]["table"]:
        for metric in ("where_used", "preserved_invariants", "leakage_note"):
            rows.append(
                {
                    "section": "mask_family_accounting",
                    "key": entry["mask_family"],
                    "metric": metric,
                    "value": entry[metric],
                }
            )
    return rows


def _build_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append(
        "# Privacy LLM Obfuscation — Nonlinear Island Security Proxies (Stage 5.2)"
    )
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Three lightweight proxies on the operator-compatible mask scheme used"
        " by the Stage 5.2 nonlinear islands. None of these proves formal"
        " security — they are naive-observer bounds, recorded so the paper"
        " can quote them directly under the security section."
    )
    out.append("")

    # 1. Permutation recovery
    out.append("## Permutation Recovery Proxy")
    out.append("")
    out.append(payload["permutation_recovery_proxy"]["interpretation"])
    out.append("")
    headers = [
        "strategy",
        "top-1 recovery",
        "top-5 recovery",
        "mean signature error",
    ]
    rows = []
    for name, m in payload["permutation_recovery_proxy"]["per_strategy"].items():
        rows.append(
            [
                name,
                m["permutation_recovery_top1"],
                m["permutation_recovery_top5"],
                m["mean_channel_signature_error"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    chance = 1.0 / payload["permutation_recovery_proxy"]["hidden_size"]
    out.append(
        f"- Random-chance top-1 baseline at hidden_size="
        f"{payload['permutation_recovery_proxy']['hidden_size']}: ``≈ {chance:.4f}``."
    )
    out.append("")

    # 2. Island linkability
    out.append("## Island Linkability Proxy")
    out.append("")
    out.append(payload["island_linkability_proxy"]["interpretation"])
    out.append("")
    headers = [
        "strategy",
        "mean pair-cos",
        "mean pair-L2",
    ]
    rows = []
    for name in payload["island_linkability_proxy"]["linkability_rank_high_to_low"]:
        m = payload["island_linkability_proxy"]["per_strategy"][name]
        rows.append(
            [
                name,
                m["mean_pairwise_cosine"],
                m["mean_pairwise_l2"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    out.append(
        f"- Linkability rank (high → low):"
        f" {' > '.join(payload['island_linkability_proxy']['linkability_rank_high_to_low'])}"
    )
    out.append("")

    # 3. Mask family accounting
    out.append("## Mask Family Security Accounting")
    out.append("")
    headers = ["mask family", "where used", "preserved invariants", "leakage note"]
    rows = []
    for entry in payload["mask_family_accounting"]["table"]:
        rows.append(
            [
                entry["mask_family"],
                entry["where_used"],
                entry["preserved_invariants"],
                entry["leakage_note"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    for note in payload["mask_family_accounting"]["summary_notes"]:
        out.append(f"- {note}")
    out.append("")

    # 4. Interpretation
    out.append("## Interpretation")
    out.append("")
    p = payload["permutation_recovery_proxy"]["per_strategy"]
    out.append(
        f"- Fixed permutation top-1 = **{p['fixed_permutation']['permutation_recovery_top1']:.3f}** vs"
        f" fresh = **{p['fresh_permutation_per_session']['permutation_recovery_top1']:.3f}** vs"
        f" dense sandwich = **{p['dense_sandwich_reference']['permutation_recovery_top1']:.3f}**."
        " Fresh permutation per session removes the cross-session signature"
        " alignment; the dense sandwich erases the column statistics entirely."
    )
    l = payload["island_linkability_proxy"]["per_strategy"]
    out.append(
        f"- Island linkability: `fixed_perm_no_pad` mean cosine = "
        f"**{l['fixed_perm_no_pad']['mean_pairwise_cosine']:.4f}** vs"
        f" `fresh_perm_with_linear_boundary_pad` = "
        f"**{l['fresh_perm_with_linear_boundary_pad']['mean_pairwise_cosine']:.4f}**."
        " Linear-boundary pad collapses naive linkability."
    )
    out.append(
        "- Mask family accounting (Proxy 3) catalogues the per-family"
        " leakage profile — *what is preserved by design*, not what an"
        " adversary can or cannot recover beyond that."
    )
    out.append("")

    # 5. Limitations
    out.append("## Limitations")
    out.append("")
    for lim in payload["global_limitations"]:
        out.append(f"- {lim}")
    out.append("")

    # 6. Next stage plan
    out.append("## Next Stage Plan")
    out.append("")
    out.append(
        "- **Stage 5.3** — Adaptive / learned-inverter attacker that uses"
        " more than per-channel statistics (e.g. low-rank reconstruction,"
        " supervised inverter trained against a known model)."
    )
    out.append(
        "- **Stage 6.4** — Qwen / TinyLlama migration. The nonlinear-island"
        " proxies here motivate the freshness + dense-sandwich rules that"
        " the Qwen wrapper will be required to enforce."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = NonlinearIslandSecurityConfig(
        output_dir=str(args.output_dir),
        num_sessions=args.num_sessions,
        num_samples_per_session=args.num_samples_per_session,
        num_trials=args.num_trials,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pad_scale=args.pad_scale,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    payload = run_nonlinear_island_security_experiments(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "nonlinear_island_security.json", payload)
    write_csv(out_dir / "nonlinear_island_security.csv", _flatten(payload), CSV_FIELDS)
    write_text(
        out_dir / "nonlinear_island_security.md", _build_markdown(payload)
    )

    p = payload["permutation_recovery_proxy"]["per_strategy"]
    l = payload["island_linkability_proxy"]["per_strategy"]
    print(
        f"perm_recovery fixed_top1={p['fixed_permutation']['permutation_recovery_top1']:.3f}"
        f" fresh_top1={p['fresh_permutation_per_session']['permutation_recovery_top1']:.3f}"
        f" sandwich_top1={p['dense_sandwich_reference']['permutation_recovery_top1']:.3f}"
        f" fixed_no_pad_cos={l['fixed_perm_no_pad']['mean_pairwise_cosine']:.4f}"
        f" fresh_pad_cos={l['fresh_perm_with_linear_boundary_pad']['mean_pairwise_cosine']:.4f}"
        f" output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
