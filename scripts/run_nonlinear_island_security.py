#!/usr/bin/env python
"""Stage 5.2b — Nonlinear island security proxy experiments + emitter."""

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
    run_nonlinear_island_security,
)
from pllo.experiments.report_utils import (
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sessions", type=int, default=16)
    parser.add_argument("--samples-per-session", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--pad-scale", type=float, default=1.0)
    parser.add_argument("--permutation-pool-size", type=int, default=4)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


# Long-format CSV per spec: section,strategy,metric,value,notes
CSV_FIELDS = ("section", "strategy", "metric", "value", "notes")


def _flatten_for_csv(payload: dict) -> list[dict]:
    rows: list[dict] = []

    # Proxy 1 — permutation recovery
    for strategy, m in payload["permutation_recovery"]["per_strategy"].items():
        notes = m["expected_risk_level"]
        for metric in (
            "top1_recovery_rate",
            "top5_recovery_rate",
            "mean_correct_rank",
            "mean_signature_error",
        ):
            rows.append(
                {
                    "section": "permutation_recovery",
                    "strategy": strategy,
                    "metric": metric,
                    "value": m[metric],
                    "notes": notes,
                }
            )
    rows.append(
        {
            "section": "permutation_recovery",
            "strategy": "_meta",
            "metric": "random_chance_top1",
            "value": payload["permutation_recovery"]["random_chance_top1"],
            "notes": "1 / hidden_size",
        }
    )

    # Proxy 2 — island linkability
    for strategy, m in payload["island_linkability"]["per_strategy"].items():
        notes = m["expected_linkability"]
        if m["view"] == "activation_input_visible":
            # Single-view strategy: emit its top-level pairwise metrics.
            for metric in (
                "mean_pairwise_cosine",
                "max_pairwise_cosine",
                "min_pairwise_cosine",
                "mean_pairwise_l2",
                "max_pairwise_l2",
                "min_pairwise_l2",
            ):
                rows.append(
                    {
                        "section": "island_linkability",
                        "strategy": strategy,
                        "metric": metric,
                        "value": m[metric],
                        "notes": notes,
                    }
                )
        else:
            # Dual-view strategies: emit per-view metrics with the view name
            # prefixed onto the metric.
            for view_key, view in m.items():
                if not isinstance(view, dict):
                    continue
                if "mean_pairwise_cosine" not in view:
                    continue
                for metric in (
                    "mean_pairwise_cosine",
                    "max_pairwise_cosine",
                    "min_pairwise_cosine",
                    "mean_pairwise_l2",
                    "max_pairwise_l2",
                    "min_pairwise_l2",
                ):
                    rows.append(
                        {
                            "section": "island_linkability",
                            "strategy": strategy,
                            "metric": f"{view_key}.{metric}",
                            "value": view[metric],
                            "notes": notes,
                        }
                    )

    # Proxy 3 — mask family accounting
    for entry in payload["mask_family_accounting"]["table"]:
        for field in (
            "used_for",
            "correctness_role",
            "preserved_statistics",
            "gpu_visible_leakage",
            "mitigation",
            "security_strength_relative_to_dense",
            "notes",
        ):
            rows.append(
                {
                    "section": "mask_family_accounting",
                    "strategy": entry["mask_family"],
                    "metric": field,
                    "value": entry[field],
                    "notes": "",
                }
            )

    return rows


def _build_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append(
        "# Privacy LLM Obfuscation — Nonlinear Island Security Proxies (Stage 5.2b)"
    )
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Three lightweight proxies over the operator-compatible mask scheme"
        " used by the Stage 5.2a nonlinear islands. None of these constitute"
        " a formal security proof — each is a naive-observer upper bound,"
        " recorded so the paper's security section can quote it directly."
    )
    out.append("")

    # 2. Threat model
    out.append("## Threat Model for Proxy Experiments")
    out.append("")
    out.append(payload["threat_model"])
    out.append("")

    # 3. Permutation recovery
    out.append("## Permutation Recovery Proxy")
    out.append("")
    out.append(
        "Channel signature ``(mean, std, median, q25, q75, mean_abs)`` over a"
        " synthetic activation distribution with per-channel mean offset and"
        " scale, matched to the plaintext reference signature by greedy"
        " cosine nearest-neighbour. Random-chance top-1 baseline is"
        f" ``1 / hidden_size = {payload['permutation_recovery']['random_chance_top1']:.4f}``."
    )
    out.append("")
    headers = [
        "strategy",
        "top-1",
        "top-5",
        "mean rank",
        "sig error",
        "risk level",
    ]
    rows = []
    for name in payload["permutation_recovery"]["ranking_by_top1_descending"]:
        m = payload["permutation_recovery"]["per_strategy"][name]
        rows.append(
            [
                name,
                m["top1_recovery_rate"],
                m["top5_recovery_rate"],
                m["mean_correct_rank"],
                m["mean_signature_error"],
                m["expected_risk_level"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    out.append(
        f"- Recovery ranking (high → low):"
        f" {' > '.join(payload['permutation_recovery']['ranking_by_top1_descending'])}"
    )
    out.append("")

    # 4. Island linkability
    out.append("## Island Linkability Proxy")
    out.append("")
    out.append(
        "Same plaintext input is run repeatedly through each strategy; the"
        " pairwise cosine and L2 distance of the GPU-visible tensor across"
        " trials is reported below. Dual-view strategies record both"
        " ``boundary_input_visible`` (post-pad-and-mask Linear input) and"
        " ``activation_input_visible`` (``Z P``, no pad) — see notes."
    )
    out.append("")
    headers = ["strategy", "view", "mean cos", "mean L2", "expected linkability"]
    rows = []
    for name, m in payload["island_linkability"]["per_strategy"].items():
        if m["view"] == "activation_input_visible":
            rows.append(
                [
                    name,
                    m["view"],
                    m["mean_pairwise_cosine"],
                    m["mean_pairwise_l2"],
                    m["expected_linkability"],
                ]
            )
        else:
            for view_key, view in m.items():
                if not isinstance(view, dict) or "mean_pairwise_cosine" not in view:
                    continue
                rows.append(
                    [
                        name,
                        view_key,
                        view["mean_pairwise_cosine"],
                        view["mean_pairwise_l2"],
                        m["expected_linkability"],
                    ]
                )
    out.append(markdown_table(headers, rows))
    out.append("")
    out.append(
        f"- Main-metric linkability ranking (high → low):"
        f" {' > '.join(payload['island_linkability']['linkability_rank_high_to_low'])}"
    )
    for note in payload["island_linkability"]["notes"]:
        out.append(f"- {note}")
    out.append("")

    # 5. Mask family accounting
    out.append("## Mask Family Security Accounting")
    out.append("")
    headers = [
        "mask family",
        "used for",
        "preserved statistics",
        "gpu-visible leakage",
        "mitigation",
        "strength vs dense",
    ]
    rows = []
    for entry in payload["mask_family_accounting"]["table"]:
        rows.append(
            [
                entry["mask_family"],
                entry["used_for"],
                entry["preserved_statistics"],
                entry["gpu_visible_leakage"],
                entry["mitigation"],
                entry["security_strength_relative_to_dense"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    for note in payload["mask_family_accounting"]["summary_notes"]:
        out.append(f"- {note}")
    out.append("")

    # 6. Interpretation
    out.append("## Interpretation")
    out.append("")
    p = payload["permutation_recovery"]["per_strategy"]
    out.append(
        f"- Permutation recovery: fixed top-1 = **{p['fixed_permutation']['top1_recovery_rate']:.3f}**"
        f" vs fresh top-1 = **{p['fresh_permutation_per_session']['top1_recovery_rate']:.3f}**"
        f" vs sandwich top-1 = **{p['dense_sandwich_reference']['top1_recovery_rate']:.3f}**"
        f" (random chance = **{payload['permutation_recovery']['random_chance_top1']:.4f}**)."
        " Fixed permutation enables cross-session signature accumulation; dense"
        " sandwich destroys per-channel statistics."
    )
    l = payload["island_linkability"]["main_metric_per_strategy"]["values"]
    out.append(
        f"- Island linkability: `fixed_perm_no_pad` mean cos ="
        f" **{l['fixed_perm_no_pad']:.4f}** vs"
        f" `fresh_perm_with_linear_boundary_pad` ="
        f" **{l['fresh_perm_with_linear_boundary_pad']:.4f}** vs"
        f" `dense_to_perm_to_dense_sandwich` ="
        f" **{l['dense_to_perm_to_dense_sandwich']:.4f}**."
        " Note: under `fixed_perm_with_linear_boundary_pad`, the boundary view"
        " is well-protected (fresh pad / mask), but the activation view"
        " ``Z P`` remains fully linkable because P is fixed — pad at the Linear"
        " boundary does NOT protect the activation island for a fixed permutation."
    )
    out.append(
        "- Mask family accounting (Proxy 3) catalogues per-family preserved"
        " invariants and leakage — *what is preserved by design*, not what"
        " an adversary can or cannot recover beyond that."
    )
    out.append("")

    # 7. Limitations
    out.append("## Limitations")
    out.append("")
    for lim in payload["limitations"]:
        out.append(f"- {lim}")
    out.append("")

    # 8. Next stage plan
    out.append("## Next Stage Plan")
    out.append("")
    out.append(
        "- **Stage 5.2c** — Workload profiler integration: extend the Stage"
        " 5.0.1 cost model to count the trusted-side norm / activation"
        " operations replaced by the Stage 5.2a islands, and compare the"
        " three architectures' boundary-call counts under"
        " ``ours_with_islands`` vs ``ours_current``."
    )
    out.append(
        "- **Stage 5.3** — Wrapper selective integration: replace the trusted"
        " LayerNorm / GELU shortcut with the Stage 5.2a islands in the GPT-2"
        " / BERT / T5 wrappers behind a feature flag, gated on the Stage 5.2b"
        " linkability + recovery results documented above."
    )
    out.append(
        "- **Stage 5.4** — Adaptive / learned-inverter attacker that goes"
        " beyond per-channel statistics (the Stage 5.2b proxies are naive-"
        "observer bounds only)."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = NonlinearIslandSecurityConfig(
        output_dir=str(args.output_dir),
        num_sessions=args.num_sessions,
        samples_per_session=args.samples_per_session,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        pad_scale=args.pad_scale,
        permutation_pool_size=args.permutation_pool_size,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    payload = run_nonlinear_island_security(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "nonlinear_island_security.json", payload)
    write_csv(
        out_dir / "nonlinear_island_security.csv",
        _flatten_for_csv(payload),
        CSV_FIELDS,
    )
    write_text(
        out_dir / "nonlinear_island_security.md", _build_markdown(payload)
    )

    g = payload["global_summary"]
    print(
        f"fixed_top1={g['fixed_perm_recovery_top1']:.3f}"
        f" fresh_top1={g['fresh_perm_recovery_top1']:.3f}"
        f" sandwich_top1={g['sandwich_perm_recovery_top1']:.3f}"
        f" random={g['random_chance_top1']:.4f}"
        f" fixed_no_pad_cos={g['fixed_perm_no_pad_linkability_cos']:.4f}"
        f" fresh_pad_cos={g['fresh_perm_with_pad_linkability_cos']:.4f}"
        f" sandwich_cos={g['dense_sandwich_linkability_cos']:.4f}"
        f" output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
