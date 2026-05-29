#!/usr/bin/env python
"""Stage 6.3 — Security proxy experiments (pad linkability, mask freshness,
boundary leakage accounting, cache leakage proxy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    SecurityProxyConfig,
    run_security_proxy_experiments,
)
from pllo.experiments.report_utils import (
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--pad-scale", type=float, default=1.0)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


CSV_FIELDS = (
    "section",
    "key",
    "metric",
    "value",
)


def _flatten_for_csv(payload: dict) -> list[dict]:
    rows: list[dict] = []

    # Pad linkability
    for name, m in payload["pad_linkability_proxy"]["per_strategy"].items():
        for metric_name in (
            "mean_pairwise_cosine",
            "max_pairwise_cosine",
            "min_pairwise_cosine",
            "mean_pairwise_l2",
            "max_pairwise_l2",
            "min_pairwise_l2",
        ):
            rows.append(
                {
                    "section": "pad_linkability_proxy",
                    "key": name,
                    "metric": metric_name,
                    "value": m[metric_name],
                }
            )

    # Mask freshness
    for m in payload["mask_freshness_audit"]["per_mask"]:
        for metric_name in (
            "num_generated",
            "num_unique_fingerprints",
            "unexpected_reuse_count",
            "condition_number_mean",
            "condition_number_max",
            "condition_number_min",
        ):
            rows.append(
                {
                    "section": "mask_freshness_audit",
                    "key": m["mask_name"],
                    "metric": metric_name,
                    "value": m[metric_name],
                }
            )

    # Boundary leakage accounting (one row per item, metric=visibility)
    for item in payload["boundary_leakage_accounting"]["gpu_visible"]:
        rows.append(
            {
                "section": "boundary_leakage_accounting",
                "key": item["name"],
                "metric": "visibility",
                "value": item["visibility"],
            }
        )
    for item in payload["boundary_leakage_accounting"]["trusted_only"]:
        rows.append(
            {
                "section": "boundary_leakage_accounting",
                "key": item["name"],
                "metric": "visibility",
                "value": item["visibility"],
            }
        )

    # Cache leakage proxy
    cache = payload["cache_leakage_proxy"]
    for cache_kind, sub in cache.items():
        if not isinstance(sub, dict):
            continue
        for matching_kind in (
            "plain_to_plain_baseline",
            "obfuscated_to_plain",
            "obfuscated_to_plain_v",
        ):
            matching = sub.get(matching_kind)
            if not matching:
                continue
            for metric_name in (
                "top1_match_rate",
                "mean_correct_rank",
                "mean_cosine_correct_pair",
                "mean_cosine_best_wrong_pair",
            ):
                rows.append(
                    {
                        "section": f"cache_leakage_proxy/{cache_kind}",
                        "key": matching_kind,
                        "metric": metric_name,
                        "value": matching[metric_name],
                    }
                )
    return rows


def _build_markdown(payload: dict) -> str:
    out: list[str] = []
    out.append("# Privacy LLM Obfuscation — Security Proxy Experiments (Stage 6.3)")
    out.append("")

    out.append("## Experiment scope")
    out.append("")
    out.append(
        "Four lightweight security proxies over the mask + pad scheme used in"
        " Stages 1–6.2. None of these constitutes a formal security proof;"
        " each is a naive-observer upper bound."
    )
    out.append("")

    # ---- 1. Pad linkability ----
    out.append("## Pad linkability proxy")
    out.append("")
    out.append(payload["pad_linkability_proxy"]["summary_note"])
    out.append("")
    headers = [
        "strategy",
        "mean pair-cos",
        "max pair-cos",
        "min pair-cos",
        "mean pair-L2",
        "interpretation",
    ]
    rows = []
    for name in payload["pad_linkability_proxy"]["ranking_by_mean_cosine_descending"]:
        m = payload["pad_linkability_proxy"]["per_strategy"][name]
        rows.append(
            [
                name,
                m["mean_pairwise_cosine"],
                m["max_pairwise_cosine"],
                m["min_pairwise_cosine"],
                m["mean_pairwise_l2"],
                m["interpretation"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")
    out.append(
        f"- Ranking (high → low linkability):"
        f" {' > '.join(payload['pad_linkability_proxy']['ranking_by_mean_cosine_descending'])}"
    )
    out.append("")

    # ---- 2. Mask freshness audit ----
    out.append("## Mask freshness audit")
    out.append("")
    out.append(payload["mask_freshness_audit"]["summary_note"])
    out.append("")
    headers = [
        "mask",
        "expected policy",
        "generated",
        "unique fingerprints",
        "unexpected reuse",
        "cond mean",
        "cond max",
        "cond min",
    ]
    rows = []
    for m in payload["mask_freshness_audit"]["per_mask"]:
        rows.append(
            [
                m["mask_name"],
                m["expected_policy"],
                m["num_generated"],
                m["num_unique_fingerprints"],
                m["unexpected_reuse_count"],
                m["condition_number_mean"],
                m["condition_number_max"],
                m["condition_number_min"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    # ---- 3. Boundary leakage accounting ----
    out.append("## Boundary leakage accounting")
    out.append("")
    for note in payload["boundary_leakage_accounting"]["summary_notes"]:
        out.append(f"- {note}")
    out.append("")

    out.append("### GPU-visible tensors")
    out.append("")
    headers = ["name", "contains plaintext", "architecture scope", "leakage note"]
    rows = [
        [it["name"], it["contains_plaintext"], it["architecture_scope"], it["leakage_note"]]
        for it in payload["boundary_leakage_accounting"]["gpu_visible"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("### Trusted-only tensors")
    out.append("")
    rows = [
        [it["name"], it["contains_plaintext"], it["architecture_scope"], it["leakage_note"]]
        for it in payload["boundary_leakage_accounting"]["trusted_only"]
    ]
    out.append(markdown_table(headers, rows))
    out.append("")

    # ---- 4. Cache leakage proxy ----
    out.append("## Cache leakage proxy")
    out.append("")
    out.append(payload["cache_leakage_proxy"]["interpretation_note"])
    out.append("")
    headers = [
        "cache kind",
        "matching",
        "top1 match rate",
        "mean correct rank",
        "cos correct pair",
        "cos best wrong pair",
        "queries",
    ]
    rows = []
    for cache_kind in ("kv_cache", "encoder_memory_cache"):
        sub = payload["cache_leakage_proxy"][cache_kind]
        for matching_kind, matching in sub.items():
            if not isinstance(matching, dict):
                continue
            if "top1_match_rate" not in matching:
                continue
            rows.append(
                [
                    cache_kind,
                    matching_kind,
                    matching["top1_match_rate"],
                    matching["mean_correct_rank"],
                    matching["mean_cosine_correct_pair"],
                    matching["mean_cosine_best_wrong_pair"],
                    matching["num_queries"],
                ]
            )
    out.append(markdown_table(headers, rows))
    out.append("")

    # ---- 5. Interpretation ----
    out.append("## Interpretation")
    out.append("")
    pl = payload["pad_linkability_proxy"]["per_strategy"]
    fixed_no = pl["fixed_mask_no_pad"]["mean_pairwise_cosine"]
    fresh_fresh = pl["fresh_mask_fresh_pad"]["mean_pairwise_cosine"]
    out.append(
        f"- `fixed_mask_no_pad` mean pairwise cosine: **{fixed_no:.4f}** vs"
        f" `fresh_mask_fresh_pad`: **{fresh_fresh:.4f}** — fresh mask + fresh"
        " pad collapses the naive linkability signal toward zero."
    )
    cache = payload["cache_leakage_proxy"]
    kv_plain = cache["kv_cache"]["plain_to_plain_baseline"]["top1_match_rate"]
    kv_obf = cache["kv_cache"]["obfuscated_to_plain"]["top1_match_rate"]
    enc_plain = cache["encoder_memory_cache"]["plain_to_plain_baseline"]["top1_match_rate"]
    enc_obf = cache["encoder_memory_cache"]["obfuscated_to_plain"]["top1_match_rate"]
    out.append(
        f"- KV cache: plain↔plain top1={kv_plain:.4f}; obf↔plain top1={kv_obf:.4f}."
        " Naive cosine-matching cannot recover plaintext KV from K_tilde / V_tilde."
    )
    out.append(
        f"- Encoder memory cache: plain↔plain top1={enc_plain:.4f};"
        f" obf↔plain top1={enc_obf:.4f}. Same naive bound applies."
    )
    out.append("")

    # ---- 6. Limitations ----
    out.append("## Limitations")
    out.append("")
    for lim in payload["global_limitations"]:
        out.append(f"- {lim}")
    out.append("")

    # ---- 7. Next stage plan ----
    out.append("## Next stage plan")
    out.append("")
    out.append(
        "- **Stage 5.1** — GPU-side LayerNorm primitive (replaces a"
        " trusted-side leakage point counted under `trusted_only` above)."
    )
    out.append(
        "- **Stage 5.2** — GELU / activation primitive feasibility."
    )
    out.append(
        "- **Stage 5.3** — stronger leakage experiments (adaptive observer,"
        " learned inverter); current experiments here are only naive-observer"
        " proxies."
    )
    out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = SecurityProxyConfig(
        output_dir=str(args.output_dir),
        num_trials=args.num_trials,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        pad_scale=args.pad_scale,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    payload = run_security_proxy_experiments(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "security_proxy_experiments.json", payload)
    write_csv(
        out_dir / "security_proxy_experiments.csv",
        _flatten_for_csv(payload),
        CSV_FIELDS,
    )
    write_text(
        out_dir / "security_proxy_experiments.md", _build_markdown(payload)
    )
    pl = payload["pad_linkability_proxy"]["per_strategy"]
    cache = payload["cache_leakage_proxy"]
    print(
        f"num_trials={cfg.num_trials} hidden={cfg.hidden_size} "
        f"fixed_mask_no_pad_cos={pl['fixed_mask_no_pad']['mean_pairwise_cosine']:.4f} "
        f"fresh_mask_fresh_pad_cos={pl['fresh_mask_fresh_pad']['mean_pairwise_cosine']:.4f} "
        f"kv_obf_to_plain_top1={cache['kv_cache']['obfuscated_to_plain']['top1_match_rate']:.4f} "
        f"enc_obf_to_plain_top1={cache['encoder_memory_cache']['obfuscated_to_plain']['top1_match_rate']:.4f} "
        f"output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
