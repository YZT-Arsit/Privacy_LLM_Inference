#!/usr/bin/env python
"""Stage 5.6 — Stronger attackers runner.

Drives :func:`pllo.experiments.stronger_attackers.run_stronger_attackers`
and emits ``outputs/stronger_attackers.{json,csv,md}``.

Default behaviour: synthetic-token + synthetic-model fallback; pytest
never reads from the network. Real tokenizer / real model loading is
opt-in via ``--attempt-tokenizer-load`` + ``--attempt-real-model-load``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.stronger_attackers import (  # noqa: E402
    StrongerAttackersConfig,
    run_stronger_attackers,
)
from pllo.ops.mitigation_bundles import VALID_MITIGATION_BUNDLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--prompt-max-length", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--attacker-trials", type=int, default=32)
    p.add_argument("--timing-noise-std", type=float, default=0.05)
    p.add_argument("--attempt-real-model-load", action="store_true")
    p.add_argument("--attempt-tokenizer-load", action="store_true")
    p.add_argument("--model-id", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--max-layers", type=int, default=2)
    p.add_argument(
        "--inter-block-mask-mode",
        choices=("plain_boundary", "masked_boundary_experimental"),
        default="plain_boundary",
    )
    p.add_argument(
        "--constant-time-decode-mode",
        choices=("off", "proxy_equalized"),
        default="off",
        help=(
            "Stage 5.6 extension. proxy_equalized pads simulated per-step"
            " latency to a per-method upper bound; PROXY ONLY — does not"
            " sleep or change real wall-time."
        ),
    )
    p.add_argument(
        "--mitigation-bundle",
        choices=list(VALID_MITIGATION_BUNDLES),
        default="fresh_perm_plus_sandwich_plus_pad",
    )
    p.add_argument("--use-pad", action="store_true", default=True)
    p.add_argument("--no-use-pad", dest="use_pad", action="store_false")
    p.add_argument("--synthetic-vocab-size", type=int, default=64)
    p.add_argument("--synthetic-hidden-size", type=int, default=32)
    p.add_argument("--synthetic-intermediate-size", type=int, default=64)
    p.add_argument("--synthetic-num-query-heads", type=int, default=4)
    p.add_argument("--synthetic-num-kv-heads", type=int, default=2)
    p.add_argument("--synthetic-head-dim", type=int, default=8)
    p.add_argument(
        "--stage-5-5b-artifact",
        default="outputs/real_token_activation_attacks.json",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV emitter (long format)
# ---------------------------------------------------------------------------


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ml = report["model_loading"]
    for k in (
        "load_status", "resolved_model_id", "model_family", "fallback_used",
    ):
        rows.append({
            "section": "model_loading", "attack": "n/a", "scope": "n/a",
            "metric": k, "value": ml.get(k), "notes": "",
        })
    tl = report.get("tokenizer_loading", {})
    for k in ("tokenizer_status", "tokenizer_id", "tokenizer_error"):
        rows.append({
            "section": "tokenizer_loading", "attack": "n/a", "scope": "n/a",
            "metric": k, "value": tl.get(k), "notes": "",
        })

    bb = report["blackbox_attacker"]
    for k, v in bb["prompt_linkability"].items():
        rows.append({
            "section": "blackbox", "attack": "prompt_linkability",
            "scope": "generation", "metric": k, "value": v, "notes": "",
        })
    for k, v in bb["prompt_class_inference"].items():
        rows.append({
            "section": "blackbox", "attack": "prompt_class_inference",
            "scope": "generation", "metric": k, "value": v, "notes": "",
        })
    for k, v in bb["mitigation_mode_distinguishability"].items():
        rows.append({
            "section": "blackbox", "attack": "mode_distinguishability",
            "scope": "generation", "metric": k, "value": v, "notes": "",
        })

    tm = report["timing_sidechannel_proxy"]
    for sub in (
        "prompt_length_leakage", "decode_step_leakage",
        "method_distinguishability", "mitigation_distinguishability",
    ):
        for k, v in tm[sub].items():
            rows.append({
                "section": "timing", "attack": sub,
                "scope": "prefill_decode", "metric": k, "value": v, "notes": "",
            })
    for entry in tm["boundary_call_pattern"]:
        rows.append({
            "section": "timing", "attack": "boundary_call_pattern",
            "scope": entry["method"], "metric": "per_forward_boundary_calls",
            "value": entry["per_forward_boundary_calls"],
            "notes": entry["boundary_call_formula"],
        })

    ib = report["inter_block_masking_gap"]
    rows.append({
        "section": "inter_block", "attack": "gap_analysis",
        "scope": "model_level", "metric": "current_plain_boundary_detected",
        "value": int(bool(ib["current_plain_boundary_detected"])), "notes": "",
    })
    for t in ib["affected_tensors"]:
        rows.append({
            "section": "inter_block", "attack": "gap_analysis",
            "scope": "model_level", "metric": "affected_tensor",
            "value": t, "notes": "inter_block_plain_recovered",
        })
    rows.append({
        "section": "inter_block", "attack": "gap_analysis",
        "scope": "model_level", "metric": "accounting_risk_level",
        "value": ib["accounting_risk_level"], "notes": "",
    })
    for k, v in ib["single_transition_probe"].items():
        if k == "note":
            continue
        rows.append({
            "section": "inter_block", "attack": "single_transition",
            "scope": "synthetic", "metric": k, "value": v, "notes": "",
        })
    rows.append({
        "section": "inter_block", "attack": "single_transition",
        "scope": "synthetic", "metric": "probe_status",
        "value": ib["single_transition_probe_status"], "notes": "",
    })
    rows.append({
        "section": "inter_block", "attack": "experimental",
        "scope": "model_level",
        "metric": "masked_boundary_experimental_status",
        "value": ib["masked_boundary_experimental_status"], "notes": "",
    })

    ibc = report.get("inter_block_closure_summary") or {}
    if ibc:
        for k in (
            "status", "masked_boundary_experimental_status",
        ):
            rows.append({
                "section": "inter_block_closure", "attack": "summary",
                "scope": "model_level", "metric": k, "value": ibc.get(k),
                "notes": "",
            })
        for tensor_key in ("boundary_input", "final"):
            for phase, suffix in (("before", "_before"), ("after", "_after")):
                d = ibc.get(tensor_key + suffix) or {}
                for k, v in d.items():
                    rows.append({
                        "section": "inter_block_closure",
                        "attack": f"{tensor_key}_{phase}",
                        "scope": "prefill", "metric": k, "value": v,
                        "notes": "",
                    })
    ct = report.get("constant_time_decode_summary") or {}
    if ct:
        for k, v in ct.items():
            rows.append({
                "section": "constant_time_decode", "attack": "summary",
                "scope": "decode_step", "metric": k, "value": v, "notes": "",
            })
    for k, v in report["overall_risk_summary"].items():
        rows.append({
            "section": "overall_risk_summary", "attack": "summary",
            "scope": "n/a", "metric": k, "value": v, "notes": "",
        })
    for k, v in report["recommendation"].items():
        rows.append({
            "section": "recommendation", "attack": "summary",
            "scope": "n/a", "metric": k, "value": v, "notes": "",
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["section", "attack", "scope", "metric", "value", "notes"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    bb = report["blackbox_attacker"]
    tm = report["timing_sidechannel_proxy"]
    ib = report["inter_block_masking_gap"]
    overall = report["overall_risk_summary"]
    rec = report["recommendation"]
    cmp = report["comparison_with_prior_stages"]

    lines.append("# Stronger Attackers (Stage 5.6)\n")

    lines.append("## Experiment Scope\n")
    lines.append(
        "Stage 5.6 ships three proxy attackers that DO NOT require paired"
        " plaintext / visible internal supervision:"
        " (1) a **black-box query attacker** that sees only the generated"
        " token sequence + per-step logits summaries — no internal traces;"
        " (2) a **model-based timing / boundary-call side-channel proxy**"
        " driven by Stage 5.2c op-count formulas + the Stage 5.2c cost"
        " model + Gaussian noise — NOT a real TEE wall-time measurement;"
        " and (3) an **inter-block residual masking gap analysis** that"
        " quantifies the Stage 6.4c structural plain-boundary issue"
        " surfaced by Stage 5.5b and runs a single-transition math probe"
        " showing the orthogonal-mask fix is numerically correct."
    )
    lines.append("")

    lines.append("## Threat Model\n")
    lines.append(
        "- **Black-box attacker** sees: generated tokens, per-step top-1 /"
        " top-5 token IDs, entropy, max logit, top-1 margin, output length."
        " Does NOT see internal hidden states, masks, permutations, KV"
        " cache contents, or any `per_layer_traces`."
    )
    lines.append(
        "- **Timing attacker** sees: simulated per-forward latency under"
        " a Stage 5.2c cost model. Does NOT see real TEE wall-time, real"
        " GPU utilisation, real CPU/cache counters, or hardware-level"
        " side channels."
    )
    lines.append(
        "- **Inter-block analyst** reads the Stage 5.5b artifact and"
        " verifies the structural plain-boundary finding for the Stage"
        " 6.4c model wrapper. Does NOT execute any new wrapper code."
    )
    lines.append(
        "- All attackers are PROXY attackers, not formal security proofs."
    )
    lines.append("")

    lines.append("## Black-Box Query Attacker\n")
    lines.append("**Prompt linkability** (same vs different prompt via signatures):")
    lines.append("")
    lines.append("| metric | value |\n|---|---|")
    for k, v in bb["prompt_linkability"].items():
        lines.append(f"| {k} | {_fmt(v, 4)} |")
    lines.append("")
    lines.append(
        "> Same-prompt similarity is trivially 1.0 because greedy"
        " generation is deterministic — that's a property of greedy"
        " decoding, NOT a finding against the obfuscation envelope."
    )
    lines.append("")
    lines.append("**Prompt class inference** (4-way nearest-neighbour classifier):")
    lines.append("")
    lines.append("| metric | value |\n|---|---|")
    for k, v in bb["prompt_class_inference"].items():
        lines.append(f"| {k} | {_fmt(v, 4)} |")
    lines.append("")
    lines.append("**Mitigation-mode distinguishability** (nonlinear_mode × bundle × use_pad):")
    lines.append("")
    lines.append("| metric | value |\n|---|---|")
    for k, v in bb["mitigation_mode_distinguishability"].items():
        lines.append(f"| {k} | {_fmt(v, 4)} |")
    lines.append("")
    lines.append(
        "> If `mode_classification_accuracy` ≈ `random_chance_baseline`,"
        " an API attacker cannot tell from the output which mitigation"
        " configuration produced it. Stage 6.4c verified that obfuscated"
        " and plain greedy outputs are byte-identical, so this metric is"
        " bounded at random chance by construction."
    )
    lines.append("")

    lines.append("## Timing / Boundary-Call Side-Channel Proxy\n")
    lines.append(
        "Simulated latency = Stage 5.2c per-forward op-counts plugged into"
        " the Stage 5.2c cost model (`gpu_flops_per_ms`,"
        " `tee_to_gpu_flops_ratio`, `tee_call_overhead_ms`,"
        " `tee_bytes_per_ms`) + Gaussian noise with"
        f" std = {report['config']['timing_noise_std']} × |mean|."
    )
    lines.append("")
    lines.append(
        "**This is NOT a real TEE wall-time measurement**; the wider"
        " profile keeps `wall_time_source = projected_from_op_counts` and"
        " `implemented = False`."
    )
    lines.append("")
    lines.append(
        "| sub-attack | accuracy | random_chance | correlation | risk |"
    )
    lines.append("|---|---|---|---|---|")
    for sub, label in (
        ("prompt_length_leakage", "Prompt length"),
        ("decode_step_leakage", "Decode step"),
        ("method_distinguishability", "Method"),
        ("mitigation_distinguishability", "Mitigation bundle"),
    ):
        d = tm[sub]
        acc = d.get("length_bucket_accuracy") or d.get("step_accuracy") \
            or d.get("method_accuracy") or d.get("mitigation_accuracy")
        rc = d["random_chance_baseline"]
        corr = d.get("correlation_latency_length") \
            or d.get("correlation_latency_step")
        lines.append(
            f"| {label} | {_fmt(acc, 3)} | {_fmt(rc, 3)}"
            f" | {_fmt(corr if corr is not None else 'n/a', 3)}"
            f" | {d['risk_level']} |"
        )
    lines.append("")
    lines.append("**Boundary-call pattern (static structural leakage):**")
    lines.append("")
    lines.append("| method | per_forward_boundary_calls | formula |")
    lines.append("|---|---|---|")
    for e in tm["boundary_call_pattern"]:
        lines.append(
            f"| {e['method']} | {e['per_forward_boundary_calls']}"
            f" | {e['boundary_call_formula']} |"
        )
    lines.append("")

    lines.append("## Inter-Block Residual Masking Gap\n")
    lines.append("| field | value |\n|---|---|")
    lines.append(
        f"| current_plain_boundary_detected | "
        f"{ib['current_plain_boundary_detected']} |"
    )
    lines.append(
        f"| affected_tensors | {', '.join(ib['affected_tensors'])} |"
    )
    lines.append(
        f"| accounting_risk_level | {ib['accounting_risk_level']} |"
    )
    lines.append(
        f"| single_transition_probe_status | {ib['single_transition_probe_status']} |"
    )
    lines.append(
        "| masked_boundary_experimental_status |"
        f" {ib['masked_boundary_experimental_status']} |"
    )
    lines.append(
        "| masked_boundary_experimental_default |"
        f" {ib['masked_boundary_experimental_default']} |"
    )
    lines.append(
        f"| overall_inter_block_risk_level | "
        f"{ib['overall_inter_block_risk_level']} |"
    )
    lines.append("")

    lines.append("## Single-Transition Masking Probe\n")
    lines.append(
        "Math-only verification that an orthogonal inter-block mask"
        " `N_inter` is absorbed by the next block's RMSNorm + folded Q/K/V"
        " projection — no plain inter-block transcript required."
    )
    lines.append("")
    lines.append("| metric | value |\n|---|---|")
    for k, v in ib["single_transition_probe"].items():
        if k == "note":
            continue
        lines.append(f"| {k} | {_fmt(v, 6)} |")
    lines.append("")
    lines.append(
        f"_Note_: {ib['single_transition_probe'].get('note', '')}"
    )
    lines.append("")

    # Stage 5.6 extension — inter-block closure and constant-time summaries.
    ibc = report.get("inter_block_closure_summary") or {}
    if ibc:
        lines.append("## Inter-Block Masking Mode\n")
        lines.append(
            "| field | value |\n|---|---|"
        )
        lines.append(
            f"| status | {ibc.get('status')} |"
        )
        lines.append(
            "| masked_boundary_experimental_status |"
            f" {ibc.get('masked_boundary_experimental_status')} |"
        )
        lines.append("")
        lines.append("### Plain Boundary vs Masked Boundary Experimental\n")
        lines.append(
            "| tensor | mode | risk | inter_block_plain | linear_rel_l2 | linkability_cosine |"
        )
        lines.append("|---|---|---|---|---|---|")
        for tensor_key, label in (
            ("boundary_input", "boundary_input"),
            ("final", "final"),
        ):
            before = ibc.get(f"{tensor_key}_before") or {}
            after = ibc.get(f"{tensor_key}_after") or {}
            if before.get("present"):
                lines.append(
                    f"| {label} | plain_boundary | {before['risk_level']}"
                    f" | {before['inter_block_plain']}"
                    f" | {_fmt(before['linear_rel_l2'], 4)}"
                    f" | {_fmt(before['linkability_cosine'], 4)} |"
                )
            if after.get("present"):
                lines.append(
                    f"| {label} | masked_boundary_experimental | {after['risk_level']}"
                    f" | {after['inter_block_plain']}"
                    f" | {_fmt(after['linear_rel_l2'], 4)}"
                    f" | {_fmt(after['linkability_cosine'], 4)} |"
                )
        lines.append("")
        lines.append("### Boundary Input / Final Risk Before and After\n")
        lines.append(
            f"_Note_: {ibc.get('note', '')}"
        )
        lines.append("")
    ct = report.get("constant_time_decode_summary") or {}
    if ct:
        lines.append("## Constant-Time Decode Proxy\n")
        lines.append("| field | value |\n|---|---|")
        for k in (
            "mode", "decode_step_accuracy_before", "decode_step_accuracy_after",
            "correlation_latency_step_before", "correlation_latency_step_after",
            "risk_level_before", "risk_level_after", "overhead_ms_estimate",
        ):
            lines.append(f"| {k} | {_fmt(ct.get(k), 4)} |")
        lines.append("")
        lines.append("### Decode-Step Timing Leakage Before and After\n")
        lines.append(
            f"_Limitation_: {ct.get('limitation', '')}"
        )
        lines.append("")
        lines.append("### Overhead Proxy\n")
        lines.append(
            f"Mean per-step latency padding ≈ {_fmt(ct.get('overhead_ms_estimate', 0.0), 3)}"
            f" ms (simulated). PROXY only — does not change real wall-time."
        )
        lines.append("")

    lines.append("## Comparison with Stage 5.4 / 5.5 / 5.5b\n")
    lines.append(cmp["summary"])
    lines.append("")
    lines.append("| stage | artifact |\n|---|---|")
    for k, v in cmp.items():
        if k == "summary":
            continue
        lines.append(f"| {k} | `{v}` |")
    lines.append("")

    lines.append("## Overall Risk Summary\n")
    lines.append("| dimension | level |\n|---|---|")
    for k in (
        "envelope_integrity_risk_level",
        "envelope_blackbox_risk_level",
        "envelope_timing_risk_level",
        "structural_leakage_risk_level",
        "structural_timing_risk_level",
        "structural_inter_block_risk_level",
        "overall_risk_level",
    ):
        lines.append(f"| {k} | {overall.get(k)} |")
    lines.append("")
    lines.append(
        "> **Envelope-integrity risk** is what the mitigation envelope"
        " is responsible for: can mode / bundle / use_pad be distinguished"
        " from outputs or timing? If `low`, the envelope holds under"
        " black-box + timing proxy attacks."
    )
    lines.append("")
    lines.append(
        "> **Structural-leakage risk** captures known model-wrapper /"
        " transformer properties (latency scales with prompt length and"
        " decode step; the Stage 6.4c model wrapper recovers between"
        " blocks). These are acknowledged limitations, NOT failures of"
        " the mitigation envelope; closing them requires"
        " constant-time computation and inter-block masked residual"
        " (Stage 5.6 extension / Stage 7.0)."
    )
    lines.append("")

    lines.append("## Recommendation\n")
    for k, v in rec.items():
        lines.append(f"- `{k} = \"{v}\"`")
    lines.append("")
    lines.append(
        f"**Promotion eligibility:** {overall['security_profile_detail_with_stronger_attackers_eligibility']}"
    )
    lines.append("")

    lines.append("## Limitations\n")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    # Backstop sentence guaranteeing the required honesty phrases appear.
    lines.append(
        "- _Note_: stronger proxy attacks, not formal security proofs;"
        " timing results are model-based proxies, not real TEE timing"
        " measurements; inter-block residual masking gap remains open"
        " until Stage 5.6 extension / Stage 7.0; not a real TEE measurement;"
        " not formal security."
    )
    lines.append("")

    lines.append("## Next Stage Plan\n")
    lines.append(
        "- Stage 5.6 extension (optional) — implement the full"
        " `masked_boundary_experimental` mode in"
        " ObfuscatedModernDecoderModelWrapper so the inter-block residual"
        " stays masked across layers under an orthogonal N_inter."
    )
    lines.append(
        "- Stage 7.0 (deferred) — LoRA private-training path under the"
        " same obfuscation envelope. Requires Stage 5.6 to first establish"
        " the inference-side security baseline."
    )
    lines.append(
        "- Constant-time mitigations to suppress decode-step latency"
        " leakage (timing) are deliberately deferred — they cost"
        " throughput and require explicit deployment-side opt-in."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tensor-leakage safety check
# ---------------------------------------------------------------------------


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


def _strip_traces(report: dict[str, Any]) -> dict[str, Any]:
    import torch

    def _scrub(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return {
                "_tensor_shape": list(obj.shape),
                "_tensor_redacted": True,
            }
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_scrub(v) for v in obj]
        return obj

    return _scrub(report)


def main() -> None:
    args = parse_args()
    cfg = StrongerAttackersConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        num_prompts=args.num_prompts,
        prompt_max_length=args.prompt_max_length,
        max_new_tokens=args.max_new_tokens,
        attempt_real_model_load=bool(args.attempt_real_model_load),
        attempt_tokenizer_load=bool(args.attempt_tokenizer_load),
        model_id=args.model_id,
        local_files_only=bool(args.local_files_only),
        max_layers=args.max_layers,
        inter_block_mask_mode=args.inter_block_mask_mode,
        constant_time_decode_mode=args.constant_time_decode_mode,
        mitigation_bundle=args.mitigation_bundle,
        use_pad=bool(args.use_pad),
        attacker_trials=args.attacker_trials,
        timing_noise_std=args.timing_noise_std,
        synthetic_vocab_size=args.synthetic_vocab_size,
        synthetic_hidden_size=args.synthetic_hidden_size,
        synthetic_intermediate_size=args.synthetic_intermediate_size,
        synthetic_num_attention_heads=args.synthetic_num_query_heads,
        synthetic_num_key_value_heads=args.synthetic_num_kv_heads,
        synthetic_head_dim=args.synthetic_head_dim,
        stage_5_5b_artifact=args.stage_5_5b_artifact,
    )
    report = run_stronger_attackers(cfg)
    safe_report = _strip_traces(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "stronger_attackers.json"
    csv_path = args.output_dir / "stronger_attackers.csv"
    md_path = args.output_dir / "stronger_attackers.md"

    json_text = json.dumps(safe_report, indent=2)
    assert "tensor(" not in json_text
    assert _LONG_NUMBER_ARRAY.search(json_text) is None
    json_path.write_text(json_text, encoding="utf-8")
    _write_csv(csv_path, _csv_rows(safe_report))
    md_text = _format_markdown(safe_report)
    assert "tensor(" not in md_text
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    o = report["overall_risk_summary"]
    print(
        f"envelope={o['envelope_integrity_risk_level']}"
        f" structural={o['structural_leakage_risk_level']}"
        f" overall={o['overall_risk_level']}"
    )


if __name__ == "__main__":
    main()
