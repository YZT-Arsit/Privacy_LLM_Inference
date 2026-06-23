"""Stage 8.2 -- top-conference experiment aggregator.

Reads the Group A--G compact JSON outputs under ``outputs/`` and emits a single
summary (JSON / MD / CSV) organised into the 9 reviewer-facing sections. It
NEVER reruns experiments, downloads anything, or dumps tensors. Missing files
are reported as missing, not fatal. Includes an explicit claim-discipline
section (allowed vs disallowed claims).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any

# Static claim discipline (kept in sync with the experiment design).
ALLOWED_CLAIMS = [
    "real ModelScope Qwen checkpoints (no Hugging Face remote download)",
    "real tokenized prompt inputs (not only synthetic/random tokens)",
    "masked runtime path verified by audit flags and negative controls",
    "float32 / mixed-safe (bf16 load, fp32 fold/recover/compare) correctness "
    "under the tested settings",
    "explicit leakage accounting (token recovery, masked-logit alignment, "
    "hidden-state structure)",
]
DISALLOWED_CLAIMS = [
    "production-ready TEE (this is a simulated TEE only)",
    "semantic security",
    "hiding sequence length",
    "hiding attention scores / probabilities",
    "full 3B/7B end-to-end generation unless actually run at full layers",
    "pure-bf16 exact correctness (mixed-safe fp32 recovery is required)",
]


def _load(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _glob1(outdir: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(outdir, pattern)))


def _probe_row(r: dict[str, Any], src: str) -> dict[str, Any]:
    a = r.get("audit", {}) or {}
    mr = r.get("masked_runtime", {}) or {}
    lm = r.get("latency_memory", {}) or {}
    return {
        "source": os.path.basename(src),
        "model_id": r.get("model_id"),
        "layers": f"{a.get('max_layers_executed')}/"
                  f"{a.get('num_hidden_layers_total')}",
        "input_source": a.get("input_source"),
        "prompt_count": a.get("prompt_count"),
        "seq": a.get("prefill_seq_len"),
        "decode": a.get("decode_steps"),
        "dtype": a.get("dtype"),
        "token_match_rate": mr.get("token_match_rate_vs_extracted"),
        "recovered_logits_err": mr.get("recovered_logits_max_abs_error"),
        "negative_control": r.get("negative_control"),
        "expected_to_match": r.get("expected_to_match"),
        "negative_control_passed": r.get("negative_control_passed"),
        "masked_latency_ms": lm.get("masked_runtime_latency_ms"),
        "extracted_latency_ms": lm.get("extracted_plaintext_latency_ms"),
        "hf_latency_ms": lm.get("hf_baseline_latency_ms"),
        "slowdown_masked_vs_extracted": lm.get("slowdown_masked_vs_extracted"),
        "peak_mem_mb_masked": (lm.get("peak_cuda_memory_mb", {}) or {})
            .get("masked_runtime"),
    }


def _section_probes(outdir: str) -> dict[str, Any]:
    """All probe-style reports (latency / audit / full-layer / 3B / 7B)."""
    patterns = ["eval_latency_*.json", "eval_full_layer_*.json",
                "eval_realprompts_3b_*.json", "eval_7b_smoke_*.json",
                "audit_*.json"]
    rows = []
    for pat in patterns:
        for f in _glob1(outdir, pat):
            r = _load(f)
            if r and r.get("stage", "").startswith("8.2_modelscope"):
                rows.append(_probe_row(r, f))
    return {"present": bool(rows), "rows": rows}


def _section_boundary(outdir: str) -> dict[str, Any]:
    rows = []
    for f in _glob1(outdir, "eval_output_boundary_ablation_*.json"):
        r = _load(f)
        if not r or "boundary_ablation" not in r:
            continue
        for row in r["boundary_ablation"]:
            rows.append({"source": os.path.basename(f),
                         "model_id": r.get("model_id"), **row})
    return {"present": bool(rows), "rows": rows}


def _section_attacks(outdir: str) -> dict[str, Any]:
    out: dict[str, Any] = {"present": False}
    for key, pat in (("token_recovery", "eval_attack_token_recovery_*.json"),
                     ("masked_logits", "eval_attack_masked_logits_*.json"),
                     ("hidden_structure",
                      "eval_hidden_structure_leakage_*.json")):
        rows = []
        for f in _glob1(outdir, pat):
            r = _load(f)
            block = (r or {}).get(
                {"token_recovery": "attack_token_recovery",
                 "masked_logits": "attack_masked_logits",
                 "hidden_structure": "hidden_structure_leakage"}[key])
            if block:
                rows.append({"source": os.path.basename(f),
                             "model_id": r.get("model_id"), **block})
        out[key] = rows
        if rows:
            out["present"] = True
    return out


def _section_batch(outdir: str) -> dict[str, Any]:
    rows = []
    for f in _glob1(outdir, "eval_batch_scaling_*.json"):
        r = _load(f)
        for row in (r or {}).get("rows", []):
            rows.append({"source": os.path.basename(f),
                         "model_id": r.get("model_id"), **row})
    return {"present": bool(rows), "rows": rows}


def aggregate(outdir: str) -> dict[str, Any]:
    probes = _section_probes(outdir)
    boundary = _section_boundary(outdir)
    attacks = _section_attacks(outdir)
    batch = _section_batch(outdir)

    def _pick(pred) -> list[dict[str, Any]]:
        return [row for row in probes["rows"] if pred(row)]

    full_layer = _pick(lambda r: "full_layer" in (r["source"] or "")
                        or (r["layers"] and r["layers"].split("/")[0]
                            == r["layers"].split("/")[-1]))
    big = _pick(lambda r: ("3b" in (r["source"] or "").lower()
                           or "7b" in (r["source"] or "").lower()))
    neg = _pick(lambda r: r["negative_control"]
                and r["negative_control"] != "none")
    normal = _pick(lambda r: r["negative_control"] in (None, "none"))

    return {
        "1_real_checkpoint_correctness": {
            "present": probes["present"],
            "rows": [{k: r[k] for k in ("source", "model_id", "layers",
                                        "token_match_rate",
                                        "recovered_logits_err", "dtype")}
                     for r in normal]},
        "2_real_prompt_audit_and_negative_controls": {
            "present": bool(normal or neg),
            "normal_runs": [{k: r[k] for k in (
                "source", "model_id", "input_source", "prompt_count",
                "token_match_rate", "expected_to_match",
                "negative_control_passed")} for r in normal],
            "negative_controls": [{k: r[k] for k in (
                "source", "negative_control", "token_match_rate",
                "recovered_logits_err", "expected_to_match",
                "negative_control_passed")} for r in neg]},
        "3_latency_memory_baseline": {
            "present": probes["present"],
            "rows": [{k: r[k] for k in (
                "source", "model_id", "layers", "hf_latency_ms",
                "extracted_latency_ms", "masked_latency_ms",
                "slowdown_masked_vs_extracted", "peak_mem_mb_masked")}
                for r in probes["rows"]]},
        "4_output_boundary_ablation": boundary,
        "5_leakage_attack_metrics": attacks,
        "6_batch_scaling": batch,
        "7_full_layer_0_5b": {"present": bool(full_layer), "rows": full_layer},
        "8_3b_7b_scalability": {"present": bool(big), "rows": big},
        "9_limitations_and_disallowed_claims": {
            "allowed_claims": ALLOWED_CLAIMS,
            "disallowed_claims": DISALLOWED_CLAIMS},
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    out.append("")
    return out


def render_md(agg: dict[str, Any]) -> str:
    L: list[str] = ["# Stage 8.2 — Top-Conference Experiment Summary", ""]

    L.append("## 1. Real checkpoint correctness")
    L += _md_table(agg["1_real_checkpoint_correctness"]["rows"],
                   ["model_id", "layers", "token_match_rate",
                    "recovered_logits_err", "dtype"])

    s2 = agg["2_real_prompt_audit_and_negative_controls"]
    L.append("## 2. Real-prompt audit & negative controls")
    L.append("**Normal runs**")
    L += _md_table(s2["normal_runs"], ["model_id", "input_source",
                                       "prompt_count", "token_match_rate",
                                       "negative_control_passed"])
    L.append("**Negative controls (mismatch = passing)**")
    L += _md_table(s2["negative_controls"], ["negative_control",
                                             "token_match_rate",
                                             "recovered_logits_err",
                                             "negative_control_passed"])

    L.append("## 3. Latency / memory baseline")
    L += _md_table(agg["3_latency_memory_baseline"]["rows"],
                   ["model_id", "layers", "extracted_latency_ms",
                    "masked_latency_ms", "slowdown_masked_vs_extracted",
                    "peak_mem_mb_masked"])

    L.append("## 4. Output boundary ablation")
    L += _md_table(agg["4_output_boundary_ablation"]["rows"],
                   ["model_id", "boundary_mode", "gpu_visible",
                    "tee_compute_flops", "transfer_bytes", "token_match_rate",
                    "recovered_logits_err", "latency_ms", "latency_kind"])

    L.append("## 5. Leakage / attack metrics")
    a = agg["5_leakage_attack_metrics"]
    L.append("**D1 token recovery from embeddings**")
    L += _md_table(a.get("token_recovery", []),
                   ["model_id", "vocab_size", "plaintext_top1_token_recovery",
                    "masked_top1_token_recovery", "random_baseline_top1"])
    L.append("**D2 masked-logit alignment**")
    L += _md_table(a.get("masked_logits", []),
                   ["model_id", "gpu_visible_argmax_matches_plaintext",
                    "top5_overlap_plain_vs_masked_visible",
                    "rank_correlation_plain_vs_masked_visible",
                    "recovered_argmax_matches_plaintext"])
    L.append("**D3 hidden-state structural leakage**")
    L += _md_table(a.get("hidden_structure", []),
                   ["model_id", "norm_preservation_ratio_mean",
                    "pairwise_distance_correlation",
                    "pairwise_cosine_correlation",
                    "nearest_neighbor_identity_preservation"])

    L.append("## 6. Batch scaling")
    L += _md_table(agg["6_batch_scaling"]["rows"],
                   ["model_id", "batch_size", "token_match_rate", "latency_ms",
                    "peak_cuda_memory_mb", "tokens_per_second"])

    L.append("## 7. Full-layer 0.5B")
    L += _md_table(agg["7_full_layer_0_5b"]["rows"],
                   ["model_id", "layers", "token_match_rate",
                    "recovered_logits_err"])

    L.append("## 8. 3B / 7B scalability")
    L += _md_table(agg["8_3b_7b_scalability"]["rows"],
                   ["model_id", "layers", "token_match_rate",
                    "recovered_logits_err", "masked_latency_ms"])

    L.append("## 9. Limitations & disallowed claims")
    L.append("**Allowed:**")
    L += [f"- {c}" for c in ALLOWED_CLAIMS]
    L.append("")
    L.append("**Disallowed (must NOT claim):**")
    L += [f"- {c}" for c in DISALLOWED_CLAIMS]
    L.append("")
    return "\n".join(L) + "\n"


def render_csv(agg: dict[str, Any], path: str) -> None:
    rows: list[dict[str, Any]] = []
    for r in agg["3_latency_memory_baseline"]["rows"]:
        rows.append({"section": "latency", **r})
    for r in agg["4_output_boundary_ablation"]["rows"]:
        rows.append({"section": "boundary", "model_id": r.get("model_id"),
                     "boundary_mode": r.get("boundary_mode"),
                     "tee_compute_flops": r.get("tee_compute_flops"),
                     "transfer_bytes": r.get("transfer_bytes"),
                     "token_match_rate": r.get("token_match_rate"),
                     "recovered_logits_err": r.get("recovered_logits_err")})
    for r in agg["6_batch_scaling"]["rows"]:
        rows.append({"section": "batch", "model_id": r.get("model_id"),
                     "batch_size": r.get("batch_size"),
                     "token_match_rate": r.get("token_match_rate"),
                     "latency_ms": r.get("latency_ms"),
                     "peak_mem_mb_masked": r.get("peak_cuda_memory_mb")})
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--json-out",
                    default="outputs/topconf_experiment_summary.json")
    ap.add_argument("--md-out", default="outputs/topconf_experiment_summary.md")
    ap.add_argument("--csv-out", default="outputs/topconf_experiment_tables.csv")
    args = ap.parse_args()

    agg = aggregate(args.output_dir)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, default=str)
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write(render_md(agg))
    render_csv(agg, args.csv_out)

    present = [k for k, v in agg.items()
               if isinstance(v, dict) and v.get("present")]
    print(f"Wrote: {args.json_out}")
    print(f"Wrote: {args.md_out}")
    print(f"Wrote: {args.csv_out}")
    print("sections with data:", ", ".join(present) or "(none yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
