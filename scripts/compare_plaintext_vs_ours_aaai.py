"""AAAI headline comparison: plaintext-GPU vs ours (A_rightmul, TDX + H800).

Consolidates the per-dataset generation reports + the separate eval artifacts
(GSM8K exact match, MT-Bench / IFEval / HumanEval preservation, sandboxed
HumanEval pass@1, sensitive-prompt leakage scan) + the attestation evidence +
worker-health snapshots into one utility / performance / security / paper-readiness
table (``aaai_main_results.json`` / ``.csv`` / ``.md``).

The AAAI comparison is ONLY plaintext-GPU vs ours (A_rightmul + linear boundary
pad + TDX boundary client + H800 folded worker). No LoRA, no amulet_secure_R, no
pure-TEE. Missing artifacts are reported as ``null`` (never silently treated as a
pass). stdlib only; no torch, no network.

Layout discovery (per dataset ``D``): a report is looked up in this order under a
results dir ``R``::

    R/D/report.json   ->   R/D_generation.json   ->   R/**/<D>*generation.json

Example::

    python scripts/compare_plaintext_vs_ours_aaai.py \\
      --plaintext-dir outputs/aaai/qwen/plaintext_local \\
      --ours-dir      outputs/aaai/qwen/folded_remote_unstaged \\
      --datasets ifeval gsm8k mt_bench humaneval sensitive_prompt_1024 \\
      --attestation-evidence-json <EVIDENCE> \\
      --gsm8k-scored outputs/aaai/qwen/folded_remote_unstaged/gsm8k/gsm8k_scored.json \\
      --preservation-dir outputs/aaai/qwen/preservation \\
      --code-eval-json outputs/aaai/qwen/code_eval_humaneval.json \\
      --security-json  outputs/aaai/qwen/sensitive_security_report.json \\
      --output-dir outputs/aaai/qwen/main_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except Exception:                                            # noqa: BLE001
        return None


def _find_report(root, dataset):
    if not root:
        return None
    root = Path(root)
    for cand in (root / dataset / "report.json",
                 root / ("%s_generation.json" % dataset),
                 root / dataset / ("%s_generation.json" % dataset)):
        if cand.exists():
            return _load_json(cand)
    hits = sorted(root.rglob("*%s*generation.json" % dataset))
    return _load_json(hits[0]) if hits else None


def _utility(dataset, plain_rep, ours_rep, *, gsm8k_scored, preservation,
             code_eval):
    out = {"dataset": dataset}
    if dataset == "gsm8k":
        g = gsm8k_scored or {}
        out["gsm8k_exact_match_ours"] = g.get("exact_match_accuracy")
        if preservation and "gsm8k" in preservation:
            pg = preservation["gsm8k"]
            out["gsm8k_exact_match_plaintext"] = pg.get("plaintext_exact_match")
            out["gsm8k_exact_match_delta"] = pg.get("exact_match_delta")
    if preservation:
        agg = (preservation.get("aggregate")
               or (preservation.get("turn1", {}) or {}).get("aggregate"))
        if agg:
            out["exact_response_match_rate"] = agg.get(
                "exact_response_match_rate")
            out["mean_normalized_edit_distance"] = agg.get(
                "mean_normalized_edit_distance")
            out["finish_reason_match_rate"] = agg.get("finish_reason_match_rate")
        if dataset == "mt_bench":
            out["mt_bench_turn2_complete"] = preservation.get("turn2_complete")
    if dataset == "humaneval" and code_eval:
        out["humaneval_pass@1_plaintext"] = code_eval.get("pass@1_plaintext")
        out["humaneval_pass@1_ours"] = code_eval.get("pass@1_ours")
        out["humaneval_pass@1_delta"] = code_eval.get("pass@1_delta")
    if dataset == "ifeval":
        out["ifeval_note"] = ("preservation reported; run the official IFEval "
                              "checker for the strict score")
    return out


def _performance(ours_rep):
    r = ours_rep or {}
    return {
        "ttft_s": r.get("prefill_latency_s") or r.get("time_to_first_token_s"),
        "tpot_s": r.get("latency_per_generated_token_s"),
        "end_to_end_latency_s": (r.get("online_generation_latency_s")
                                 or r.get("latency_s_online_only")),
        "tokens_per_sec": r.get("tokens_per_sec"),
        "generated_tokens": r.get("generated_tokens"),
        "boundary_calls_per_generated_token": r.get(
            "boundary_calls_per_generated_token"),
        "worker_retries_total": r.get("worker_retry_count"),
        "worker_reconnects_total": r.get("worker_reconnects_total"),
        "peak_gpu_memory_gb": (r.get("peak_gpu_memory_gb")
                               or r.get("resident_weight_memory_gb")),
        "failed_examples": r.get("failed_examples"),
    }


def _security(ours_rep, evidence):
    r = ours_rep or {}
    ev = evidence or {}
    quote_valid = bool(ev.get("paper_facing") is True
                       or ev.get("runtime_hash_bound") is True
                       or ev.get("verified") is True)
    return {
        "tdx_quote_valid": quote_valid,
        "runtime_hash_binds_nonlinear_backend": bool(
            ev.get("runtime_hash_binds_nonlinear_backend")
            or r.get("attestation_runtime_hash_binds_nonlinear_backend")),
        "compatible_masks_verified": r.get("compatible_masks_verified"),
        "nonlinear_trusted_calls": r.get("nonlinear_trusted_calls"),
        "nonlinear_trusted_calls_zero": (
            r.get("nonlinear_trusted_calls") == 0
            if r.get("nonlinear_trusted_calls") is not None else None),
        "schedule_full_coverage_verified": r.get(
            "schedule_full_coverage_verified"),
        "raw_prompt_not_in_response_jsonl": (
            r.get("response_jsonl_contains_raw_prompt") is False),
        "h800_worker_tee_used_on_gpu": r.get("h800_worker_tee_used_on_gpu"),
        "worker_has_mask_secrets": r.get("worker_has_mask_secrets"),
        "raw_mask_or_pad_on_gpu": bool(r.get("worker_has_mask_secrets")),
        "gpu_visible_plaintext_fields": r.get("gpu_visible_plaintext_fields"),
    }


def _paper_ready(ours_rep, sec, security_json):
    r = ours_rep or {}
    blockers = []
    if r.get("paper_ready") is not True:
        blockers.append("ours report paper_ready != true")
    if r.get("paper_facing_generation") is False:
        blockers.append("paper_facing_generation contract unmet: %s"
                        % r.get("paper_facing_generation_violations"))
    if r.get("failed_examples"):
        blockers.append("%s failed example(s)" % r.get("failed_examples"))
    if not sec.get("tdx_quote_valid"):
        blockers.append("tdx quote not valid / not attached")
    if sec.get("nonlinear_trusted_calls_zero") is False:
        blockers.append("nonlinear_trusted_calls != 0")
    if security_json is not None and security_json.get("leakage_pass") is False:
        blockers.append("sensitive leakage scan failed")
    return {"paper_ready": (not blockers), "blockers": blockers}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plaintext-dir", required=True)
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--worker-health-jsonl", default=None)
    ap.add_argument("--dataset-card-dir", default=None)
    ap.add_argument("--gsm8k-scored", default=None)
    ap.add_argument("--preservation-dir", default=None,
                    help="dir with <dataset>_preservation.json files")
    ap.add_argument("--code-eval-json", default=None)
    ap.add_argument("--security-json", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    evidence = _load_json(args.attestation_evidence_json)
    gsm8k_scored = _load_json(args.gsm8k_scored)
    code_eval = _load_json(args.code_eval_json)
    security_json = _load_json(args.security_json)

    per_dataset = []
    global_blockers = []
    for ds in args.datasets:
        plain_rep = _find_report(args.plaintext_dir, ds)
        ours_rep = _find_report(args.ours_dir, ds)
        preservation = None
        if args.preservation_dir:
            preservation = _load_json(
                Path(args.preservation_dir) / ("%s_preservation.json" % ds))
        card = None
        if args.dataset_card_dir:
            card = _load_json(
                Path(args.dataset_card_dir) / ("%s_card.json" % ds))
        util = _utility(ds, plain_rep, ours_rep, gsm8k_scored=gsm8k_scored,
                        preservation=preservation,
                        code_eval=(code_eval if ds == "humaneval" else None))
        perf = _performance(ours_rep)
        sec = _security(ours_rep, evidence)
        pr = _paper_ready(ours_rep, sec,
                          security_json if ds == "sensitive_prompt_1024"
                          else None)
        global_blockers += ["%s: %s" % (ds, b) for b in pr["blockers"]]
        per_dataset.append({
            "dataset": ds,
            "plaintext_report_found": plain_rep is not None,
            "ours_report_found": ours_rep is not None,
            "num_examples": (card or {}).get("num_examples")
            or (ours_rep or {}).get("num_examples"),
            "utility": util, "performance": perf, "security": sec,
            "paper_ready": pr["paper_ready"], "blockers": pr["blockers"]})

    results = {
        "stage": "aaai_main_results",
        "comparison": ["plaintext_local",
                       "ours(A_rightmul, linear pad, TDX boundary, H800 worker)"],
        "excluded": ["lora", "amulet_secure_R", "pure_tee"],
        "attestation_evidence_json": args.attestation_evidence_json,
        "attestation_paper_facing": (evidence or {}).get("paper_facing"),
        "per_dataset": per_dataset,
        "global_paper_ready": (not global_blockers),
        "global_blockers": global_blockers,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aaai_main_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # CSV: one row per dataset, flattened headline numbers
    cols = ["dataset", "ours_report_found", "paper_ready",
            "exact_response_match_rate", "gsm8k_exact_match_delta",
            "humaneval_pass@1_delta", "ttft_s", "tpot_s",
            "end_to_end_latency_s", "tdx_quote_valid",
            "compatible_masks_verified", "nonlinear_trusted_calls"]
    lines = [",".join(cols)]
    for d in per_dataset:
        row = {"dataset": d["dataset"],
               "ours_report_found": d["ours_report_found"],
               "paper_ready": d["paper_ready"],
               **d["utility"], **d["performance"], **d["security"]}
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    (out / "aaai_main_results.csv").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")

    # MD summary
    md = ["# AAAI main results: plaintext-GPU vs ours (A_rightmul)", "",
          "global_paper_ready: **%s**" % results["global_paper_ready"], ""]
    if global_blockers:
        md += ["## Blockers"] + ["- %s" % b for b in global_blockers] + [""]
    md += ["| dataset | paper_ready | exact_match | tpot_s | tdx_quote |",
           "| --- | --- | --- | --- | --- |"]
    for d in per_dataset:
        md.append("| %s | %s | %s | %s | %s |" % (
            d["dataset"], d["paper_ready"],
            d["utility"].get("exact_response_match_rate"),
            d["performance"].get("tpot_s"),
            d["security"].get("tdx_quote_valid")))
    (out / "aaai_main_results.md").write_text("\n".join(md) + "\n",
                                              encoding="utf-8")

    print("=== AAAI main results ===")
    print("datasets=%d global_paper_ready=%s" % (len(per_dataset),
                                                 results["global_paper_ready"]))
    for b in global_blockers:
        print("BLOCKER: %s" % b)
    print("written: %s" % (out / "aaai_main_results.json"))
    return 0 if results["global_paper_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
