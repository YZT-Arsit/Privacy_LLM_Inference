"""Plaintext-GPU vs ours (A_rightmul) generation PRESERVATION evaluator.

The AAAI claim is that obfuscated greedy decoding reproduces the plaintext greedy
output deterministically, so the headline metric is *preservation* (ideally
exact). This script compares a plaintext response set against the ours response
set for one dataset and writes a preservation delta table (JSON + CSV + MD) plus
divergence case studies. It NEVER calls an external GPT-4 / Claude judge API; for
MT-Bench it writes a FastChat-compatible judge-ready JSONL that an offline judge
can consume.

Datasets:
* **ifeval**            -- preservation metrics (+ optional official-checker export
  path note); the official IFEval instruction checker is a separate offline step.
* **gsm8k**             -- plaintext exact-match, ours exact-match, AND the delta
  (never ours alone), plus preservation metrics.
* **mt_bench**          -- 80 questions, turn-1 and turn-2 compared SEPARATELY,
  per-category metrics, divergence case studies, and a FastChat-compatible judge
  file. Missing a turn for a two-turn question is a failure.
* **humaneval**         -- completion preservation (the pass@1 scorer is the
  separate sandboxed ``evaluate_code_generation.py`` / ``evaluate_humaneval_pass1``
  step).
* **sensitive_prompt_1024** -- preservation only (the leakage scan is the separate
  ``evaluate_sensitive_prompt_security.py`` step).

If ours diverges from plaintext the paper does not automatically fail, but the
preservation delta table makes the divergence explicit. stdlib only; no torch, no
network, no external judge API.

Example (single-turn)::

    python scripts/evaluate_generation_preservation.py --dataset gsm8k \\
      --dataset-jsonl <GSM8K_JSONL> \\
      --plaintext-responses <PLAIN.jsonl> --ours-responses <OURS.jsonl> \\
      --output-json out.json --output-csv out.csv --output-md out.md

Example (mt_bench, two-turn)::

    python scripts/evaluate_generation_preservation.py --dataset mt_bench \\
      --dataset-jsonl <MTBENCH_JSONL> \\
      --plaintext-turn1 <P1.jsonl> --plaintext-turn2 <P2.jsonl> \\
      --ours-turn1 <O1.jsonl> --ours-turn2 <O2.jsonl> \\
      --judge-jsonl out_judge.jsonl --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.aaai_preservation import (  # noqa: E402
    aggregate_preservation, case_studies, compare_responses)
from pllo.benchmarks.generation_datasets import (  # noqa: E402
    gsm8k_exact_match, load_dataset)

ALL_DATASETS = ("ifeval", "gsm8k", "mt_bench", "humaneval",
                "sensitive_prompt_1024")


def _load_jsonl(path):
    rows = []
    p = Path(path) if path else None
    if not p or not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return rows


def _by_id(rows, *, skip_failed=True):
    out = {}
    for r in rows:
        if skip_failed and r.get("status") == "failed":
            continue
        out[str(r.get("id", r.get("question_id")))] = r
    return out


def _pairwise(plain_by_id, ours_by_id):
    """Per-example preservation for the ids common to both sides."""
    common = sorted(set(plain_by_id) & set(ours_by_id))
    per = []
    for rid in common:
        cmp = compare_responses(plain_by_id[rid], ours_by_id[rid])
        per.append({"id": rid, **cmp})
    missing_plain = sorted(set(ours_by_id) - set(plain_by_id))
    missing_ours = sorted(set(plain_by_id) - set(ours_by_id))
    return per, common, missing_plain, missing_ours


def _gsm8k_block(dataset_jsonl, plain_by_id, ours_by_id):
    """plaintext EM, ours EM, delta -- never ours alone."""
    rows = load_dataset("gsm8k", dataset_jsonl)
    gold = {r["id"]: r.get("reference") for r in rows}

    def _em(by_id):
        scored = correct = 0
        for rid, g in gold.items():
            if g is None or rid not in by_id:
                continue
            scored += 1
            correct += int(gsm8k_exact_match(by_id[rid].get("response"), g))
        return (correct / scored) if scored else None, scored, correct

    p_acc, p_n, p_c = _em(plain_by_id)
    o_acc, o_n, o_c = _em(ours_by_id)
    return {
        "plaintext_exact_match": p_acc, "plaintext_num_scored": p_n,
        "ours_exact_match": o_acc, "ours_num_scored": o_n,
        "exact_match_delta": (None if (p_acc is None or o_acc is None)
                              else round(p_acc - o_acc, 6)),
    }


def _per_category(dataset_jsonl, dataset, per_by_id):
    """Per-category aggregate (uses the dataset's category field if present)."""
    rows = load_dataset(dataset, dataset_jsonl) if dataset_jsonl else []
    cats = {}
    for r in rows:
        cat = r.get("category") or (r.get("meta", {}) or {}).get("category")
        if cat is None:
            continue
        cats.setdefault(str(cat), []).append(str(r["id"]))
    out = {}
    for cat, ids in cats.items():
        sub = [per_by_id[i] for i in ids if i in per_by_id]
        if sub:
            out[cat] = aggregate_preservation(sub)
    return out


def _write_outputs(report, args):
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2),
                                      encoding="utf-8")
    if args.output_csv:
        agg = report.get("aggregate") or {}
        keys = sorted(agg)
        lines = ["metric,value"] + ["%s,%s" % (k, agg[k]) for k in keys]
        Path(args.output_csv).write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    if args.output_md:
        agg = report.get("aggregate") or {}
        md = ["# Generation preservation: %s" % report["dataset"], "",
              "_plaintext-GPU vs ours (A_rightmul)_", "",
              "| metric | value |", "| --- | --- |"]
        for k in sorted(agg):
            md.append("| %s | %s |" % (k, agg[k]))
        if "gsm8k" in report:
            g = report["gsm8k"]
            md += ["", "## GSM8K exact match", "",
                   "| side | exact_match |", "| --- | --- |",
                   "| plaintext | %s |" % g.get("plaintext_exact_match"),
                   "| ours | %s |" % g.get("ours_exact_match"),
                   "| delta | %s |" % g.get("exact_match_delta")]
        Path(args.output_md).write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(ALL_DATASETS))
    ap.add_argument("--dataset-jsonl", default=None)
    # single-turn response files
    ap.add_argument("--plaintext-responses", default=None)
    ap.add_argument("--ours-responses", default=None)
    # mt_bench two-turn response files
    ap.add_argument("--plaintext-turn1", default=None)
    ap.add_argument("--plaintext-turn2", default=None)
    ap.add_argument("--ours-turn1", default=None)
    ap.add_argument("--ours-turn2", default=None)
    ap.add_argument("--judge-jsonl", default=None,
                    help="(mt_bench) FastChat-compatible judge-ready JSONL")
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    report = {"stage": "generation_preservation", "dataset": args.dataset}

    if args.dataset == "mt_bench":
        return _mt_bench(args, report)

    plain = _by_id(_load_jsonl(args.plaintext_responses))
    ours = _by_id(_load_jsonl(args.ours_responses))
    per, common, miss_p, miss_o = _pairwise(plain, ours)
    report["aggregate"] = aggregate_preservation(per)
    report["num_compared"] = len(common)
    report["missing_in_plaintext"] = miss_p
    report["missing_in_ours"] = miss_o
    report["case_studies"] = case_studies(plain, ours, max_cases=args.max_cases)

    if args.dataset == "gsm8k" and args.dataset_jsonl:
        report["gsm8k"] = _gsm8k_block(args.dataset_jsonl, plain, ours)
    if args.dataset == "ifeval":
        report["ifeval_note"] = ("preservation only; run the official IFEval "
                                 "instruction checker on the responses JSONL for "
                                 "the strict score")
    if args.dataset == "humaneval":
        report["humaneval_note"] = ("preservation only; run "
                                    "evaluate_humaneval_pass1.py for sandboxed "
                                    "pass@1")
    if args.dataset == "sensitive_prompt_1024":
        report["sensitive_note"] = ("preservation only; run "
                                    "evaluate_sensitive_prompt_security.py for "
                                    "the GPU-visible leakage scan")

    _write_outputs(report, args)
    print(json.dumps({"dataset": args.dataset, "num_compared": len(common),
                      **{k: report["aggregate"].get(k) for k in (
                          "exact_response_match_rate",
                          "mean_normalized_edit_distance",
                          "finish_reason_match_rate")},
                      **({"gsm8k": report.get("gsm8k")}
                         if "gsm8k" in report else {})}, indent=2))
    return 0


def _mt_bench(args, report) -> int:
    # turn-1 / turn-2 compared SEPARATELY (all 80 questions, no sampling).
    if not (args.plaintext_turn1 and args.ours_turn1):
        print("ERROR: mt_bench needs --plaintext-turn1 and --ours-turn1 "
              "(and turn2 files for two-turn questions)", file=sys.stderr)
        return 3
    rows = load_dataset("mt_bench", args.dataset_jsonl) if args.dataset_jsonl \
        else []
    two_turn_ids = {str(r["id"]) for r in rows
                    if len(r.get("turns", []) or []) > 1}

    p1, o1 = _by_id(_load_jsonl(args.plaintext_turn1)), \
        _by_id(_load_jsonl(args.ours_turn1))
    per1, common1, mp1, mo1 = _pairwise(p1, o1)
    report["turn1"] = {"aggregate": aggregate_preservation(per1),
                       "num_compared": len(common1),
                       "missing_in_plaintext": mp1, "missing_in_ours": mo1}
    report["turn1_per_category"] = _per_category(
        args.dataset_jsonl, "mt_bench", {e["id"]: e for e in per1})

    p2 = _by_id(_load_jsonl(args.plaintext_turn2)) if args.plaintext_turn2 else {}
    o2 = _by_id(_load_jsonl(args.ours_turn2)) if args.ours_turn2 else {}
    per2, common2, mp2, mo2 = _pairwise(p2, o2)
    report["turn2"] = {"aggregate": aggregate_preservation(per2),
                       "num_compared": len(common2),
                       "missing_in_plaintext": mp2, "missing_in_ours": mo2}
    report["turn2_per_category"] = _per_category(
        args.dataset_jsonl, "mt_bench", {e["id"]: e for e in per2})

    # a two-turn question MUST have a turn-2 response on both sides
    missing_turn2 = sorted(two_turn_ids - (set(p2) & set(o2))) if two_turn_ids \
        else []
    report["two_turn_questions"] = len(two_turn_ids)
    report["missing_turn2"] = missing_turn2
    report["turn2_complete"] = (not missing_turn2)
    report["case_studies_turn1"] = case_studies(p1, o1, max_cases=args.max_cases)
    report["case_studies_turn2"] = case_studies(p2, o2, max_cases=args.max_cases)
    # combined headline aggregate over both turns
    report["aggregate"] = aggregate_preservation(per1 + per2)

    # FastChat-compatible judge-ready JSONL (no external API). One row per
    # question listing BOTH model answers (plaintext, ours) for offline judging.
    if args.judge_jsonl:
        Path(args.judge_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.judge_jsonl, "w", encoding="utf-8") as fh:
            for r in rows:
                rid = str(r["id"])
                turns = r.get("turns", []) or []
                fh.write(json.dumps({
                    "question_id": rid, "category": r.get("category"),
                    "turns": turns,
                    "answers": {
                        "plaintext_gpu": [p1.get(rid, {}).get("response"),
                                          p2.get(rid, {}).get("response")],
                        "ours_a_rightmul": [o1.get(rid, {}).get("response"),
                                            o2.get(rid, {}).get("response")]},
                    "judge": "offline_no_external_api"}, ensure_ascii=False)
                    + "\n")
        report["judge_jsonl"] = args.judge_jsonl

    _write_outputs(report, args)
    print(json.dumps({
        "dataset": "mt_bench",
        "turn1_exact": report["turn1"]["aggregate"].get(
            "exact_response_match_rate"),
        "turn2_exact": report["turn2"]["aggregate"].get(
            "exact_response_match_rate"),
        "turn2_complete": report["turn2_complete"],
        "missing_turn2": missing_turn2}, indent=2))
    return 0 if report.get("turn2_complete", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
