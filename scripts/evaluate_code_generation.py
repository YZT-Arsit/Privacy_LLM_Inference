"""Evaluate code-generation responses (HumanEval / MBPP) -> pass@1, sandboxed.

Reads the dataset JSONL + a response JSONL (``response`` = completion), extracts
code, runs the dataset tests in an ISOLATED CPU subprocess with a timeout, and
reports pass@1. With ``--ours-responses`` it also compares plaintext vs ours
(pass@1 delta + exact/edit/token preservation). NO model, NO GPU, NO network; the
raw prompt is never written to the report.

Example::

    python scripts/evaluate_code_generation.py --dataset humaneval \\
      --dataset-jsonl <HE_JSONL> --plaintext-responses <PLAIN.jsonl> \\
      --ours-responses <OURS.jsonl> --timeout 10 \\
      --output-json outputs/aaai/qwen/code_eval_humaneval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.aaai_preservation import compare_responses  # noqa: E402
from pllo.benchmarks.code_eval import (  # noqa: E402
    evaluate_humaneval_example, evaluate_mbpp_example, pass_at_1)
from pllo.benchmarks.generation_datasets import load_dataset  # noqa: E402


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


def _by_id(rows):
    out = {}
    for r in rows:
        if r.get("status") == "failed":
            continue
        out[str(r.get("id"))] = r
    return out


def _score(dataset, ds_by_id, resp_by_id, timeout):
    per = []
    for rid, ex in ds_by_id.items():
        rec = resp_by_id.get(rid)
        completion = (rec or {}).get("response", "")
        if dataset == "humaneval":
            res = evaluate_humaneval_example(
                prompt=ex.get("prompt", ""), completion=completion,
                test=ex.get("test"), entry_point=ex.get("entry_point"),
                timeout=timeout)
        else:
            res = evaluate_mbpp_example(
                completion=completion,
                test_list=(ex.get("meta", {}) or {}).get("test_list")
                or ex.get("test_list", []), timeout=timeout)
        per.append({"id": rid, "passed": bool(res.get("passed")),
                    "error_type": res.get("error_type")})
    return per, pass_at_1(per)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--plaintext-responses", required=True)
    ap.add_argument("--ours-responses", default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    ds_by_id = _by_id(load_dataset(args.dataset, args.dataset_jsonl))
    plain = _by_id(_load_jsonl(args.plaintext_responses))
    p_per, p_pass = _score(args.dataset, ds_by_id, plain, args.timeout)

    report = {"stage": "code_generation_eval", "dataset": args.dataset,
              "pass@1_plaintext": p_pass["pass@1"],
              "plaintext_num": p_pass["num"],
              "plaintext_failed_cases": p_pass["failed_cases"]}

    if args.ours_responses:
        ours = _by_id(_load_jsonl(args.ours_responses))
        o_per, o_pass = _score(args.dataset, ds_by_id, ours, args.timeout)
        # preservation plaintext vs ours
        common = sorted(set(plain) & set(ours))
        cmp = [compare_responses(plain[i], ours[i]) for i in common]
        exact = (sum(1 for c in cmp if c["exact_response_match"]) / len(cmp)
                 if cmp else None)
        ned = ([c["normalized_edit_distance"] for c in cmp])
        tmr = [c["token_exact_match_rate"] for c in cmp
               if c["token_exact_match_rate"] is not None]
        report.update({
            "pass@1_ours": o_pass["pass@1"],
            "pass@1_delta": (None if (p_pass["pass@1"] is None
                                      or o_pass["pass@1"] is None)
                             else round(p_pass["pass@1"] - o_pass["pass@1"], 6)),
            "ours_failed_cases": o_pass["failed_cases"],
            "exact_completion_match_rate": exact,
            "normalized_edit_distance_mean":
                (round(sum(ned) / len(ned), 6) if ned else None),
            "token_match_rate_mean":
                (round(sum(tmr) / len(tmr), 6) if tmr else None),
            "num_compared": len(common)})

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2),
                                      encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in (
        "dataset", "pass@1_plaintext", "pass@1_ours", "pass@1_delta",
        "exact_completion_match_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
