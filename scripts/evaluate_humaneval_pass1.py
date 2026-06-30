"""HumanEval pass@1 (sandboxed, CPU) -- plaintext-GPU vs ours (A_rightmul).

Extracts the candidate function from each response, runs the HumanEval unit test
in an ISOLATED CPU subprocess with a hard timeout, and reports pass@1 for the
plaintext baseline and (optionally) for ours, plus the delta and the failed
cases. This is a scoring step fully separate from model inference / the TEE-GPU
security claim: NO model, NO GPU, NO network, and the raw prompt is never written
to the report. Untrusted model output is executed, so it runs sandboxed
(subprocess + timeout) -- do not run on a host where that is unacceptable.

This is the HumanEval-focused companion to ``evaluate_code_generation.py`` (which
also handles MBPP); it exists as the explicitly-named pass@1 entrypoint.

Example::

    python scripts/evaluate_humaneval_pass1.py \\
      --dataset-jsonl <HE_JSONL> \\
      --plaintext-responses <PLAIN.jsonl> --ours-responses <OURS.jsonl> \\
      --timeout 10 --output-json outputs/aaai/qwen/humaneval_pass1.json
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
    evaluate_humaneval_example, pass_at_1)
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


def _score(ds_by_id, resp_by_id, timeout):
    per = []
    for rid, ex in ds_by_id.items():
        completion = (resp_by_id.get(rid) or {}).get("response", "")
        res = evaluate_humaneval_example(
            prompt=ex.get("prompt", ""), completion=completion,
            test=ex.get("test"), entry_point=ex.get("entry_point"),
            timeout=timeout)
        per.append({"id": rid, "passed": bool(res.get("passed")),
                    "error_type": res.get("error_type")})
    return per, pass_at_1(per)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--plaintext-responses", required=True)
    ap.add_argument("--ours-responses", default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    ds_by_id = _by_id(load_dataset("humaneval", args.dataset_jsonl))
    plain = _by_id(_load_jsonl(args.plaintext_responses))
    p_per, p_pass = _score(ds_by_id, plain, args.timeout)
    report = {"stage": "humaneval_pass1", "dataset": "humaneval",
              "num_problems": len(ds_by_id),
              "pass@1_plaintext": p_pass["pass@1"],
              "plaintext_num": p_pass["num"],
              "plaintext_failed_cases": p_pass["failed_cases"]}

    if args.ours_responses:
        ours = _by_id(_load_jsonl(args.ours_responses))
        o_per, o_pass = _score(ds_by_id, ours, args.timeout)
        common = sorted(set(plain) & set(ours))
        cmp = [compare_responses(plain[i], ours[i]) for i in common]
        exact = (sum(1 for c in cmp if c["exact_response_match"]) / len(cmp)
                 if cmp else None)
        report.update({
            "pass@1_ours": o_pass["pass@1"],
            "ours_num": o_pass["num"],
            "pass@1_delta": (None if (p_pass["pass@1"] is None
                                      or o_pass["pass@1"] is None)
                             else round(p_pass["pass@1"] - o_pass["pass@1"], 6)),
            "ours_failed_cases": o_pass["failed_cases"],
            "exact_completion_match_rate": exact,
            "num_compared": len(common)})

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2),
                                      encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in (
        "pass@1_plaintext", "pass@1_ours", "pass@1_delta",
        "exact_completion_match_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
