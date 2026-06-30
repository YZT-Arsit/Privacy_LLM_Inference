"""Unified AAAI generation benchmark runner: IFEval / GSM8K / MT-Bench.

One ``--dataset`` flag drives the three AAAI generation benchmarks over the SAME
folded-remote / TDX-boundary-client pipeline. This is a thin, honest ADAPTER on
top of ``scripts/run_ifeval_generation.py`` (the fully-featured single-turn runner
with the obfuscation schedule, TDX provenance, worker-health capture, and the
``--paper-facing-generation`` gate): every heavy/paper-facing capability is reused
verbatim, never re-implemented.

* **ifeval**   -- normalise to ``{id, prompt}`` JSONL, run once.
* **gsm8k**    -- normalise (parse the ``#### <n>`` gold), run once, then score
  exact match (last-number extraction) over the responses JSONL.
* **mt_bench** -- TWO-turn: run turn 1, build the turn-2 prompts from turn-1
  responses, run turn 2, and write a judge-ready JSONL
  (``{question_id, category, turns, responses}``).

All passthrough flags (``--backend``, ``--model-path``, ``--gpu-worker-url``,
``--nonlinear-backend``, ``--seq-len``, ``--max-new-tokens``, ``--tdx-boundary-
client``, ``--attestation-evidence-json``, ``--paper-facing-generation``, ...) are
forwarded unchanged to the inner runner, so the AAAI paper-facing contract is
enforced identically for every dataset. No downloads; all paths are CLI args.

Examples (AAAI A_rightmul mainline; run on the trusted TDX guest)::

    python scripts/run_generation_benchmark.py --dataset gsm8k \\
      --input-jsonl <GSM8K_JSONL> --backend folded_remote \\
      --model-path <MODEL> --gpu-worker-url http://127.0.0.1:18082 \\
      --embedding-path <EMB_ARTIFACT> --nonlinear-backend A_rightmul \\
      --seq-len 1024 --max-new-tokens 512 --require-real --use-chat-template \\
      --tdx-boundary-client --attestation-evidence-json <EVIDENCE> \\
      --paper-facing-generation \\
      --output-dir outputs/aaai/gsm8k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_datasets import (  # noqa: E402
    DATASETS, extract_gsm8k_answer, gsm8k_exact_match, load_dataset)

_INNER = REPO_ROOT / "scripts" / "run_ifeval_generation.py"


def _write_prompts(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "prompt": r["prompt"]},
                                ensure_ascii=False) + "\n")


def _run_inner(input_jsonl, resp_jsonl, report_json, passthrough) -> int:
    cmd = [sys.executable, str(_INNER), "--input-jsonl", str(input_jsonl),
           "--output-response-jsonl", str(resp_jsonl),
           "--output-report-json", str(report_json)] + list(passthrough)
    print("[gen-bench] -> %s" % " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="dir for the responses JSONL, the inner report JSON, and "
                         "(gsm8k) the scored report / (mt_bench) the judge JSONL")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--no-gsm8k-instruction", action="store_true", default=False,
                    help="do not prepend the GSM8K 'show reasoning + #### answer' "
                         "instruction to each question")
    # everything else is forwarded verbatim to run_ifeval_generation.py
    args, passthrough = ap.parse_known_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset

    rows = load_dataset(
        ds, args.input_jsonl, max_examples=args.max_examples,
        **({"add_instruction": not args.no_gsm8k_instruction}
           if ds == "gsm8k" else {}))
    if not rows:
        print("ERROR: no examples in %s" % args.input_jsonl, file=sys.stderr)
        return 3

    if ds in ("ifeval", "gsm8k"):
        prompts = out / ("%s_prompts.jsonl" % ds)
        _write_prompts(rows, prompts)
        resp = out / ("%s_responses.jsonl" % ds)
        report = out / ("%s_generation.json" % ds)
        rc = _run_inner(prompts, resp, report, passthrough)
        if rc != 0:
            print("ERROR: inner generation runner failed (rc=%d)" % rc,
                  file=sys.stderr)
            return rc
        if ds == "gsm8k":
            return _score_gsm8k(rows, resp, report, out)
        return 0

    # ---- mt_bench: two-turn ----
    return _run_mt_bench(rows, out, passthrough)


def _score_gsm8k(rows, resp_path, report_path, out) -> int:
    by_id = {r["id"]: r for r in rows}
    responses = _read_jsonl(resp_path)
    n = correct = scored = 0
    per_example = []
    for rec in responses:
        rid = str(rec.get("id"))
        gold = (by_id.get(rid) or {}).get("reference")
        pred = extract_gsm8k_answer(rec.get("response"))
        ok = gsm8k_exact_match(rec.get("response"), gold)
        n += 1
        if gold is not None:
            scored += 1
            correct += int(ok)
        per_example.append({"id": rid, "gold": gold, "predicted": pred,
                            "exact_match": bool(ok)})
    acc = (correct / scored) if scored else None
    inner = {}
    try:
        inner = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        pass
    scored_report = {
        "stage": "gsm8k_generation_scored", "dataset": "gsm8k",
        "num_responses": n, "num_scored": scored, "num_correct": correct,
        "exact_match_accuracy": acc,
        "per_example": per_example,
        "generation_report": str(report_path),
        "nonlinear_backend": inner.get("nonlinear_backend"),
        "paper_ready": inner.get("paper_ready"),
        "paper_facing_generation": inner.get("paper_facing_generation"),
        "paper_facing_generation_violations":
            inner.get("paper_facing_generation_violations"),
    }
    sp = out / "gsm8k_scored.json"
    sp.write_text(json.dumps(scored_report, indent=2), encoding="utf-8")
    print("=== gsm8k scored ===")
    print("num_scored=%d num_correct=%d exact_match_accuracy=%s paper_ready=%s"
          % (scored, correct, acc, scored_report["paper_ready"]))
    # honour the inner paper-facing verdict for the exit code
    if inner.get("paper_facing_generation") is False:
        return 1
    return 0


def _run_mt_bench(rows, out, passthrough) -> int:
    # turn 1
    t1_prompts = out / "mt_bench_turn1_prompts.jsonl"
    _write_prompts([{"id": r["id"], "prompt": r["turns"][0]} for r in rows],
                   t1_prompts)
    t1_resp = out / "mt_bench_turn1_responses.jsonl"
    t1_report = out / "mt_bench_turn1_generation.json"
    rc = _run_inner(t1_prompts, t1_resp, t1_report, passthrough)
    if rc != 0:
        print("ERROR: mt_bench turn-1 failed (rc=%d)" % rc, file=sys.stderr)
        return rc
    t1_by_id = {str(rec.get("id")): rec.get("response", "")
                for rec in _read_jsonl(t1_resp)}

    # turn 2 (only for examples that actually have a second turn)
    two_turn = [r for r in rows if len(r["turns"]) > 1]
    t2_by_id = {}
    t2_report = None
    if two_turn:
        t2_rows = []
        for r in two_turn:
            r1 = t1_by_id.get(r["id"], "")
            convo = ("%s\n\n%s\n\n%s" % (r["turns"][0], r1, r["turns"][1]))
            t2_rows.append({"id": r["id"], "prompt": convo})
        t2_prompts = out / "mt_bench_turn2_prompts.jsonl"
        _write_prompts(t2_rows, t2_prompts)
        t2_resp = out / "mt_bench_turn2_responses.jsonl"
        t2_report = out / "mt_bench_turn2_generation.json"
        rc = _run_inner(t2_prompts, t2_resp, t2_report, passthrough)
        if rc != 0:
            print("ERROR: mt_bench turn-2 failed (rc=%d)" % rc, file=sys.stderr)
            return rc
        t2_by_id = {str(rec.get("id")): rec.get("response", "")
                    for rec in _read_jsonl(t2_resp)}

    # judge-ready JSONL: one row per question with both turns + both responses
    judge = out / "mt_bench_judge.jsonl"
    with open(judge, "w", encoding="utf-8") as fh:
        for r in rows:
            resp = [t1_by_id.get(r["id"], "")]
            if len(r["turns"]) > 1:
                resp.append(t2_by_id.get(r["id"], ""))
            fh.write(json.dumps({
                "question_id": r["id"], "category": r.get("category"),
                "turns": r["turns"], "responses": resp}, ensure_ascii=False) + "\n")

    inner = {}
    try:
        inner = json.loads(Path(t1_report).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        pass
    summary = {
        "stage": "mt_bench_generation", "dataset": "mt_bench",
        "num_questions": len(rows), "num_two_turn": len(two_turn),
        "judge_jsonl": str(judge),
        "turn1_report": str(t1_report),
        "turn2_report": str(t2_report) if t2_report else None,
        "nonlinear_backend": inner.get("nonlinear_backend"),
        "paper_ready": inner.get("paper_ready"),
        "paper_facing_generation": inner.get("paper_facing_generation"),
        "paper_facing_generation_violations":
            inner.get("paper_facing_generation_violations"),
    }
    (out / "mt_bench_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print("=== mt_bench ===")
    print("num_questions=%d num_two_turn=%d judge_jsonl=%s paper_ready=%s"
          % (len(rows), len(two_turn), judge, summary["paper_ready"]))
    if inner.get("paper_facing_generation") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
