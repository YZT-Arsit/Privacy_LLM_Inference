"""AAAI validator: plaintext-GPU vs ours (A_rightmul TDX/H800) generation.

Reads the plaintext + ours reports and response JSONLs (and the dataset card +
attestation evidence) and checks the AAAI contract, then computes utility
PRESERVATION (plaintext vs ours, the paper's primary metric) plus a performance
table. Emits ``<out>.json`` / ``.md`` and a ``<out>.csv`` table.

Checks (fail -> non-zero exit):
* ids align; completed == expected (from the card); failed == 0 unless --allow-failed;
* seq_len==1024, max_new_tokens==512, EOS on, greedy, no dry_run / mock on both;
* OURS: A_rightmul, nonlinear_trusted_calls==0, trusted_nonlinear_ops_count==0,
  single TEE entry/exit, compatible_masks_verified, pad coverage, valid bound TDX
  evidence, worker health present, TDX boundary client, runtime hash binds A_rightmul.

Preservation: exact response / token match, normalized edit distance, length ratio,
ROUGE-L, chrF, finish-reason match (per dataset; per turn for MT-Bench). GSM8K also
reports numeric exact-match for plaintext and ours + delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations  # noqa: E402
from pllo.benchmarks.aaai_preservation import (  # noqa: E402
    aggregate_preservation, case_studies, compare_responses)
from pllo.benchmarks.run_state import recount_status_from_jsonl  # noqa: E402


def _load_json(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None


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


def _ok_records(rows):
    """Map (id, turn_index) -> record for status==ok rows (latest wins)."""
    out = {}
    for r in rows:
        if r.get("status") == "failed":
            continue
        key = (str(r.get("id")), int(r.get("turn_index") or 0))
        out[key] = r
    return out


def _completed_total(rep, resp_rows, *, dataset):
    """Resume-aware completed count: prefer report.completed_total, else recount
    the response JSONL (latest-status-per-id; per-question for MT-Bench)."""
    rep = rep or {}
    if rep.get("completed_total") is not None:
        return int(rep["completed_total"])
    rc = _recount_rows(resp_rows, dataset=dataset)
    return rc["completed_total"]


def _recount_rows(rows, *, dataset):
    """Recount completion from already-loaded response rows (latest-status-per-id;
    per-question for MT-Bench). Mirrors run_state.recount_status_from_jsonl."""
    mt = (dataset == "mt_bench")
    if mt:
        per = {}
        for r in rows:
            rid = r.get("id")
            if rid is None or r.get("status") == "skipped":
                continue
            per.setdefault(str(rid), {})[int(r.get("turn_index") or 0)] = \
                r.get("status", "ok")
        ok = [rid for rid, t in per.items()
              if t and all(s == "ok" for s in t.values())]
        failed = [rid for rid, t in per.items()
                  if rid not in ok and any(s == "failed" for s in t.values())]
        return {"completed_total": len(ok), "failed_total": len(failed)}
    latest = {}
    for r in rows:
        rid = r.get("id")
        if rid is None or r.get("status") == "skipped":
            continue
        latest[str(rid)] = r.get("status", "ok")
    ok = [k for k, v in latest.items() if v in (None, "ok", "completed")]
    failed = [k for k, v in latest.items() if v == "failed"]
    return {"completed_total": len(ok), "failed_total": len(failed)}


def _general_checks(rep, label, expected_n, *, completed_total=None):
    c = []

    def chk(name, ok, detail=""):
        c.append({"check": "%s:%s" % (label, name), "ok": bool(ok),
                  "detail": detail})
    chk("seq_len==1024", int(rep.get("seq_len") or 0) == 1024)
    chk("max_new_tokens==512", int(rep.get("max_new_tokens") or 0) == 512)
    chk("eos_stop_on", rep.get("stop_on_eos") is True)
    chk("greedy", (rep.get("decoding") or "greedy") == "greedy")
    chk("not_dry_run", rep.get("dry_run") is not True)
    chk("not_mock", rep.get("mock_runtime") is not True)
    if expected_n is not None:
        # resume-aware: completed_total (this run + prior runs), NOT the per-run
        # completed_examples (which would falsely fail after a resume).
        got = completed_total if completed_total is not None else int(
            rep.get("completed_total") or rep.get("completed_examples") or 0)
        chk("completed==expected", int(got) == int(expected_n),
            "completed_total=%s expected=%s" % (got, expected_n))
    return c


def validate(plaintext_rep, ours_rep, plaintext_resp, ours_resp, *, dataset,
             card, evidence, expected_mr_td, allow_failed):
    checks = []
    expected_n = (card or {}).get("num_examples")

    # resume-aware completion totals (this run + prior runs) for both sides
    p_completed = _completed_total(plaintext_rep, plaintext_resp, dataset=dataset)
    o_completed = _completed_total(ours_rep, ours_resp, dataset=dataset)

    # general (both sides)
    checks += _general_checks(plaintext_rep or {}, "plaintext", expected_n,
                              completed_total=p_completed)
    checks += _general_checks(ours_rep or {}, "ours", expected_n,
                              completed_total=o_completed)

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # failure count: prefer the resume-aware failed_total (a later success clears
    # an earlier failure), else the report's per-run failed_examples.
    for label, rep, resp in (("plaintext", plaintext_rep, plaintext_resp),
                             ("ours", ours_rep, ours_resp)):
        rep = rep or {}
        failed = rep.get("failed_total")
        if failed is None:
            failed = _recount_rows(resp, dataset=dataset)["failed_total"]
        chk("%s:failed_examples_zero" % label, int(failed) == 0 or allow_failed,
            "failed_total=%d" % int(failed))

    # ours AAAI contract
    ours_viol = aaai_generation_violations(ours_rep or {}, evidence=evidence,
                                           expected_mr_td=expected_mr_td)
    chk("ours:aaai_paper_facing", len(ours_viol) == 0, "; ".join(ours_viol))

    # id alignment
    pr = _ok_records(plaintext_resp)
    orr = _ok_records(ours_resp)
    p_ids = {k[0] for k in pr}
    o_ids = {k[0] for k in orr}
    chk("ids_align", p_ids == o_ids,
        "plaintext_only=%s ours_only=%s" % (sorted(p_ids - o_ids)[:5],
                                            sorted(o_ids - p_ids)[:5]))

    # preservation (per aligned (id,turn))
    common = sorted(set(pr) & set(orr))
    per_example = [compare_responses(pr[k], orr[k]) for k in common]
    preservation = aggregate_preservation(per_example)

    # per-turn (MT-Bench)
    per_turn = {}
    if dataset == "mt_bench":
        for turn in sorted({k[1] for k in common}):
            rows = [compare_responses(pr[k], orr[k]) for k in common
                    if k[1] == turn]
            per_turn["turn_%d" % turn] = aggregate_preservation(rows)

    # gsm8k numeric accuracy (both sides) + delta
    gsm8k = {}
    if dataset == "gsm8k":
        def _acc(records):
            s = c = 0
            for k, r in records.items():
                if r.get("gold_answer") is not None:
                    s += 1
                    c += int(bool(r.get("exact_match")))
            return (c / s) if s else None, s, c
        pa, ps, pc = _acc(pr)
        oa, os_, oc = _acc(orr)
        gsm8k = {"plaintext_accuracy": pa, "plaintext_scored": ps,
                 "ours_accuracy": oa, "ours_scored": os_,
                 "accuracy_delta": (None if (pa is None or oa is None)
                                    else round(pa - oa, 6))}

    cases = case_studies({k[0]: pr[k] for k in common if k[1] == 0},
                         {k[0]: orr[k] for k in common if k[1] == 0},
                         max_cases=20)

    # performance (best-effort from reports)
    def _perf(rep):
        rep = rep or {}
        keys = ("ttft_s", "tpot_s", "end_to_end_latency_s", "tokens_per_sec",
                "worker_forward_time_s", "boundary_client_time_s",
                "network_serialization_time_s", "peak_gpu_memory_mb",
                "nonlinear_trusted_bytes", "nonlinear_accelerator_bytes",
                "boundary_calls", "gpu_calls", "generated_tokens_total")
        return {k: rep.get(k) for k in keys if k in rep}
    performance = {"plaintext": _perf(plaintext_rep), "ours": _perf(ours_rep)}

    failed = [c for c in checks if not c["ok"]]
    return {
        "stage": "aaai_generation_validation", "dataset": dataset,
        "expected_examples": expected_n,
        "num_checks": len(checks), "num_failed": len(failed),
        "passed": len(failed) == 0,
        "failed_checks": failed, "checks": checks,
        "ours_aaai_violations": ours_viol,
        "utility_preservation": preservation,
        "utility_preservation_per_turn": per_turn,
        "gsm8k_accuracy": gsm8k,
        "performance": performance,
        "divergence_case_studies": cases,
        "num_compared": len(common),
    }


def _render_md(r, run_id):
    p = r["utility_preservation"]
    L = ["# AAAI generation validation (%s)" % run_id, "",
         "_dataset=%s passed=%s checks=%d failed=%d compared=%d_"
         % (r["dataset"], r["passed"], r["num_checks"], r["num_failed"],
            r["num_compared"]), "",
         "## Utility preservation (plaintext vs ours)", "",
         "| metric | value |", "| --- | --- |"]
    for k, v in (p or {}).items():
        L.append("| %s | %s |" % (k, v))
    if r.get("gsm8k_accuracy"):
        L += ["", "## GSM8K accuracy", "", "| field | value |", "| --- | --- |"]
        for k, v in r["gsm8k_accuracy"].items():
            L.append("| %s | %s |" % (k, v))
    if r["failed_checks"]:
        L += ["", "## FAILED checks", ""]
        for c in r["failed_checks"]:
            L.append("- `%s`: %s" % (c["check"], c["detail"]))
    if r.get("divergence_case_studies"):
        L += ["", "## Divergence case studies (%d)"
              % len(r["divergence_case_studies"]), ""]
        for cs in r["divergence_case_studies"][:5]:
            L.append("- id=%s edit_dist=%s token_match=%s"
                     % (cs["id"], cs["normalized_edit_distance"],
                        cs.get("token_exact_match_rate")))
    return "\n".join(L) + "\n"


def _render_csv(r, path):
    p = r["utility_preservation"] or {}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric", "value"])
        for k, v in p.items():
            w.writerow([r["dataset"], k, v])
        for k, v in (r.get("gsm8k_accuracy") or {}).items():
            w.writerow([r["dataset"], "gsm8k_" + k, v])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["ifeval", "gsm8k", "mt_bench"])
    ap.add_argument("--plaintext-report", required=True)
    ap.add_argument("--plaintext-responses", required=True)
    ap.add_argument("--ours-report", required=True)
    ap.add_argument("--ours-responses", required=True)
    ap.add_argument("--dataset-card", default=None)
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--allow-failed", action="store_true", default=False)
    ap.add_argument("--run-id", default="aaai")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    rep = validate(
        _load_json(args.plaintext_report), _load_json(args.ours_report),
        _load_jsonl(args.plaintext_responses), _load_jsonl(args.ours_responses),
        dataset=args.dataset, card=_load_json(args.dataset_card),
        evidence=_load_json(args.attestation_evidence_json),
        expected_mr_td=args.expected_mr_td, allow_failed=args.allow_failed)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rep, indent=2, default=str),
                                      encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_render_md(rep, args.run_id),
                                        encoding="utf-8")
    if args.output_csv:
        _render_csv(rep, args.output_csv)

    print(json.dumps({k: rep[k] for k in (
        "dataset", "num_checks", "num_failed", "passed", "num_compared")},
        indent=2))
    if rep["failed_checks"]:
        print("\nFAILED:", file=sys.stderr)
        for c in rep["failed_checks"]:
            print("  - %s: %s" % (c["check"], c["detail"]), file=sys.stderr)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
