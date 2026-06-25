"""E9 pairwise / aggregate utility preservation (pure parsing).

A single E9 metric value does NOT show that the private path preserves utility --
you need the SAME task measured on a plaintext baseline and on the
folded/TDX/LoRA candidate, then a bounded drop. This module computes that
comparison and an aggregate over the required datasets.

Honesty gate: ``paper_ready`` is True only when BOTH inputs are themselves
``paper_ready`` and not ``dry_run`` (i.e. real runs); ``utility_preserved``
additionally requires the drop to be within the abs/rel thresholds. stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["metric_of", "pairwise_preservation", "aggregate_preservation",
           "render_pairwise_md", "render_aggregate_md", "REQUIRED_DATASETS"]

# Canonical paper datasets an aggregate "utility preserved" claim should cover.
REQUIRED_DATASETS = ("mmlu", "gsm8k", "boolq", "sst2")
_DATASET_ALIASES = {"agnews": "sst2", "ag_news": "sst2", "cmmlu": "mmlu",
                    "ceval": "mmlu", "cnndm": "cnndm", "xsum": "xsum"}


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def metric_of(report):
    if isinstance(report, dict):
        for k in ("metric_value", "accuracy", "score"):
            v = report.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _both_paper_ready(baseline, candidate) -> bool:
    return bool(_g(baseline, "paper_ready") is True
                and _g(candidate, "paper_ready") is True
                and _g(baseline, "dry_run") is not True
                and _g(candidate, "dry_run") is not True)


def pairwise_preservation(baseline, candidate, *, max_abs_drop=0.02,
                          max_rel_drop=0.05, dataset=None,
                          task_type=None) -> dict:
    """Compare a plaintext baseline E9 report to a candidate E9 report."""
    bm = metric_of(baseline)
    cm = metric_of(candidate)
    delta_abs = (None if (bm is None or cm is None) else round(bm - cm, 6))
    delta_rel = (None if (delta_abs is None or not bm)
                 else round(delta_abs / bm, 6))
    within = None
    if delta_abs is not None:
        within = bool(delta_abs <= max_abs_drop
                      and (delta_rel is None or delta_rel <= max_rel_drop))
    paper_ready = _both_paper_ready(baseline, candidate)
    utility_preserved = bool(within) and paper_ready
    return {
        "stage": "e9_pairwise_utility_preservation",
        "dataset": dataset or _g(candidate, "dataset") or _g(baseline, "dataset"),
        "task_type": (task_type or _g(candidate, "task_type")
                      or _g(baseline, "task_type")),
        "metric_name": _g(candidate, "metric_name") or _g(baseline,
                                                          "metric_name"),
        "baseline_backend": _g(baseline, "backend"),
        "candidate_backend": _g(candidate, "backend"),
        # the candidate determines the nonlinear design under test (the baseline
        # is plaintext); this tags the report for the per-design utility claim.
        "nonlinear_backend": (_g(candidate, "nonlinear_backend")
                              or _g(baseline, "nonlinear_backend")),
        "baseline_metric": bm, "candidate_metric": cm,
        "delta_abs": delta_abs, "delta_rel": delta_rel,
        "max_abs_drop": max_abs_drop, "max_rel_drop": max_rel_drop,
        "within_threshold": within,
        "utility_preserved": utility_preserved,
        "paper_ready": paper_ready,
        "dry_run": not paper_ready,
        "note": "utility_preserved requires the drop within abs/rel thresholds "
                "AND both inputs paper_ready (real, not dry_run).",
    }


def _norm_dataset(name):
    if not name:
        return None
    low = str(name).lower()
    return _DATASET_ALIASES.get(low, low)


def aggregate_preservation(pairwise_reports, *,
                           required_datasets=REQUIRED_DATASETS) -> dict:
    """Aggregate pairwise reports; overall preserved only if every REQUIRED
    dataset is present, within threshold, and paper_ready."""
    rows = []
    by_dataset = {}
    backends = set()
    for pr in pairwise_reports:
        if not isinstance(pr, dict):
            continue
        ds = _norm_dataset(pr.get("dataset"))
        if pr.get("nonlinear_backend"):
            backends.add(pr.get("nonlinear_backend"))
        row = {
            "dataset": pr.get("dataset"), "norm_dataset": ds,
            "task_type": pr.get("task_type"),
            "nonlinear_backend": pr.get("nonlinear_backend"),
            "baseline_metric": pr.get("baseline_metric"),
            "candidate_metric": pr.get("candidate_metric"),
            "delta_abs": pr.get("delta_abs"), "delta_rel": pr.get("delta_rel"),
            "within_threshold": pr.get("within_threshold"),
            "utility_preserved": pr.get("utility_preserved"),
            "paper_ready": pr.get("paper_ready"),
        }
        rows.append(row)
        if ds:
            by_dataset[ds] = row

    required = [_norm_dataset(d) for d in required_datasets]
    missing = [d for d in required if d not in by_dataset]
    covered = [d for d in required if d in by_dataset]
    all_pass = bool(required) and not missing and all(
        by_dataset[d].get("utility_preserved") is True for d in covered)
    paper_ready = bool(rows) and all(r.get("paper_ready") is True for r in rows) \
        and not missing
    return {
        "stage": "e9_aggregate_utility_preservation",
        "required_datasets": list(required),
        "covered_datasets": covered, "missing_datasets": missing,
        # one design only when all pairwise rows agree; "mixed" otherwise so a
        # claim cannot be silently backed by cross-design evidence.
        "nonlinear_backend": (next(iter(backends)) if len(backends) == 1
                              else ("mixed" if backends else None)),
        "rows": rows,
        "utility_preserved": all_pass and paper_ready,
        "all_within_threshold": bool(rows) and all(
            r.get("within_threshold") is True for r in rows),
        "paper_ready": paper_ready, "dry_run": not paper_ready,
        "note": "overall utility_preserved requires every required dataset "
                "present, within threshold, and paper_ready.",
    }


def render_pairwise_md(r: dict) -> str:
    return "\n".join([
        "# E9 pairwise utility preservation", "",
        "_dataset=%s task=%s metric=%s (paper_ready=%s)_"
        % (r["dataset"], r["task_type"], r["metric_name"], r["paper_ready"]), "",
        "| field | value |", "| --- | --- |",
        "| baseline_backend | %s |" % r["baseline_backend"],
        "| candidate_backend | %s |" % r["candidate_backend"],
        "| baseline_metric | %s |" % r["baseline_metric"],
        "| candidate_metric | %s |" % r["candidate_metric"],
        "| delta_abs | %s |" % r["delta_abs"],
        "| delta_rel | %s |" % r["delta_rel"],
        "| max_abs_drop | %s |" % r["max_abs_drop"],
        "| max_rel_drop | %s |" % r["max_rel_drop"],
        "| within_threshold | %s |" % r["within_threshold"],
        "| **utility_preserved** | **%s** |" % r["utility_preserved"], ""])


def render_aggregate_md(r: dict) -> str:
    L = ["# E9 aggregate utility preservation", "",
         "_required=%s covered=%s missing=%s paper_ready=%s_"
         % (r["required_datasets"], r["covered_datasets"],
            r["missing_datasets"], r["paper_ready"]), "",
         "| dataset | baseline | candidate | delta_abs | delta_rel | "
         "within | preserved | paper_ready |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in r["rows"]:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (row["dataset"], row["baseline_metric"],
                    row["candidate_metric"], row["delta_abs"], row["delta_rel"],
                    row["within_threshold"], row["utility_preserved"],
                    row["paper_ready"]))
    L += ["", "- **overall utility_preserved=%s**" % r["utility_preserved"], ""]
    return "\n".join(L)


def load_json(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None
