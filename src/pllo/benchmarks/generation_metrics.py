"""Generation utility benchmark: task-aware metrics + runner + pairwise (stdlib).

The paper's core claim is privacy-preserving *autoregressive generation*, so this
module scores longer-output benchmarks under deterministic greedy decoding and
compares the protected folded path to a plaintext baseline:

* ``generation_exact`` (GSM8K)   -> numeric_exact_match, extracted_number,
  output_length_tokens, latency_s.
* ``summarization`` (CNN/DM, XSum) -> rouge1 / rouge2 / rougeL (+
  ``rouge_unavailable`` when the ``rouge_score`` package is absent and the pure-
  Python LCS/overlap fallback is used), output_length_tokens, latency_s.
* ``open_ended`` (custom JSONL)  -> exact_text_match + normalized_edit_similarity
  vs a reference (when present), output_length_tokens, latency_s.

Pairwise preservation (plaintext_local vs folded_remote, CURRENT design only):
metric_abs_drop, metric_rel_drop, exact_output_match_rate, length_delta_mean,
latency_ratio, audit_passed.

Constraints baked in: CURRENT design only (``trusted_shortcut`` refused), no
downloads, no LLM judge, no subjective quality scoring. stdlib only -- predictors
(torch) are injected by the runner script and never imported here. ``rouge_score``
is used IF already importable; it is never installed/downloaded here.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pllo.benchmarks import metrics as M
# single-source the edit-distance helpers shipped with the preservation module
from pllo.benchmarks.generation_preservation import (
    exact_text_match,
    normalized_edit_similarity,
)

__all__ = [
    "STAGE_BENCHMARK",
    "STAGE_PAIRWISE",
    "TASK_TYPES",
    "PRIMARY_METRIC",
    "SUPPORTED_BACKENDS",
    "validate_example",
    "load_examples",
    "assert_current_only",
    "extract_number",
    "numeric_exact_match",
    "rouge_scores",
    "rouge_available",
    "output_length_tokens",
    "exact_text_match",
    "normalized_edit_similarity",
    "score_example",
    "stub_generate",
    "run_generation_utility_benchmark",
    "pairwise_generation_preservation",
    "summarize_pairwise",
    "render_benchmark_md",
    "render_benchmark_csv",
    "render_pairwise_md",
    "render_pairwise_csv",
    "render_summary_md",
    "render_summary_csv",
    "load_json",
]

STAGE_BENCHMARK = "generation_utility_benchmark"
STAGE_PAIRWISE = "generation_pairwise_preservation"
STAGE_SUMMARY = "generation_preservation_summary"

SUPPORTED_BACKENDS = ("plaintext_local", "folded_remote")

# Canonical generation task types + their primary scalar metric (for drop calc).
TASK_TYPES = ("generation_exact", "summarization", "open_ended")
PRIMARY_METRIC = {
    "generation_exact": "numeric_exact_match",
    "summarization": "rougeL",
    "open_ended": "normalized_edit_similarity",
}
_TASK_ALIASES = {
    "generation": "open_ended", "open-ended": "open_ended", "custom": "open_ended",
    "gen": "open_ended", "summ": "summarization", "cnndm": "summarization",
    "xsum": "summarization", "gsm8k": "generation_exact",
    "numeric": "generation_exact",
}


def _norm_task(t) -> str:
    s = str(t or "open_ended").strip().lower()
    s = _TASK_ALIASES.get(s, s)
    return s if s in TASK_TYPES else "open_ended"


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def validate_example(ex: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if not isinstance(ex, dict):
        return ["example is not an object"]
    if not str(ex.get("id", "")).strip():
        problems.append("missing non-empty 'id'")
    if not str(ex.get("prompt", "")).strip():
        problems.append("missing non-empty 'prompt'")
    return problems


def _reference_of(ex: Dict[str, Any]):
    for k in ("reference", "answer"):
        v = ex.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return None


def load_examples(path) -> List[Dict[str, Any]]:
    """Load normalized generation examples (one JSON object per JSONL line).

    Required: ``id``, ``prompt``. Optional: ``task_type`` (default open_ended),
    ``dataset_name``, ``reference``/``answer``, ``category``."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            ex = json.loads(ln)
            probs = validate_example(ex)
            if probs:
                raise ValueError("invalid example %r: %s"
                                 % (ex.get("id"), "; ".join(probs)))
            ex["task_type"] = _norm_task(ex.get("task_type"))
            ex.setdefault("dataset_name", ex.get("dataset", "custom"))
            ex.setdefault("category", "uncategorized")
            out.append(ex)
    return out


def assert_current_only(nonlinear_backend) -> str:
    """Refuse anything but the ``current`` design (no trusted_shortcut here)."""
    from pllo.experiments.nonlinear_designs import normalize_nonlinear_backend
    nb = normalize_nonlinear_backend(nonlinear_backend or "current")
    if nb != "current":
        raise ValueError(
            "generation utility benchmark is CURRENT-ONLY; the trusted_shortcut "
            "design is not paper-facing here (got %r). Run with "
            "--nonlinear-backend current." % nb)
    return nb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def extract_number(text) -> Optional[str]:
    """Canonical numeric answer extracted from free text (GSM8K-style)."""
    return M.extract_numeric_answer(text)


def numeric_exact_match(pred, reference) -> bool:
    """True iff the extracted numbers of pred and reference match."""
    p = extract_number(pred)
    g = extract_number(reference)
    return bool(p is not None and g is not None and p == g)


def output_length_tokens(token_ids) -> Optional[int]:
    return None if token_ids is None else len(token_ids)


def rouge_available() -> bool:
    """True iff the optional ``rouge_score`` package is importable (never
    installed/downloaded here)."""
    import importlib.util
    return importlib.util.find_spec("rouge_score") is not None


def _words(s) -> List[str]:
    return str(s or "").strip().lower().split()


def _ngram_f1(pred_words: List[str], ref_words: List[str], n: int) -> float:
    if len(pred_words) < n or len(ref_words) < n:
        return 0.0
    pg = Counter(tuple(pred_words[i:i + n]) for i in range(len(pred_words) - n + 1))
    rg = Counter(tuple(ref_words[i:i + n]) for i in range(len(ref_words) - n + 1))
    overlap = sum((pg & rg).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(pg.values())
    rec = overlap / sum(rg.values())
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def rouge_scores(pred, reference) -> Dict[str, Any]:
    """ROUGE-1/2/L F1. Uses ``rouge_score`` if importable; otherwise a pure-
    Python n-gram-overlap / LCS fallback (``rouge_unavailable=True``).

    Never downloads. ``rouge_unavailable`` makes the approximation explicit."""
    pred = "" if pred is None else str(pred)
    reference = "" if reference is None else str(reference)
    if rouge_available():
        try:
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            s = sc.score(reference, pred)
            return {
                "rouge1": round(float(s["rouge1"].fmeasure), 6),
                "rouge2": round(float(s["rouge2"].fmeasure), 6),
                "rougeL": round(float(s["rougeL"].fmeasure), 6),
                "rouge_unavailable": False,
            }
        except Exception:                                    # noqa: BLE001
            pass
    pw, rw = _words(pred), _words(reference)
    return {
        "rouge1": round(_ngram_f1(pw, rw, 1), 6),
        "rouge2": round(_ngram_f1(pw, rw, 2), 6),
        # rougeL fallback == LCS-based token F1 (pllo.benchmarks.metrics.rouge_l)
        "rougeL": round(M.rouge_l(pred, reference), 6),
        "rouge_unavailable": True,
    }


def score_example(task_type: str, pred_text: str, reference,
                  token_ids) -> Dict[str, Any]:
    """Per-example objective metrics for one generation (task-type aware)."""
    tt = _norm_task(task_type)
    row: Dict[str, Any] = {
        "task_type": tt,
        "output_length_tokens": output_length_tokens(token_ids),
        "has_reference": reference is not None and str(reference) != "",
    }
    if tt == "generation_exact":
        row["extracted_number"] = extract_number(pred_text)
        row["numeric_exact_match"] = (numeric_exact_match(pred_text, reference)
                                      if row["has_reference"] else None)
    elif tt == "summarization":
        rs = rouge_scores(pred_text, reference or "")
        row.update(rs)
    else:  # open_ended
        if row["has_reference"]:
            row["exact_text_match"] = exact_text_match(pred_text, reference)
            row["normalized_edit_similarity"] = round(
                normalized_edit_similarity(pred_text, reference), 6)
        else:
            row["exact_text_match"] = None
            row["normalized_edit_similarity"] = None
    return row


# ---------------------------------------------------------------------------
# Deterministic stub (dry-run; never a paper result)
# ---------------------------------------------------------------------------


def stub_generate(example: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    """Deterministic, gold-blind stub generation for dry runs / tests."""
    p = str(example.get("prompt", "")).strip()
    text = ("gen: " + p)[: max(1, int(max_new_tokens) * 4)]
    token_ids = [ord(c) % 256 for c in text][: max(1, int(max_new_tokens))]
    return {"text": text, "token_ids": token_ids}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def _rate(flags) -> Optional[float]:
    flags = [f for f in flags if f is not None]
    return round(sum(1 for f in flags if f) / len(flags), 6) if flags else None


def _aggregate(task_type: str, rows: List[Dict[str, Any]]
               ) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """Return (primary_metric_name, primary_metric_value, extra_aggregate)."""
    tt = _norm_task(task_type)
    lengths = _mean([r.get("output_length_tokens") for r in rows])
    extra: Dict[str, Any] = {"mean_output_length_tokens": lengths,
                             "num_scored": len(rows)}
    if tt == "generation_exact":
        val = _rate([r.get("numeric_exact_match") for r in rows])
        return "numeric_exact_match", val, extra
    if tt == "summarization":
        extra["rouge1"] = _mean([r.get("rouge1") for r in rows])
        extra["rouge2"] = _mean([r.get("rouge2") for r in rows])
        extra["rougeL"] = _mean([r.get("rougeL") for r in rows])
        extra["rouge_unavailable"] = bool(rows) and any(
            r.get("rouge_unavailable") for r in rows)
        return "rougeL", extra["rougeL"], extra
    # open_ended
    extra["exact_text_match_rate"] = _rate(
        [r.get("exact_text_match") for r in rows])
    val = _mean([r.get("normalized_edit_similarity") for r in rows])
    return "normalized_edit_similarity", val, extra


# ---------------------------------------------------------------------------
# Per-backend runner
# ---------------------------------------------------------------------------


def _stat(stats, key, default=None):
    return stats.get(key, default) if isinstance(stats, dict) else default


def run_generation_utility_benchmark(examples: List[Dict[str, Any]], *,
                                     backend: str, predictor=None,
                                     model_name: str = "stub",
                                     nonlinear_backend: str = "current",
                                     seq_len: int = 256,
                                     max_new_tokens: int = 64,
                                     dataset_name: Optional[str] = None,
                                     task_type: Optional[str] = None,
                                     dry_run: Optional[bool] = None,
                                     max_records: int = 200) -> Dict[str, Any]:
    """Greedy generation over ``examples`` (single task_type) -> honest report."""
    nb = assert_current_only(nonlinear_backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("unknown backend %r (allowed: %s)"
                         % (backend, ", ".join(SUPPORTED_BACKENDS)))
    is_dry = (predictor is None) if dry_run is None else bool(dry_run)

    tt = _norm_task(task_type or (examples[0].get("task_type")
                                  if examples else "open_ended"))
    ds = dataset_name or (examples[0].get("dataset_name") if examples else None)

    gens: List[Dict[str, Any]] = []
    total_latency = 0.0
    for ex in examples:
        prompt = str(ex.get("prompt", ""))
        ex_tt = _norm_task(ex.get("task_type") or tt)
        ref = _reference_of(ex)
        t0 = time.perf_counter()
        g = predictor.generate(prompt) if predictor is not None \
            else stub_generate(ex, max_new_tokens)
        dt = time.perf_counter() - t0
        total_latency += dt
        text = "" if g.get("text") is None else str(g.get("text"))
        toks = g.get("token_ids")
        toks = None if toks is None else [int(t) for t in toks]
        row = {
            "id": ex.get("id"),
            "dataset_name": ex.get("dataset_name", ds),
            "category": ex.get("category", "uncategorized"),
            "text": text,
            "token_ids": toks,
            "reference": ref,
            "latency_s": (None if is_dry else round(dt, 6)),
        }
        row.update(score_example(ex_tt, text, ref, toks))
        gens.append(row)

    stats = {}
    if predictor is not None and hasattr(predictor, "stats"):
        try:
            stats = predictor.stats() or {}
        except Exception:                                    # noqa: BLE001
            stats = {}

    metric_name, metric_value, extra = _aggregate(tt, gens)
    n = len(gens)
    tokens_available = bool(n) and all(g["token_ids"] is not None for g in gens)
    report = {
        "stage": STAGE_BENCHMARK,
        "backend": backend,
        "nonlinear_backend": nb,
        "model_name": model_name,
        "dataset_name": ds,
        "task_type": tt,
        "decoding": "greedy",
        "num_examples": n,
        "seq_len": int(seq_len),
        "max_new_tokens": int(max_new_tokens),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "token_ids_available": tokens_available,
        "latency_s_total": (None if is_dry else round(total_latency, 6)),
        "latency_s_mean": (None if (is_dry or not n)
                           else round(total_latency / n, 6)),
        "audit_passed": _stat(stats, "audit_passed"),
        "tee_used_on_gpu": _stat(stats, "tee_used_on_gpu"),
        "worker_has_mask_secrets": _stat(stats, "worker_has_mask_secrets"),
        "gpu_visible_plaintext_fields": _stat(stats,
                                              "gpu_visible_plaintext_fields"),
        "leaked_secret_fields": _stat(stats, "leaked_secret_fields"),
        "dry_run": is_dry,
        "paper_ready": (not is_dry),
        "generations": gens[: max(0, int(max_records))],
        "num_generations_recorded": min(n, max(0, int(max_records))),
    }
    report.update(extra)
    return report


# ---------------------------------------------------------------------------
# Pairwise preservation (plaintext baseline vs folded candidate)
# ---------------------------------------------------------------------------


def pairwise_generation_preservation(baseline: Dict[str, Any],
                                     candidate: Dict[str, Any], *,
                                     max_abs_drop: float = 0.05,
                                     max_rel_drop: float = 0.10) -> Dict[str, Any]:
    """Compare plaintext baseline vs folded (current) candidate generation
    reports. ``utility_preserved`` requires the metric drop within thresholds,
    both reports real (paper_ready), and the candidate audit not failed."""
    nb = assert_current_only(candidate.get("nonlinear_backend") or "current")

    bm = baseline.get("metric_value")
    cm = candidate.get("metric_value")
    abs_drop = (None if (bm is None or cm is None)
                else round(float(bm) - float(cm), 6))
    rel_drop = (None if (abs_drop is None or not bm)
                else round(abs_drop / float(bm), 6))

    b_by_id = {g.get("id"): g for g in baseline.get("generations", [])}
    matched = 0
    exact_out = 0
    length_deltas: List[int] = []
    for cg in candidate.get("generations", []):
        bg = b_by_id.get(cg.get("id"))
        if bg is None:
            continue
        matched += 1
        if exact_text_match(bg.get("text"), cg.get("text")):
            exact_out += 1
        bl = bg.get("output_length_tokens")
        cl = cg.get("output_length_tokens")
        if bl is not None and cl is not None:
            length_deltas.append(cl - bl)
    exact_output_match_rate = (round(exact_out / matched, 6) if matched else None)
    length_delta_mean = _mean(length_deltas)

    b_lat = baseline.get("latency_s_mean")
    c_lat = candidate.get("latency_s_mean")
    latency_ratio = (round(c_lat / b_lat, 6)
                     if (b_lat and c_lat) else None)

    candidate_audit = candidate.get("audit_passed")
    audit_ok = candidate_audit is not False
    both_paper_ready = bool(baseline.get("paper_ready") is True
                            and candidate.get("paper_ready") is True
                            and baseline.get("dry_run") is not True
                            and candidate.get("dry_run") is not True)
    within = None
    if abs_drop is not None:
        within = bool(abs_drop <= max_abs_drop
                      and (rel_drop is None or rel_drop <= max_rel_drop))
    utility_preserved = bool(within) and audit_ok and both_paper_ready

    return {
        "stage": STAGE_PAIRWISE,
        "baseline_backend": baseline.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "nonlinear_backend": nb,
        "dataset_name": candidate.get("dataset_name")
        or baseline.get("dataset_name"),
        "task_type": candidate.get("task_type") or baseline.get("task_type"),
        "model_name": candidate.get("model_name") or baseline.get("model_name"),
        "decoding": "greedy",
        "seq_len": candidate.get("seq_len"),
        "max_new_tokens": candidate.get("max_new_tokens"),
        "metric_name": candidate.get("metric_name") or baseline.get("metric_name"),
        "baseline_metric": bm,
        "candidate_metric": cm,
        "metric_abs_drop": abs_drop,
        "metric_rel_drop": rel_drop,
        "max_abs_drop": max_abs_drop,
        "max_rel_drop": max_rel_drop,
        "within_threshold": within,
        "num_compared": matched,
        "exact_output_match_rate": exact_output_match_rate,
        "length_delta_mean": length_delta_mean,
        "baseline_latency_s_mean": b_lat,
        "candidate_latency_s_mean": c_lat,
        "latency_ratio": latency_ratio,
        "audit_passed": candidate_audit,
        "candidate_tee_used_on_gpu": candidate.get("tee_used_on_gpu"),
        "candidate_worker_has_mask_secrets": candidate.get(
            "worker_has_mask_secrets"),
        "utility_preserved": utility_preserved,
        "paper_ready": both_paper_ready,
        "dry_run": not both_paper_ready,
        "note": "utility_preserved requires the metric drop within abs/rel "
                "thresholds, both reports real (paper_ready), and the candidate "
                "audit not failed. No LLM judge / no subjective quality scoring.",
    }


def summarize_pairwise(pairwise_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine several pairwise reports into one final summary table."""
    rows = []
    for r in pairwise_reports:
        if not isinstance(r, dict):
            continue
        rows.append({
            "dataset_name": r.get("dataset_name"),
            "task_type": r.get("task_type"),
            "metric_name": r.get("metric_name"),
            "baseline_metric": r.get("baseline_metric"),
            "candidate_metric": r.get("candidate_metric"),
            "metric_abs_drop": r.get("metric_abs_drop"),
            "metric_rel_drop": r.get("metric_rel_drop"),
            "exact_output_match_rate": r.get("exact_output_match_rate"),
            "length_delta_mean": r.get("length_delta_mean"),
            "latency_ratio": r.get("latency_ratio"),
            "audit_passed": r.get("audit_passed"),
            "max_new_tokens": r.get("max_new_tokens"),
            "utility_preserved": r.get("utility_preserved"),
            "paper_ready": r.get("paper_ready"),
        })
    all_preserved = bool(rows) and all(x["utility_preserved"] is True
                                       for x in rows)
    all_paper_ready = bool(rows) and all(x["paper_ready"] is True for x in rows)
    return {
        "stage": STAGE_SUMMARY,
        "num_datasets": len(rows),
        "rows": rows,
        "all_utility_preserved": all_preserved,
        "all_paper_ready": all_paper_ready,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_benchmark_md(r: Dict[str, Any]) -> str:
    L = ["# Generation utility benchmark (%s / %s)"
         % (r.get("dataset_name"), r["backend"]), "",
         "- backend=`%s`  nonlinear_backend=`%s`  task_type=%s  decoding=%s"
         % (r["backend"], r["nonlinear_backend"], r["task_type"], r["decoding"]),
         "- model_name=`%s`  num_examples=%s  seq_len=%s  max_new_tokens=%s"
         % (r["model_name"], r["num_examples"], r["seq_len"],
            r["max_new_tokens"]),
         "- **metric `%s` = %s**  mean_output_length_tokens=%s  latency_s_mean=%s"
         % (r["metric_name"], r["metric_value"],
            r.get("mean_output_length_tokens"), r["latency_s_mean"])]
    if r["task_type"] == "summarization":
        L.append("- rouge1=%s rouge2=%s rougeL=%s rouge_unavailable=%s"
                 % (r.get("rouge1"), r.get("rouge2"), r.get("rougeL"),
                    r.get("rouge_unavailable")))
    L += ["- audit_passed=%s  tee_used_on_gpu=%s  worker_has_mask_secrets=%s"
          % (r["audit_passed"], r["tee_used_on_gpu"],
             r["worker_has_mask_secrets"]),
          "- **dry_run=%s  paper_ready=%s**" % (r["dry_run"], r["paper_ready"]),
          ""]
    return "\n".join(L)


def _csv_cols_for(task_type: str) -> List[str]:
    base = ["id", "dataset_name", "category", "output_length_tokens", "latency_s"]
    if task_type == "generation_exact":
        return base + ["numeric_exact_match", "extracted_number"]
    if task_type == "summarization":
        return base + ["rouge1", "rouge2", "rougeL", "rouge_unavailable"]
    return base + ["exact_text_match", "normalized_edit_similarity"]


def render_benchmark_csv(r: Dict[str, Any]) -> str:
    cols = _csv_cols_for(r["task_type"])
    lines = [",".join(cols)]
    for g in r["generations"]:
        lines.append(",".join(str(g.get(c)) for c in cols))
    return "\n".join(lines) + "\n"


def render_pairwise_md(r: Dict[str, Any]) -> str:
    return "\n".join([
        "# Generation pairwise preservation (%s vs %s)"
        % (r["candidate_backend"], r["baseline_backend"]), "",
        "- dataset=%s  task_type=%s  metric=`%s`  decoding=%s"
        % (r["dataset_name"], r["task_type"], r["metric_name"], r["decoding"]),
        "- nonlinear_backend=`%s`  max_new_tokens=%s  num_compared=%s"
        % (r["nonlinear_backend"], r["max_new_tokens"], r["num_compared"]),
        "- **dry_run=%s  paper_ready=%s**" % (r["dry_run"], r["paper_ready"]),
        "", "| field | value |", "| --- | --- |",
        "| baseline_metric | %s |" % r["baseline_metric"],
        "| candidate_metric | %s |" % r["candidate_metric"],
        "| metric_abs_drop | %s |" % r["metric_abs_drop"],
        "| metric_rel_drop | %s |" % r["metric_rel_drop"],
        "| exact_output_match_rate | %s |" % r["exact_output_match_rate"],
        "| length_delta_mean | %s |" % r["length_delta_mean"],
        "| latency_ratio | %s |" % r["latency_ratio"],
        "| audit_passed | %s |" % r["audit_passed"],
        "| within_threshold | %s |" % r["within_threshold"],
        "| **utility_preserved** | **%s** |" % r["utility_preserved"], ""])


def render_pairwise_csv(r: Dict[str, Any]) -> str:
    cols = ["dataset_name", "task_type", "metric_name", "baseline_metric",
            "candidate_metric", "metric_abs_drop", "metric_rel_drop",
            "exact_output_match_rate", "length_delta_mean", "latency_ratio",
            "audit_passed", "max_new_tokens", "utility_preserved", "paper_ready"]
    return ",".join(cols) + "\n" + ",".join(str(r.get(c)) for c in cols) + "\n"


def render_summary_md(s: Dict[str, Any]) -> str:
    L = ["# Generation preservation summary (current design)", "",
         "- datasets=%s  all_utility_preserved=%s  all_paper_ready=%s"
         % (s["num_datasets"], s["all_utility_preserved"], s["all_paper_ready"]),
         "", "| dataset | task | metric | base | cand | abs_drop | rel_drop | "
         "exact_out | len_delta | lat_ratio | audit | newtok | preserved |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
         "--- | --- |"]
    for x in s["rows"]:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
                 "%s |" % (x["dataset_name"], x["task_type"], x["metric_name"],
                          x["baseline_metric"], x["candidate_metric"],
                          x["metric_abs_drop"], x["metric_rel_drop"],
                          x["exact_output_match_rate"], x["length_delta_mean"],
                          x["latency_ratio"], x["audit_passed"],
                          x["max_new_tokens"], x["utility_preserved"]))
    L.append("")
    return "\n".join(L)


def render_summary_csv(s: Dict[str, Any]) -> str:
    cols = ["dataset_name", "task_type", "metric_name", "baseline_metric",
            "candidate_metric", "metric_abs_drop", "metric_rel_drop",
            "exact_output_match_rate", "length_delta_mean", "latency_ratio",
            "audit_passed", "max_new_tokens", "utility_preserved", "paper_ready"]
    lines = [",".join(cols)]
    for x in s["rows"]:
        lines.append(",".join(str(x.get(c)) for c in cols))
    return "\n".join(lines) + "\n"


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
