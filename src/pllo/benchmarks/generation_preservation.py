"""Open-ended generation preservation benchmark (protected vs plaintext).

Does the protected inference path (``folded_remote`` under the **current** design)
reproduce the plaintext model's open-ended generations under deterministic greedy
decoding? This is a *preservation* benchmark, not a quality benchmark: every
metric is an objective string/token comparison between the protected output and
the plaintext baseline -- there is NO LLM judge and NO subjective quality score.

Input: JSONL of ``{id, prompt, category}``. Same model / tokenizer / seq_len /
max_new_tokens / greedy decoding for both backends.

Per-backend report (one per backend): the captured generations + latency +
(folded) audit flags. Pairwise report (baseline=plaintext vs candidate=folded):

* ``exact_text_match``           -- generated strings identical;
* ``exact_token_match``          -- generated token-id sequences identical
                                    (only when both backends expose token ids);
* ``normalized_edit_similarity`` -- 1 - charLevenshtein / max(len) in [0, 1];
* ``output_length_delta``        -- candidate - baseline (chars and tokens);
* latency + ``audit_passed`` (folded) are carried from the per-backend reports.

Constraints baked in: **current design only** (``trusted_shortcut`` is refused
here), no downloads, no LLM judge, no subjective quality scoring. stdlib only --
the predictors (torch) are injected by the runner script, never imported here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "STAGE_BENCHMARK",
    "STAGE_PAIRWISE",
    "validate_example",
    "load_examples",
    "levenshtein",
    "normalized_edit_similarity",
    "exact_text_match",
    "exact_token_match",
    "stub_generate",
    "assert_current_only",
    "run_generation_benchmark",
    "compare_generation",
    "pairwise_generation_preservation",
    "render_benchmark_md",
    "render_benchmark_csv",
    "render_pairwise_md",
    "render_pairwise_csv",
    "load_json",
]

STAGE_BENCHMARK = "generation_preservation_benchmark"
STAGE_PAIRWISE = "generation_preservation_pairwise"

# Backends this preservation benchmark compares (current design only).
SUPPORTED_BACKENDS = ("plaintext_local", "folded_remote")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def validate_example(ex: Dict[str, Any]) -> List[str]:
    """Return a list of problems ([] == valid) for a ``{id, prompt, category}``
    example. ``category`` is optional (defaults to ``uncategorized``)."""
    problems: List[str] = []
    if not isinstance(ex, dict):
        return ["example is not an object"]
    if not str(ex.get("id", "")).strip():
        problems.append("missing non-empty 'id'")
    if not str(ex.get("prompt", "")).strip():
        problems.append("missing non-empty 'prompt'")
    return problems


def load_examples(path) -> List[Dict[str, Any]]:
    """Load one ``{id, prompt, category}`` example per non-empty JSONL line."""
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
            ex.setdefault("category", "uncategorized")
            out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Objective comparison metrics (no judge, no subjective scoring)
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein edit distance (insert/delete/substitute)."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost)  # substitution
        prev = cur
    return prev[len(b)]


def normalized_edit_similarity(a: str, b: str) -> float:
    """1 - charLevenshtein / max(len) in [0, 1]; 1.0 when both are empty."""
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - (levenshtein(a, b) / m)


def exact_text_match(a: str, b: str) -> bool:
    """Generated strings are byte-for-byte identical (no normalization)."""
    return ("" if a is None else str(a)) == ("" if b is None else str(b))


def exact_token_match(a: Optional[List[int]],
                      b: Optional[List[int]]) -> Optional[bool]:
    """Token-id sequences identical; None when either is unavailable."""
    if a is None or b is None:
        return None
    return [int(x) for x in a] == [int(x) for x in b]


# ---------------------------------------------------------------------------
# current-only guard
# ---------------------------------------------------------------------------


def assert_current_only(nonlinear_backend) -> str:
    """Refuse anything but the ``current`` design (no trusted_shortcut here)."""
    from pllo.experiments.nonlinear_designs import normalize_nonlinear_backend
    nb = normalize_nonlinear_backend(nonlinear_backend or "current")
    if nb != "current":
        raise ValueError(
            "generation preservation benchmark is CURRENT-ONLY; the "
            "trusted_shortcut design is not supported here (got %r). Run it "
            "with --nonlinear-backend current." % nb)
    return nb


# ---------------------------------------------------------------------------
# Deterministic stub generator (dry-run; never a paper result)
# ---------------------------------------------------------------------------


def stub_generate(example: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    """Deterministic, gold-blind stub generation for dry runs / tests.

    NOT a model output and NOT a quality signal -- a fixed transform of the
    prompt so both backends produce identical stub text (preservation trivially
    holds in dry-run, where the report is labeled paper_ready=False)."""
    p = str(example.get("prompt", "")).strip()
    text = ("gen: " + p)[: max(1, int(max_new_tokens) * 4)]
    token_ids = [ord(c) % 256 for c in text][: max(1, int(max_new_tokens))]
    return {"text": text, "token_ids": token_ids}


# ---------------------------------------------------------------------------
# Per-backend generation runner
# ---------------------------------------------------------------------------


def _stat(stats: Dict[str, Any], key, default=None):
    return stats.get(key, default) if isinstance(stats, dict) else default


def run_generation_benchmark(examples: List[Dict[str, Any]], *, backend: str,
                             predictor=None, model_name: str = "stub",
                             nonlinear_backend: str = "current",
                             seq_len: int = 256, max_new_tokens: int = 64,
                             dry_run: Optional[bool] = None,
                             max_records: int = 200) -> Dict[str, Any]:
    """Run greedy generation over ``examples`` and return an honest per-backend
    report. ``predictor`` must expose ``generate(prompt) -> {text, token_ids}``
    (and optionally ``stats()``); when None the deterministic stub is used and
    the report is ``dry_run=True, paper_ready=False``."""
    nb = assert_current_only(nonlinear_backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("unknown backend %r (allowed: %s)"
                         % (backend, ", ".join(SUPPORTED_BACKENDS)))
    is_dry = (predictor is None) if dry_run is None else bool(dry_run)

    gens: List[Dict[str, Any]] = []
    total_latency = 0.0
    for ex in examples:
        prompt = str(ex.get("prompt", ""))
        t0 = time.perf_counter()
        if predictor is not None:
            g = predictor.generate(prompt)
        else:
            g = stub_generate(ex, max_new_tokens)
        dt = time.perf_counter() - t0
        total_latency += dt
        text = "" if g.get("text") is None else str(g.get("text"))
        toks = g.get("token_ids")
        toks = None if toks is None else [int(t) for t in toks]
        gens.append({
            "id": ex.get("id"),
            "category": ex.get("category", "uncategorized"),
            "text": text,
            "token_ids": toks,
            "num_chars": len(text),
            "num_tokens": (None if toks is None else len(toks)),
            "latency_s": (None if is_dry else round(dt, 6)),
        })

    stats = {}
    if predictor is not None and hasattr(predictor, "stats"):
        try:
            stats = predictor.stats() or {}
        except Exception:                                    # noqa: BLE001
            stats = {}

    n = len(gens)
    tokens_available = bool(n) and all(g["token_ids"] is not None for g in gens)
    report = {
        "stage": STAGE_BENCHMARK,
        "backend": backend,
        "nonlinear_backend": nb,
        "model_name": model_name,
        "decoding": "greedy",
        "num_examples": n,
        "seq_len": int(seq_len),
        "max_new_tokens": int(max_new_tokens),
        "token_ids_available": tokens_available,
        "categories": sorted({g["category"] for g in gens}),
        "latency_s_total": (None if is_dry else round(total_latency, 6)),
        "latency_s_mean": (None if (is_dry or not n)
                           else round(total_latency / n, 6)),
        # security/audit flags (folded only; None / not-applicable for plaintext)
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
    return report


# ---------------------------------------------------------------------------
# Pairwise preservation (baseline=plaintext vs candidate=folded)
# ---------------------------------------------------------------------------


def compare_generation(baseline_gen: Dict[str, Any],
                       candidate_gen: Dict[str, Any]) -> Dict[str, Any]:
    """One per-example preservation row (objective metrics only)."""
    bt = "" if baseline_gen.get("text") is None else str(baseline_gen["text"])
    ct = "" if candidate_gen.get("text") is None else str(candidate_gen["text"])
    btok = baseline_gen.get("token_ids")
    ctok = candidate_gen.get("token_ids")
    b_nt = None if btok is None else len(btok)
    c_nt = None if ctok is None else len(ctok)
    return {
        "id": candidate_gen.get("id"),
        "category": candidate_gen.get("category",
                                      baseline_gen.get("category",
                                                       "uncategorized")),
        "exact_text_match": exact_text_match(bt, ct),
        "exact_token_match": exact_token_match(btok, ctok),
        "normalized_edit_similarity": round(
            normalized_edit_similarity(bt, ct), 6),
        "baseline_num_chars": len(bt),
        "candidate_num_chars": len(ct),
        "output_char_length_delta": len(ct) - len(bt),
        "baseline_num_tokens": b_nt,
        "candidate_num_tokens": c_nt,
        "output_token_length_delta": (None if (b_nt is None or c_nt is None)
                                      else c_nt - b_nt),
    }


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (round(sum(xs) / len(xs), 6) if xs else None)


def _rate(flags) -> Optional[float]:
    flags = [f for f in flags if f is not None]
    return (round(sum(1 for f in flags if f) / len(flags), 6) if flags else None)


def pairwise_generation_preservation(baseline_report: Dict[str, Any],
                                     candidate_report: Dict[str, Any], *,
                                     min_exact_token_match_rate: float = 0.95,
                                     min_edit_similarity: float = 0.95,
                                     min_exact_text_match_rate: float = 0.90
                                     ) -> Dict[str, Any]:
    """Compare a plaintext-baseline generation report to a folded candidate
    report, matching examples by ``id``. ``generation_preserved`` requires both
    reports paper_ready (real, not dry_run), the candidate audit not failed, and
    the preservation thresholds met (token-exact rate when token ids are
    available, else edit-similarity + text-exact rate)."""
    nb = assert_current_only(candidate_report.get("nonlinear_backend")
                             or "current")

    b_by_id = {g.get("id"): g for g in baseline_report.get("generations", [])}
    rows: List[Dict[str, Any]] = []
    for cg in candidate_report.get("generations", []):
        bg = b_by_id.get(cg.get("id"))
        if bg is None:
            continue
        rows.append(compare_generation(bg, cg))

    tokens_available = bool(rows) and all(
        r["exact_token_match"] is not None for r in rows)

    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        c = r["category"]
        by_cat.setdefault(c, {"rows": []})["rows"].append(r)
    by_category = {}
    for c, blk in by_cat.items():
        rs = blk["rows"]
        by_category[c] = {
            "count": len(rs),
            "exact_text_match_rate": _rate([r["exact_text_match"] for r in rs]),
            "exact_token_match_rate": _rate(
                [r["exact_token_match"] for r in rs]),
            "mean_normalized_edit_similarity": _mean(
                [r["normalized_edit_similarity"] for r in rs]),
        }

    sims = [r["normalized_edit_similarity"] for r in rows]
    exact_text_rate = _rate([r["exact_text_match"] for r in rows])
    exact_token_rate = _rate([r["exact_token_match"] for r in rows])
    aggregate = {
        "num_compared": len(rows),
        "exact_text_match_rate": exact_text_rate,
        "exact_token_match_rate": exact_token_rate,
        "mean_normalized_edit_similarity": _mean(sims),
        "min_normalized_edit_similarity": (round(min(sims), 6) if sims
                                           else None),
        "mean_output_char_length_delta": _mean(
            [r["output_char_length_delta"] for r in rows]),
        "mean_abs_output_char_length_delta": _mean(
            [abs(r["output_char_length_delta"]) for r in rows]),
        "mean_output_token_length_delta": _mean(
            [r["output_token_length_delta"] for r in rows]),
        "by_category": by_category,
    }

    both_paper_ready = bool(baseline_report.get("paper_ready") is True
                            and candidate_report.get("paper_ready") is True
                            and baseline_report.get("dry_run") is not True
                            and candidate_report.get("dry_run") is not True)
    candidate_audit = candidate_report.get("audit_passed")
    audit_ok = candidate_audit is not False        # None (n/a) or True both pass

    if tokens_available:
        thresholds_met = (exact_token_rate is not None
                          and exact_token_rate >= min_exact_token_match_rate)
    else:
        mean_sim = aggregate["mean_normalized_edit_similarity"]
        thresholds_met = (mean_sim is not None
                          and mean_sim >= min_edit_similarity
                          and exact_text_rate is not None
                          and exact_text_rate >= min_exact_text_match_rate)

    generation_preserved = bool(rows) and thresholds_met and audit_ok \
        and both_paper_ready

    return {
        "stage": STAGE_PAIRWISE,
        "baseline_backend": baseline_report.get("backend"),
        "candidate_backend": candidate_report.get("backend"),
        "nonlinear_backend": nb,
        "decoding": "greedy",
        "model_name": candidate_report.get("model_name")
        or baseline_report.get("model_name"),
        "seq_len": candidate_report.get("seq_len"),
        "max_new_tokens": candidate_report.get("max_new_tokens"),
        "num_baseline": len(baseline_report.get("generations", [])),
        "num_candidate": len(candidate_report.get("generations", [])),
        "token_ids_available": tokens_available,
        "rows": rows,
        "aggregate": aggregate,
        "candidate_audit_passed": candidate_audit,
        "candidate_tee_used_on_gpu": candidate_report.get("tee_used_on_gpu"),
        "candidate_worker_has_mask_secrets": candidate_report.get(
            "worker_has_mask_secrets"),
        "baseline_latency_s_mean": baseline_report.get("latency_s_mean"),
        "candidate_latency_s_mean": candidate_report.get("latency_s_mean"),
        "preservation_thresholds": {
            "min_exact_token_match_rate": min_exact_token_match_rate,
            "min_edit_similarity": min_edit_similarity,
            "min_exact_text_match_rate": min_exact_text_match_rate,
            "basis": "exact_token_match_rate" if tokens_available
            else "edit_similarity+exact_text_match_rate",
        },
        "thresholds_met": thresholds_met,
        "generation_preserved": generation_preserved,
        "paper_ready": both_paper_ready,
        "dry_run": not both_paper_ready,
        "note": "generation_preserved requires both reports real (paper_ready), "
                "the candidate audit not failed, and the preservation thresholds "
                "met. No LLM judge / no subjective quality scoring -- all metrics "
                "are objective string/token comparisons.",
    }


# ---------------------------------------------------------------------------
# Renderers (MD / CSV)
# ---------------------------------------------------------------------------


def render_benchmark_md(r: Dict[str, Any]) -> str:
    L = ["# Generation preservation benchmark (%s)" % r["backend"], "",
         "- backend=`%s`  nonlinear_backend=`%s`  decoding=%s"
         % (r["backend"], r["nonlinear_backend"], r["decoding"]),
         "- model_name=`%s`  num_examples=%s  seq_len=%s  max_new_tokens=%s"
         % (r["model_name"], r["num_examples"], r["seq_len"],
            r["max_new_tokens"]),
         "- token_ids_available=%s  latency_s_mean=%s"
         % (r["token_ids_available"], r["latency_s_mean"]),
         "- audit_passed=%s  tee_used_on_gpu=%s  worker_has_mask_secrets=%s"
         % (r["audit_passed"], r["tee_used_on_gpu"],
            r["worker_has_mask_secrets"]),
         "- **dry_run=%s  paper_ready=%s**" % (r["dry_run"], r["paper_ready"]),
         "", "## Sample generations", "",
         "| id | category | num_chars | num_tokens |",
         "| --- | --- | --- | --- |"]
    for g in r["generations"][:20]:
        L.append("| %s | %s | %s | %s |"
                 % (g["id"], g["category"], g["num_chars"], g["num_tokens"]))
    L.append("")
    return "\n".join(L)


def render_benchmark_csv(r: Dict[str, Any]) -> str:
    lines = ["id,category,num_chars,num_tokens,latency_s"]
    for g in r["generations"]:
        lines.append("%s,%s,%s,%s,%s" % (
            g["id"], g["category"], g["num_chars"], g["num_tokens"],
            g["latency_s"]))
    return "\n".join(lines) + "\n"


def render_pairwise_md(r: Dict[str, Any]) -> str:
    a = r["aggregate"]
    L = ["# Generation preservation (pairwise: %s vs %s)"
         % (r["candidate_backend"], r["baseline_backend"]), "",
         "- nonlinear_backend=`%s`  decoding=%s  num_compared=%s"
         % (r["nonlinear_backend"], r["decoding"], a["num_compared"]),
         "- token_ids_available=%s  candidate_audit_passed=%s"
         % (r["token_ids_available"], r["candidate_audit_passed"]),
         "- **dry_run=%s  paper_ready=%s**" % (r["dry_run"], r["paper_ready"]),
         "", "## Aggregate", "", "| metric | value |", "| --- | --- |",
         "| exact_text_match_rate | %s |" % a["exact_text_match_rate"],
         "| exact_token_match_rate | %s |" % a["exact_token_match_rate"],
         "| mean_normalized_edit_similarity | %s |"
         % a["mean_normalized_edit_similarity"],
         "| min_normalized_edit_similarity | %s |"
         % a["min_normalized_edit_similarity"],
         "| mean_output_char_length_delta | %s |"
         % a["mean_output_char_length_delta"],
         "| mean_output_token_length_delta | %s |"
         % a["mean_output_token_length_delta"],
         "| **generation_preserved** | **%s** |" % r["generation_preserved"],
         "", "## By category", "",
         "| category | count | exact_text | exact_token | mean_edit_sim |",
         "| --- | --- | --- | --- | --- |"]
    for c, blk in sorted(a["by_category"].items()):
        L.append("| %s | %s | %s | %s | %s |"
                 % (c, blk["count"], blk["exact_text_match_rate"],
                    blk["exact_token_match_rate"],
                    blk["mean_normalized_edit_similarity"]))
    L.append("")
    return "\n".join(L)


def render_pairwise_csv(r: Dict[str, Any]) -> str:
    cols = ["id", "category", "exact_text_match", "exact_token_match",
            "normalized_edit_similarity", "output_char_length_delta",
            "output_token_length_delta", "baseline_num_chars",
            "candidate_num_chars"]
    lines = [",".join(cols)]
    for row in r["rows"]:
        lines.append(",".join(str(row.get(c)) for c in cols))
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
