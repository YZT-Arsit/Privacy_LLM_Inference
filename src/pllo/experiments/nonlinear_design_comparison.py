"""E15: dual-design comparison across the two nonlinear designs.

The advisor requires that the *same* full experiment suite run under BOTH
nonlinear designs (``current`` / ``trusted_shortcut``) so either can be chosen
later with complete results. This module consolidates the already-produced
per-design experiment report JSONs into five comparison tables -- correctness,
security, performance, deployment, recommendation -- plus a Markdown renderer.

CRITICAL honesty rule: a concrete recommendation is emitted for an axis only
when BOTH backends have COMPLETE evidence for that axis. If any required
evidence is missing, ``recommendation_status="insufficient_evidence"`` and
``missing_evidence`` lists what is missing. A design whose security_status is
``not_formally_claimed`` (the ``trusted_shortcut`` design) is never recommended
for security over ``current`` unless its security evidence is complete AND
favorable -- be conservative.

stdlib only. Defensive: any key may be missing -- ``_g`` returns ``None``; the
comparison never crashes on missing keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pllo.experiments.nonlinear_designs import (
    list_nonlinear_backends,
    nonlinear_backend_metadata,
    normalize_nonlinear_backend,
)

__all__ = [
    "load_json",
    "classify_reports",
    "build_comparison",
    "render_md",
]

DEFAULT_BACKENDS = ["current", "trusted_shortcut"]


def load_json(path: "str | Path | None") -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _first(reports, *keys):
    """First non-None value of any of ``keys`` over a list of report dicts."""
    for r in reports or []:
        if not isinstance(r, dict):
            continue
        for k in keys:
            v = r.get(k)
            if v is not None:
                return v
    return None


def _stage_of(report) -> str:
    return str(_g(report, "stage") or "")


# ---------------------------------------------------------------------------
# Classification of a backend's reports into buckets by stage/fields
# ---------------------------------------------------------------------------


def classify_reports(reports: List[dict]) -> Dict[str, List[dict]]:
    """Bucket a backend's reports by what they measure.

    Buckets: decode, build, pairwise_utility, lora_utility, security_negative,
    security_transcript, attestation, setup_cost, public_benchmark, other.
    A report can land in multiple buckets when several signals apply.
    """
    buckets: Dict[str, List[dict]] = {
        "decode": [], "build": [], "pairwise_utility": [], "lora_utility": [],
        "security_negative": [], "security_transcript": [], "attestation": [],
        "setup_cost": [], "public_benchmark": [], "other": [],
    }
    for r in reports or []:
        if not isinstance(r, dict):
            continue
        stage = _stage_of(r).lower()
        placed = False
        if ("folded_package_build" in stage
                or ("folded_lora" in stage and "build" in stage)
                or r.get("folded_weight_size_gb") is not None
                or r.get("folded_weight_generation_time_s") is not None):
            buckets["build"].append(r)
            placed = True
        if "pairwise" in stage and "utility" in stage:
            buckets["pairwise_utility"].append(r)
            placed = True
        if "e10" in stage or "lora_utility" in stage:
            buckets["lora_utility"].append(r)
            placed = True
        if "negative" in stage:
            buckets["security_negative"].append(r)
            placed = True
        if "transcript" in stage:
            buckets["security_transcript"].append(r)
            placed = True
        if "setup_cost" in stage or "e4" in stage:
            buckets["setup_cost"].append(r)
            placed = True
        if "task_utility_benchmark" in stage or "public_benchmark" in stage \
                or "aggregate_utility" in stage:
            buckets["public_benchmark"].append(r)
            placed = True
        # decode-ish reports carry latency/token/boundary signals
        if (r.get("tokens_exact_match") is not None
                or r.get("latency_s") is not None
                or r.get("boundary_calls") is not None
                or "protocol" in stage or "decode" in stage
                or "remote_folded" in stage):
            buckets["decode"].append(r)
            placed = True
        # attestation signals can ride on decode reports too
        if (r.get("runtime_hash_bound") is not None
                or r.get("boundary_attested") is not None
                or _g(r, "attestation") is not None):
            buckets["attestation"].append(r)
            placed = True
        if not placed:
            buckets["other"].append(r)
    return buckets


# ---------------------------------------------------------------------------
# Per-table metric extraction (returns one dict per backend; None where absent)
# ---------------------------------------------------------------------------


def _correctness_for(reports, buckets) -> Dict[str, Any]:
    decode = buckets["decode"]
    pairwise = buckets["pairwise_utility"]
    lora = buckets["lora_utility"]
    return {
        "operator_allclose": _first(decode, "operator_allclose", "allclose"),
        "logits_error": _first(decode, "logits_error", "logits_mae",
                               "max_abs_error"),
        "token_exact_match": _first(decode, "tokens_exact_match",
                                    "token_exact_match"),
        "task_utility_delta": _first(pairwise, "delta_abs", "task_utility_delta"),
        "lora_utility_delta": _first(lora, "delta_abs", "lora_utility_delta"),
    }


def _count_truthy(report, key):
    v = _g(report, key)
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if v is None:
        return None
    return 1 if v else 0


def _security_for(reports, buckets) -> Dict[str, Any]:
    decode = buckets["decode"]
    neg = buckets["security_negative"]
    trans = buckets["security_transcript"]
    att = buckets["attestation"]

    gpu_plain = _first(decode, "gpu_visible_plaintext_fields")
    leaked = _first(decode, "leaked_secret_fields")

    transcript_pass = None
    if trans:
        # transcript scan reports use {"fail": bool}; pass == not fail
        f = _first(trans, "fail")
        if f is not None:
            transcript_pass = (f is False)
        else:
            p = _first(trans, "pass", "passed")
            transcript_pass = bool(p) if p is not None else None

    neg_all = _first(neg, "all_passed", "passed")

    return {
        "gpu_visible_plaintext_fields": _count_truthy(
            {"gpu_visible_plaintext_fields": gpu_plain},
            "gpu_visible_plaintext_fields"),
        "leaked_secret_fields": _count_truthy(
            {"leaked_secret_fields": leaked}, "leaked_secret_fields"),
        "worker_has_mask_secrets": _first(decode, "worker_has_mask_secrets"),
        "worker_has_raw_lora": _first(decode, "worker_has_raw_lora"),
        "transcript_scan": transcript_pass,
        "negative_tests": (bool(neg_all) if neg_all is not None else None),
        "attestation_binding": _first(att, "runtime_hash_bound"),
    }


def _performance_for(reports, buckets) -> Dict[str, Any]:
    decode = buckets["decode"]
    build = buckets["build"]
    setup = buckets["setup_cost"]

    prefill = _first(decode, "prefill_latency", "prefill_latency_s")
    decode_lat = _first(decode, "decode_latency", "decode_latency_s")
    total_lat = _first(decode, "latency_s")
    per_tok = _first(decode, "latency_per_token", "latency_per_token_s")
    if per_tok is None:
        # derive from latency_s / max_new_tokens when possible
        for r in decode:
            lat = _g(r, "latency_s")
            mnt = _g(r, "max_new_tokens")
            if isinstance(lat, (int, float)) and isinstance(mnt, (int, float)) \
                    and mnt:
                per_tok = lat / mnt
                break

    return {
        "prefill_latency": prefill,
        "decode_latency": decode_lat if decode_lat is not None else total_lat,
        "latency_per_token": per_tok,
        "trusted_bytes": _first(decode, "trusted_bytes"),
        "gpu_bytes": _first(decode, "gpu_bytes"),
        "boundary_calls": _first(decode, "boundary_calls"),
        "peak_gpu_memory": _first(decode, "peak_gpu_memory_mb",
                                  "peak_gpu_memory"),
        "setup_build_time": _first(build, "folded_weight_generation_time_s",
                                   "setup_build_time"),
        "package_size": _first(build, "folded_weight_size_gb", "package_size"),
        "amortized_setup_cost": _first(setup, "amortized_setup_cost"),
    }


def _deployment_for(reports, buckets) -> Dict[str, Any]:
    decode = buckets["decode"]

    def _any(pred):
        return any(pred(r) for r in reports if isinstance(r, dict))

    tdx_lite = _any(lambda r: _g(r, "boundary_mode") == "lite"
                    or "tdx_lite" in _stage_of(r).lower())
    tdx_attested = _any(lambda r: _g(r, "boundary_attested") is True
                        or _g(r, "runtime_hash_bound") is True)
    remote_h800 = _any(lambda r: _g(r, "gpu_worker_remote") is True
                       or "remote" in _stage_of(r).lower())
    lora = _any(lambda r: _g(r, "lora_enabled") is True
                or "lora" in _stage_of(r).lower())
    public_bench = bool(buckets["public_benchmark"])

    return {
        "tdx_lite_supported": tdx_lite,
        "tdx_attested_supported": tdx_attested,
        "remote_h800_supported": remote_h800,
        "lora_supported": lora,
        "public_benchmark_supported": public_bench,
    }


# ---------------------------------------------------------------------------
# Recommendation (only with complete evidence on BOTH backends per axis)
# ---------------------------------------------------------------------------


def _present(v) -> bool:
    return v is not None


def _missing_for_axis(tables, backends, axis_keys, table_name):
    """Return list of '<backend>: <table>.<key> missing' for absent metrics."""
    missing = []
    for b in backends:
        row = tables[table_name].get(b, {}) or {}
        for k in axis_keys:
            if not _present(row.get(k)):
                missing.append("%s: %s.%s missing" % (b, table_name, k))
    return missing


def _lower_better_winner(perf, backends, key):
    """Backend with the smaller numeric value (lower==better)."""
    vals = {}
    for b in backends:
        v = (perf.get(b) or {}).get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals[b] = v
    if len(vals) < len(backends):
        return None
    return min(vals, key=vals.get)


def _build_recommendation(tables, backends) -> Dict[str, Any]:
    correctness = tables["correctness"]
    security = tables["security"]
    performance = tables["performance"]

    missing: List[str] = []
    per_axis: Dict[str, Any] = {}

    # --- correctness axis ---
    corr_keys = ["operator_allclose", "token_exact_match"]
    corr_missing = _missing_for_axis(tables, backends, corr_keys, "correctness")
    if corr_missing:
        missing.extend(corr_missing)
        per_axis["correctness"] = None
    else:
        # both correct (allclose truthy + token match truthy) -> tie ("either")
        ok = all(
            bool((correctness.get(b) or {}).get("operator_allclose"))
            and bool((correctness.get(b) or {}).get("token_exact_match"))
            for b in backends)
        per_axis["correctness"] = "either_exact" if ok else "neither_complete"

    # --- security axis ---
    sec_keys = ["transcript_scan", "negative_tests", "attestation_binding",
                "worker_has_mask_secrets"]
    sec_missing = _missing_for_axis(tables, backends, sec_keys, "security")
    if sec_missing:
        missing.extend(sec_missing)
        per_axis["security"] = None
    else:
        # be conservative: a not_formally_claimed design is never recommended for
        # security unless its evidence is complete AND favorable.
        favorable = {}
        for b in backends:
            row = security.get(b) or {}
            favorable[b] = (
                row.get("transcript_scan") is True
                and row.get("negative_tests") is True
                and row.get("attestation_binding") is True
                and row.get("worker_has_mask_secrets") is False)
        claim_status = {}
        for b in backends:
            try:
                claim_status[b] = nonlinear_backend_metadata(b).get(
                    "security_status")
            except Exception:                                # noqa: BLE001
                claim_status[b] = None
        # prefer a formally-claimed favorable design; never prefer a
        # not_formally_claimed design over a formally-claimed one for security.
        formally = [b for b in backends
                    if claim_status.get(b) != "not_formally_claimed"
                    and favorable.get(b)]
        if formally:
            per_axis["security"] = formally[0]
        else:
            per_axis["security"] = "insufficient_for_security_recommendation"

    # --- latency axis ---
    lat_keys = ["decode_latency"]
    lat_missing = _missing_for_axis(tables, backends, lat_keys, "performance")
    if lat_missing:
        missing.extend(lat_missing)
        per_axis["latency"] = None
    else:
        per_axis["latency"] = _lower_better_winner(
            performance, backends, "decode_latency")

    # --- trusted_transfer axis ---
    tt_keys = ["trusted_bytes"]
    tt_missing = _missing_for_axis(tables, backends, tt_keys, "performance")
    if tt_missing:
        missing.extend(tt_missing)
        per_axis["trusted_transfer"] = None
    else:
        per_axis["trusted_transfer"] = _lower_better_winner(
            performance, backends, "trusted_bytes")

    if missing:
        return {
            "recommendation_status": "insufficient_evidence",
            "missing_evidence": missing,
            "per_axis_winners": per_axis,
            "final_recommendation": None,
            "rationale": ("Incomplete evidence on at least one axis for at least "
                          "one design; no recommendation is synthesized."),
        }

    # all axes complete: produce a conservative final recommendation
    sec_winner = per_axis.get("security")
    lat_winner = per_axis.get("latency")
    if sec_winner and sec_winner not in (
            "insufficient_for_security_recommendation",):
        final = sec_winner
        rationale = ("%s has complete + favorable security evidence and a "
                     "formally-claimed boundary; recommend it where security is "
                     "the priority (latency winner: %s)." % (sec_winner,
                                                             lat_winner))
    else:
        final = lat_winner
        rationale = ("Security evidence does not favor a not_formally_claimed "
                     "design; defaulting the recommendation to the latency "
                     "winner %s (correctness is exact for both)." % lat_winner)
    return {
        "recommendation_status": "ok",
        "missing_evidence": [],
        "per_axis_winners": per_axis,
        "final_recommendation": final,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_comparison(reports_by_backend: dict, *,
                     backends: "list | None" = None) -> dict:
    reports_by_backend = reports_by_backend if isinstance(
        reports_by_backend, dict) else {}

    if backends is None:
        backends = list(DEFAULT_BACKENDS)
    backends = [normalize_nonlinear_backend(b) for b in backends]

    # normalize the report-map keys to canonical names + merge
    norm_map: Dict[str, List[dict]] = {b: [] for b in backends}
    for raw_key, reps in reports_by_backend.items():
        try:
            canon = normalize_nonlinear_backend(raw_key)
        except Exception:                                    # noqa: BLE001
            continue
        if canon not in norm_map:
            norm_map[canon] = []
        for r in (reps or []):
            if isinstance(r, dict):
                norm_map[canon].append(r)

    classified = {b: classify_reports(norm_map.get(b, [])) for b in backends}

    correctness = {b: _correctness_for(norm_map.get(b, []), classified[b])
                   for b in backends}
    security = {b: _security_for(norm_map.get(b, []), classified[b])
                for b in backends}
    performance = {b: _performance_for(norm_map.get(b, []), classified[b])
                   for b in backends}
    deployment = {b: _deployment_for(norm_map.get(b, []), classified[b])
                  for b in backends}

    tables = {
        "correctness": correctness,
        "security": security,
        "performance": performance,
        "deployment": deployment,
    }

    # If any backend has zero reports, recommendation is insufficient_evidence by
    # construction (its metrics will all be None) -- handled by _build_recommendation,
    # but record a clear missing-evidence line too.
    recommendation = _build_recommendation(tables, backends)
    for b in backends:
        if not norm_map.get(b):
            line = "%s: no reports provided" % b
            if recommendation["recommendation_status"] != "ok":
                if line not in recommendation["missing_evidence"]:
                    recommendation["missing_evidence"].insert(0, line)

    limitations = [
        "tables are only as honest as the per-design input reports; dry-run or "
        "fixture-derived reports are NOT paper evidence",
        "a metric is None when no provided report for that design carried it",
        "no recommendation is synthesized for an axis unless BOTH designs have "
        "complete evidence for that axis",
        "the trusted_shortcut design has security_status not_formally_claimed; it "
        "is never recommended for security over current unless its security "
        "evidence is complete AND favorable",
        "trusted_bytes_due_to_nonlinear uses trusted_bytes as a proxy for the "
        "nonlinear transfer cost",
    ]

    report = {
        "stage": "e15_nonlinear_design_comparison",
        "backends": backends,
        "num_reports_by_backend": {b: len(norm_map.get(b, []))
                                   for b in backends},
        "design_metadata": {b: {
            "security_status": _safe_meta(b, "security_status"),
            "security_claim_status": _safe_meta(b, "security_claim_status"),
            "design_label": _safe_meta(b, "design_label"),
        } for b in backends},
        "correctness": correctness,
        "security": security,
        "performance": performance,
        "deployment": deployment,
        "recommendation": recommendation,
        "limitations": limitations,
    }
    return report


def _safe_meta(backend, key):
    try:
        return nonlinear_backend_metadata(backend).get(key)
    except Exception:                                        # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    return str(v)


def _table_block(title, table, backends, rows):
    L = ["## %s" % title, "",
         "| metric | " + " | ".join(backends) + " |",
         "| --- | " + " | ".join("---" for _ in backends) + " |"]
    for key in rows:
        cells = [_fmt((table.get(b) or {}).get(key)) for b in backends]
        L.append("| %s | %s |" % (key, " | ".join(cells)))
    L.append("")
    return L


def render_md(report: dict) -> str:
    backends = report.get("backends", [])
    L = ["# E15 — Nonlinear design comparison (dual-design)", "",
         "Backends: %s" % ", ".join(backends),
         "Reports per backend: %s" % json.dumps(
             report.get("num_reports_by_backend", {})), ""]

    L += _table_block("1. Correctness", report.get("correctness", {}), backends,
                      ["operator_allclose", "logits_error", "token_exact_match",
                       "task_utility_delta", "lora_utility_delta"])
    L += _table_block("2. Security", report.get("security", {}), backends,
                      ["gpu_visible_plaintext_fields", "leaked_secret_fields",
                       "worker_has_mask_secrets", "worker_has_raw_lora",
                       "transcript_scan", "negative_tests",
                       "attestation_binding"])
    L += _table_block("3. Performance", report.get("performance", {}), backends,
                      ["prefill_latency", "decode_latency", "latency_per_token",
                       "trusted_bytes", "gpu_bytes", "boundary_calls",
                       "peak_gpu_memory", "setup_build_time", "package_size",
                       "amortized_setup_cost"])
    L += _table_block("4. Deployment", report.get("deployment", {}), backends,
                      ["tdx_lite_supported", "tdx_attested_supported",
                       "remote_h800_supported", "lora_supported",
                       "public_benchmark_supported"])

    rec = report.get("recommendation", {}) or {}
    L += ["## 5. Recommendation", "",
          "- recommendation_status: %s" % rec.get("recommendation_status"),
          "- final_recommendation: %s" % _fmt(rec.get("final_recommendation")),
          "- rationale: %s" % (rec.get("rationale") or ""), ""]
    per_axis = rec.get("per_axis_winners") or {}
    if per_axis:
        L += ["Per-axis winners:", ""]
        for axis, w in per_axis.items():
            L.append("- %s: %s" % (axis, _fmt(w)))
        L.append("")
    if rec.get("missing_evidence"):
        L += ["Missing evidence:", ""]
        L += ["- %s" % m for m in rec["missing_evidence"]]
        L.append("")

    L += ["## Limitations", ""]
    L += ["- %s" % x for x in report.get("limitations", [])]
    L.append("")
    return "\n".join(L)
