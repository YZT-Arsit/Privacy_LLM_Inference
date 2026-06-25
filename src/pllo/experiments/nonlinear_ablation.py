"""E16: nonlinear-design ablation -- current vs trusted_shortcut.

A focused ablation table that isolates what changes when the nonlinear design is
swapped from ``current`` (nonlinearity in the trusted boundary) to
``trusted_shortcut`` (nonlinearity migrated to the GPU with a small trusted
shortcut). Reuses the E15 metric extraction (:mod:`nonlinear_design_comparison`)
so the two reports stay consistent, then arranges the per-row deltas an ablation
table needs.

Rows: (1) design identity, (2) nonlinear_boundary_calls, (3)
trusted_bytes_due_to_nonlinear (trusted_bytes proxy), (4)
latency_overhead_due_to_nonlinear (decode latency delta), (5)
security_difference, (6) package_size_difference, (7)
lora_compatibility_difference.

stdlib only. Defensive against missing keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pllo.experiments.nonlinear_design_comparison import (
    build_comparison,
    load_json,  # re-exported for the CLI
)
from pllo.experiments.nonlinear_designs import normalize_nonlinear_backend

__all__ = [
    "load_json",
    "build_nonlinear_ablation",
    "render_md",
    "render_csv",
    "render_latex",
]

DEFAULT_BACKENDS = ["current", "trusted_shortcut"]


def _num_delta(a, b):
    """b - a when both are real numbers (not bool), else None."""
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)):
        return b - a
    return None


def _row(metric, a_val, b_val, *, backends, delta=None, note=None):
    return {
        "metric": metric,
        backends[0]: a_val,
        backends[1]: b_val,
        "delta": delta,
        "note": note,
    }


def build_nonlinear_ablation(reports_by_backend: dict, *,
                             backends: "list | None" = None) -> dict:
    if backends is None:
        backends = list(DEFAULT_BACKENDS)
    backends = [normalize_nonlinear_backend(b) for b in backends]
    if len(backends) != 2:
        raise ValueError("nonlinear ablation compares exactly two designs; got %r"
                         % backends)
    a, b = backends[0], backends[1]

    comp = build_comparison(reports_by_backend, backends=backends)
    sec = comp["security"]
    perf = comp["performance"]
    deploy = comp["deployment"]
    meta = comp["design_metadata"]

    def pa(key):
        return (perf.get(a) or {}).get(key)

    def pb(key):
        return (perf.get(b) or {}).get(key)

    rows: List[Dict[str, Any]] = []

    # (1) design identity
    rows.append(_row("design", a, b, backends=backends,
                     note="nonlinearity in trusted boundary (current) vs "
                          "migrated to GPU + trusted shortcut (trusted_shortcut)"))

    # (2) nonlinear_boundary_calls (boundary_calls proxy)
    rows.append(_row("nonlinear_boundary_calls", pa("boundary_calls"),
                     pb("boundary_calls"), backends=backends,
                     delta=_num_delta(pa("boundary_calls"),
                                      pb("boundary_calls")),
                     note="uses total boundary_calls as a proxy"))

    # (3) trusted_bytes_due_to_nonlinear (trusted_bytes proxy)
    rows.append(_row("trusted_bytes_due_to_nonlinear", pa("trusted_bytes"),
                     pb("trusted_bytes"), backends=backends,
                     delta=_num_delta(pa("trusted_bytes"), pb("trusted_bytes")),
                     note="trusted_bytes used as a proxy for nonlinear transfer"))

    # (4) latency_overhead_due_to_nonlinear (decode latency delta)
    rows.append(_row("latency_overhead_due_to_nonlinear", pa("decode_latency"),
                     pb("decode_latency"), backends=backends,
                     delta=_num_delta(pa("decode_latency"),
                                      pb("decode_latency")),
                     note="decode-latency delta (trusted_shortcut - current)"))

    # (5) security_difference
    sec_a = {
        "security_status": (meta.get(a) or {}).get("security_status"),
        "transcript_scan": (sec.get(a) or {}).get("transcript_scan"),
        "negative_tests": (sec.get(a) or {}).get("negative_tests"),
        "attestation_binding": (sec.get(a) or {}).get("attestation_binding"),
    }
    sec_b = {
        "security_status": (meta.get(b) or {}).get("security_status"),
        "transcript_scan": (sec.get(b) or {}).get("transcript_scan"),
        "negative_tests": (sec.get(b) or {}).get("negative_tests"),
        "attestation_binding": (sec.get(b) or {}).get("attestation_binding"),
    }
    rows.append(_row("security_difference", sec_a, sec_b, backends=backends,
                     note="security_status + transcript/negative/attestation; "
                          "trusted_shortcut is not_formally_claimed"))

    # (6) package_size_difference
    rows.append(_row("package_size_difference", pa("package_size"),
                     pb("package_size"), backends=backends,
                     delta=_num_delta(pa("package_size"), pb("package_size")),
                     note="folded_weight_size_gb delta"))

    # (7) lora_compatibility_difference
    rows.append(_row("lora_compatibility_difference",
                     (deploy.get(a) or {}).get("lora_supported"),
                     (deploy.get(b) or {}).get("lora_supported"),
                     backends=backends,
                     note="lora_supported per design (presence of qualifying "
                          "reports)"))

    # deltas summary (trusted_shortcut minus current) for numeric rows
    deltas_summary = {}
    for r in rows:
        if r["delta"] is not None:
            deltas_summary[r["metric"]] = r["delta"]

    limitations = [
        "trusted_bytes_due_to_nonlinear uses trusted_bytes as a PROXY for the "
        "nonlinear-specific transfer cost; it is not an isolated nonlinear "
        "measurement",
        "nonlinear_boundary_calls uses total boundary_calls as a proxy",
        "latency_overhead_due_to_nonlinear is a decode-latency delta; compare "
        "only same-token runs",
        "trusted_shortcut has security_status not_formally_claimed -- the "
        "security_difference row is qualitative, not a security recommendation",
        "deltas are None where either design lacks the metric in its provided "
        "reports",
    ]

    return {
        "stage": "e16_nonlinear_ablation",
        "backends": backends,
        "num_reports_by_backend": comp.get("num_reports_by_backend", {}),
        "rows": rows,
        "deltas_summary": deltas_summary,
        "design_metadata": meta,
        "limitations": limitations,
    }


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(";", ":"))
    return str(v)


def render_md(report: dict) -> str:
    backends = report.get("backends", ["current", "trusted_shortcut"])
    L = ["# E16 — Nonlinear design ablation (%s vs %s)"
         % (backends[0], backends[1]), "",
         "Reports per backend: %s" % json.dumps(
             report.get("num_reports_by_backend", {})), "",
         "| metric | %s | %s | delta | note |" % (backends[0], backends[1]),
         "| --- | --- | --- | --- | --- |"]
    for r in report.get("rows", []):
        L.append("| %s | %s | %s | %s | %s |" % (
            _fmt(r.get("metric")), _fmt(r.get(backends[0])),
            _fmt(r.get(backends[1])), _fmt(r.get("delta")),
            _fmt(r.get("note"))))
    L += ["", "## Deltas summary (%s - %s)" % (backends[1], backends[0]), ""]
    ds = report.get("deltas_summary", {})
    if ds:
        for k, v in ds.items():
            L.append("- %s: %s" % (k, _fmt(v)))
    else:
        L.append("- (no numeric deltas available)")
    L += ["", "## Limitations", ""]
    L += ["- %s" % x for x in report.get("limitations", [])]
    L.append("")
    return "\n".join(L)


def _csv_field(v):
    s = _fmt(v)
    if any(c in s for c in (",", '"', "\n")):
        s = '"' + s.replace('"', '""') + '"'
    return s


def render_csv(report: dict) -> str:
    backends = report.get("backends", ["current", "trusted_shortcut"])
    L = [",".join(["metric", backends[0], backends[1], "delta", "note"])]
    for r in report.get("rows", []):
        L.append(",".join(_csv_field(x) for x in (
            r.get("metric"), r.get(backends[0]), r.get(backends[1]),
            r.get("delta"), r.get("note"))))
    return "\n".join(L) + "\n"


def _tex_escape(s):
    s = str(s)
    for x, y in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}")):
        s = s.replace(x, y)
    return s


def render_latex(report: dict) -> str:
    backends = report.get("backends", ["current", "trusted_shortcut"])
    L = ["%% E16 nonlinear design ablation (%s vs %s)"
         % (backends[0], backends[1]),
         r"\begin{table}[h]", r"\centering",
         r"\caption{Nonlinear design ablation: %s vs %s}"
         % (_tex_escape(backends[0]), _tex_escape(backends[1])),
         r"\begin{tabular}{llll}", r"\hline",
         " & ".join(_tex_escape(h) for h in
                    ["Metric", backends[0], backends[1], "Delta"]) + r" \\",
         r"\hline"]
    for r in report.get("rows", []):
        L.append(" & ".join(_tex_escape(_fmt(x)) for x in (
            r.get("metric"), r.get(backends[0]), r.get(backends[1]),
            r.get("delta"))) + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)
