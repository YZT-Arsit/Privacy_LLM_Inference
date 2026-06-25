"""E14 ablation report -- correctness/security/performance DELTAS across axes.

Given a set of already-produced experiment report JSONs (decode reports, folded-
package build reports, LoRA decode reports, attested vs non-attested decode
reports, and a max_new_tokens scaling sweep), compute the per-axis DELTAS that an
ablation table needs. Each axis is only computed when its required inputs are
present; otherwise it is marked ``{"available": False, "reason": ...}``.

Honest-labeling discipline: deltas are only as honest as the inputs. The report
is ``paper_ready`` only when EVERY provided input report is itself paper_ready and
not a dry_run. Fixture- or dry-run-derived deltas are NEVER paper_ready.

stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build_ablation_report", "render_md", "load_json"]


def load_json(path):
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


def _num_delta(a, b):
    """b - a when both are numbers, else None."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return None


def _both_present(*reports):
    return all(isinstance(r, dict) for r in reports)


def _report_paper_ready(r):
    """A single input report is paper-honest iff paper_ready True and not dry_run."""
    return _g(r, "paper_ready") is True and _g(r, "dry_run") is not True


# ---------------------------------------------------------------------------
# individual ablation axes
# ---------------------------------------------------------------------------


def _axis_boundary(full_ref, lite):
    if not _both_present(full_ref, lite):
        return {"available": False,
                "reason": "requires both full-reference and lite decode reports"}
    metrics = ("tokens_exact_match", "latency_s", "trusted_bytes", "gpu_bytes",
               "boundary_calls")
    deltas = {}
    for m in metrics:
        a = _g(full_ref, m)
        b = _g(lite, m)
        deltas[m] = {"full_reference": a, "lite": b, "delta": _num_delta(a, b)}
    same_tokens = (_g(full_ref, "tokens_exact_match")
                   == _g(lite, "tokens_exact_match"))
    return {
        "available": True,
        "axis": "boundary_full_reference_vs_lite",
        "deltas": deltas,
        "tokens_equivalent": same_tokens,
        "limitations": (
            "lite boundary does NOT hold the full security argument; latency/byte "
            "savings come at a weaker boundary -- compare only same-token runs."),
    }


def _axis_storage(f32, bf16):
    if not _both_present(f32, bf16):
        return {"available": False,
                "reason": "requires both F32 and BF16 folded-build reports "
                          "(artifacts must exist)"}
    metrics = ("folded_weight_size_gb",)
    deltas = {}
    for m in metrics:
        a = _g(f32, m)
        b = _g(bf16, m)
        deltas[m] = {"f32": a, "bf16": b, "delta": _num_delta(a, b)}
    # optional correctness / probe deltas if the build reports carry them
    for m in ("probe_max_abs_err", "tokens_exact_match", "folded_package_valid"):
        a = _g(f32, m)
        b = _g(bf16, m)
        if a is not None or b is not None:
            deltas[m] = {"f32": a, "bf16": b, "delta": _num_delta(a, b)}
    return {
        "available": True,
        "axis": "folded_storage_f32_vs_bf16",
        "deltas": deltas,
        "limitations": (
            "BF16 trades folded-storage size for reduced precision; verify the "
            "correctness/probe deltas before claiming the smaller package."),
    }


def _axis_lora_rankmask(on, off, *, safe_fixture_mode):
    if not safe_fixture_mode:
        return {"available": False,
                "reason": "requires safe fixture mode"}
    if not _both_present(on, off):
        return {"available": False,
                "reason": "requires both rank-mask-on and rank-mask-off reports"}
    metrics = ("tokens_exact_match", "latency_s", "trusted_bytes", "gpu_bytes",
               "rank", "padded_rank")
    deltas = {}
    for m in metrics:
        a = _g(off, m)
        b = _g(on, m)
        deltas[m] = {"rankmask_off": a, "rankmask_on": b, "delta": _num_delta(a, b)}
    return {
        "available": True,
        "axis": "lora_rankmask_on_vs_off",
        "safe_fixture_mode": True,
        "deltas": deltas,
        "limitations": (
            "safe-fixture only: these are fixture deltas, NOT a real-adapter "
            "security measurement; rank-mask off LEAKS the true LoRA rank."),
    }


def _axis_attested(attested, nonattested):
    if not _both_present(attested, nonattested):
        return {"available": False,
                "reason": "requires both attested and non-attested decode reports"}
    metrics = ("latency_s", "trusted_bytes", "gpu_bytes", "boundary_calls")
    deltas = {}
    for m in metrics:
        a = _g(nonattested, m)
        b = _g(attested, m)
        deltas[m] = {"non_attested": a, "attested": b, "delta": _num_delta(a, b)}
    same_tokens = (_g(attested, "tokens_exact_match")
                   == _g(nonattested, "tokens_exact_match"))
    return {
        "available": True,
        "axis": "attested_vs_non_attested",
        "deltas": deltas,
        "tokens_equivalent": same_tokens,
        "attestation_overhead_latency_s": _num_delta(
            _g(nonattested, "latency_s"), _g(attested, "latency_s")),
        "boundary_attested": {
            "attested": _g(attested, "boundary_attested"),
            "non_attested": _g(nonattested, "boundary_attested")},
        "runtime_hash_bound": {
            "attested": _g(attested, "runtime_hash_bound"),
            "non_attested": _g(nonattested, "runtime_hash_bound")},
        "limitations": (
            "attestation adds quote/verification overhead; compare only same-token "
            "runs -- the attested run must actually carry boundary_attested."),
    }


def _axis_max_new_tokens(reports):
    reports = [r for r in (reports or []) if isinstance(r, dict)]
    if len(reports) < 2:
        return {"available": False,
                "reason": "requires >=2 decode reports at different max_new_tokens"}
    rows = []
    for r in reports:
        mnt = _g(r, "max_new_tokens")
        lat = _g(r, "latency_s")
        tb = _g(r, "trusted_bytes")
        per_tok_lat = (lat / mnt) if (isinstance(lat, (int, float))
                                      and isinstance(mnt, (int, float))
                                      and mnt) else None
        per_tok_tb = (tb / mnt) if (isinstance(tb, (int, float))
                                    and isinstance(mnt, (int, float))
                                    and mnt) else None
        rows.append({
            "max_new_tokens": mnt,
            "latency_s": lat,
            "trusted_bytes": tb,
            "latency_per_token_s": per_tok_lat,
            "trusted_bytes_per_token": per_tok_tb,
            "tokens_exact_match": _g(r, "tokens_exact_match"),
        })
    rows.sort(key=lambda x: (x["max_new_tokens"] is None, x["max_new_tokens"]))
    return {
        "available": True,
        "axis": "max_new_tokens_scaling",
        "rows": rows,
        "limitations": (
            "per-token trends assume a clean prefill/decode split; small "
            "max_new_tokens values are dominated by fixed prefill cost."),
    }


# ---------------------------------------------------------------------------
# top-level builder
# ---------------------------------------------------------------------------


def build_ablation_report(inputs: dict, *, nonlinear_backend=None) -> dict:
    inputs = inputs if isinstance(inputs, dict) else {}
    safe_fixture_mode = bool(inputs.get("safe_fixture_mode"))

    axes = {
        "boundary_full_reference_vs_lite": _axis_boundary(
            inputs.get("full_reference_decode"), inputs.get("lite_decode")),
        "folded_storage_f32_vs_bf16": _axis_storage(
            inputs.get("f32_build"), inputs.get("bf16_build")),
        "lora_rankmask_on_vs_off": _axis_lora_rankmask(
            inputs.get("lora_rankmask_on"), inputs.get("lora_rankmask_off"),
            safe_fixture_mode=safe_fixture_mode),
        "attested_vs_non_attested": _axis_attested(
            inputs.get("attested_decode"), inputs.get("nonattested_decode")),
        "max_new_tokens_scaling": _axis_max_new_tokens(
            inputs.get("max_new_tokens_decode")),
    }

    # gather every provided input report for the paper_ready gate
    provided = []
    for key in ("full_reference_decode", "lite_decode", "f32_build", "bf16_build",
                "lora_rankmask_on", "lora_rankmask_off", "attested_decode",
                "nonattested_decode"):
        r = inputs.get(key)
        if isinstance(r, dict):
            provided.append(r)
    for r in (inputs.get("max_new_tokens_decode") or []):
        if isinstance(r, dict):
            provided.append(r)

    any_dry_run = any(_g(r, "dry_run") is True for r in provided)
    all_paper_ready = bool(provided) and all(
        _report_paper_ready(r) for r in provided)

    limitations = [
        "deltas are only as honest as the input reports; dry-run inputs are "
        "NOT paper-ready",
        "each axis is computed only when its required input reports are present",
        "compare only same-token runs -- a delta over differing outputs is "
        "meaningless",
    ]
    if any_dry_run:
        limitations.append(
            "at least one input report is dry_run -> this ablation is NOT "
            "paper-ready")
    if safe_fixture_mode:
        limitations.append(
            "safe_fixture_mode is on -> the LoRA rank-mask axis is fixture-only")

    report = {
        "stage": "e14_ablation_report",
        "axes": axes,
        "axes_available": sorted(
            name for name, a in axes.items() if a.get("available")),
        "axes_unavailable": sorted(
            name for name, a in axes.items() if not a.get("available")),
        "num_input_reports": len(provided),
        "any_input_dry_run": any_dry_run,
        "safe_fixture_mode": safe_fixture_mode,
        "paper_ready": all_paper_ready and not any_dry_run,
        "dry_run": any_dry_run,
        "limitations": limitations,
    }

    if nonlinear_backend is not None:
        try:
            from pllo.experiments.nonlinear_designs import (
                nonlinear_design_report_fields, normalize_nonlinear_backend)
            report["nonlinear_backend"] = normalize_nonlinear_backend(
                nonlinear_backend)
            report.update(nonlinear_design_report_fields(nonlinear_backend))
        except Exception as exc:                            # noqa: BLE001
            report["nonlinear_backend_error"] = str(exc)

    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    return str(v)


def render_md(report: dict) -> str:
    L = ["# E14 ablation report", "",
         "- paper_ready: %s" % report.get("paper_ready"),
         "- dry_run (any input): %s" % report.get("any_input_dry_run"),
         "- input reports: %d" % report.get("num_input_reports", 0),
         "- axes available: %s"
         % (", ".join(report.get("axes_available") or []) or "none"),
         "- axes unavailable: %s"
         % (", ".join(report.get("axes_unavailable") or []) or "none"),
         ""]
    if report.get("nonlinear_backend"):
        L.append("- nonlinear design: %s" % report.get("nonlinear_backend"))
        L.append("")

    for name, axis in (report.get("axes") or {}).items():
        L.append("## %s" % name)
        L.append("")
        if not axis.get("available"):
            L.append("- unavailable: %s" % axis.get("reason"))
            L.append("")
            continue
        if name == "max_new_tokens_scaling":
            L += ["| max_new_tokens | latency_s | latency/token | "
                  "trusted_bytes/token | tokens_exact_match |",
                  "| --- | --- | --- | --- | --- |"]
            for row in axis.get("rows", []):
                L.append("| %s | %s | %s | %s | %s |" % (
                    _fmt(row.get("max_new_tokens")), _fmt(row.get("latency_s")),
                    _fmt(row.get("latency_per_token_s")),
                    _fmt(row.get("trusted_bytes_per_token")),
                    _fmt(row.get("tokens_exact_match"))))
        else:
            L += ["| metric | A | B | delta |", "| --- | --- | --- | --- |"]
            for metric, d in (axis.get("deltas") or {}).items():
                vals = [v for k, v in d.items() if k != "delta"]
                a = vals[0] if len(vals) > 0 else None
                b = vals[1] if len(vals) > 1 else None
                L.append("| %s | %s | %s | %s |" % (
                    metric, _fmt(a), _fmt(b), _fmt(d.get("delta"))))
        if axis.get("limitations"):
            L += ["", "- limitations: %s" % axis["limitations"]]
        L.append("")

    L += ["## Limitations", ""]
    L += ["- %s" % x for x in report.get("limitations", [])]
    L.append("")
    return "\n".join(L)
