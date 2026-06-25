"""Latency / overhead baselines -- pure parsing of decode report JSON.

Builds a paper-ready latency comparison across the deployment backends
(plaintext H800, folded local / remote, TDX lite / attested, folded-LoRA,
TEE-only CPU estimate). Computes per-token latency, throughput, transport-byte
accounting, and overhead ratios vs the plaintext-H800 and folded-local
baselines. Renders Markdown, CSV and a LaTeX ``tabular``.

stdlib + numpy only. Defensive: any key may be missing -- value is ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "BACKENDS", "parse_backend_row", "build_latency_table",
    "render_md", "render_csv", "render_latex",
]

BACKENDS = [
    "plaintext_h800",
    "folded_h800_local",
    "folded_h800_remote",
    "tdx_lite_remote",
    "tdx_attested_remote",
    "folded_lora_remote",
    "tdx_attested_folded_lora_remote",
    "tee_only_cpu_estimate",
]


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _sum_calls(v):
    if isinstance(v, dict):
        total = 0.0
        any_num = False
        for x in v.values():
            n = _num(x)
            if n is not None:
                total += n
                any_num = True
        return total if any_num else None
    return _num(v)


def parse_backend_row(name: str, report: dict) -> dict:
    report = report if isinstance(report, dict) else {}
    prefill = _num(_g(report, "prefill_latency_s"))
    decode = _num(_g(report, "decode_latency_s"))
    total = _num(_g(report, "total_latency_s"))
    latency_s = _num(_g(report, "latency_s"))
    if total is None:
        if prefill is not None or decode is not None:
            total = (prefill or 0.0) + (decode or 0.0)
        else:
            total = latency_s

    max_new = _num(_g(report, "max_new_tokens"))
    latency_per_token_s = None
    tokens_per_s = None
    if total is not None and max_new and max_new > 0:
        latency_per_token_s = total / max_new
        tokens_per_s = (max_new / total) if total else None

    package_size_gb = _num(_g(report, "package_size_gb"))
    lora_enabled = bool(_g(report, "lora_enabled"))
    folded_lora_package_size_mb = (
        package_size_gb * 1024.0
        if (lora_enabled and package_size_gb is not None) else None)

    setup_load_time_s = _num(_g(report, "setup_load_time_s"))
    if setup_load_time_s is None:
        setup_load_time_s = _num(_g(report, "setup_time_s"))
    if setup_load_time_s is None:
        setup_load_time_s = _num(_g(report, "build_time_s"))

    return {
        "backend": name,
        "provided": bool(report),
        "stage": _g(report, "stage"),
        "dry_run": _g(report, "dry_run"),
        "paper_ready": (_g(report, "dry_run") is False and bool(report)),
        "prefill_latency_s": prefill,
        "decode_latency_s": decode,
        "total_latency_s": total,
        "latency_per_token_s": latency_per_token_s,
        "tokens_per_s": tokens_per_s,
        "max_new_tokens": max_new,
        "trusted_bytes": _num(_g(report, "trusted_bytes")),
        "gpu_bytes": _num(_g(report, "gpu_bytes")),
        "boundary_calls": _sum_calls(_g(report, "boundary_calls")),
        "gpu_calls": _sum_calls(_g(report, "gpu_calls")),
        "peak_gpu_memory_mb": _num(_g(report, "peak_gpu_memory_mb")),
        "setup_load_time_s": setup_load_time_s,
        "folded_package_size_gb": package_size_gb,
        "folded_lora_package_size_mb": folded_lora_package_size_mb,
        "lora_enabled": lora_enabled,
    }


def build_latency_table(reports_by_backend: dict) -> dict:
    rows = []
    for name in BACKENDS:
        rep = reports_by_backend.get(name)
        if rep is None:
            continue
        rows.append(parse_backend_row(name, rep))

    by_name = {r["backend"]: r for r in rows}
    base_plain = _g(by_name, "plaintext_h800", "total_latency_s")
    base_local = _g(by_name, "folded_h800_local", "total_latency_s")
    tee_only = by_name.get("tee_only_cpu_estimate")
    tee_only_trusted = _g(tee_only, "trusted_bytes") if tee_only else None

    for r in rows:
        tot = r["total_latency_s"]
        r["overhead_vs_plaintext_h800"] = (
            round(tot / base_plain, 6)
            if (tot is not None and base_plain) else None)
        r["overhead_vs_folded_h800_local"] = (
            round(tot / base_local, 6)
            if (tot is not None and base_local) else None)
        # trusted-compute reduction vs a TEE-only (CPU) estimate
        if tee_only_trusted and r["trusted_bytes"] is not None:
            r["trusted_compute_reduction_vs_tee_only_estimate"] = (
                round(r["trusted_bytes"] / tee_only_trusted, 6)
                if tee_only_trusted else None)
        else:
            r["trusted_compute_reduction_vs_tee_only_estimate"] = None

    return {
        "stage": "latency_baselines",
        "backends_present": [r["backend"] for r in rows],
        "baseline_plaintext_h800_total_latency_s": base_plain,
        "baseline_folded_h800_local_total_latency_s": base_local,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_COLS = [
    ("backend", "backend"),
    ("total_latency_s", "total_latency_s"),
    ("latency_per_token_s", "latency_per_token_s"),
    ("tokens_per_s", "tokens_per_s"),
    ("overhead_vs_plaintext_h800", "overhead_vs_plaintext"),
    ("peak_gpu_memory_mb", "peak_gpu_memory_mb"),
    ("dry_run", "dry_run"),
    ("paper_ready", "paper_ready"),
]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    return str(v)


def render_md(table: dict) -> str:
    headers = [h for _, h in _COLS]
    L = ["# Latency / overhead baselines", "",
         "Baselines: plaintext_h800 total=%s s, folded_h800_local total=%s s."
         % (_fmt(table.get("baseline_plaintext_h800_total_latency_s")),
            _fmt(table.get("baseline_folded_h800_local_total_latency_s"))), "",
         "| " + " | ".join(headers) + " |",
         "| " + " | ".join("---" for _ in headers) + " |"]
    for r in table["rows"]:
        L.append("| " + " | ".join(_fmt(r.get(k)) for k, _ in _COLS) + " |")
    L += [""]
    return "\n".join(L)


def render_csv(table: dict) -> str:
    headers = [h for _, h in _COLS]
    out = [",".join(headers)]
    for r in table["rows"]:
        cells = []
        for k, _ in _COLS:
            cells.append(_fmt(r.get(k)).replace(",", ";"))
        out.append(",".join(cells))
    return "\n".join(out) + "\n"


def _tex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "%", r"\%")


def render_latex(table: dict) -> str:
    headers = [_tex_escape(h) for _, h in _COLS]
    L = [r"\begin{tabular}{l" + "r" * (len(_COLS) - 1) + "}",
         r"\hline",
         " & ".join(headers) + r" \\",
         r"\hline"]
    for r in table["rows"]:
        cells = []
        for k, _ in _COLS:
            cells.append(_tex_escape(_fmt(r.get(k))))
        L.append(" & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}"]
    return "\n".join(L)


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
