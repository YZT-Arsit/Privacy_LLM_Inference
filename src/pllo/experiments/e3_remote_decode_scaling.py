"""E3: scaling / performance sweep for remote package-backed decode.

Aggregation + reporting for a sweep of the already-working remote folded-package
decode (``run_tee_gpu_protocol_demo.build_remote_folded_package_decode_report``)
over ``max_new_tokens`` (and optionally ``seq_len``). This module holds NO
protocol logic: the runner injects a ``decode_fn(seq_len, max_new_tokens)`` that
returns one demo report dict; here we extract the required per-row fields, compute
``latency_per_token_s``, and build the pass/fail + latency/bytes/boundary-call +
security tables. Injection keeps the aggregation unit-testable without H800/TDX.

stdlib only (no torch / numpy).
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "E3_EXPERIMENT", "E3_STAGE", "E3_ROW_FIELDS",
    "make_row", "run_e3_scaling", "row_pass", "row_security_ok",
    "build_e3_summary", "render_e3_csv", "render_e3_md",
]

E3_EXPERIMENT = "E3"
E3_STAGE = "remote_package_decode_scaling"

# Per-row fields the experiment must emit (in CSV column order).
E3_ROW_FIELDS = [
    "experiment", "stage", "boundary_mode", "gpu_worker_remote", "gpu_backend",
    "max_new_tokens", "seq_len", "tokens_exact_match", "token_match_rate",
    "package_backed_prefill", "package_backed_decode", "folded_package_loaded",
    "folded_package_valid", "worker_has_mask_secrets", "tee_used_on_gpu",
    "gpu_visible_plaintext_fields", "leaked_secret_fields", "audit_passed",
    "latency_s", "latency_per_token_s", "trusted_bytes", "gpu_bytes",
    "boundary_calls", "gpu_calls", "peak_gpu_memory_mb",
]


def make_row(report: dict, *, seq_len: int, max_new_tokens: int) -> dict:
    """Extract one E3 row from a demo decode report (does not mutate it)."""
    n = int(max_new_tokens)
    latency = report.get("latency_s")
    lpt = (float(latency) / n) if (latency is not None and n > 0) else None
    return {
        "experiment": E3_EXPERIMENT,
        "stage": E3_STAGE,
        "source_stage": report.get("stage"),
        "boundary_mode": report.get("boundary_mode"),
        "gpu_worker_remote": report.get("gpu_worker_remote"),
        "gpu_backend": report.get("gpu_backend"),
        "max_new_tokens": n,
        "seq_len": int(report.get("seq_len", seq_len)),
        "tokens_exact_match": report.get("tokens_exact_match"),
        "token_match_rate": report.get("token_match_rate"),
        "reference_basis": report.get("reference_basis"),
        "package_backed_prefill": report.get("package_backed_prefill"),
        "package_backed_decode": report.get("package_backed_decode"),
        "folded_package_loaded": report.get("folded_package_loaded"),
        "folded_package_valid": report.get("folded_package_valid"),
        "worker_has_mask_secrets": report.get("worker_has_mask_secrets"),
        "tee_used_on_gpu": report.get("tee_used_on_gpu"),
        "gpu_visible_plaintext_fields": report.get("gpu_visible_plaintext_fields"),
        "leaked_secret_fields": report.get("leaked_secret_fields"),
        "audit_passed": report.get("audit_passed"),
        "latency_s": latency,
        "latency_per_token_s": lpt,
        "trusted_bytes": report.get("trusted_bytes"),
        "gpu_bytes": report.get("gpu_bytes"),
        "boundary_calls": report.get("boundary_calls"),
        "gpu_calls": report.get("gpu_calls"),
        "peak_gpu_memory_mb": report.get("peak_gpu_memory_mb"),
    }


def run_e3_scaling(decode_fn: Callable[..., dict], *, seq_lens: list[int],
                   max_new_tokens_list: list[int]) -> list[dict]:
    """Drive ``decode_fn(seq_len=..., max_new_tokens=...)`` over the grid and
    collect one E3 row per (seq_len, max_new_tokens). ``decode_fn`` returns a demo
    decode report dict (the runner injects the real remote decode)."""
    rows: list[dict] = []
    for sl in seq_lens:
        for n in max_new_tokens_list:
            report = decode_fn(seq_len=int(sl), max_new_tokens=int(n))
            rows.append(make_row(report, seq_len=int(sl), max_new_tokens=int(n)))
    return rows


def row_security_ok(row: dict) -> bool:
    """No mask secrets / no GPU TEE / no GPU-visible plaintext / no leaks / audit
    not failed."""
    return bool(
        not row.get("worker_has_mask_secrets")
        and not row.get("tee_used_on_gpu")
        and not row.get("gpu_visible_plaintext_fields")
        and not row.get("leaked_secret_fields")
        and row.get("audit_passed") is not False)


def row_pass(row: dict) -> bool:
    """A row passes when it is package-backed, the package loaded, security holds,
    and correctness (if a reference/expected basis was present) is not violated.
    ``tokens_exact_match is None`` (no reference) does not fail the row."""
    compute_ok = bool(
        row.get("package_backed_decode")
        and row.get("folded_package_loaded")
        and row.get("tokens_exact_match") is not False)
    return bool(compute_ok and row_security_ok(row))


def _boundary_call_total(calls: Any) -> int | None:
    if isinstance(calls, dict):
        return int(sum(int(v) for v in calls.values()))
    return None


def build_e3_summary(rows: list[dict]) -> dict:
    """Pass/fail per row + latency / bytes / boundary-call / security tables."""
    passfail = [{
        "seq_len": r["seq_len"], "max_new_tokens": r["max_new_tokens"],
        "pass": row_pass(r), "tokens_exact_match": r.get("tokens_exact_match"),
        "token_match_rate": r.get("token_match_rate"),
    } for r in rows]

    latency = [{
        "seq_len": r["seq_len"], "max_new_tokens": r["max_new_tokens"],
        "latency_s": r.get("latency_s"),
        "latency_per_token_s": r.get("latency_per_token_s"),
        "peak_gpu_memory_mb": r.get("peak_gpu_memory_mb"),
    } for r in rows]

    bytes_tbl = [{
        "seq_len": r["seq_len"], "max_new_tokens": r["max_new_tokens"],
        "trusted_bytes": r.get("trusted_bytes"), "gpu_bytes": r.get("gpu_bytes"),
    } for r in rows]

    boundary_tbl = [{
        "seq_len": r["seq_len"], "max_new_tokens": r["max_new_tokens"],
        "boundary_calls": r.get("boundary_calls"),
        "boundary_calls_total": _boundary_call_total(r.get("boundary_calls")),
        "gpu_calls": r.get("gpu_calls"),
        "gpu_calls_total": _boundary_call_total(r.get("gpu_calls")),
    } for r in rows]

    security_tbl = [{
        "seq_len": r["seq_len"], "max_new_tokens": r["max_new_tokens"],
        "worker_has_mask_secrets": r.get("worker_has_mask_secrets"),
        "tee_used_on_gpu": r.get("tee_used_on_gpu"),
        "gpu_visible_plaintext_fields": r.get("gpu_visible_plaintext_fields"),
        "leaked_secret_fields": r.get("leaked_secret_fields"),
        "audit_passed": r.get("audit_passed"),
        "security_ok": row_security_ok(r),
    } for r in rows]

    n_pass = sum(1 for p in passfail if p["pass"])
    return {
        "num_rows": len(rows),
        "num_pass": n_pass,
        "all_pass": bool(rows) and n_pass == len(rows),
        "all_security_ok": all(row_security_ok(r) for r in rows),
        "passfail": passfail,
        "latency_table": latency,
        "bytes_table": bytes_tbl,
        "boundary_call_table": boundary_tbl,
        "security_table": security_tbl,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    if isinstance(v, (list, dict)):
        import json
        return json.dumps(v, separators=(";", ":"))
    return str(v)


def render_e3_csv(rows: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(E3_ROW_FIELDS)
    for r in rows:
        w.writerow([_fmt(r.get(k)) for k in E3_ROW_FIELDS])
    return buf.getvalue()


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return out


def render_e3_md(rows: list[dict], summary: dict, meta: dict) -> str:
    L = ["# E3 — Remote package-backed decode scaling", "",
         "- gpu_backend: `%s`" % meta.get("gpu_backend"),
         "- gpu_worker_url: `%s`" % meta.get("gpu_worker_url"),
         "- boundary_mode: **%s**" % meta.get("boundary_mode"),
         "- model_name: `%s`  dtype: %s  device: %s"
         % (meta.get("model_name"), meta.get("dtype"), meta.get("device")),
         "- dry_run: %s" % meta.get("dry_run"),
         "- **all_pass: %s**  (%d/%d rows)  all_security_ok: %s"
         % (summary["all_pass"], summary["num_pass"], summary["num_rows"],
            summary["all_security_ok"]),
         ""]
    L += ["## Pass / fail", ""]
    L += _md_table(
        ["seq_len", "max_new_tokens", "pass", "tokens_exact_match",
         "token_match_rate"],
        [[p["seq_len"], p["max_new_tokens"], p["pass"],
          p["tokens_exact_match"], p["token_match_rate"]]
         for p in summary["passfail"]])
    L += ["", "## Latency scaling", ""]
    L += _md_table(
        ["seq_len", "max_new_tokens", "latency_s", "latency_per_token_s",
         "peak_gpu_memory_mb"],
        [[t["seq_len"], t["max_new_tokens"], t["latency_s"],
          t["latency_per_token_s"], t["peak_gpu_memory_mb"]]
         for t in summary["latency_table"]])
    L += ["", "## Bytes scaling", ""]
    L += _md_table(
        ["seq_len", "max_new_tokens", "trusted_bytes", "gpu_bytes"],
        [[t["seq_len"], t["max_new_tokens"], t["trusted_bytes"], t["gpu_bytes"]]
         for t in summary["bytes_table"]])
    L += ["", "## Boundary / GPU call scaling", ""]
    L += _md_table(
        ["seq_len", "max_new_tokens", "boundary_calls_total", "gpu_calls_total"],
        [[t["seq_len"], t["max_new_tokens"], t["boundary_calls_total"],
          t["gpu_calls_total"]] for t in summary["boundary_call_table"]])
    L += ["", "## Security audit", ""]
    L += _md_table(
        ["seq_len", "max_new_tokens", "worker_has_mask_secrets",
         "tee_used_on_gpu", "gpu_visible_plaintext_fields", "leaked_secret_fields",
         "audit_passed", "security_ok"],
        [[t["seq_len"], t["max_new_tokens"], t["worker_has_mask_secrets"],
          t["tee_used_on_gpu"], t["gpu_visible_plaintext_fields"] or "[]",
          t["leaked_secret_fields"] or "[]", t["audit_passed"], t["security_ok"]]
         for t in summary["security_table"]])
    L += ["",
          "_Each row reuses the validated remote folded-package decode path; the "
          "GPU worker sees only masked embeddings + public metadata. No TDX "
          "attestation is claimed by this scaling run._", ""]
    return "\n".join(L)
