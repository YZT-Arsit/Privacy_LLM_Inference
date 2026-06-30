"""Sensitive-span + secret leakage scan for GPU-visible artifacts.

The core security claim is that the untrusted GPU never sees the raw user input or
any raw secret. This module checks that on real evidence:

* **sensitive spans** (fabricated PII from the synthetic stress set) must NOT
  appear in any GPU-visible channel: the transcript's
  ``boundary_to_worker`` / ``worker_to_boundary`` entries, the worker health
  records, or error logs. (They MAY appear in the trusted-side final response,
  which the GPU never sees -- that is not a leak.)
* **forbidden field names** (input_ids / plaintext / raw mask / N / N_inv / raw
  pad / recovery / token ids) must not appear as keys on GPU-visible channels
  (reuses :mod:`pllo.security.transcript_scanner`).
* **allowed** GPU-visible artifacts: folded ``*_tilde`` weights, ``xpad_tilde`` /
  ``cpad_tilde``, masked-basis pre-staged artifacts, public schedule commitments.

stdlib only. ``attack hook`` helpers expose only GPU-visible metadata (no secrets)
and a cheap nearest-neighbour reconstruction probe that SHOULD fail.
"""

from __future__ import annotations

from typing import Any

from pllo.security.transcript_scanner import scan_transcript

__all__ = [
    "find_spans_in_text",
    "scan_object_for_spans",
    "scan_gpu_visible_for_spans",
    "scan_sensitive_leakage",
    "ALLOWED_GPU_VISIBLE_HINTS",
]

# substrings whose presence as a KEY is fine on a GPU-visible channel
ALLOWED_GPU_VISIBLE_HINTS = ("_tilde", "xpad_tilde", "cpad_tilde", "commitment",
                             "slot_id", "folded", "shape", "dtype")

_GPU_VISIBLE_DIRECTIONS = ("boundary_to_worker", "worker_to_boundary")


def find_spans_in_text(text: str | None, spans: list[str]) -> list[str]:
    if not text:
        return []
    t = str(text)
    return [s for s in spans if s and str(s) in t]


def scan_object_for_spans(obj: Any, spans: list[str]) -> list[str]:
    """Recursively scan any JSON-like object for sensitive spans (in strings,
    dict keys/values, list items). Returns the set of spans found."""
    found: set[str] = set()

    def _walk(o):
        if isinstance(o, str):
            found.update(find_spans_in_text(o, spans))
        elif isinstance(o, dict):
            for k, v in o.items():
                found.update(find_spans_in_text(str(k), spans))
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)
    _walk(obj)
    return sorted(found)


def scan_gpu_visible_for_spans(transcript_entries: list[dict],
                               spans: list[str]) -> list[dict]:
    """Find sensitive spans on GPU-visible transcript directions only."""
    leaks = []
    for e in (transcript_entries or []):
        direction = e.get("direction") if isinstance(e, dict) else None
        if direction not in _GPU_VISIBLE_DIRECTIONS:
            continue
        hit = scan_object_for_spans(e, spans)
        if hit:
            leaks.append({"direction": direction, "spans": hit,
                          "method": e.get("method")})
    return leaks


def scan_sensitive_leakage(*, dataset_rows: list[dict],
                           transcript_entries: list[dict] | None = None,
                           worker_health: list[dict] | None = None,
                           report: dict | None = None,
                           error_logs: list[str] | None = None) -> dict[str, Any]:
    """Full sensitive-leakage scan. ``leakage_pass`` is True iff NO sensitive span
    and NO forbidden field reaches a GPU-visible channel / worker log / report."""
    all_spans: list[str] = []
    for r in dataset_rows:
        all_spans.extend(r.get("sensitive_spans", []) or [])
    all_spans = sorted(set(s for s in all_spans if s))

    leaked_fields: list[str] = []
    leaked_examples: list[dict] = []

    # 1. spans on GPU-visible transcript channels
    tx_leaks = scan_gpu_visible_for_spans(transcript_entries or [], all_spans)
    if tx_leaks:
        leaked_examples.extend({"source": "transcript", **l} for l in tx_leaks)

    # 2. spans in worker health (GPU-visible) and report (must not echo prompt)
    for src, obj in (("worker_health", worker_health), ("report", report)):
        if obj is None:
            continue
        hit = scan_object_for_spans(obj, all_spans)
        if hit:
            leaked_examples.append({"source": src, "spans": hit})

    # 3. spans in error logs
    for ln in (error_logs or []):
        hit = find_spans_in_text(ln, all_spans)
        if hit:
            leaked_examples.append({"source": "error_log", "spans": hit})

    # 4. forbidden FIELD NAMES on GPU-visible channels (input_ids/plaintext/mask)
    name_report = scan_transcript(transcript_entries or [],
                                  allowlist=list(ALLOWED_GPU_VISIBLE_HINTS))
    if name_report.get("forbidden_fields_found"):
        leaked_fields = list(name_report["forbidden_fields_found"])

    leakage_pass = (not leaked_examples) and (not leaked_fields)
    return {
        "stage": "sensitive_prompt_security_scan",
        "num_sensitive_spans": len(all_spans),
        "num_transcript_entries": len(transcript_entries or []),
        "leakage_pass": leakage_pass,
        "leaked_fields": leaked_fields,
        "leaked_examples": leaked_examples,
        "gpu_visible_span_leaks": len(leaked_examples),
        "allowed_gpu_visible_hints": list(ALLOWED_GPU_VISIBLE_HINTS),
        "raw_input_protected": leakage_pass,
    }
