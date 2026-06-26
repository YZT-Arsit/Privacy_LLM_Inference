"""Extended GPU-channel audit for the strict length-hiding decode mode.

This is a NON-MEASURED defense-in-depth layer (deliberately NOT in
``attestation.DEFAULT_TRUSTED_BOUNDARY_PATHS``, so adding it does not change the
attestation runtime hash). It wraps the canonical, hash-bound
:func:`pllo.security.transcript_scanner.scan_transcript` and additionally rejects
a few names that the strict length-hiding / generation-config path must never let
reach the GPU but that the canonical substring set does not already cover:

* ``token_ids`` / ``generated_token_history`` / ``token_history`` -- the trusted
  token history (prompt + generated + dummy) is trusted-only;
* ``eos_decision`` / ``finish_reason`` -- the true stop step is trusted-only;
* ``plaintext_logits`` / ``recovered_logits`` -- recovered logits stay trusted;
* ``dummy_token_id`` / ``dummy_token`` -- the length-hiding dummy id is trusted;
* ``pad`` / ``inverse`` / ``prg_seed`` / ``schedule_secret`` -- belt-and-braces
  mask/secret names.

The canonical scanner still runs first (so every name IT forbids is still
forbidden); this only ADDS names. Trusted-side artifacts and folded ``*_tilde``
tensors remain allowed. Standard library only.
"""

from __future__ import annotations

from typing import Any

from pllo.security.transcript_scanner import (
    FORBIDDEN_GPU_VISIBLE,
    PUBLIC_METADATA_ALLOWED,
    load_transcript_jsonl,
    scan_transcript,
)

__all__ = [
    "EXTENDED_FORBIDDEN_GPU_VISIBLE",
    "LENGTH_HIDING_EXTRA_FORBIDDEN",
    "forbidden_names_in_payload",
    "scan_length_hiding_transcript",
    "audit_gpu_request_payloads",
    "load_transcript_jsonl",
]

# names the strict length-hiding / generation-config path must never expose, on
# top of the canonical set (substring match, lowercased).
LENGTH_HIDING_EXTRA_FORBIDDEN = frozenset({
    "token_ids", "generated_token_history", "token_history", "generated_tokens",
    "eos_decision", "eos_token_id", "finish_reason", "stop_reason",
    "plaintext_logits", "plain_logits", "recovered_logits", "logits_plain",
    "dummy_token_id", "dummy_token", "sampling_decision",
    "inverse", "pad", "prg_seed", "schedule_secret", "mask_secret",
})

EXTENDED_FORBIDDEN_GPU_VISIBLE = frozenset(
    set(FORBIDDEN_GPU_VISIBLE) | set(LENGTH_HIDING_EXTRA_FORBIDDEN))

_SAFE = frozenset({"a_tilde", "b_tilde", "lora_a_tilde", "lora_b_tilde"})
_GPU_VISIBLE_DIRECTIONS = frozenset({"boundary_to_worker", "worker_to_boundary"})


def _is_safe(name_lc: str) -> bool:
    return name_lc in _SAFE or "_tilde" in name_lc


def _match_extra(name: str, allowlist) -> str | None:
    """Match ONLY the length-hiding extra forbidden substrings (the canonical set
    is handled by ``scan_transcript``). Safe/tilde + allowlist applied first."""
    name_lc = str(name).lower()
    if _is_safe(name_lc):
        return None
    for allowed in (allowlist or []):
        if allowed and str(allowed).lower() in name_lc:
            return None
    for bad in sorted(LENGTH_HIDING_EXTRA_FORBIDDEN):    # deterministic order
        if bad in name_lc:
            return bad
    return None


def forbidden_names_in_payload(payload: Any, *, allowlist=None) -> list[str]:
    """Recursively scan a (decoded) GPU request/response payload's KEYS for any
    forbidden name (canonical + length-hiding extras). Values are not scanned
    (the wire payload base64-encodes tensors); this mirrors the name-only
    philosophy of the transcript scanner. Returns the sorted matched names."""
    found: set[str] = set()
    full = EXTENDED_FORBIDDEN_GPU_VISIBLE
    allow = [str(a).lower() for a in (allowlist or [])]

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                klc = str(k).lower()
                if klc in PUBLIC_METADATA_ALLOWED or _is_safe(klc):
                    _walk(v)
                    continue
                if not any(a and a in klc for a in allow):
                    for bad in sorted(full):            # deterministic order
                        if bad in klc:
                            found.add(bad)
                            break
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(payload)
    return sorted(found)


def scan_length_hiding_transcript(entries, *, allowlist=None) -> dict:
    """Run the canonical scan PLUS the length-hiding extra-forbidden check over a
    metadata-only transcript. Returns a merged report; ``fail`` is True if EITHER
    layer found a leak. Only the GPU-visible directions are scanned."""
    base = scan_transcript(entries, allowlist=allowlist)
    extra_leaks: list[dict] = []
    for entry in (entries or []):
        direction = (entry.get("direction") if isinstance(entry, dict)
                     else getattr(entry, "direction", None))
        if direction not in _GPU_VISIBLE_DIRECTIONS:
            continue
        seq = entry.get("seq") if isinstance(entry, dict) else \
            getattr(entry, "seq", None)
        mt = entry.get("message_type") if isinstance(entry, dict) else \
            getattr(entry, "message_type", None)
        specs = (entry.get("tensor_specs") if isinstance(entry, dict)
                 else getattr(entry, "tensor_specs", None)) or []
        for spec in specs:
            name = spec.get("name") if isinstance(spec, dict) else \
                getattr(spec, "name", None)
            if name is None:
                continue
            m = _match_extra(name, allowlist)
            if m is not None:
                extra_leaks.append({"seq": seq, "message_type": mt,
                                    "direction": direction, "field": name,
                                    "kind": "tensor", "matched_forbidden": m})
        keys = (entry.get("public_metadata_keys") if isinstance(entry, dict)
                else getattr(entry, "public_metadata_keys", None)) or []
        for key in keys:
            if str(key).lower() in PUBLIC_METADATA_ALLOWED:
                continue
            m = _match_extra(key, allowlist)
            if m is not None:
                extra_leaks.append({"seq": seq, "message_type": mt,
                                    "direction": direction, "field": key,
                                    "kind": "metadata", "matched_forbidden": m})
    all_leaks = list(base.get("leaks", [])) + extra_leaks
    forbidden_found = sorted({l["matched_forbidden"] for l in all_leaks})
    return {
        "stage": "length_hiding_transcript_scan",
        "scanned_entries": base.get("scanned_entries", 0),
        "gpu_visible_entries": base.get("gpu_visible_entries", 0),
        "canonical_leaks": base.get("leaks", []),
        "extra_leaks": extra_leaks,
        "leaks": all_leaks,
        "leak_count": len(all_leaks),
        "forbidden_fields_found": forbidden_found,
        "fail": bool(all_leaks),
        "allowlist_used": list(allowlist or []),
    }


def audit_gpu_request_payloads(payloads, *, allowlist=None) -> dict:
    """Audit a list of decoded GPU request payloads (e.g. ``encode_message(req)``)
    for any forbidden name in their keys. Returns a report with ``fail`` /
    ``forbidden_fields_found`` / per-payload matches."""
    per = []
    found: set[str] = set()
    for i, payload in enumerate(payloads or []):
        names = forbidden_names_in_payload(payload, allowlist=allowlist)
        if names:
            per.append({"index": i, "forbidden_fields_found": names})
            found.update(names)
    return {
        "stage": "length_hiding_request_payload_audit",
        "num_payloads": len(payloads or []),
        "forbidden_fields_found": sorted(found),
        "per_payload": per,
        "fail": bool(found),
    }
