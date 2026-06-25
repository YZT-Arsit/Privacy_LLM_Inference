"""Scanner for GPU-channel transcripts.

Given a metadata-only transcript (produced by
:class:`~pllo.security.transcript_recorder.TranscriptRecorder`), the scanner
verifies that nothing the untrusted GPU worker could see -- on EITHER GPU-visible
direction -- carries a forbidden (secret / plaintext) name. Only tensor names
and public-metadata KEYS are checked (the transcript never carries values).

Folded artifacts (``a_tilde`` / ``b_tilde`` / anything containing ``_tilde``)
are explicitly ALLOWED: they bake the masks into the weights and reveal no raw
adapter. Trusted-side artifacts are never scanned -- only the two GPU-visible
directions are.

numpy + standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "FORBIDDEN_GPU_VISIBLE",
    "PUBLIC_METADATA_ALLOWED",
    "scan_transcript",
    "load_transcript_jsonl",
]

# Substrings (lowercased) that must NEVER appear on a GPU-visible channel.
FORBIDDEN_GPU_VISIBLE = frozenset({
    "input_ids", "prompt", "raw_text", "raw_logits", "recovered_logits",
    "logits_plain", "label", "labels", "ground_truth", "answer_key",
    "raw_lora", "lora_a", "lora_b", "lora_a_raw", "lora_b_raw",
    "optimizer_state", "optim", "gradient", "grad", "mask_seed",
    "mask_secret", "secret", "n0", "n_0", "vocab_mask", "adapter",
    "training_example", "target_label",
})

# Public model hyper-parameters / protocol metadata that are fine to expose.
PUBLIC_METADATA_ALLOWED = frozenset({
    "hidden_size", "intermediate_size", "num_heads", "num_key_value_heads",
    "head_dim", "rope_theta", "rms_norm_eps", "attention_bias", "mlp_bias",
    "mask_family", "fold_dtype", "rope_max_pos", "num_layers", "vocab_size",
    "seq_len", "max_new_tokens", "dtype", "positions", "position",
    "batch_size", "session_id", "step", "model_name", "model_type",
})

# Folded / masked tokens that look adapter-ish but are SAFE.
_SAFE = frozenset({"a_tilde", "b_tilde", "lora_a_tilde", "lora_b_tilde"})

# The two channel directions the GPU worker can observe.
_GPU_VISIBLE_DIRECTIONS = frozenset({"boundary_to_worker", "worker_to_boundary"})


def _is_safe(name_lc: str) -> bool:
    """Folded artifacts are allowed: exact safe token or anything ``_tilde``."""
    if name_lc in _SAFE:
        return True
    return "_tilde" in name_lc


def _match_forbidden(name: str, allowlist) -> str | None:
    """Return the matched forbidden substring, or ``None`` if the name is clean.

    The tilde/safe check and the optional allowlist are applied first."""
    name_lc = str(name).lower()
    if _is_safe(name_lc):
        return None
    if allowlist:
        for allowed in allowlist:
            if allowed and str(allowed).lower() in name_lc:
                return None
    for bad in FORBIDDEN_GPU_VISIBLE:
        if bad in name_lc:
            return bad
    return None


def _entry_get(entry: Any, key: str, default=None):
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def scan_transcript(entries, *, allowlist=None) -> dict:
    """Scan a transcript for forbidden names on the GPU-visible channels.

    ``entries`` -- a list of transcript entries (dicts or ``TranscriptEntry``).
    ``allowlist`` -- optional list of allowed substrings (checked before the
    forbidden set). Returns a structured report dict (see module docstring)."""
    allowlist = list(allowlist or [])
    scanned = 0
    gpu_visible = 0
    leaks: list[dict] = []

    for entry in (entries or []):
        scanned += 1
        direction = _entry_get(entry, "direction")
        # Only the two GPU-visible directions are scanned; trusted-side
        # artifacts (any other direction) are out of scope by construction.
        if direction not in _GPU_VISIBLE_DIRECTIONS:
            continue
        gpu_visible += 1
        seq = _entry_get(entry, "seq")
        msg_type = _entry_get(entry, "message_type")

        # Tensor names.
        for spec in (_entry_get(entry, "tensor_specs") or []):
            name = spec.get("name") if isinstance(spec, dict) else \
                getattr(spec, "name", None)
            if name is None:
                continue
            matched = _match_forbidden(name, allowlist)
            if matched is not None:
                leaks.append({"seq": seq, "message_type": msg_type,
                              "direction": direction, "field": name,
                              "kind": "tensor", "matched_forbidden": matched})

        # Public-metadata keys.
        for key in (_entry_get(entry, "public_metadata_keys") or []):
            key_lc = str(key).lower()
            # Explicitly-allowed public metadata is always fine.
            if key_lc in PUBLIC_METADATA_ALLOWED:
                continue
            matched = _match_forbidden(key, allowlist)
            if matched is not None:
                leaks.append({"seq": seq, "message_type": msg_type,
                              "direction": direction, "field": key,
                              "kind": "metadata", "matched_forbidden": matched})

    forbidden_found = sorted({l["matched_forbidden"] for l in leaks})
    return {
        "stage": "security_transcript_scan",
        "scanned_entries": scanned,
        "gpu_visible_entries": gpu_visible,
        "leaks": leaks,
        "leak_count": len(leaks),
        "forbidden_fields_found": forbidden_found,
        "fail": bool(leaks),
        "allowlist_used": allowlist,
    }


def load_transcript_jsonl(path: str | Path) -> list[dict]:
    """Load a transcript JSONL file into a list of entry dicts."""
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
