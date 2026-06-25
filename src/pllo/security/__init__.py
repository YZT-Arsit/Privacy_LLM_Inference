"""Security transcript recorder + scanner (Task D).

A *transcript* is a metadata-only record of every message that crossed the
trusted boundary <-> untrusted GPU worker channel: message type, direction,
public-metadata KEYS, and tensor name/shape/dtype SPECS only -- never tensor
contents, never secret values. The scanner then verifies that nothing on the
GPU-visible channels matches a forbidden (secret / plaintext) name.

stdlib + numpy only.
"""

from __future__ import annotations

from pllo.security.transcript_recorder import (
    TranscriptEntry,
    TranscriptRecorder,
    record_message,
)
from pllo.security.transcript_scanner import (
    FORBIDDEN_GPU_VISIBLE,
    PUBLIC_METADATA_ALLOWED,
    load_transcript_jsonl,
    scan_transcript,
)

__all__ = [
    "TranscriptEntry",
    "TranscriptRecorder",
    "record_message",
    "FORBIDDEN_GPU_VISIBLE",
    "PUBLIC_METADATA_ALLOWED",
    "scan_transcript",
    "load_transcript_jsonl",
]
