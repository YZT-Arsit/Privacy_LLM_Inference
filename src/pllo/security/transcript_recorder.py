"""Metadata-only transcript recorder for the GPU-channel protocol.

The recorder writes a structural transcript of everything that crossed to /
from the untrusted GPU worker. By construction it stores ONLY:

* the message type + direction,
* the KEYS of any public-metadata dict (never the values),
* the name / shape / dtype of any tensor (never its contents),
* a byte count.

It NEVER stores tensor contents or scalar/secret values, so a transcript JSONL
can be serialised and audited without carrying any plaintext or mask secret.

numpy + standard library only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TranscriptEntry",
    "TranscriptRecorder",
    "record_message",
]

# The two GPU-visible channel directions.
_DIRECTIONS = ("boundary_to_worker", "worker_to_boundary")

# Map the protocol-layer recorder vocabulary ("inbound"/"outbound", relative to
# the GPU worker) onto the transcript direction vocabulary.
_DIRECTION_ALIASES = {
    "inbound": "boundary_to_worker",
    "outbound": "worker_to_boundary",
    "boundary_to_worker": "boundary_to_worker",
    "worker_to_boundary": "worker_to_boundary",
}


@dataclass
class TranscriptEntry:
    """One metadata-only record of a message on the GPU channel.

    ``tensor_specs`` holds one ``{"name", "shape", "dtype"}`` dict per tensor --
    shapes / dtypes ONLY, never tensor contents. ``public_metadata_keys`` holds
    the KEYS of the public-metadata dict, never the values."""
    seq: int
    message_type: str
    direction: str
    public_metadata_keys: list = field(default_factory=list)
    tensor_specs: list = field(default_factory=list)
    byte_count: int = 0
    notes: str = ""


def _normalise_direction(direction: str) -> str:
    d = _DIRECTION_ALIASES.get(str(direction))
    if d is None:
        raise ValueError(
            "direction must be one of %s (or inbound/outbound); got %r"
            % (list(_DIRECTIONS), direction))
    return d


def _tensor_spec(name: str, obj: Any) -> dict | None:
    """Build a {name, shape, dtype} spec from a numpy array / array-like / spec.

    Returns ``None`` for objects that are not tensor-like. Stores SHAPE + DTYPE
    only -- never the contents."""
    # Already a pre-built spec dict.
    if isinstance(obj, dict):
        if "shape" in obj or "dtype" in obj:
            shape = obj.get("shape")
            try:
                shape = [int(x) for x in shape] if shape is not None else None
            except (TypeError, ValueError):
                shape = list(shape) if shape is not None else None
            return {"name": str(obj.get("name", name)),
                    "shape": shape,
                    "dtype": (None if obj.get("dtype") is None
                              else str(obj.get("dtype")))}
        return None
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is None and dtype is None:
        return None
    try:
        shape_list = [int(x) for x in shape] if shape is not None else None
    except (TypeError, ValueError):
        shape_list = list(shape) if shape is not None else None
    return {"name": str(name),
            "shape": shape_list,
            "dtype": (None if dtype is None else str(dtype))}


class TranscriptRecorder:
    """Accumulates :class:`TranscriptEntry` records (metadata only)."""

    def __init__(self) -> None:
        self.entries: list[TranscriptEntry] = []

    def record(self, message_type: str, direction: str, *,
               public_metadata: dict | None = None,
               tensors: dict | None = None,
               byte_count: int = 0,
               notes: str = "") -> TranscriptEntry:
        """Record one message, extracting metadata KEYS + tensor SPECS only.

        ``public_metadata`` -- a dict whose KEYS (not values) are kept.
        ``tensors`` -- a dict ``{name: ndarray|array-like|spec-dict}`` from which
        only name/shape/dtype are kept. ``byte_count`` -- optional byte count."""
        meta_keys: list[str] = []
        if public_metadata:
            meta_keys = [str(k) for k in public_metadata.keys()]
        specs: list[dict] = []
        if tensors:
            for name, obj in tensors.items():
                spec = _tensor_spec(str(name), obj)
                if spec is not None:
                    specs.append(spec)
        entry = TranscriptEntry(
            seq=len(self.entries),
            message_type=str(message_type),
            direction=_normalise_direction(direction),
            public_metadata_keys=meta_keys,
            tensor_specs=specs,
            byte_count=int(byte_count),
            notes=str(notes),
        )
        self.entries.append(entry)
        return entry

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.entries]

    def to_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(json.dumps(asdict(e), default=str))
                fh.write("\n")
        return p


def _is_tensor_like(v: Any) -> bool:
    return hasattr(v, "shape") and hasattr(v, "dtype")


def record_message(recorder: TranscriptRecorder, direction: str,
                   msg: Any) -> TranscriptEntry:
    """Introspect a protocol message and record it metadata-only.

    Works on dataclass message objects (with ndarray fields like
    ``masked_embeddings`` / ``masked_logits`` + scalar public fields) and on
    plain dicts. ndarray-like fields become tensor specs (name/shape/dtype);
    every other field name becomes a public-metadata key. No values are kept
    except the byte count of the tensor fields."""
    # Resolve a (name -> value) iterable defensively.
    if isinstance(msg, dict):
        items = list(msg.items())
        message_type = str(msg.get("message_type") or msg.get("type")
                           or "dict")
    elif hasattr(msg, "__dataclass_fields__"):
        from dataclasses import fields as _dc_fields
        items = [(f.name, getattr(msg, f.name, None)) for f in _dc_fields(msg)]
        message_type = type(msg).__name__
    else:
        # Fall back to public attributes.
        items = [(k, getattr(msg, k)) for k in dir(msg)
                 if not k.startswith("_") and not callable(getattr(msg, k))]
        message_type = type(msg).__name__

    public_metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    byte_count = 0
    for name, value in items:
        if _is_tensor_like(value):
            tensors[name] = value
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                try:
                    byte_count += int(nbytes)
                except (TypeError, ValueError):
                    pass
        else:
            # Keep the KEY only (never the value).
            public_metadata[name] = None
    return recorder.record(message_type, direction,
                           public_metadata=public_metadata,
                           tensors=tensors, byte_count=byte_count)
