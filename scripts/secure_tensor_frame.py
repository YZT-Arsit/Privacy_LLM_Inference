"""Secure tensor-frame codec (PHASE 1.2) — replaces torch.load(pickle) on network payloads.

A fixed binary frame that carries a dict of tensors WITHOUT executing pickle. Deserialization
validates every field against an explicit per-operation spec BEFORE allocating any tensor from
attacker-controlled bytes.

Frame layout (all big-endian, no pickle, no code execution):
    magic      : 4 bytes  b"STF1"
    header_len : uint32
    header     : json utf-8  {"op": str, "tensors": [{"name","dtype","shape"}]}  (no other keys)
    body       : concatenated raw tensor bytes, in header order

Nested dicts are flattened with "/" (e.g. {"A": {"0.q_proj": t}} -> name "A/0.q_proj") and
un-flattened on decode.

Threat model note: this codec is for NETWORK-CONTROLLED payloads. Durable LOCAL encrypted
optimizer checkpoints inside the TDX may still use torch serialization, but ONLY after the bytes
are authenticated/decrypted and are therefore not attacker-controlled (see `decode_local_trusted`).
"""
from __future__ import annotations
import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

MAGIC = b"STF1"

# dtype whitelist (name -> (torch dtype, itemsize bytes))
_DTYPES = {
    "float32": (torch.float32, 4), "float64": (torch.float64, 8),
    "bfloat16": (torch.bfloat16, 2), "float16": (torch.float16, 2),
    "int64": (torch.int64, 8), "int32": (torch.int32, 4),
    "uint8": (torch.uint8, 1), "bool": (torch.bool, 1),
}
_TORCH_TO_NAME = {v[0]: k for k, v in _DTYPES.items()}
_HEADER_KEYS = {"op", "tensors"}
_TENSOR_KEYS = {"name", "dtype", "shape"}


class FrameError(ValueError):
    """A malformed / policy-violating frame (fail-closed)."""


@dataclass
class FrameSpec:
    """Per-operation validation policy."""
    op: str
    allowed_names: Optional[Set[str]] = None      # exact whitelist; None => any that pass prefix check
    allowed_name_prefixes: tuple = ()             # e.g. ("A/", "B/", "gA/", "gB/", "logits")
    expected_names: Optional[Set[str]] = None     # target set that MUST be present exactly (no more/less)
    max_tensors: int = 4096
    allowed_dtypes: Set[str] = field(default_factory=lambda: set(_DTYPES))
    max_rank: int = 3
    max_numel: int = 200_000_000                  # per-tensor element cap
    max_tensor_bytes: int = 800_000_000
    max_total_bytes: int = 1_200_000_000
    forbid_substrings: tuple = ()                 # e.g. mask-secret hints must never appear


def _flatten(d: Dict, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif torch.is_tensor(v):
            out[key] = v
        else:
            raise FrameError(f"non-tensor value at {key!r}: {type(v).__name__}")
    return out


def _unflatten(flat: Dict[str, torch.Tensor]) -> Dict:
    root: Dict = {}
    for k, v in flat.items():
        if "/" in k:
            top, sub = k.split("/", 1)
            root.setdefault(top, {})[sub] = v
        else:
            root[k] = v
    return root


def encode(op: str, tensors: Dict) -> bytes:
    """Serialize a (possibly nested) dict of tensors into a secure frame."""
    flat = _flatten(tensors)
    meta: List[dict] = []
    body = bytearray()
    for name, t in flat.items():
        t = t.detach().cpu().contiguous()
        dn = _TORCH_TO_NAME.get(t.dtype)
        if dn is None:
            raise FrameError(f"dtype {t.dtype} not serializable")
        meta.append({"name": name, "dtype": dn, "shape": list(t.shape)})
        # bit-exact raw bytes
        body += bytes(t.view(torch.uint8).reshape(-1).numpy().tobytes()) if t.numel() else b""
    header = json.dumps({"op": op, "tensors": meta}).encode()
    return MAGIC + struct.pack(">I", len(header)) + header + bytes(body)


def decode(blob: bytes, spec: FrameSpec) -> Dict:
    """Validate + deserialize a secure frame per `spec`. Raises FrameError on any violation,
    BEFORE allocating tensors where practical (header is fully validated first)."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 8:
        raise FrameError("frame too short")
    if bytes(blob[:4]) != MAGIC:
        raise FrameError("bad magic (not a secure frame; refusing pickle/legacy)")
    (hlen,) = struct.unpack(">I", blob[4:8])
    if hlen <= 0 or 8 + hlen > len(blob):
        raise FrameError("bad header length")
    try:
        header = json.loads(blob[8:8 + hlen].decode())
    except Exception as e:  # noqa: BLE001
        raise FrameError(f"header not json: {e}")
    if not isinstance(header, dict) or set(header) - _HEADER_KEYS:
        raise FrameError("unknown/extra header keys")
    if header.get("op") != spec.op:
        raise FrameError(f"op mismatch: {header.get('op')!r} != {spec.op!r}")
    meta = header.get("tensors")
    if not isinstance(meta, list):
        raise FrameError("tensors not a list")
    if len(meta) > spec.max_tensors:
        raise FrameError(f"too many tensors: {len(meta)} > {spec.max_tensors}")

    # ---- validate every tensor descriptor BEFORE touching the body ----
    names_seen: Set[str] = set()
    total = 0
    plan = []
    for m in meta:
        if not isinstance(m, dict) or set(m) - _TENSOR_KEYS:
            raise FrameError("tensor descriptor has unknown/extra keys")
        name, dt, shape = m.get("name"), m.get("dtype"), m.get("shape")
        if not isinstance(name, str) or not name:
            raise FrameError("bad tensor name")
        if name in names_seen:
            raise FrameError(f"duplicate tensor name: {name}")
        names_seen.add(name)
        for bad in spec.forbid_substrings:
            if bad in name.lower():
                raise FrameError(f"forbidden substring {bad!r} in name {name!r}")
        if spec.allowed_names is not None:
            if name not in spec.allowed_names:
                raise FrameError(f"name not in whitelist: {name}")
        elif spec.allowed_name_prefixes and not any(name.startswith(p) for p in spec.allowed_name_prefixes):
            raise FrameError(f"name fails prefix whitelist: {name}")
        if dt not in spec.allowed_dtypes:
            raise FrameError(f"dtype not allowed: {dt}")
        if not isinstance(shape, list) or len(shape) > spec.max_rank:
            raise FrameError(f"rank too high or bad shape: {shape}")
        numel = 1
        for s in shape:
            if not isinstance(s, int) or s < 0 or s > spec.max_numel:
                raise FrameError(f"bad dim: {s}")
            numel *= s
        if numel > spec.max_numel:
            raise FrameError(f"numel too large: {numel}")
        nbytes = numel * _DTYPES[dt][1]
        if nbytes > spec.max_tensor_bytes:
            raise FrameError(f"tensor too large: {nbytes} bytes")
        total += nbytes
        if total > spec.max_total_bytes:
            raise FrameError(f"total payload too large: {total} bytes")
        plan.append((name, dt, shape, nbytes))

    if spec.expected_names is not None and names_seen != spec.expected_names:
        missing = spec.expected_names - names_seen
        extra = names_seen - spec.expected_names
        raise FrameError(f"target set mismatch (missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]})")

    body = blob[8 + hlen:]
    if len(body) != total:
        raise FrameError(f"body length {len(body)} != declared {total} (truncated/padded)")

    # ---- now allocate from validated body ----
    flat: Dict[str, torch.Tensor] = {}
    off = 0
    for name, dt, shape, nbytes in plan:
        raw = bytearray(body[off:off + nbytes]); off += nbytes
        tdt = _DTYPES[dt][0]
        if nbytes == 0:
            t = torch.empty(shape, dtype=tdt)
        else:
            t = torch.frombuffer(raw, dtype=torch.uint8).view(tdt).reshape(shape).clone()
        flat[name] = t
    return _unflatten(flat)


def decode_local_trusted(blob: bytes):
    """Deserialize a DURABLE LOCAL checkpoint that has ALREADY been authenticated/decrypted
    inside the TDX (NOT network-attacker-controlled). torch.load is acceptable here by policy;
    this function exists to make the distinction explicit and auditable."""
    import io
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
