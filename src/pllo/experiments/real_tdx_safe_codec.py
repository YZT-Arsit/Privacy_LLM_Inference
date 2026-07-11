"""Safe tensor wire codec for the real-TDX training boundary (NO pickle).

The untrusted GPU/host controls request payloads. The trusted guest MUST NOT call
``torch.load`` / pickle on them (arbitrary code execution). This codec frames a
nested message as:

    b"PLLOSAFE" | version(1) | header_len(4 LE) | header_json | raw tensor bytes*

``header_json`` = {"structure": <message with tensor leaves replaced by
{"__t__": i}>, "tensors": [{"dtype","shape","nbytes"}...]}. Tensors are raw
little-endian bytes reconstructed with ``torch.frombuffer`` (no deserialization of
code). Every limit is checked on the header BEFORE any allocation.

Whitelist dtypes only; bounded rank/dims/count/bytes; overflow-safe numel*itemsize;
exact-length payload (reject truncated/extra); positional refs used exactly once;
explicit NaN/Inf policy; dtype/shape re-verified after decode.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import torch

MAGIC = b"PLLOSAFE"
VERSION = 1

_DTYPE = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16, "int32": torch.int32, "int64": torch.int64}
_ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int32": 4, "int64": 8}


class CodecError(Exception):
    """Any malformed / out-of-policy payload. Fail-closed; no partial decode."""


@dataclass(frozen=True)
class Limits:
    max_tensors: int = 512
    max_rank: int = 6
    max_dim: int = 1_000_000
    max_tensor_bytes: int = 512 * 1024 * 1024      # 512 MiB / tensor
    max_total_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB / request
    max_header_bytes: int = 4 * 1024 * 1024        # 4 MiB header
    forbid_nan_inf: bool = False                   # explicit policy


def _dtype_name(t: torch.Tensor) -> str:
    name = str(t.dtype).replace("torch.", "")
    if name not in _DTYPE:
        raise CodecError(f"dtype {name!r} not in whitelist {sorted(_DTYPE)}")
    return name


def encode_message(obj) -> bytes:
    tensors: list[torch.Tensor] = []
    meta: list[dict] = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            name = _dtype_name(o)
            t = o.detach().contiguous().cpu()
            raw = t.flatten().view(torch.uint8).numpy().tobytes()
            meta.append({"dtype": name, "shape": list(t.shape), "nbytes": len(raw)})
            tensors.append(raw)
            return {"__t__": len(tensors) - 1}
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        raise CodecError(f"unserializable value of type {type(o).__name__}")

    structure = walk(obj)
    header = json.dumps({"structure": structure, "tensors": meta}).encode()
    out = bytearray()
    out += MAGIC + bytes([VERSION]) + struct.pack("<I", len(header)) + header
    for raw in tensors:
        out += raw
    return bytes(out)


def decode_message(data: bytes, *, limits: Limits = Limits()):
    if not isinstance(data, (bytes, bytearray)):
        raise CodecError("payload not bytes")
    if len(data) < len(MAGIC) + 1 + 4:
        raise CodecError("truncated frame")
    if data[:len(MAGIC)] != MAGIC:
        raise CodecError("bad magic (refusing to deserialize; not a safe frame)")
    off = len(MAGIC)
    version = data[off]; off += 1
    if version != VERSION:
        raise CodecError(f"unsupported version {version}")
    (hlen,) = struct.unpack("<I", data[off:off + 4]); off += 4
    if hlen > limits.max_header_bytes:
        raise CodecError("header too large")
    if off + hlen > len(data):
        raise CodecError("truncated header")
    try:
        header = json.loads(data[off:off + hlen].decode())
    except Exception as exc:
        raise CodecError(f"bad header json: {exc}")
    off += hlen
    meta = header.get("tensors")
    structure = header.get("structure")
    if not isinstance(meta, list) or structure is None:
        raise CodecError("malformed header")
    if len(meta) > limits.max_tensors:
        raise CodecError(f"too many tensors ({len(meta)} > {limits.max_tensors})")

    # ---- validate ALL tensor metadata BEFORE allocating anything ----
    total = 0
    for i, m in enumerate(meta):
        if set(m) < {"dtype", "shape", "nbytes"}:
            raise CodecError(f"tensor {i} missing fields")
        dt = m["dtype"]
        if dt not in _DTYPE:
            raise CodecError(f"tensor {i} dtype {dt!r} not whitelisted")
        shape = m["shape"]
        if not isinstance(shape, list) or len(shape) > limits.max_rank:
            raise CodecError(f"tensor {i} rank {len(shape) if isinstance(shape, list) else '?'} > {limits.max_rank}")
        numel = 1
        for d in shape:
            if not isinstance(d, int) or d < 0 or d > limits.max_dim:
                raise CodecError(f"tensor {i} bad dim {d}")
            numel *= d
            if numel * _ITEMSIZE[dt] > limits.max_tensor_bytes:  # overflow-safe running check
                raise CodecError(f"tensor {i} exceeds max_tensor_bytes")
        nbytes = m["nbytes"]
        if not isinstance(nbytes, int) or nbytes != numel * _ITEMSIZE[dt]:
            raise CodecError(f"tensor {i} nbytes {nbytes} != numel*itemsize {numel * _ITEMSIZE[dt]}")
        total += nbytes
        if total > limits.max_total_bytes:
            raise CodecError("request exceeds max_total_bytes")
    if off + total != len(data):
        raise CodecError(f"payload length mismatch (extra/truncated bytes): "
                         f"declared {off + total}, got {len(data)}")

    # ---- materialize tensors ----
    # Memory ownership (Gate 2 §1-A): torch.frombuffer VIEWS the request buffer, so
    # we must clone into fresh, contiguous, service-owned storage before the tensor
    # reaches trusted computation or state. The per-tensor `bytearray(...)` copy plus
    # `.clone()` guarantees (a) no alias to the mutable request bytes, (b) no alias
    # between decoded tensors, (c) contiguous standard layout. The transient view and
    # the source chunk are released immediately after cloning.
    tensors = []
    for m in meta:
        dt = m["dtype"]; nb = m["nbytes"]
        chunk = bytearray(data[off:off + nb]); off += nb
        view = torch.frombuffer(chunk, dtype=torch.uint8).view(_DTYPE[dt]).reshape(m["shape"])
        if tuple(view.shape) != tuple(m["shape"]) or str(view.dtype).replace("torch.", "") != dt:
            raise CodecError("post-decode dtype/shape re-verification failed")
        if limits.forbid_nan_inf and view.is_floating_point() and not torch.isfinite(view).all():
            raise CodecError("NaN/Inf present but forbidden by policy")
        owned = view.clone().contiguous()     # fresh service-owned, unaliased storage
        if owned.data_ptr() == view.data_ptr() or not owned.is_contiguous():
            raise CodecError("failed to obtain unaliased contiguous storage")
        tensors.append(owned)
        del view, chunk                        # release the request-buffer view

    used = [0] * len(tensors)

    def rebuild(o):
        if isinstance(o, dict):
            if "__t__" in o and len(o) == 1:
                idx = o["__t__"]
                if not isinstance(idx, int) or not (0 <= idx < len(tensors)):
                    raise CodecError("tensor ref out of range")
                used[idx] += 1
                if used[idx] > 1:
                    raise CodecError("duplicate tensor ref")
                return tensors[idx]
            return {k: rebuild(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rebuild(v) for v in o]
        return o

    result = rebuild(structure)
    if any(u != 1 for u in used):
        raise CodecError("unused/unexpected tensor payload (ref mismatch)")
    return result
