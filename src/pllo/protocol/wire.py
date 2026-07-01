"""JSON wire serialization for the TEE <-> GPU protocol messages.

Encodes the protocol dataclasses (which carry numpy arrays) to JSON-safe dicts
and back, so the trusted boundary and the untrusted GPU worker can talk over
HTTP across machines. numpy arrays are encoded as base64 of their raw bytes plus
dtype + shape (exact, lossless). Only the known GPU-channel message types can be
decoded -- an unknown ``__msgtype__`` is rejected.

stdlib + numpy only.
"""

from __future__ import annotations

import base64
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest,
    BoundaryInitResponse,
    MaskedDecodeRequest,
    MaskedDecodeResponse,
    MaskedPrefillRequest,
    MaskedPrefillResponse,
)

__all__ = [
    "MESSAGE_TYPES",
    "encode_value",
    "decode_value",
    "encode_message",
    "decode_message",
]

# Whitelist of decodable message types (request + response, both directions).
MESSAGE_TYPES = {
    cls.__name__: cls
    for cls in (
        BoundaryInitRequest, BoundaryInitResponse,
        MaskedPrefillRequest, MaskedPrefillResponse,
        MaskedDecodeRequest, MaskedDecodeResponse,
    )
}


def _bf16_tensor_or_none(obj: Any):
    """Return ``obj`` if it is a torch bfloat16 tensor, else None (lazy torch)."""
    mod = getattr(type(obj), "__module__", "") or ""
    if mod.split(".", 1)[0] != "torch":
        return None
    import torch                                            # lazy; both sides have it
    if isinstance(obj, torch.Tensor) and obj.dtype == torch.bfloat16:
        return obj
    return None


def encode_value(obj: Any) -> Any:
    """Recursively encode a value to JSON-safe form (ndarrays -> base64 dict).

    bfloat16 torch tensors are encoded losslessly as their raw 2-byte bits with a
    ``dtype == "bfloat16"`` tag (numpy has no native bfloat16). This lets the GPU
    worker return masked logits in their native bf16 compute dtype -- HALF the
    wire size of the fp32 upcast, and BIT-IDENTICAL, since the trusted boundary
    upcasts bf16 -> fp32 for recovery exactly as it did with the fp32 payload."""
    bf16 = _bf16_tensor_or_none(obj)
    if bf16 is not None:
        import torch
        t = bf16.detach().to("cpu").contiguous()
        raw = t.view(torch.uint16).numpy().tobytes()       # raw bf16 bit pattern
        return {
            "__ndarray__": True,
            "dtype": "bfloat16",
            "shape": list(t.shape),
            "b64": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_value(v) for v in obj]
    return obj


def decode_value(obj: Any) -> Any:
    """Inverse of :func:`encode_value`."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            raw = base64.b64decode(obj["b64"].encode("ascii"))
            if obj["dtype"] == "bfloat16":
                import torch                                # lazy; both sides have it
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
                return t.reshape(obj["shape"]).clone()     # owns data, writable
            arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"]).copy()      # writable, owns data
        return {k: decode_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_value(v) for v in obj]
    return obj


def encode_message(msg: Any) -> dict[str, Any]:
    """Encode a protocol message dataclass to a JSON-safe dict."""
    if not is_dataclass(msg):
        raise TypeError(f"not a dataclass message: {type(msg)!r}")
    out: dict[str, Any] = {"__msgtype__": type(msg).__name__}
    for f in fields(msg):
        out[f.name] = encode_value(getattr(msg, f.name))
    return out


def decode_message(payload: dict[str, Any]) -> Any:
    """Reconstruct a protocol message from a decoded dict (whitelisted types)."""
    tname = payload.get("__msgtype__")
    cls = MESSAGE_TYPES.get(tname)
    if cls is None:
        raise ValueError(f"unknown or forbidden message type {tname!r}")
    kwargs = {k: decode_value(v) for k, v in payload.items()
              if k != "__msgtype__"}
    return cls(**kwargs)
