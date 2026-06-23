"""Message schemas for the TEE-boundary <-> untrusted-GPU-worker protocol.

This module defines the *only* objects that are allowed to cross from the
trusted boundary domain to the untrusted GPU worker, and back. Every field here
is **public** by construction:

* the GPU worker receives masked embeddings + public metadata only;
* it returns masked logits + a public KV-cache length only.

It NEVER receives the raw prompt, ``input_ids``, generated token ids, recovered
logits, tokenizer output, or any mask secret (perm / signs / scale / seed /
``MaskHandles``). The audit functions in :mod:`pllo.protocol.security_audit`
verify exactly that against the message objects recorded in a
:class:`ProtocolTrace`.

``RecoveredTokenResponse`` is the trusted boundary's per-step output to the
client. It is listed here for completeness of the protocol but is a
*trusted-side* message and is never placed on the GPU channel.

numpy + standard library only -- no torch, no transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterator

import numpy as np

__all__ = [
    "BoundaryInitRequest",
    "BoundaryInitResponse",
    "MaskedPrefillRequest",
    "MaskedPrefillResponse",
    "MaskedDecodeRequest",
    "MaskedDecodeResponse",
    "RecoveredTokenResponse",
    "ProtocolTrace",
    "GPU_INBOUND_TYPES",
    "GPU_OUTBOUND_TYPES",
    "iter_named_values",
    "message_array_nbytes",
]


# ---------------------------------------------------------------------------
# GPU-channel messages (trusted boundary -> untrusted GPU worker)
# ---------------------------------------------------------------------------


@dataclass
class BoundaryInitRequest:
    """Trusted -> GPU. Public model metadata + the folded/masked LM head.

    The folded head ``W_tilde = N^{-1} @ W @ M_vocab`` bakes the masks into the
    weights but is not itself a recoverable secret (you cannot read off the
    permutation / signs / scale from it without the trusted handles). All other
    fields are public hyper-parameters."""
    session_id: str
    hidden_size: int
    vocab_size: int
    num_layers: int
    dtype: str
    gpu_backend: str                       # "mock" | "qwen7b"
    folded_lm_head: np.ndarray | None = None   # [H, V] public folded artifact
    public_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryInitResponse:
    """GPU -> trusted. Acknowledgement + the GPU's own TEE flag (always False)."""
    session_id: str
    ok: bool
    gpu_backend: str
    tee_used_on_gpu: bool                   # MUST be False (worker is untrusted)
    notes: str = ""


@dataclass
class MaskedPrefillRequest:
    """Trusted -> GPU. Masked prompt embeddings + public positions only."""
    session_id: str
    masked_embeddings: np.ndarray          # [B, T, H] (masked)
    positions: list[int]                   # public absolute positions
    batch_size: int
    seq_len: int


@dataclass
class MaskedPrefillResponse:
    """GPU -> trusted. Masked last-position logits + public KV length."""
    session_id: str
    masked_logits: np.ndarray              # [B, V] (masked)
    kv_cache_len: int


@dataclass
class MaskedDecodeRequest:
    """Trusted -> GPU. One masked token embedding + its public position."""
    session_id: str
    masked_embedding: np.ndarray           # [B, 1, H] (masked)
    position: int                          # public absolute position
    step: int


@dataclass
class MaskedDecodeResponse:
    """GPU -> trusted. Masked next-token logits + public KV length."""
    session_id: str
    masked_logits: np.ndarray              # [B, V] (masked)
    kv_cache_len: int


# ---------------------------------------------------------------------------
# Trusted-side message (NEVER placed on the GPU channel)
# ---------------------------------------------------------------------------


@dataclass
class RecoveredTokenResponse:
    """Trusted boundary -> client. The recovered next token for one step.

    Holds plaintext (token ids); it stays inside the trusted domain. We record
    only the *nbytes* of the recovered logits here, not the array itself, so
    that a ``ProtocolTrace`` can be inspected/serialised without carrying
    plaintext logits around."""
    step: int
    next_token_ids: list[int]
    recovered_logits_nbytes: int


# ---------------------------------------------------------------------------
# Trace (audit subject)
# ---------------------------------------------------------------------------

GPU_INBOUND_TYPES = (
    BoundaryInitRequest,
    MaskedPrefillRequest,
    MaskedDecodeRequest,
)
GPU_OUTBOUND_TYPES = (
    BoundaryInitResponse,
    MaskedPrefillResponse,
    MaskedDecodeResponse,
)


@dataclass
class ProtocolTrace:
    """Everything that crossed the trusted/untrusted boundary, for auditing.

    ``gpu_inbound_messages`` / ``gpu_outbound_messages`` hold the *exact* objects
    sent to / received from the untrusted GPU worker. The security audit scans
    these for plaintext or mask-secret leaks."""
    boundary_backend: str
    gpu_backend: str
    max_new_tokens: int
    tee_used_on_gpu: bool
    boundary_calls: dict[str, int] = field(default_factory=dict)
    gpu_calls: dict[str, int] = field(default_factory=dict)
    trusted_bytes: int = 0
    gpu_bytes: int = 0
    gpu_inbound_messages: list[Any] = field(default_factory=list)
    gpu_outbound_messages: list[Any] = field(default_factory=list)
    recovered_tokens: list[int] = field(default_factory=list)

    def record_gpu_inbound(self, msg: Any) -> None:
        self.gpu_inbound_messages.append(msg)
        self.gpu_bytes += message_array_nbytes(msg)
        self.gpu_calls[type(msg).__name__] = (
            self.gpu_calls.get(type(msg).__name__, 0) + 1)

    def record_gpu_outbound(self, msg: Any) -> None:
        self.gpu_outbound_messages.append(msg)
        self.gpu_bytes += message_array_nbytes(msg)

    def bump_boundary(self, op: str) -> None:
        self.boundary_calls[op] = self.boundary_calls.get(op, 0) + 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def iter_named_values(obj: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, leaf_value)`` for every leaf inside ``obj``.

    Recurses through dataclasses, dicts, lists and tuples; everything else is a
    leaf. Used by the security audit to scan messages structurally."""
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            yield from iter_named_values(getattr(obj, f.name),
                                         f"{path}.{f.name}" if path else f.name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_named_values(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_named_values(v, f"{path}[{i}]")
    else:
        yield path, obj


def message_array_nbytes(obj: Any) -> int:
    """Total bytes of all ndarray leaves inside ``obj`` (for byte accounting)."""
    total = 0
    for _, v in iter_named_values(obj):
        if isinstance(v, np.ndarray):
            total += int(v.nbytes)
    return total
