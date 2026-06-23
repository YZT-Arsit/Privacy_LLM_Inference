"""Untrusted GPU worker backends + a process-isolated worker client.

The GPU worker is the **untrusted** side of the protocol. It only ever sees the
:class:`~pllo.protocol.tee_gpu_messages.MaskedPrefillRequest` /
:class:`~pllo.protocol.tee_gpu_messages.MaskedDecodeRequest` messages (masked
embeddings + public metadata + the folded LM head) and returns masked logits.

Two backends:

* ``MockGpuBackend`` -- numpy identity "decoder": next-token logits depend only
  on the current masked hidden state via the folded head. Lets the whole
  trusted/untrusted round trip run with no torch and exact recovery. It stores
  only a masked KV cache (the masked hidden states it was handed).
* ``Qwen7BGpuBackend`` -- lazy-torch adapter for the real masked Qwen pipeline.
  Constructible with numpy only (no torch import at construction); a real
  forward requires CUDA + a checkpoint and is meant to run on the GPU server.
  ``tee_used`` is ``False`` either way -- the model never enters the TEE.

``LocalGpuWorker`` runs a chosen backend in a separate ``spawn`` process and
talks to it over a ``Pipe`` -- this is the ``local_two_process`` deployment and
lets a recorder capture exactly what crosses to the untrusted side.

numpy + standard library only (torch is imported lazily by the qwen backend).
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
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
    "GpuBackend",
    "MockGpuBackend",
    "Qwen7BGpuBackend",
    "make_gpu_backend",
    "LocalGpuWorker",
]


# ---------------------------------------------------------------------------
# Backend interface + implementations
# ---------------------------------------------------------------------------


class GpuBackend(ABC):
    """Untrusted masked backend. Sees masked tensors + public metadata only."""

    name: str
    tee_used: bool = False                  # the GPU side is NEVER a TEE

    @abstractmethod
    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse: ...

    @abstractmethod
    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse: ...

    @abstractmethod
    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used}


class MockGpuBackend(GpuBackend):
    """Identity-decoder mock: ``masked_logits = masked_hidden @ folded_head``.

    With an identity decoder the next-token logits depend only on the last
    masked hidden state, so ``X_tilde @ W_tilde = (X @ N) @ (N^{-1} W M) =
    L @ M`` -- the trusted side recovers the exact plaintext logits. Stores only
    the masked hidden states it receives (the "masked KV cache")."""

    name = "mock"
    tee_used = False

    def __init__(self) -> None:
        self._folded_head: np.ndarray | None = None
        self._session: str | None = None
        self._masked_kv: list[np.ndarray] = []   # masked hidden states only

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        if req.folded_lm_head is None:
            raise ValueError("mock backend requires a folded LM head")
        self._folded_head = np.asarray(req.folded_lm_head)
        self._session = req.session_id
        self._masked_kv = []
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False,
            notes="mock identity decoder; masked KV cache only; no TEE")

    def _logits(self, masked_hidden_last: np.ndarray) -> np.ndarray:
        assert self._folded_head is not None
        return masked_hidden_last @ self._folded_head        # [B, V]

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        x = np.asarray(req.masked_embeddings)                # [B, T, H] masked
        for t in range(x.shape[1]):                          # store masked KV
            self._masked_kv.append(x[:, t, :])
        masked_logits = self._logits(x[:, -1, :])
        return MaskedPrefillResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv))

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        x = np.asarray(req.masked_embedding)                 # [B, 1, H] masked
        self._masked_kv.append(x[:, -1, :])
        masked_logits = self._logits(x[:, -1, :])
        return MaskedDecodeResponse(
            session_id=req.session_id, masked_logits=masked_logits,
            kv_cache_len=len(self._masked_kv))


class Qwen7BGpuBackend(GpuBackend):
    """Lazy-torch adapter for the real masked Qwen pipeline (GPU server only).

    Construction is numpy-only (no torch import) so the class can be imported and
    audited in a CPU/test environment. A real ``prefill``/``decode`` requires
    CUDA + a ModelScope checkpoint and the Stage 8.4 masked pipeline; that path
    is intended to run on the GPU server. ``tee_used`` is always ``False``."""

    name = "qwen7b"
    tee_used = False

    def __init__(self, model_path: str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16") -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._session: str | None = None
        self._public: dict[str, Any] = {}

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        self._session = req.session_id
        self._public = dict(req.public_metadata)
        return BoundaryInitResponse(
            session_id=req.session_id, ok=True, gpu_backend=self.name,
            tee_used_on_gpu=False,
            notes="qwen7b masked pipeline runs on the untrusted GPU; not a TEE")

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "tee_used": self.tee_used,
                "model_path": self.model_path, "device": self.device,
                "dtype": self.dtype}

    def _require_pipeline(self):  # pragma: no cover - GPU server only
        raise NotImplementedError(
            "Qwen7BGpuBackend.prefill/decode requires CUDA + a ModelScope "
            "checkpoint + the Stage 8.4 masked pipeline; run on the GPU server. "
            "The mock backend exercises the protocol/audit locally.")

    def prefill(self, req: MaskedPrefillRequest):  # pragma: no cover
        self._require_pipeline()

    def decode(self, req: MaskedDecodeRequest):  # pragma: no cover
        self._require_pipeline()


def make_gpu_backend(name: str, **kwargs: Any) -> GpuBackend:
    if name == "mock":
        return MockGpuBackend()
    if name == "qwen7b":
        return Qwen7BGpuBackend(**kwargs)
    raise ValueError(f"unknown gpu backend {name!r}; expected 'mock' or 'qwen7b'")


# ---------------------------------------------------------------------------
# Process-isolated worker (the untrusted domain)
# ---------------------------------------------------------------------------


def _gpu_worker_loop(conn: Any, backend_name: str,
                     backend_kwargs: dict[str, Any]) -> None:
    """Untrusted worker process: build a backend, service masked-tensor RPCs."""
    backend = make_gpu_backend(backend_name, **backend_kwargs)
    while True:
        try:
            method, payload = conn.recv()
        except EOFError:
            break
        try:
            if method == "shutdown":
                conn.send(("ok", None))
                break
            if method == "init":
                conn.send(("ok", backend.init(payload)))
            elif method == "prefill":
                conn.send(("ok", backend.prefill(payload)))
            elif method == "decode":
                conn.send(("ok", backend.decode(payload)))
            elif method == "describe":
                conn.send(("ok", backend.describe()))
            else:
                conn.send(("error", f"unknown method {method!r}"))
        except Exception as exc:  # noqa: BLE001 - propagate to parent
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
    conn.close()


class LocalGpuWorker:
    """Drives a GPU backend in a separate (untrusted) ``spawn`` process.

    Optionally takes a ``recorder`` callback ``(direction, method, message)``
    invoked for every object crossing the boundary, so the orchestrator can log
    the exact GPU-inbound / GPU-outbound traffic for the security audit."""

    def __init__(self, backend_name: str = "mock",
                 backend_kwargs: dict[str, Any] | None = None,
                 recorder: Any = None) -> None:
        self.backend_name = backend_name
        self.backend_kwargs = dict(backend_kwargs or {})
        self._recorder = recorder
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_gpu_worker_loop,
            args=(child_conn, backend_name, self.backend_kwargs), daemon=True)
        self._proc.start()
        child_conn.close()
        self._closed = False

    def _rpc(self, method: str, payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("gpu worker is closed")
        if self._recorder is not None and payload is not None:
            self._recorder("inbound", method, payload)
        self._parent_conn.send((method, payload))
        status, result = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"GPU worker error: {result}")
        if self._recorder is not None and result is not None:
            self._recorder("outbound", method, result)
        return result

    def init(self, req: BoundaryInitRequest) -> BoundaryInitResponse:
        return self._rpc("init", req)

    def prefill(self, req: MaskedPrefillRequest) -> MaskedPrefillResponse:
        return self._rpc("prefill", req)

    def decode(self, req: MaskedDecodeRequest) -> MaskedDecodeResponse:
        return self._rpc("decode", req)

    def describe(self) -> dict[str, Any]:
        return self._rpc("describe", {})

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._parent_conn.send(("shutdown", None))
            self._parent_conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            pass
        finally:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)

    def __enter__(self) -> "LocalGpuWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
