"""Process-isolated trusted runtime.

Runs the reference :class:`SimulatedTrustedRuntime` inside a separate Python
process and talks to it over a multiprocessing ``Pipe``. This models the real
deployment where the trusted runtime is an isolated domain (a TDX guest) that
the untrusted host drives by message passing -- and it measures the IPC cost of
the boundary. Because the worker IS the simulated runtime, results are
numerically identical. Mask handles never cross the pipe (they stay in the
worker, i.e. the trusted domain). numpy only.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from pllo.tee.runtime_api import (
    AttestationReport,
    MaskedEmbeddingPacket,
    MaskedLogitsPacket,
    SamplingResult,
    TEEConfig,
    TrustedRuntime,
)

__all__ = ["ProcessTrustedRuntime"]


def _worker_loop(conn: Any, config: TEEConfig) -> None:
    """Trusted-domain worker: own the masks + embedding table, service RPCs."""
    from pllo.tee.simulated_runtime import SimulatedTrustedRuntime
    rt = SimulatedTrustedRuntime(config)
    while True:
        try:
            method, payload = conn.recv()
        except EOFError:
            break
        try:
            if method == "shutdown":
                conn.send(("ok", None))
                break
            if method == "attest":
                conn.send(("ok", rt.attest()))
            elif method == "setup_masks":
                rt.setup_masks(payload)
                conn.send(("ok", None))     # handles NEVER leave the worker
            elif method == "embed_and_mask":
                conn.send(("ok", rt.embed_and_mask(payload)))
            elif method == "recover_logits":
                conn.send(("ok", rt.recover_logits(payload)))
            elif method == "sample":
                conn.send(("ok", rt.sample(payload)))
            else:
                conn.send(("error", f"unknown method {method!r}"))
        except Exception as exc:  # noqa: BLE001 - propagate to parent
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
    conn.close()


class ProcessTrustedRuntime(TrustedRuntime):
    """Drives a :class:`SimulatedTrustedRuntime` in a separate process."""

    def __init__(self, config: TEEConfig) -> None:
        self.config = config
        # spawn: deterministic fresh import; never inherits parent ML state.
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_loop, args=(child_conn, config), daemon=True)
        self._proc.start()
        child_conn.close()
        self._closed = False

    def _rpc(self, method: str, payload: Any = None) -> Any:
        if self._closed:
            raise RuntimeError("runtime is closed")
        self._parent_conn.send((method, payload))
        status, result = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"TEE worker error: {result}")
        return result

    def attest(self) -> AttestationReport:
        return self._rpc("attest")

    def setup_masks(self, seed: int | None = None):
        # Returns None: in the process backend mask handles stay in the worker
        # (the trusted domain) and are never copied to the untrusted parent.
        return self._rpc("setup_masks", seed)

    def embed_and_mask(self, input_ids: np.ndarray) -> MaskedEmbeddingPacket:
        return self._rpc("embed_and_mask", np.asarray(input_ids))

    def recover_logits(self, packet: MaskedLogitsPacket) -> np.ndarray:
        return self._rpc("recover_logits", packet)

    def sample(self, recovered_logits: np.ndarray) -> SamplingResult:
        return self._rpc("sample", np.asarray(recovered_logits))

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

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
