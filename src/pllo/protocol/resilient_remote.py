"""Resilient HTTP client for the untrusted H800 GPU worker.

Wraps :class:`pllo.protocol.remote.RemoteGpuWorker` (or any object exposing
``health/init/prefill/decode/close``) with the robustness the AAAI generation
benchmark needs on a memory-constrained server whose worker may drop, time out, or
return a transient 5xx:

* per-call **retry with exponential backoff** (+ jitter-free deterministic delays
  so runs are reproducible and tests are stable);
* **reconnect** on connection-refused / timeout / 5xx (the inner client's TCP
  connection is closed so the next attempt redials);
* per-request timeout;
* measured ``retry_count`` / ``reconnect_count`` / ``last_error`` / ``worker_url``
  for the report.

Security: this layer only ever retries on *transport* failures. It never logs the
request payload, prompt, plaintext, or token ids -- only the method name, attempt
number, and a sanitized error class/message. The boundary's own audit
(``RemoteGpuWorker._audit_encoded``) still runs on every forwarded request.

stdlib only (time / typing). No torch, no model code.
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = [
    "WorkerUnavailable",
    "is_retriable_error",
    "ResilientRemoteGpuWorker",
]


class WorkerUnavailable(RuntimeError):
    """A GPU-worker request failed after exhausting all retries."""


_RETRIABLE_SUBSTRINGS = (
    "connection refused", "connection reset", "connection aborted",
    "timed out", "timeout", "temporarily unavailable", "broken pipe",
    "http 500", "http 502", "http 503", "http 504",
    "bad gateway", "service unavailable", "gateway timeout",
    "remote end closed", "connection failed",
)


def is_retriable_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient transport failure worth retrying.

    Connection/OS/timeout errors are always retriable; a generic ``RuntimeError``
    is retriable only when its message names a 5xx / connection-level failure (so
    a deterministic 4xx / validation error is NOT retried)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _RETRIABLE_SUBSTRINGS)


class ResilientRemoteGpuWorker:
    """Retry/backoff/reconnect wrapper around a remote GPU-worker client."""

    def __init__(self, url: str, gpu_backend: str = "qwen7b_folded_package", *,
                 max_retries: int = 5, backoff_base_sec: float = 0.5,
                 backoff_max_sec: float = 30.0, per_request_timeout: float = 120.0,
                 recorder: Any = None, request_worker_timing: bool = False,
                 sleep_fn: Callable[[float], None] | None = None,
                 client_factory: Callable[[], Any] | None = None) -> None:
        self.url = url.rstrip("/") if url else url
        self.gpu_backend = gpu_backend
        self.max_retries = int(max_retries)
        self.backoff_base_sec = float(backoff_base_sec)
        self.backoff_max_sec = float(backoff_max_sec)
        self.per_request_timeout = float(per_request_timeout)
        self._recorder = recorder
        self._request_worker_timing = bool(request_worker_timing)
        self._sleep = sleep_fn or time.sleep
        # client_factory lets tests inject a fake transport; default builds the
        # real RemoteGpuWorker lazily (so importing this module needs no network).
        self._factory = client_factory or self._default_factory
        self._client: Any = None
        # measured robustness counters (public, no secrets)
        self.retry_count = 0
        self.reconnect_count = 0
        self.request_count = 0
        self.last_error: str | None = None

    def _default_factory(self):
        from pllo.protocol.remote import RemoteGpuWorker
        return RemoteGpuWorker(
            self.url, self.gpu_backend, recorder=self._recorder,
            timeout=self.per_request_timeout,
            request_worker_timing=self._request_worker_timing)

    def _ensure_client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def _drop_client(self) -> None:
        """Close + discard the inner client so the next call redials."""
        c = self._client
        self._client = None
        if c is not None:
            try:
                c.close()
            except Exception:                                # noqa: BLE001
                pass

    def _backoff_delay(self, attempt: int) -> float:
        """Deterministic exponential backoff (attempt is 1-based)."""
        return min(self.backoff_max_sec,
                   self.backoff_base_sec * (2 ** (attempt - 1)))

    def _call_with_retry(self, method: str, invoke: Callable[[Any], Any]) -> Any:
        """Invoke ``invoke(client)`` with retry/backoff/reconnect."""
        self.request_count += 1
        last: BaseException | None = None
        for attempt in range(1, self.max_retries + 2):       # 1 try + max_retries
            try:
                return invoke(self._ensure_client())
            except Exception as exc:                         # noqa: BLE001
                last = exc
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                retriable = is_retriable_error(exc)
                # transport failure -> drop the connection so we redial next time
                self._drop_client()
                self.reconnect_count += 1
                if not retriable or attempt > self.max_retries:
                    break
                self.retry_count += 1
                delay = self._backoff_delay(attempt)
                if self._recorder is not None:
                    # method name + attempt only; never the payload
                    try:
                        self._recorder("retry", method,
                                       {"attempt": attempt, "delay_sec": delay,
                                        "error_class": type(exc).__name__})
                    except Exception:                        # noqa: BLE001
                        pass
                self._sleep(delay)
        raise WorkerUnavailable(
            "GPU worker %s failed after %d attempt(s) on %s: %s"
            % (self.url, self.max_retries + 1, method, last)) from last

    # -- worker API (drop-in for RemoteGpuWorker) -----------------------------
    def health(self) -> dict:
        return self._call_with_retry("health", lambda c: c.health())

    def init(self, req):
        return self._call_with_retry("init", lambda c: c.init(req))

    def prefill(self, req):
        return self._call_with_retry("prefill", lambda c: c.prefill(req))

    def decode(self, req):
        return self._call_with_retry("decode", lambda c: c.decode(req))

    def wait_healthy(self, *, attempts: int | None = None) -> dict:
        """Block (with backoff) until ``/health`` succeeds or retries exhaust."""
        saved = self.max_retries
        if attempts is not None:
            self.max_retries = int(attempts)
        try:
            return self.health()
        finally:
            self.max_retries = saved

    def stats(self) -> dict:
        return {"worker_url": self.url, "retry_count": self.retry_count,
                "reconnect_count": self.reconnect_count,
                "request_count": self.request_count,
                "last_error": self.last_error}

    def close(self) -> None:
        self._drop_client()

    def __enter__(self) -> "ResilientRemoteGpuWorker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
