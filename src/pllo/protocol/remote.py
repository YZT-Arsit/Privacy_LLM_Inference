"""Cross-machine transport for the TEE <-> GPU protocol (stdlib HTTP).

Two halves:

* :class:`GpuWorkerServer` -- runs on the **untrusted** H800/GPU server. A
  threaded stdlib HTTP server exposing ``/health`` + ``/init`` + ``/prefill`` +
  ``/decode``. It accepts only masked/public protocol messages, **rejects** any
  request body containing a forbidden plaintext field, runs the chosen GPU
  backend, and returns masked logits. It always reports ``tee_used_on_gpu=False``.
* :class:`RemoteGpuWorker` -- the client used by the trusted boundary (inside the
  TDX guest). Same ``init/prefill/decode/close`` interface as ``LocalGpuWorker``,
  but POSTs encoded messages to the remote worker. It imports no model code.

stdlib + numpy only. The qwen7b backend lazy-imports torch **inside the server
process** (never in the boundary client).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest,
    MaskedDecodeRequest,
    MaskedPrefillRequest,
)
from pllo.protocol.wire import decode_message, encode_message

from pllo.runtime.obfuscation_schedule import (  # noqa: E402
    SCHEDULE_SECRET_FIELDS,
    audit_gpu_payload_no_schedule_secrets,
    audit_worker_package_no_schedule_secrets,
)

__all__ = [
    "FORBIDDEN_WIRE_FIELDS",
    "forbidden_fields_in_payload",
    "audit_gpu_payload_no_schedule_secrets",
    "audit_worker_package_no_schedule_secrets",
    "GpuWorkerServer",
    "RemoteGpuWorker",
    "run_gpu_worker_server",
]

# Names that must NEVER appear in a request body crossing to the GPU worker.
# Covers both generation-stage and LoRA-training-stage plaintext/secret fields.
FORBIDDEN_WIRE_FIELDS = frozenset({
    # generation-stage
    "raw_prompt", "prompt", "input_ids", "input_id", "generated_token_ids",
    "generated_tokens", "recovered_logits", "tokenizer_output",
    # training-stage (LoRA)
    "labels", "label", "train_examples", "training_examples",
    "tokenized_examples", "plain_hidden", "lora_a", "lora_b", "delta_w",
    "lora_grad_a", "lora_grad_b", "grad_a", "grad_b", "optimizer_state",
    "adam_m", "adam_v", "adapter_update",
    # mask secrets
    "mask_secret", "mask_secrets", "mask_handles", "handles", "residual_perm",
    "residual_inv_perm", "in_perm", "in_signs", "vocab_perm", "out_perm",
    "vocab_scale", "out_scale", "residual_signs", "seed",
    # precomputed-obfuscation-schedule secrets (also caught by the schedule
    # audit's substring matcher; listed here for the server's exact-match guard)
    *SCHEDULE_SECRET_FIELDS,
})

# Path -> message class expected in the request body.
_ROUTES = {
    "/init": BoundaryInitRequest,
    "/prefill": MaskedPrefillRequest,
    "/decode": MaskedDecodeRequest,
}


def forbidden_fields_in_payload(payload: Any,
                                names: frozenset[str] = FORBIDDEN_WIRE_FIELDS
                                ) -> list[str]:
    """Return dotted paths of any forbidden field names found anywhere in the
    (decoded JSON) request body. Empty == clean."""
    found: list[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                here = f"{path}.{k}" if path else str(k)
                if k in names:
                    found.append(here)
                walk(v, here)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(payload, "")
    return found


# ---------------------------------------------------------------------------
# Server (untrusted GPU side)
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence default logging
        pass

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        # Persistent connection ONLY when the client explicitly opts in via the
        # private header (pllo persistent transport). Default clients (no header)
        # get the historical close-after-response behaviour byte-for-byte.
        keep_alive = self.headers.get("X-PLLO-KeepAlive", "") == "1"
        self.close_connection = not keep_alive
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive" if keep_alive else "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            srv = self.server
            # Measured nonlinear-design execution evidence (empty until a run);
            # lets a trusted client retrieve the design B lift counters post-run
            # without any plaintext/secret crossing the channel.
            ev_fn = getattr(srv.backend, "nonlinear_execution_evidence", None)
            self._send_json(200, {
                "status": "ok", "gpu_backend": srv.gpu_backend_name,
                "tee_used_on_gpu": False,
                "nonlinear_backend": getattr(
                    srv.backend, "nonlinear_backend", None),
                "nonlinear_execution_evidence": ev_fn() if callable(ev_fn)
                else {},
                # measured server-side by compute backends; None until any run
                "peak_gpu_memory_mb": getattr(
                    srv.backend, "peak_gpu_memory_mb", None)})
        else:
            self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        srv = self.server
        t_handler0 = time.perf_counter()
        path = self.path.rstrip("/") or "/"
        if path not in _ROUTES:
            self._send_json(404, {"error": "not_found", "path": self.path})
            return
        # Worker-side timing ONLY when the client explicitly opts in (profiling
        # mode). Default clients get no timing work and a worker_timing=None field.
        want_timing = self.headers.get("X-PLLO-WorkerTiming", "") == "1"
        try:
            t_parse0 = time.perf_counter()
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            parse_s = time.perf_counter() - t_parse0
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": "bad_request", "detail": str(exc)})
            return

        # Defense-in-depth: refuse any plaintext/secret field outright.
        if srv.audit:
            bad = forbidden_fields_in_payload(payload)
            if bad:
                self._send_json(400, {"error": "forbidden_field", "fields": bad,
                                      "tee_used_on_gpu": False})
                return

        try:
            t_dec0 = time.perf_counter()
            msg = decode_message(payload)
            decode_s = time.perf_counter() - t_dec0
            if not isinstance(msg, _ROUTES[path]):
                raise ValueError(
                    f"expected {_ROUTES[path].__name__} at {path}, got "
                    f"{type(msg).__name__}")
            with srv.lock:
                try:
                    srv.backend.collect_worker_timing = bool(want_timing)
                except Exception:                            # noqa: BLE001
                    pass
                if path == "/init":
                    result = srv.backend.init(msg)
                elif path == "/prefill":
                    result = srv.backend.prefill(msg)
                else:
                    result = srv.backend.decode(msg)
            t_enc0 = time.perf_counter()
            encoded = encode_message(result)
            encode_s = time.perf_counter() - t_enc0
            encoded = self._attach_worker_timing(
                encoded, want_timing=want_timing, parse_s=parse_s,
                decode_s=decode_s, encode_s=encode_s, t_handler0=t_handler0)
            self._send_json(200, encoded)
        except NotImplementedError as exc:
            self._send_json(501, {"error": "not_implemented", "detail": str(exc),
                                  "tee_used_on_gpu": False})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": "worker_error", "detail": str(exc)})

    def _attach_worker_timing(self, encoded: dict, *, want_timing: bool,
                              parse_s, decode_s, encode_s, t_handler0) -> dict:
        """Merge the worker's forward timing (already on ``encoded.worker_timing``)
        with the server handler's stage timings, or strip it. Default (no opt-in):
        worker_timing is forced to None so responses match the historical path."""
        if "worker_timing" not in encoded:
            return encoded
        if not want_timing:
            encoded["worker_timing"] = None
            return encoded
        wt = encoded.get("worker_timing")
        if not isinstance(wt, dict):
            encoded["worker_timing"] = None
            return encoded
        from pllo.protocol.worker_timing import (
            audit_worker_timing_no_secrets, merge_server_timing)
        # response size: the encoded body sans the (about-to-be-added) server
        # stage fields -- dominated by the masked-logits base64, so it is the wire
        # size to within a few bytes; documented as approximate.
        try:
            resp_bytes = len(json.dumps(encoded).encode("utf-8"))
        except Exception:                                    # noqa: BLE001
            resp_bytes = None
        total = time.perf_counter() - t_handler0
        merged = merge_server_timing(
            wt, total_s=total, parse_s=parse_s, decode_s=decode_s,
            encode_s=encode_s, response_bytes=resp_bytes)
        # GPU->trusted direction must also carry no secret-named field.
        audit_worker_timing_no_secrets(merged)
        encoded["worker_timing"] = merged
        return encoded


class GpuWorkerServer:
    """Threaded stdlib HTTP server hosting an (untrusted) GPU backend."""

    def __init__(self, host: str = "0.0.0.0", port: int = 18080,
                 backend_name: str = "mock",
                 backend_kwargs: dict[str, Any] | None = None,
                 audit: bool = True) -> None:
        from pllo.protocol.gpu_worker import make_gpu_backend  # lazy
        self.backend = make_gpu_backend(backend_name, **(backend_kwargs or {}))
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        # attach state for the handler
        self._httpd.backend = self.backend
        self._httpd.gpu_backend_name = backend_name
        self._httpd.audit = audit
        self._httpd.lock = threading.Lock()
        self.tee_used_on_gpu = bool(getattr(self.backend, "tee_used", False))

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        return t

    def shutdown(self) -> None:
        try:
            self._httpd.shutdown()
        finally:
            self._httpd.server_close()

    def __enter__(self) -> "GpuWorkerServer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def run_gpu_worker_server(host: str, port: int, backend_name: str,
                          backend_kwargs: dict[str, Any] | None = None,
                          audit: bool = True) -> None:
    """Blocking: start the GPU worker server and serve until interrupted."""
    server = GpuWorkerServer(host, port, backend_name, backend_kwargs, audit)
    print(f"[gpu_worker_server] backend={backend_name} "
          f"listening on {host}:{server.port} tee_used_on_gpu="
          f"{server.tee_used_on_gpu} audit={audit}", flush=True)
    print(f"[gpu_worker_server] endpoints: /health /init /prefill /decode",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Client (trusted boundary side)
# ---------------------------------------------------------------------------


class RemoteGpuWorker:
    """HTTP client to a remote :class:`GpuWorkerServer`.

    Drop-in for ``LocalGpuWorker`` (``init/prefill/decode/close``). Optional
    ``recorder(direction, method, message)`` lets the boundary log the exact
    objects sent/received for the security audit. Imports no model code."""

    def __init__(self, url: str, gpu_backend: str = "mock",
                 recorder: Any = None, timeout: float = 60.0,
                 persistent: bool = False,
                 request_worker_timing: bool = False) -> None:
        self.url = url.rstrip("/")
        self.gpu_backend = gpu_backend
        self._recorder = recorder
        self.timeout = timeout
        # Ask the (untrusted) worker to return its PUBLIC forward-timing metadata
        # so the client can split gpu_worker_roundtrip into network vs worker
        # compute. Default off -> the worker does no timing work and the response
        # carries worker_timing=None. The trusted client never sends a secret.
        self.request_worker_timing = bool(request_worker_timing)
        # Optional persistent transport: reuse ONE TCP connection across decode
        # steps (keep-alive) to drop per-token connection-setup overhead. Default
        # off -> historical one-connection-per-request behaviour. Profiling
        # decides whether this matters (it does not when GPU compute dominates).
        self.persistent = bool(persistent)
        self._conn = None
        self._closed = False

    def _audit_encoded(self, path: str, msg: Any):
        encoded = encode_message(msg)
        # Defense-in-depth (client side): no forbidden/secret field -- including a
        # precomputed obfuscation schedule's secret material -- may ride a GPU
        # request. Exact-name match (same matcher the server uses) so legitimate
        # masked payloads are never false-flagged; raise loudly, no silent pass.
        bad = forbidden_fields_in_payload(encoded)
        if bad:
            from pllo.runtime.obfuscation_schedule import ScheduleSecretLeak
            raise ScheduleSecretLeak(
                "forbidden/secret field(s) on GPU request to %s: %s"
                % (path, bad))
        return encoded

    def _persistent_conn(self):
        import http.client
        from urllib.parse import urlparse
        if self._conn is None:
            u = urlparse(self.url)
            self._conn = http.client.HTTPConnection(
                u.hostname, u.port or 80, timeout=self.timeout)
        return self._conn

    def _post_persistent(self, path: str, body: bytes) -> dict:
        headers = {"Content-Type": "application/json", "Connection": "keep-alive",
                   "X-PLLO-KeepAlive": "1"}
        if self.request_worker_timing:
            headers["X-PLLO-WorkerTiming"] = "1"
        for attempt in (1, 2):                     # reconnect once on a dropped conn
            try:
                conn = self._persistent_conn()
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read().decode("utf-8")
                if resp.status >= 400:
                    raise RuntimeError(
                        "GPU worker HTTP %d at %s: %s" % (resp.status, path, raw))
                return json.loads(raw)
            except (ConnectionError, OSError) as exc:
                self._close_conn()
                if attempt == 2:
                    raise RuntimeError("persistent GPU worker connection failed "
                                       "at %s: %s" % (path, exc)) from None
        raise RuntimeError("unreachable")

    def _close_conn(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:                                    # noqa: BLE001
            pass
        self._conn = None

    def _post(self, path: str, msg: Any) -> Any:
        if self._recorder is not None:
            self._recorder("inbound", path, msg)
        encoded = self._audit_encoded(path, msg)
        body = json.dumps(encoded).encode("utf-8")
        if self.persistent:
            out = self._post_persistent(path, body)
        else:
            headers = {"Content-Type": "application/json"}
            if self.request_worker_timing:
                headers["X-PLLO-WorkerTiming"] = "1"
            req = urllib.request.Request(
                self.url + path, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                raise RuntimeError(
                    f"GPU worker HTTP {exc.code} at {path}: {detail}") from None
        result = decode_message(out)
        if self._recorder is not None:
            self._recorder("outbound", path, result)
        return result

    def health(self) -> dict:
        req = urllib.request.Request(self.url + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def init(self, req):
        return self._post("/init", req)

    def prefill(self, req):
        return self._post("/prefill", req)

    def decode(self, req):
        return self._post("/decode", req)

    def close(self) -> None:
        self._closed = True
        self._close_conn()

    def __enter__(self) -> "RemoteGpuWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
