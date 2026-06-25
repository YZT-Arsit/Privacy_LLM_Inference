"""Worker-side (untrusted GPU) timing instrumentation for the folded forward.

The per-token decode profiler localises the bottleneck to ``gpu_worker_roundtrip``
-- but that single client-observed stage folds together the network round trip AND
the worker's own folded-forward compute. This module lets the **untrusted worker**
hand back a PUBLIC, non-secret profiling-metadata dict so the trusted client can
split the round trip into ``network_protocol_overhead_s`` vs
``worker_backend_forward_s`` (and, when the folded backend can be split at low
cost, into per-layer / attention / MLP / nonlinear / LM-head sub-totals).

Security: the metadata is numeric timings + public backend identifiers ONLY. It
NEVER carries mask / pad / inverse / seed / raw tensors. :func:`audit_worker_timing_no_secrets`
re-uses the schedule secret-substring matcher and raises on any violation; the
server runs it before sending and the client re-runs it on receipt
(defense-in-depth).

GPU kernels are asynchronous, so wrapping a sub-region in a bare wall-clock
under-reports it (you measure the launch, not the compute). :class:`WorkerTimer`
therefore supports three methods: ``cuda_event`` (per-region ``torch.cuda.Event``
pairs resolved with a SINGLE final ``synchronize`` -- accurate, no per-op sync to
pollute the very thing being measured; preferred on CUDA), ``cuda_synchronize``
(explicit sync at each region boundary; fallback), and ``wall_clock`` (CPU is
synchronous, so this is accurate there -- used by the mock/CPU path). The legacy
``wall_clock_unsynchronized`` label marks UNRELIABLE substage timing (CUDA without
sync); the client refuses to pick a bottleneck from unreliable substages.

The backend also reports ``worker_unattributed_forward_s`` =
``worker_backend_forward_s - known_substage_sum``. On the real folded path the
per-token cost is dominated by per-layer weight movement (disk shard load + the
H2D ``.to(device)`` copy), NOT the 1-token matmuls -- so the named compute
substages are genuinely small and the remainder is large; surfacing it stops a
tiny-but-measured ``attention`` from being mislabelled the bottleneck.

Mostly stdlib; ``cuda_event`` lazily imports torch (only when that method is
chosen on the GPU host), so the module stays importable on the trusted boundary.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "WORKER_TIMING_KEYS",
    "WORKER_FORWARD_UNATTRIBUTED",
    "WorkerTimer",
    "empty_worker_timing",
    "coarse_forward_metadata",
    "merge_server_timing",
    "synthetic_worker_timing",
    "audit_worker_timing_no_secrets",
    "substage_reliable",
]

# Every key a worker-timing dict may carry. All optional -- a coarse backend
# (req. 11) fills only worker_total_s + worker_backend_forward_s (+ identifiers)
# and leaves the per-layer breakdown None.
WORKER_TIMING_KEYS = (
    # server-stage timings (measured in the HTTP handler)
    "worker_total_s",
    "worker_request_parse_s",
    "worker_payload_decode_s",
    "worker_payload_encode_s",
    "worker_response_bytes",
    # forward-level timing + public identifiers (measured in the backend)
    "worker_backend_forward_s",
    "worker_backend_name",
    "worker_device",
    "worker_dtype",
    "worker_prefill_or_decode",
    # optional low-cost folded-forward breakdown (None if not split)
    "worker_num_layers",
    "worker_layer_total_s",
    "worker_lm_head_s",
    "worker_attention_total_s",
    "worker_mlp_total_s",
    "worker_nonlinear_total_s",
    "per_layer_timing_summary",
    # attribution + timing provenance (so a bottleneck is only ever read off
    # accurately-measured fields)
    "worker_known_substage_total_s",
    "worker_unattributed_forward_s",
    "worker_timing_method",
    "worker_timing_is_cuda_synchronized",
)

# Named compute substages whose sum is the "known" (attributed) forward time.
_SUBSTAGE_KEYS = ("worker_attention_total_s", "worker_mlp_total_s",
                  "worker_nonlinear_total_s", "worker_lm_head_s")
# Timing methods whose substage numbers are trustworthy for bottleneck selection.
_RELIABLE_METHODS = ("cuda_event", "cuda_synchronize", "wall_clock")
# Synthetic bottleneck name for "most of the forward is outside named substages".
WORKER_FORWARD_UNATTRIBUTED = "worker_backend_forward_unattributed"


def _round(x: Optional[float]) -> Optional[float]:
    return round(x, 9) if isinstance(x, (int, float)) else None


def empty_worker_timing() -> Dict[str, Any]:
    """A fully-defaulted worker-timing dict (all keys present, all ``None``)."""
    return {k: None for k in WORKER_TIMING_KEYS}


def _summary(xs: List[float]) -> Optional[Dict[str, Any]]:
    """mean_s / max_s / min_s (+ count) of per-layer times -- NOT the full
    per-layer array (keeps the trace small even at 28+ layers)."""
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return {"mean_s": round(sum(xs) / len(xs), 9), "max_s": round(max(xs), 9),
            "min_s": round(min(xs), 9), "count": len(xs)}


class WorkerTimer:
    """Accumulates folded-forward sub-region timings on the untrusted worker.

    ``region(name)`` accumulates time into a named total (``attention`` / ``mlp``
    / ``nonlinear`` / ``lm_head``); ``layer()`` appends one elapsed time per folded
    layer. ``method`` selects how GPU time is measured:

    * ``cuda_event`` -- record a ``torch.cuda.Event`` pair around each region and
      resolve all of them with a SINGLE ``synchronize`` at ``forward_metadata``
      time (accurate device time; no per-op sync polluting the measurement).
    * ``cuda_synchronize`` -- explicit ``sync`` (injected) at each region boundary.
    * ``wall_clock`` -- plain ``perf_counter`` (accurate on CPU, which is
      synchronous; used by the mock/CPU path).

    With ``enabled=False`` (or no timer threaded in) every context is a no-op --
    the default path is byte-for-byte unchanged."""

    def __init__(self, *, enabled: bool = True, method: str = "wall_clock",
                 sync: Optional[Callable[[], None]] = None) -> None:
        self.enabled = bool(enabled)
        self.method = str(method)
        self._sync = sync or (lambda: None)
        self._totals: Dict[str, float] = {}
        self._layers: List[float] = []
        # cuda_event mode: deferred (start, end) event pairs, resolved once.
        self._events: Dict[str, List[Any]] = {}
        self._layer_events: List[Any] = []
        self._torch = None
        self._resolved = False
        if self.enabled and self.method == "cuda_event":
            try:
                import torch
                self._torch = torch
            except Exception:                                # noqa: BLE001
                # torch missing -> degrade to explicit-sync wall clock
                self.method = "cuda_synchronize"

    # -- region / layer timing -------------------------------------------
    def _new_event(self):
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    @contextmanager
    def region(self, name: str):
        if not self.enabled:
            yield
            return
        if self.method == "cuda_event":
            s = self._new_event()
            try:
                yield
            finally:
                e = self._new_event()
                self._events.setdefault(name, []).append((s, e))
            return
        if self.method == "cuda_synchronize":
            self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.method == "cuda_synchronize":
                self._sync()
            self._totals[name] = self._totals.get(name, 0.0) + \
                (time.perf_counter() - t)

    @contextmanager
    def layer(self):
        if not self.enabled:
            yield
            return
        if self.method == "cuda_event":
            s = self._new_event()
            try:
                yield
            finally:
                self._layer_events.append((s, self._new_event()))
            return
        if self.method == "cuda_synchronize":
            self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.method == "cuda_synchronize":
                self._sync()
            self._layers.append(time.perf_counter() - t)

    def _resolve(self) -> None:
        """Resolve deferred CUDA events into seconds with a SINGLE synchronize."""
        if self._resolved or self.method != "cuda_event":
            return
        self._torch.cuda.synchronize()
        for name, pairs in self._events.items():
            self._totals[name] = sum(s.elapsed_time(e) for s, e in pairs) / 1000.0
        self._layers = [s.elapsed_time(e) / 1000.0
                        for s, e in self._layer_events]
        self._resolved = True

    def total(self, name: str) -> Optional[float]:
        self._resolve()
        return self._totals.get(name)

    @property
    def is_cuda_synchronized(self) -> bool:
        return self.method in ("cuda_event", "cuda_synchronize")

    def forward_metadata(self, *, phase: str, backend_name: str, device: Any,
                         dtype: Any, forward_s: Optional[float],
                         num_layers: Optional[int] = None) -> Dict[str, Any]:
        """Forward-level metadata from the recorded regions/layers (server-stage
        fields are merged later by :func:`merge_server_timing`)."""
        self._resolve()
        layer_total = round(sum(self._layers), 9) if self._layers else None
        out = empty_worker_timing()
        out.update({
            "worker_backend_forward_s": _round(forward_s),
            "worker_backend_name": backend_name,
            "worker_device": str(device) if device is not None else None,
            "worker_dtype": str(dtype) if dtype is not None else None,
            "worker_prefill_or_decode": phase,
            "worker_num_layers": (int(num_layers) if num_layers is not None
                                  else (len(self._layers) or None)),
            "worker_layer_total_s": layer_total,
            "worker_lm_head_s": _round(self._totals.get("lm_head")),
            "worker_attention_total_s": _round(self._totals.get("attention")),
            "worker_mlp_total_s": _round(self._totals.get("mlp")),
            "worker_nonlinear_total_s": _round(self._totals.get("nonlinear")),
            "per_layer_timing_summary": _summary(self._layers),
            "worker_timing_method": self.method,
            "worker_timing_is_cuda_synchronized": self.is_cuda_synchronized,
        })
        _fill_attribution(out)
        return out


def _fill_attribution(meta: Dict[str, Any]) -> None:
    """Compute ``worker_known_substage_total_s`` (sum of named compute substages)
    and ``worker_unattributed_forward_s`` = forward - known, in place."""
    subs = [meta.get(k) for k in _SUBSTAGE_KEYS]
    subs = [s for s in subs if isinstance(s, (int, float))]
    known = round(sum(subs), 9) if subs else None
    meta["worker_known_substage_total_s"] = known
    fwd = meta.get("worker_backend_forward_s")
    meta["worker_unattributed_forward_s"] = (
        round(fwd - known, 9)
        if isinstance(fwd, (int, float)) and known is not None else None)


def substage_reliable(meta: Optional[Dict[str, Any]]) -> bool:
    """Whether a worker-timing dict's named substages can be trusted for picking a
    bottleneck (accurate timing method)."""
    if not isinstance(meta, dict):
        return False
    return meta.get("worker_timing_method") in _RELIABLE_METHODS


def coarse_forward_metadata(*, phase: str, backend_name: str, device: Any,
                            dtype: Any, forward_s: Optional[float],
                            num_layers: Optional[int] = None,
                            timing_method: str = "wall_clock",
                            is_cuda_synchronized: bool = False) -> Dict[str, Any]:
    """Forward-level metadata for backends that can NOT split the forward at low
    cost (req. 11): only the total forward time + public identifiers; the per-layer
    breakdown stays ``None``. ``timing_method``/``is_cuda_synchronized`` describe
    how ``worker_backend_forward_s`` was measured (a synchronize-bracketed total is
    accurate even without a per-substage split)."""
    out = empty_worker_timing()
    out.update({
        "worker_backend_forward_s": _round(forward_s),
        "worker_backend_name": backend_name,
        "worker_device": str(device) if device is not None else None,
        "worker_dtype": str(dtype) if dtype is not None else None,
        "worker_prefill_or_decode": phase,
        "worker_num_layers": int(num_layers) if num_layers is not None else None,
        "worker_timing_method": timing_method,
        "worker_timing_is_cuda_synchronized": bool(is_cuda_synchronized),
    })
    _fill_attribution(out)               # known=None (no substages) -> unattr=None
    return out


def merge_server_timing(forward_meta: Optional[Dict[str, Any]], *,
                        total_s: Optional[float], parse_s: Optional[float],
                        decode_s: Optional[float], encode_s: Optional[float],
                        response_bytes: Optional[int]) -> Dict[str, Any]:
    """Merge the backend's forward-level metadata with the server handler's stage
    timings (parse / decode / encode / total) + the response size."""
    out = empty_worker_timing()
    out.update(forward_meta or {})
    out["worker_total_s"] = _round(total_s)
    out["worker_request_parse_s"] = _round(parse_s)
    out["worker_payload_decode_s"] = _round(decode_s)
    out["worker_payload_encode_s"] = _round(encode_s)
    out["worker_response_bytes"] = (int(response_bytes)
                                    if response_bytes is not None else None)
    return out


def synthetic_worker_timing(*, phase: str, num_layers: int = 28,
                            forward_s: float = 4.65,
                            backend_name: str = "mock",
                            device: str = "cpu", dtype: str = "float32",
                            timing_method: str = "cuda_event"
                            ) -> Dict[str, Any]:
    """A SYNTHETIC worker-timing dict for local mock runs (no real worker).

    Clearly labelled mock (``worker_backend_name='mock'``). The shape MIRRORS the
    measured server reality: the 1-token matmul substages are small while the
    per-layer total is large (dominated by weight movement: disk shard load + H2D
    copy), so ``worker_unattributed_forward_s`` is large. NOT a measurement of any
    real backend (req. 11 bars faking real GPU timing -- this is only for the mock
    probe / tests; it exercises the client merge / aggregate / bottleneck / audit)."""
    fwd = float(forward_s)
    layer_total = fwd * 0.79            # ~3.69 of 4.65: mostly per-layer H2D
    per_layer = layer_total / max(1, num_layers)
    # small, accurately-measured compute substages (the real matmuls are cheap)
    attn = 0.025
    mlp = 0.013
    nonlin = 0.007
    head = 0.02
    out = empty_worker_timing()
    out.update({
        "worker_total_s": round(fwd * 1.012, 9),
        "worker_request_parse_s": round(fwd * 0.0004, 9),
        "worker_payload_decode_s": round(fwd * 0.0008, 9),
        "worker_payload_encode_s": round(fwd * 0.0012, 9),
        "worker_response_bytes": 4096,
        "worker_backend_forward_s": round(fwd, 9),
        "worker_backend_name": backend_name,
        "worker_device": device,
        "worker_dtype": dtype,
        "worker_prefill_or_decode": phase,
        "worker_num_layers": int(num_layers),
        "worker_layer_total_s": round(layer_total, 9),
        "worker_lm_head_s": round(head, 9),
        "worker_attention_total_s": round(attn, 9),
        "worker_mlp_total_s": round(mlp, 9),
        "worker_nonlinear_total_s": round(nonlin, 9),
        "per_layer_timing_summary": {"mean_s": round(per_layer, 9),
                                     "max_s": round(per_layer * 1.1, 9),
                                     "min_s": round(per_layer * 0.9, 9),
                                     "count": int(num_layers)},
        "worker_timing_method": timing_method,
        "worker_timing_is_cuda_synchronized": timing_method in (
            "cuda_event", "cuda_synchronize"),
    })
    _fill_attribution(out)
    return out


def _array_like(v: Any) -> bool:
    """True for anything tensor/ndarray-shaped (duck-typed; no numpy/torch import
    required) -- worker timing must be scalars/strings/counts/durations only."""
    return hasattr(v, "shape") and hasattr(v, "dtype")


def _bad_value_paths(meta: Optional[Dict[str, Any]],
                     max_list_len: int = 8) -> List[str]:
    """Dotted paths of any NON-scalar value: a tensor/ndarray, or a list/tuple
    long enough to be a smuggled vector (a timing dict carries no such payload)."""
    found: List[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, "%s.%s" % (path, k) if path else str(k))
        elif isinstance(o, (list, tuple)):
            if len(o) > max_list_len:
                found.append("%s (len=%d)" % (path or "<root>", len(o)))
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i))
        elif _array_like(o):
            found.append("%s (array-like)" % (path or "<root>"))

    walk(meta or {}, "")
    return found


def audit_worker_timing_no_secrets(meta: Optional[Dict[str, Any]]
                                   ) -> Dict[str, Any]:
    """Raise :class:`ScheduleSecretLeak` if a worker-timing dict carries a
    secret-named field OR a non-scalar value (tensor/ndarray/long list); return an
    audit record when clean. Re-uses the schedule secret-substring matcher so the
    same blocklist guards both directions, and additionally enforces that the
    metadata is scalars/strings/counts/durations only (no smuggled vectors)."""
    from pllo.runtime.obfuscation_schedule import (
        ScheduleSecretLeak, _secret_key_paths)
    paths = _secret_key_paths(meta or {})
    if paths:
        raise ScheduleSecretLeak(
            "worker timing metadata contains secret field(s): %s" % paths)
    bad = _bad_value_paths(meta)
    if bad:
        raise ScheduleSecretLeak(
            "worker timing metadata contains non-scalar value(s): %s" % bad)
    return {"audit": "worker_timing_no_secrets", "ok": True}
