"""Per-token, per-stage profiler for the folded_remote autoregressive decode.

The point of this module is to LOCATE the per-token online latency bottleneck
before optimizing anything. It times each decode step across the 9 stages of the
folded_remote path and produces (a) a per-token trace and (b) aggregate target
metrics that are written verbatim into the run report.

The 9 stages (``DECODE_STAGES``):
  1. prompt_token_prep                -- build the next-token tensor
  2. trusted_input_embedding          -- boundary embed + mask the input
  3. schedule_slot_lookup             -- consume the step's fresh obfuscation slot
  4. http_request_serialization       -- build + serialize the masked request
  5. gpu_worker_roundtrip             -- send to the untrusted worker + receive
  6. trusted_nonlinear_restore_logits -- recover masked logits on the boundary
  7. sampling                         -- pick the next token (greedy)
  8. kv_cache_update                  -- per-step KV bookkeeping
  9. output_json_writing              -- per-step output flush (usually 0)

Counters (boundary_calls / gpu_calls / trusted_bytes / gpu_bytes) are read via an
injected ``counters()`` callback so this module needs neither torch nor a worker;
it is pure stdlib and unit-tested with a synthetic decode loop. Honest by design:
``avg_gpu_compute_s_per_token`` is the GPU's *internal* compute time, which the
trusted client cannot observe -- it is reported as ``None`` unless a worker hands
it back, and ``gpu_worker_roundtrip`` (which folds in network + GPU) is reported
separately so the bottleneck is not mislabelled.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "DECODE_STAGES",
    "TRUSTED_STAGES",
    "DecodeProfiler",
    "empty_target_metrics",
    "simulate_mock_decode",
]

DECODE_STAGES = (
    "prompt_token_prep",
    "trusted_input_embedding",
    "schedule_slot_lookup",
    "http_request_serialization",
    "gpu_worker_roundtrip",
    "trusted_nonlinear_restore_logits",
    "sampling",
    "kv_cache_update",
    "output_json_writing",
)

# Stages that run inside the trusted boundary (their sum ~ trusted compute/token).
TRUSTED_STAGES = (
    "prompt_token_prep", "trusted_input_embedding", "schedule_slot_lookup",
    "trusted_nonlinear_restore_logits", "sampling", "kv_cache_update",
)

# All target metric keys -- present (defaulted) even when profiling is disabled,
# so report consumers never KeyError.
_TARGET_KEYS = (
    "latency_per_generated_token_s",
    "boundary_calls_per_generated_token",
    "gpu_calls_per_generated_token",
    "trusted_bytes_per_generated_token",
    "gpu_bytes_per_generated_token",
    "avg_http_roundtrip_s",
    "avg_trusted_compute_s_per_token",
    "avg_gpu_compute_s_per_token",
    "schedule_precompute_latency_s",
    "online_decode_latency_s",
    "prefill_latency_s",
    "decode_latency_s",
    "bottleneck_stage",
    "stage_total_s",
    "total_boundary_calls",
    "total_gpu_calls",
    "generated_tokens",
    "boundary_calls_reduced",
    "boundary_calls_reduction_note",
    # worker-side timing split of gpu_worker_roundtrip (None unless the worker
    # returned its public timing metadata for this run)
    "avg_worker_total_s_per_token",
    "avg_worker_backend_forward_s_per_token",
    "avg_network_protocol_overhead_s_per_token",
    "avg_worker_known_substage_total_s_per_token",
    "avg_worker_unattributed_forward_s_per_token",
    "worker_timing_method",
    "worker_timing_is_cuda_synchronized",
    "worker_bottleneck_stage",
)

# Measured baseline (server, current backend): ~2 boundary calls / generated
# token. boundary_calls_reduced is only true when measurably BELOW the threshold
# (a prefill that does 1 boundary call makes the per-token average ~1.875 over a
# short run -- that is NOT a reduction, so the threshold sits well under 2.0).
_BASELINE_BOUNDARY_CALLS_PER_TOKEN = 2.0
_BOUNDARY_REDUCED_THRESHOLD = 1.5


def empty_target_metrics() -> Dict[str, Any]:
    """Target metrics with safe defaults (profiling disabled / no rows)."""
    return {k: None for k in _TARGET_KEYS}


def _now() -> float:
    return time.perf_counter()


def _delta(before, after):
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 9) if xs else None


def _boundary_reduction_note(bpt: Optional[float]) -> Optional[str]:
    """Honest one-liner about the boundary-call rate vs the ~2/token baseline."""
    if bpt is None:
        return None
    if bpt < _BOUNDARY_REDUCED_THRESHOLD:
        return ("reduced to %.3f boundary calls/generated token (below the %.1f "
                "threshold; baseline ~%.1f)"
                % (bpt, _BOUNDARY_REDUCED_THRESHOLD,
                   _BASELINE_BOUNDARY_CALLS_PER_TOKEN))
    return ("still ~%.3f boundary calls/generated token (baseline ~%.1f; NOT "
            "reduced below the %.1f threshold -- each token still needs a fresh "
            "mask embedding + a logits recovery)"
            % (bpt, _BASELINE_BOUNDARY_CALLS_PER_TOKEN,
               _BOUNDARY_REDUCED_THRESHOLD))


class DecodeProfiler:
    """Times decode steps and emits a per-token trace + aggregate metrics.

    Usage per step::

        prof.begin_step(step_id, phase="decode")
        with prof.stage("trusted_input_embedding"):
            ...
        ...
        prof.end_step(token_id=tok)
    """

    def __init__(self, *, counters: Optional[Callable[[], Dict[str, Any]]] = None,
                 enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._counters = counters or (lambda: {})
        self._rows: List[Dict[str, Any]] = []
        self._cur: Optional[Dict[str, Any]] = None

    # -- per-step API -----------------------------------------------------
    def begin_step(self, step_id: int, phase: str = "decode") -> None:
        if not self.enabled:
            return
        snap = self._counters() or {}
        self._cur = {
            "step_id": int(step_id),
            "phase": str(phase),
            "token_id": None,
            "boundary_calls_before": snap.get("boundary_calls"),
            "gpu_calls_before": snap.get("gpu_calls"),
            "_tb_before": snap.get("trusted_bytes"),
            "_gb_before": snap.get("gpu_bytes"),
            "stage_timings": {s: 0.0 for s in DECODE_STAGES},
            "_t0": _now(),
        }

    def set_worker_timing(self, meta: Optional[Dict[str, Any]]) -> None:
        """Attach the worker's PUBLIC forward-timing metadata to the current step
        (call between ``begin_step`` and ``end_step``). Used to split the
        client-observed ``gpu_worker_roundtrip`` into network vs worker compute."""
        if not self.enabled or self._cur is None or not meta:
            return
        self._cur["worker_timings"] = dict(meta)

    @contextmanager
    def stage(self, name: str):
        if not self.enabled or self._cur is None:
            yield
            return
        t = _now()
        try:
            yield
        finally:
            self._cur["stage_timings"][name] = \
                self._cur["stage_timings"].get(name, 0.0) + (_now() - t)

    def end_step(self, token_id: Any = None) -> None:
        if not self.enabled or self._cur is None:
            return
        snap = self._counters() or {}
        c = self._cur
        c["token_id"] = token_id
        c["boundary_calls_after"] = snap.get("boundary_calls")
        c["gpu_calls_after"] = snap.get("gpu_calls")
        c["trusted_bytes_delta"] = _delta(c.pop("_tb_before"),
                                          snap.get("trusted_bytes"))
        c["gpu_bytes_delta"] = _delta(c.pop("_gb_before"),
                                      snap.get("gpu_bytes"))
        c["total_step_latency_s"] = round(_now() - c.pop("_t0"), 9)
        c["stage_timings"] = {k: round(v, 9)
                              for k, v in c["stage_timings"].items()}
        # derived: client-observed roundtrip minus the worker's own total time =
        # network + protocol overhead (None unless the worker reported its total).
        wt = c.get("worker_timings")
        rt = c["stage_timings"].get("gpu_worker_roundtrip")
        if (isinstance(wt, dict)
                and isinstance(wt.get("worker_total_s"), (int, float))
                and isinstance(rt, (int, float))):
            c["network_protocol_overhead_s"] = round(
                rt - wt["worker_total_s"], 9)
        else:
            c["network_protocol_overhead_s"] = None
        self._rows.append(c)
        self._cur = None

    # -- views ------------------------------------------------------------
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def _count_total(self, key_before: str, key_after: str) -> Optional[int]:
        total = 0
        seen = False
        for r in self._rows:
            d = _delta(r.get(key_before), r.get(key_after))
            if d is not None:
                total += d
                seen = True
        return total if seen else None

    def aggregate(self, *, generated_tokens: Optional[int] = None,
                  schedule_precompute_latency_s: Optional[float] = None,
                  avg_gpu_compute_s_per_token: Optional[float] = None
                  ) -> Dict[str, Any]:
        """Compute the target metrics from the recorded rows."""
        out = empty_target_metrics()
        if not self._rows:
            out["schedule_precompute_latency_s"] = schedule_precompute_latency_s
            return out
        n = int(generated_tokens or len(self._rows))
        prefill = [r for r in self._rows if r["phase"] == "prefill"]
        decode = [r for r in self._rows if r["phase"] != "prefill"]
        prefill_lat = round(sum(r["total_step_latency_s"] for r in prefill), 9)
        decode_lat = round(sum(r["total_step_latency_s"] for r in decode), 9)
        online = round(prefill_lat + decode_lat, 9)

        stage_total = {s: round(sum(r["stage_timings"].get(s, 0.0)
                                    for r in self._rows), 9)
                       for s in DECODE_STAGES}
        bottleneck = max(stage_total, key=lambda s: stage_total[s]) \
            if stage_total else None

        # trusted/gpu byte totals from per-row deltas
        tb_total = sum(r["trusted_bytes_delta"] for r in self._rows
                       if isinstance(r.get("trusted_bytes_delta"), (int, float)))
        gb_total = sum(r["gpu_bytes_delta"] for r in self._rows
                       if isinstance(r.get("gpu_bytes_delta"), (int, float)))
        bcalls = self._count_total("boundary_calls_before",
                                   "boundary_calls_after")
        gcalls = self._count_total("gpu_calls_before", "gpu_calls_after")

        bpt = (round(bcalls / n, 6) if (bcalls is not None and n) else None)
        out.update({
            "generated_tokens": n,
            "prefill_latency_s": prefill_lat,
            "decode_latency_s": decode_lat,
            "online_decode_latency_s": online,
            "latency_per_generated_token_s": (round(online / n, 9)
                                              if n else None),
            "total_boundary_calls": bcalls,
            "total_gpu_calls": gcalls,
            "boundary_calls_per_generated_token": bpt,
            "gpu_calls_per_generated_token": (round(gcalls / n, 6)
                                              if (gcalls is not None and n)
                                              else None),
            "trusted_bytes_per_generated_token": (round(tb_total / n, 3)
                                                  if n else None),
            "gpu_bytes_per_generated_token": (round(gb_total / n, 3)
                                              if n else None),
            "avg_http_roundtrip_s": _mean(
                [r["stage_timings"].get("gpu_worker_roundtrip")
                 for r in self._rows]),
            "avg_trusted_compute_s_per_token": _mean(
                [sum(r["stage_timings"].get(s, 0.0) for s in TRUSTED_STAGES)
                 for r in self._rows]),
            # the GPU's INTERNAL compute time is not observable from the trusted
            # client; only a worker that reports it can fill this in.
            "avg_gpu_compute_s_per_token": avg_gpu_compute_s_per_token,
            "schedule_precompute_latency_s": schedule_precompute_latency_s,
            "bottleneck_stage": bottleneck,
            "stage_total_s": stage_total,
            "boundary_calls_reduced": (bool(bpt < _BOUNDARY_REDUCED_THRESHOLD)
                                       if bpt is not None else False),
            "boundary_calls_reduction_note": _boundary_reduction_note(bpt),
        })
        out.update(self._worker_aggregate())
        return out

    def _worker_aggregate(self) -> Dict[str, Any]:
        """Aggregate the per-step worker timing (when present) into per-token
        averages + a worker-internal bottleneck. All None when the worker returned
        no timing (e.g. the client did not opt in)."""
        out = {
            "avg_worker_total_s_per_token": None,
            "avg_worker_backend_forward_s_per_token": None,
            "avg_network_protocol_overhead_s_per_token": None,
            "avg_worker_known_substage_total_s_per_token": None,
            "avg_worker_unattributed_forward_s_per_token": None,
            "worker_timing_method": None,
            "worker_timing_is_cuda_synchronized": None,
            "worker_bottleneck_stage": None,
        }
        wt_rows = [r.get("worker_timings") for r in self._rows
                   if isinstance(r.get("worker_timings"), dict)]
        if not wt_rows:
            return out

        def _wmean(key):
            return _mean([w.get(key) for w in wt_rows])

        out["avg_worker_total_s_per_token"] = _wmean("worker_total_s")
        out["avg_worker_backend_forward_s_per_token"] = _wmean(
            "worker_backend_forward_s")
        out["avg_network_protocol_overhead_s_per_token"] = _mean(
            [r.get("network_protocol_overhead_s") for r in self._rows])
        out["avg_worker_known_substage_total_s_per_token"] = _wmean(
            "worker_known_substage_total_s")
        out["avg_worker_unattributed_forward_s_per_token"] = _wmean(
            "worker_unattributed_forward_s")
        # provenance: the dominant method across rows (they are homogeneous in
        # practice -- one backend per run)
        methods = [w.get("worker_timing_method") for w in wt_rows
                   if w.get("worker_timing_method")]
        out["worker_timing_method"] = methods[0] if methods else None
        out["worker_timing_is_cuda_synchronized"] = bool(
            wt_rows[0].get("worker_timing_is_cuda_synchronized"))
        out["worker_bottleneck_stage"] = self._worker_bottleneck(wt_rows, _wmean)
        return out

    @staticmethod
    def _worker_bottleneck(wt_rows, _wmean) -> Optional[str]:
        """Pick the worker-internal bottleneck WITHOUT being fooled by tiny
        accurately-but-partially-measured substages.

        Rules (req. 4/5):
          * a substage is only a candidate when its timing method is reliable;
          * if the named substages account for < half the forward (i.e. most of the
            forward is weight movement / unattributed), the bottleneck is the
            per-layer total (if it dominates) or ``worker_backend_forward_unattributed``
            -- never attention/mlp/nonlinear/lm_head;
          * otherwise the largest reliable substage wins.
        """
        from pllo.protocol.worker_timing import (
            WORKER_FORWARD_UNATTRIBUTED, substage_reliable)
        fwd = _wmean("worker_backend_forward_s")
        layer_total = _wmean("worker_layer_total_s")
        known = _wmean("worker_known_substage_total_s")
        reliable = all(substage_reliable(w) for w in wt_rows)

        if not reliable:
            # substage numbers can't be trusted -> only coarse fields
            cand = {k: v for k, v in (("worker_layer_total_s", layer_total),
                                      ("worker_backend_forward_s", fwd))
                    if v is not None}
            return max(cand, key=cand.get) if cand else None

        # substages reliable: do they explain most of the forward?
        if fwd is not None and known is not None and known < 0.5 * fwd:
            if layer_total is not None and layer_total >= max(known, 0.5 * fwd):
                return "worker_layer_total_s"
            return WORKER_FORWARD_UNATTRIBUTED

        cand: Dict[str, float] = {}
        for key in ("worker_attention_total_s", "worker_mlp_total_s",
                    "worker_nonlinear_total_s", "worker_lm_head_s",
                    "worker_request_parse_s", "worker_payload_decode_s",
                    "worker_payload_encode_s"):
            m = _wmean(key)
            if m is not None:
                cand[key] = m
        if not cand:
            m = _wmean("worker_backend_forward_s")
            if m is not None:
                cand["worker_backend_forward_s"] = m
        return max(cand, key=cand.get) if cand else None

    def write_trace(self, path, *, limit: Optional[int] = None) -> str:
        """Write the per-token trace as JSONL (one row per decode step)."""
        rows = self._rows if limit is None else self._rows[:max(0, int(limit))]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        return str(p)


def _busy(n: int) -> int:
    s = 0
    for i in range(int(n)):
        s += i * i
    return s


def simulate_mock_decode(profiler: "DecodeProfiler", counters: Dict[str, Any], *,
                         n_tokens: int, hidden_size: int, on_step=None,
                         worker_timing_fn=None, busy_iters: int = 2000) -> None:
    """Run a SYNTHETIC profiled decode (no model/GPU) over ``n_tokens`` steps,
    exercising all 9 stages with tiny CPU work so the profiler's trace + metrics
    can be validated locally. Mirrors the real folded path's per-step boundary
    (2/token) and gpu (1/token) call pattern. ``on_step(kind, step, phase)`` is
    invoked inside the ``schedule_slot_lookup`` (kind='schedule') and
    ``http_request_serialization`` (kind='serialize') stages for the caller to
    consume a schedule slot / audit a synthetic GPU payload.
    ``worker_timing_fn(step, phase)`` (optional) returns a SYNTHETIC worker-timing
    dict attached to the step (mock only). This is a MOCK helper -- it is NOT a
    substitute for measuring the real runtime."""
    for step in range(int(n_tokens)):
        phase = "prefill" if step == 0 else "decode"
        profiler.begin_step(step, phase)
        if step > 0:
            with profiler.stage("prompt_token_prep"):
                _busy(busy_iters // 4)
        with profiler.stage("trusted_input_embedding"):
            _busy(busy_iters)
            counters["boundary_calls"] += 1
            counters["trusted_bytes"] += hidden_size * 4
        with profiler.stage("schedule_slot_lookup"):
            if on_step:
                on_step("schedule", step, phase)
        with profiler.stage("http_request_serialization"):
            if on_step:
                on_step("serialize", step, phase)
            counters["gpu_bytes"] += hidden_size * 4
        with profiler.stage("gpu_worker_roundtrip"):
            _busy(busy_iters * 2)               # roundtrip dominates per token
            counters["gpu_calls"] += 1
        if worker_timing_fn is not None:
            profiler.set_worker_timing(worker_timing_fn(step, phase))
        with profiler.stage("trusted_nonlinear_restore_logits"):
            _busy(busy_iters)
            if step > 0:
                counters["boundary_calls"] += 1   # recover boundary call
            counters["trusted_bytes"] += 256
        with profiler.stage("sampling"):
            _busy(busy_iters // 6)
        if step > 0:
            with profiler.stage("kv_cache_update"):
                pass
        profiler.end_step(token_id=None)
