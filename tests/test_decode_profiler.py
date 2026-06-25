"""Per-token decode profiler + persistent transport tests (local, no model/GPU).

Covers the 9-stage profiler, per-token trace JSONL, aggregate target metrics +
bottleneck localisation, the mock decode simulator, the optional persistent HTTP
transport (functional, against a mock worker), and the probe / ifeval scripts in
mock mode. No Qwen weights, no server paths.

Run: python -m pytest tests/test_decode_profiler.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.decode_profiler import (  # noqa: E402
    DECODE_STAGES,
    DecodeProfiler,
    empty_target_metrics,
    simulate_mock_decode,
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


# ---- profiler core --------------------------------------------------------

def test_profiler_records_stages_and_counter_deltas() -> None:
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    p.begin_step(0, "prefill")
    with p.stage("trusted_input_embedding"):
        counters["boundary_calls"] += 1
        counters["trusted_bytes"] += 100
    with p.stage("gpu_worker_roundtrip"):
        counters["gpu_calls"] += 1
        counters["gpu_bytes"] += 50
    p.end_step(token_id=7)
    rows = p.rows()
    assert len(rows) == 1
    r = rows[0]
    assert r["step_id"] == 0 and r["phase"] == "prefill" and r["token_id"] == 7
    assert r["boundary_calls_before"] == 0 and r["boundary_calls_after"] == 1
    assert r["gpu_calls_after"] == 1
    assert r["trusted_bytes_delta"] == 100 and r["gpu_bytes_delta"] == 50
    assert set(r["stage_timings"]) == set(DECODE_STAGES)
    assert r["total_step_latency_s"] >= 0.0


def test_profiler_disabled_no_rows() -> None:
    p = DecodeProfiler(enabled=False)
    p.begin_step(0, "decode")
    with p.stage("sampling"):
        pass
    p.end_step()
    assert p.rows() == []
    assert set(empty_target_metrics()) and all(
        v is None for v in empty_target_metrics().values())


def test_simulate_mock_decode_and_aggregate() -> None:
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    consumed = []
    simulate_mock_decode(
        p, counters, n_tokens=6, hidden_size=32,
        on_step=lambda kind, step, phase: consumed.append((kind, step)))
    agg = p.aggregate(generated_tokens=6, schedule_precompute_latency_s=0.5)
    assert agg["generated_tokens"] == 6
    assert agg["gpu_calls_per_generated_token"] == 1.0
    # 1 boundary call on prefill + 2 per decode step (5) = 11 over 6 tokens
    assert agg["total_boundary_calls"] == 11
    assert agg["bottleneck_stage"] == "gpu_worker_roundtrip"   # by construction
    assert agg["schedule_precompute_latency_s"] == 0.5
    assert agg["avg_gpu_compute_s_per_token"] is None          # not observable
    assert agg["online_decode_latency_s"] >= 0.0
    assert agg["prefill_latency_s"] >= 0.0 and agg["decode_latency_s"] >= 0.0
    # on_step invoked for schedule + serialize each step
    assert ("schedule", 0) in consumed and ("serialize", 5) in consumed


def test_profiler_write_trace_jsonl(tmp_path) -> None:
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    simulate_mock_decode(p, counters, n_tokens=4, hidden_size=8)
    out = tmp_path / "trace.jsonl"
    p.write_trace(out)
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 4
    assert lines[0]["phase"] == "prefill" and lines[1]["phase"] == "decode"
    assert "stage_timings" in lines[0] and "total_step_latency_s" in lines[0]


# ---- persistent HTTP transport (functional, mock worker) ------------------

@pytest.fixture()
def mock_server():
    from pllo.protocol.remote import GpuWorkerServer
    srv = GpuWorkerServer("127.0.0.1", 0, "mock", audit=True)
    srv.start_background()
    try:
        yield "http://127.0.0.1:%d" % srv.port
    finally:
        srv.shutdown()


def _drive(url, persistent):
    import numpy as np
    from pllo.protocol.remote import RemoteGpuWorker
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    H, V = 4, 5
    head = np.ones((H, V), dtype=np.float32)
    c = RemoteGpuWorker(url, "mock", persistent=persistent)
    try:
        c.init(BoundaryInitRequest(session_id="s", hidden_size=H, vocab_size=V,
                                   num_layers=1, dtype="float32",
                                   gpu_backend="mock", folded_lm_head=head))
        pre = c.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=np.ones((1, 3, H), np.float32),
            positions=[0, 1, 2], batch_size=1, seq_len=3))
        outs = [pre.masked_logits is not None]
        for step in range(1, 4):
            dec = c.decode(MaskedDecodeRequest(
                session_id="s", masked_embedding=np.ones((1, 1, H), np.float32),
                position=2 + step, step=step))
            outs.append(dec.masked_logits is not None)
        return outs
    finally:
        c.close()


def test_persistent_transport_multi_request(mock_server) -> None:
    # multiple init/prefill/decode over ONE reused keep-alive connection
    assert _drive(mock_server, persistent=True) == [True, True, True, True]


def test_non_persistent_transport_still_works(mock_server) -> None:
    assert _drive(mock_server, persistent=False) == [True, True, True, True]


def test_remote_post_rejects_forbidden_field(mock_server) -> None:
    from pllo.protocol.remote import RemoteGpuWorker
    from pllo.runtime.obfuscation_schedule import ScheduleSecretLeak

    class _Leaky:                                    # encodes to a dict with a secret
        def __init__(self): self.session_id = "s"; self.mask_secret = [1, 2]
    # monkeypatch encode to passthrough dict-like for this crafted object
    import pllo.protocol.remote as rmod
    orig = rmod.encode_message
    rmod.encode_message = lambda m: {"session_id": "s", "mask_secret": [1, 2]}
    try:
        c = RemoteGpuWorker(mock_server, "mock")
        with pytest.raises(ScheduleSecretLeak):
            c._post("/decode", _Leaky())
    finally:
        rmod.encode_message = orig


# ---- probe + ifeval scripts (mock) ----------------------------------------

def test_probe_emits_trace_and_metrics(tmp_path) -> None:
    mod = _load("probe_p", "scripts/run_precomputed_schedule_probe.py")
    oj = tmp_path / "probe.json"
    tr = tmp_path / "trace.jsonl"
    rc = _main(mod, ["x", "--hidden-size", "32", "--seq-len", "16",
                     "--max-new-tokens", "5", "--device", "cpu",
                     "--precompute-obfuscation-schedule", "--schedule-max-steps",
                     "16", "--schedule-seed", "2035", "--mock-runtime",
                     "--trace-decode-steps", "--trace-output-jsonl", str(tr),
                     "--output-json", str(oj)])
    assert rc == 0
    r = json.loads(oj.read_text())
    for k in ("latency_per_generated_token_s", "boundary_calls_per_generated_token",
              "gpu_calls_per_generated_token", "avg_http_roundtrip_s",
              "bottleneck_stage", "prefill_latency_s", "decode_latency_s",
              "online_decode_latency_s", "schedule_precompute_latency_s"):
        assert k in r
    assert r["bottleneck_stage"] == "gpu_worker_roundtrip"
    assert r["gpu_calls_per_generated_token"] == 1.0
    assert r["schedule_secret_leaked_to_gpu"] is False
    assert r["online_remask_still_performed"] is True
    assert sum(1 for _ in open(tr)) == 5            # one trace row per token


def test_ifeval_emits_target_metrics_and_trace(tmp_path) -> None:
    mod = _load("ifeval_p", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Write a poem."}) + "\n",
                  encoding="utf-8")
    rep = tmp_path / "rep.json"
    tr = tmp_path / "tr.jsonl"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend", "folded_remote",
                     "--mock-runtime", "--max-new-tokens", "6",
                     "--precompute-obfuscation-schedule", "--trace-decode-steps",
                     "--trace-output-jsonl", str(tr),
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep)])
    assert rc == 0
    r = json.loads(rep.read_text())
    for k in ("latency_per_generated_token_s", "boundary_calls_per_generated_token",
              "gpu_calls_per_generated_token", "trusted_bytes_per_generated_token",
              "gpu_bytes_per_generated_token", "avg_http_roundtrip_s",
              "avg_trusted_compute_s_per_token", "avg_gpu_compute_s_per_token",
              "online_decode_latency_s", "prefill_latency_s", "decode_latency_s",
              "schedule_precompute_latency_s", "bottleneck_stage"):
        assert k in r, "missing %s" % k
    assert r["bottleneck_stage"] == "gpu_worker_roundtrip"
    assert r["online_remask_still_performed"] is True
    assert r["schedule_used_for_metadata_only"] is True
    assert tr.is_file() and sum(1 for _ in open(tr)) >= 1
