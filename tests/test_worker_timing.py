"""Worker-side timing instrumentation tests (local, no model/GPU).

Covers the ``WorkerTimer`` + metadata builders + secret audit, the end-to-end
opt-in over the mock HTTP worker (the server returns its public forward timing,
the client splits the roundtrip), the decode profiler's worker-timing merge +
aggregate (worker_bottleneck_stage, network overhead), and the probe / ifeval
scripts emitting the worker fields with synthetic timing. No Qwen weights, no
server paths.

Run: python -m pytest tests/test_worker_timing.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.decode_profiler import DecodeProfiler  # noqa: E402
from pllo.protocol.worker_timing import (  # noqa: E402
    WORKER_TIMING_KEYS,
    WorkerTimer,
    audit_worker_timing_no_secrets,
    coarse_forward_metadata,
    empty_worker_timing,
    merge_server_timing,
    synthetic_worker_timing,
)
from pllo.runtime.obfuscation_schedule import ScheduleSecretLeak  # noqa: E402


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


# ---- WorkerTimer + builders ----------------------------------------------

def test_empty_and_coarse_metadata_keys() -> None:
    e = empty_worker_timing()
    assert set(e) == set(WORKER_TIMING_KEYS)
    assert all(v is None for v in e.values())
    c = coarse_forward_metadata(phase="decode", backend_name="qwen7b",
                                device="cuda:0", dtype="bfloat16",
                                forward_s=4.5, num_layers=28)
    assert c["worker_backend_forward_s"] == 4.5
    assert c["worker_prefill_or_decode"] == "decode"
    assert c["worker_num_layers"] == 28
    # coarse backend leaves the fine breakdown None (req. 11)
    assert c["worker_attention_total_s"] is None
    assert c["per_layer_timing_summary"] is None


def test_worker_timer_regions_and_layers() -> None:
    t = WorkerTimer(enabled=True)
    for _ in range(3):                                  # 3 folded layers
        with t.layer():
            with t.region("nonlinear"):
                pass
            with t.region("attention"):
                pass
            with t.region("mlp"):
                pass
    with t.region("lm_head"):
        pass
    meta = t.forward_metadata(phase="prefill", backend_name="qwen7b_folded_package",
                              device="cpu", dtype="float32", forward_s=0.01,
                              num_layers=3)
    assert meta["worker_num_layers"] == 3
    assert meta["worker_attention_total_s"] is not None
    assert meta["worker_mlp_total_s"] is not None
    assert meta["worker_nonlinear_total_s"] is not None
    assert meta["worker_lm_head_s"] is not None
    s = meta["per_layer_timing_summary"]
    assert s["count"] == 3 and s["max_s"] >= s["min_s"]
    assert "mean_s" in s


def test_worker_timer_disabled_is_noop() -> None:
    t = WorkerTimer(enabled=False)
    with t.layer():
        with t.region("attention"):
            pass
    meta = t.forward_metadata(phase="decode", backend_name="x", device="cpu",
                              dtype="float32", forward_s=None)
    assert meta["worker_num_layers"] is None
    assert meta["worker_attention_total_s"] is None


def test_merge_server_timing_fills_all_keys() -> None:
    fwd = coarse_forward_metadata(phase="decode", backend_name="mock",
                                  device="cpu", dtype="float32", forward_s=1.0)
    merged = merge_server_timing(fwd, total_s=1.2, parse_s=0.01, decode_s=0.02,
                                 encode_s=0.03, response_bytes=4096)
    assert set(merged) >= set(WORKER_TIMING_KEYS)
    assert merged["worker_total_s"] == 1.2
    assert merged["worker_request_parse_s"] == 0.01
    assert merged["worker_response_bytes"] == 4096
    assert merged["worker_backend_forward_s"] == 1.0


def test_audit_worker_timing_clean_and_secret() -> None:
    assert audit_worker_timing_no_secrets(
        synthetic_worker_timing(phase="decode"))["ok"] is True
    with pytest.raises(ScheduleSecretLeak):
        audit_worker_timing_no_secrets({"worker_total_s": 1.0,
                                        "mask_secret": [1, 2, 3]})
    with pytest.raises(ScheduleSecretLeak):
        audit_worker_timing_no_secrets({"prg_seed": 7})
    # non-scalar values (smuggled tensor / long vector) are rejected too
    import numpy as np
    with pytest.raises(ScheduleSecretLeak):
        audit_worker_timing_no_secrets({"worker_total_s": np.ones((3, 3))})
    with pytest.raises(ScheduleSecretLeak):
        audit_worker_timing_no_secrets({"worker_total_s": list(range(64))})


# ---- decode profiler merge + aggregate -----------------------------------

def test_profiler_merges_worker_timing_and_aggregates() -> None:
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    for step in range(3):
        phase = "prefill" if step == 0 else "decode"
        p.begin_step(step, phase)
        with p.stage("gpu_worker_roundtrip"):
            counters["gpu_calls"] += 1
        # worker reports it spent ~half the roundtrip in its own forward
        p.set_worker_timing(synthetic_worker_timing(phase=phase, forward_s=1.0))
        p.end_step(token_id=step)
    rows = p.rows()
    assert all("worker_timings" in r for r in rows)
    assert all("network_protocol_overhead_s" in r for r in rows)
    # overhead == observed roundtrip - worker_total_s (exact arithmetic)
    for r in rows:
        rt = r["stage_timings"]["gpu_worker_roundtrip"]
        wtot = r["worker_timings"]["worker_total_s"]
        assert r["network_protocol_overhead_s"] == round(rt - wtot, 9)
    agg = p.aggregate(generated_tokens=3)
    assert agg["avg_worker_total_s_per_token"] is not None
    assert agg["avg_worker_backend_forward_s_per_token"] == 1.0
    # the named matmul substages are tiny; the per-layer total (weight movement)
    # dominates -> attention must NOT be the bottleneck
    assert agg["worker_bottleneck_stage"] == "worker_layer_total_s"
    assert agg["avg_worker_known_substage_total_s_per_token"] is not None
    assert agg["avg_worker_unattributed_forward_s_per_token"] is not None
    assert agg["worker_timing_method"] == "cuda_event"
    assert agg["worker_timing_is_cuda_synchronized"] is True


def test_profiler_worker_fields_none_without_worker_timing() -> None:
    counters = {"boundary_calls": 0, "gpu_calls": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    p.begin_step(0, "prefill")
    with p.stage("gpu_worker_roundtrip"):
        counters["gpu_calls"] += 1
    p.end_step(token_id=1)
    agg = p.aggregate(generated_tokens=1)
    assert agg["avg_worker_total_s_per_token"] is None
    assert agg["worker_bottleneck_stage"] is None
    assert agg["avg_network_protocol_overhead_s_per_token"] is None


def _profiler_with_worker_timing(worker_timing):
    """A 2-step profiler whose every step carries the given worker_timing dict."""
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    for step in range(2):
        p.begin_step(step, "prefill" if step == 0 else "decode")
        with p.stage("gpu_worker_roundtrip"):
            counters["gpu_calls"] += 1
        p.set_worker_timing(dict(worker_timing))
        p.end_step(token_id=step)
    return p


def test_bottleneck_small_attention_large_layer_total() -> None:
    # attention tiny, layer_total large -> NOT attention (req. 7)
    from pllo.protocol.worker_timing import synthetic_worker_timing
    agg = _profiler_with_worker_timing(
        synthetic_worker_timing(phase="decode", forward_s=4.65)
    ).aggregate(generated_tokens=2)
    assert agg["worker_bottleneck_stage"] == "worker_layer_total_s"
    assert agg["worker_bottleneck_stage"] not in (
        "worker_attention_total_s", "worker_mlp_total_s")


def test_bottleneck_unattributed_when_no_layer_total() -> None:
    # most of forward unattributed AND no per-layer total -> forward_unattributed
    from pllo.protocol.worker_timing import (
        WORKER_FORWARD_UNATTRIBUTED, _fill_attribution, empty_worker_timing)
    wt = empty_worker_timing()
    wt.update({"worker_backend_forward_s": 4.6, "worker_total_s": 4.7,
               "worker_attention_total_s": 0.02, "worker_mlp_total_s": 0.01,
               "worker_nonlinear_total_s": 0.005, "worker_lm_head_s": 0.015,
               "worker_layer_total_s": None,      # not split into layers
               "worker_timing_method": "cuda_event",
               "worker_timing_is_cuda_synchronized": True})
    _fill_attribution(wt)
    assert wt["worker_unattributed_forward_s"] > 4.0
    agg = _profiler_with_worker_timing(wt).aggregate(generated_tokens=2)
    assert agg["worker_bottleneck_stage"] == WORKER_FORWARD_UNATTRIBUTED


def test_bottleneck_ignores_unreliable_substages() -> None:
    # unsynchronized wall-clock -> a big-looking attention must NOT be picked
    from pllo.protocol.worker_timing import _fill_attribution, empty_worker_timing
    wt = empty_worker_timing()
    wt.update({"worker_backend_forward_s": 4.6, "worker_total_s": 4.7,
               "worker_attention_total_s": 3.0,   # bogus (async launch artefact)
               "worker_mlp_total_s": 0.5, "worker_nonlinear_total_s": 0.1,
               "worker_lm_head_s": 0.1, "worker_layer_total_s": 3.6,
               "worker_timing_method": "wall_clock_unsynchronized",
               "worker_timing_is_cuda_synchronized": False})
    _fill_attribution(wt)
    agg = _profiler_with_worker_timing(wt).aggregate(generated_tokens=2)
    # unreliable -> a named substage must NOT be chosen; only coarse fields
    assert agg["worker_bottleneck_stage"] in (
        "worker_layer_total_s", "worker_backend_forward_s")
    assert agg["worker_bottleneck_stage"] not in (
        "worker_attention_total_s", "worker_mlp_total_s",
        "worker_nonlinear_total_s", "worker_lm_head_s")
    assert agg["worker_timing_is_cuda_synchronized"] is False


def test_bottleneck_largest_substage_when_substages_dominate() -> None:
    # if named substages DO account for most of forward, pick the largest one
    from pllo.protocol.worker_timing import _fill_attribution, empty_worker_timing
    wt = empty_worker_timing()
    wt.update({"worker_backend_forward_s": 1.0, "worker_total_s": 1.05,
               "worker_attention_total_s": 0.6, "worker_mlp_total_s": 0.25,
               "worker_nonlinear_total_s": 0.05, "worker_lm_head_s": 0.05,
               "worker_layer_total_s": 0.9,
               "worker_timing_method": "cuda_event",
               "worker_timing_is_cuda_synchronized": True})
    _fill_attribution(wt)                       # known=0.95 >= 0.5*1.0
    agg = _profiler_with_worker_timing(wt).aggregate(generated_tokens=2)
    assert agg["worker_bottleneck_stage"] == "worker_attention_total_s"


def _profiler_with_boundary_rate(total_boundary, n_tokens):
    """A profiler whose recorded steps sum to ``total_boundary`` boundary calls
    over ``n_tokens`` steps (1 gpu call each)."""
    counters = {"boundary_calls": 0, "gpu_calls": 0, "trusted_bytes": 0,
                "gpu_bytes": 0}
    p = DecodeProfiler(counters=lambda: dict(counters), enabled=True)
    # distribute boundary calls across steps (front-loaded; the rate is what matters)
    per = [total_boundary // n_tokens] * n_tokens
    for i in range(total_boundary - sum(per)):
        per[i] += 1
    for step in range(n_tokens):
        p.begin_step(step, "prefill" if step == 0 else "decode")
        with p.stage("trusted_input_embedding"):
            counters["boundary_calls"] += per[step]
        with p.stage("gpu_worker_roundtrip"):
            counters["gpu_calls"] += 1
        p.end_step(token_id=step)
    return p


def test_boundary_calls_reduced_false_for_1p875() -> None:
    # 30 boundary calls / 16 tokens = 1.875 -> NOT reduced (>= 1.5 threshold)
    agg = _profiler_with_boundary_rate(30, 16).aggregate(generated_tokens=16)
    assert agg["boundary_calls_per_generated_token"] == 1.875
    assert agg["boundary_calls_reduced"] is False
    assert "still" in agg["boundary_calls_reduction_note"]


def test_boundary_calls_reduced_true_below_threshold() -> None:
    # 4 boundary calls / 4 tokens = 1.0 -> reduced (< 1.5 threshold)
    agg = _profiler_with_boundary_rate(4, 4).aggregate(generated_tokens=4)
    assert agg["boundary_calls_per_generated_token"] == 1.0
    assert agg["boundary_calls_reduced"] is True
    assert "reduced to" in agg["boundary_calls_reduction_note"]


def test_ifeval_schedule_uses_real_hidden_size_dtype(tmp_path) -> None:
    """The schedule + report must size against the REAL hidden_size/dtype passed
    via CLI, never the mock float32/hidden=1 placeholder."""
    mod = _load("ifeval_hs", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Write a poem."}) + "\n",
                  encoding="utf-8")
    rep = tmp_path / "rep.json"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend", "folded_remote",
                     "--mock-runtime", "--max-new-tokens", "4",
                     "--hidden-size", "3584", "--dtype", "bfloat16",
                     "--precompute-obfuscation-schedule", "--report-schedule-stats",
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep)])
    assert rc == 0
    r = json.loads(rep.read_text())
    assert r["schedule_hidden_size"] == 3584
    assert r["schedule_dtype"] == "bfloat16"
    assert r["schedule_hidden_size_is_placeholder"] is False
    stats = r.get("schedule_stats") or {}
    assert stats.get("hidden_size") == 3584
    assert stats.get("dtype") == "bfloat16"


def test_ifeval_flags_placeholder_hidden_size(tmp_path) -> None:
    """With no --hidden-size and no real backend, the placeholder must be flagged
    (so a mock size never silently pollutes a 'real' report)."""
    mod = _load("ifeval_ph", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Hi."}) + "\n",
                  encoding="utf-8")
    rep = tmp_path / "rep.json"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend", "folded_remote",
                     "--mock-runtime", "--max-new-tokens", "3",
                     "--precompute-obfuscation-schedule",
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep)])
    assert rc == 0
    r = json.loads(rep.read_text())
    assert r["schedule_hidden_size_is_placeholder"] is True


# ---- end-to-end over the mock HTTP worker --------------------------------

@pytest.fixture()
def mock_server():
    from pllo.protocol.remote import GpuWorkerServer
    srv = GpuWorkerServer("127.0.0.1", 0, "mock", audit=True)
    srv.start_background()
    try:
        yield "http://127.0.0.1:%d" % srv.port
    finally:
        srv.shutdown()


def _init_prefill_decode(url, *, request_worker_timing):
    import numpy as np
    from pllo.protocol.remote import RemoteGpuWorker
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    H, V = 4, 5
    head = np.ones((H, V), dtype=np.float32)
    c = RemoteGpuWorker(url, "mock", request_worker_timing=request_worker_timing)
    try:
        c.init(BoundaryInitRequest(session_id="s", hidden_size=H, vocab_size=V,
                                   num_layers=1, dtype="float32",
                                   gpu_backend="mock", folded_lm_head=head))
        pre = c.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=np.ones((1, 3, H), np.float32),
            positions=[0, 1, 2], batch_size=1, seq_len=3))
        dec = c.decode(MaskedDecodeRequest(
            session_id="s", masked_embedding=np.ones((1, 1, H), np.float32),
            position=3, step=1))
        return pre, dec
    finally:
        c.close()


def test_server_returns_worker_timing_when_requested(mock_server) -> None:
    pre, dec = _init_prefill_decode(mock_server, request_worker_timing=True)
    for resp, phase in ((pre, "prefill"), (dec, "decode")):
        wt = resp.worker_timing
        assert isinstance(wt, dict)
        # server-merged stage fields are present
        for k in ("worker_total_s", "worker_request_parse_s",
                  "worker_payload_decode_s", "worker_payload_encode_s",
                  "worker_response_bytes", "worker_backend_forward_s",
                  "worker_backend_name", "worker_prefill_or_decode"):
            assert k in wt and wt[k] is not None, "missing %s" % k
        assert wt["worker_backend_name"] == "mock"
        assert wt["worker_prefill_or_decode"] == phase
        # timing provenance present (mock runs on CPU -> wall_clock)
        assert wt["worker_timing_method"] == "wall_clock"
        assert "worker_timing_is_cuda_synchronized" in wt
        # no secret rode back
        assert audit_worker_timing_no_secrets(wt)["ok"] is True


def test_server_omits_worker_timing_by_default(mock_server) -> None:
    pre, dec = _init_prefill_decode(mock_server, request_worker_timing=False)
    assert pre.worker_timing is None
    assert dec.worker_timing is None


# ---- probe + ifeval scripts emit worker fields (mock) ---------------------

def test_probe_emits_worker_timing(tmp_path) -> None:
    mod = _load("probe_wt", "scripts/run_precomputed_schedule_probe.py")
    oj = tmp_path / "probe.json"
    tr = tmp_path / "trace.jsonl"
    rc = _main(mod, ["x", "--hidden-size", "32", "--seq-len", "16",
                     "--max-new-tokens", "5", "--device", "cpu",
                     "--precompute-obfuscation-schedule", "--schedule-max-steps",
                     "16", "--schedule-seed", "2035", "--mock-runtime",
                     "--trace-decode-steps", "--trace-worker-timings",
                     "--trace-output-jsonl", str(tr), "--output-json", str(oj)])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["worker_timing_requested"] is True
    for k in ("avg_worker_total_s_per_token",
              "avg_worker_backend_forward_s_per_token",
              "avg_network_protocol_overhead_s_per_token",
              "worker_bottleneck_stage"):
        assert k in r and r[k] is not None, "missing %s" % k
    # the per-token trace carries the worker timing + derived overhead
    rows = [json.loads(l) for l in tr.read_text().splitlines() if l.strip()]
    assert any("worker_timings" in row for row in rows)
    assert any("network_protocol_overhead_s" in row for row in rows)


# ---- detailed timing through the REAL folded-package kernels (CPU) --------

def _build_pkg_and_artifact(tmp_path, *, n_layers, seed):
    builder = _load("buildpkg_t", "scripts/build_qwen7b_folded_package.py")
    pkg = tmp_path / "pkg"
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", str(n_layers), "--seed", str(seed),
                           "--write-manifest", "true"]) == 0
    embuild = _load("embart_t", "scripts/build_qwen7b_embedding_artifact.py")
    art = tmp_path / "art"
    assert _main(embuild, ["prog", "--dry-run", "--output-dir", str(art),
                           "--num-layers", str(n_layers), "--seed",
                           str(seed)]) == 0
    return pkg, art


def test_folded_package_detailed_worker_timing(tmp_path) -> None:
    """The folded-package worker, when timing is requested, splits its forward
    into per-layer / attention / MLP / nonlinear / LM-head sub-totals computed by
    the REAL torch kernels -- and carries no secret back."""
    import numpy as np
    import torch
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)

    n_layers, seq_len = 2, 4
    pkg, art = _build_pkg_and_artifact(tmp_path, n_layers=n_layers, seed=2035)
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    ids = torch.randint(0, 256, (1, seq_len))
    h_tilde = boundary.mask_embeddings(ids)
    meta = boundary.exec_metadata(seq_len=seq_len, max_new_tokens=2)

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port

    def _np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    try:
        worker = RemoteGpuWorker(url, "qwen7b_folded_package",
                                 request_worker_timing=True)
        worker.init(BoundaryInitRequest(
            session_id="t", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=n_layers,
            dtype="float32", gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))
        pre = worker.prefill(MaskedPrefillRequest(
            session_id="t", masked_embeddings=_np(h_tilde),
            positions=list(range(seq_len)), batch_size=1, seq_len=seq_len))
        tok = int(boundary.recover(torch.as_tensor(np.asarray(pre.masked_logits))
                                   .to(boundary.compute_device,
                                       boundary.fdtype)).reshape(-1).argmax())
        x = boundary.mask_token_embedding(torch.tensor([tok]))
        dec = worker.decode(MaskedDecodeRequest(
            session_id="t", masked_embedding=_np(x), position=seq_len, step=1))
        worker.close()
    finally:
        server.shutdown()

    for resp, phase in ((pre, "prefill"), (dec, "decode")):
        wt = resp.worker_timing
        assert isinstance(wt, dict)
        assert wt["worker_backend_name"] == "qwen7b_folded_package"
        assert wt["worker_prefill_or_decode"] == phase
        assert wt["worker_num_layers"] == n_layers
        # the REAL folded kernels populated the fine breakdown
        for k in ("worker_layer_total_s", "worker_lm_head_s",
                  "worker_attention_total_s", "worker_mlp_total_s",
                  "worker_nonlinear_total_s", "worker_backend_forward_s",
                  "worker_total_s", "worker_known_substage_total_s",
                  "worker_unattributed_forward_s"):
            assert isinstance(wt[k], (int, float)), "missing %s" % k
        assert wt["per_layer_timing_summary"]["count"] == n_layers
        # CPU path -> wall_clock (synchronous, accurate); attribution holds
        assert wt["worker_timing_method"] == "wall_clock"
        assert wt["worker_known_substage_total_s"] == round(
            sum(wt[k] for k in ("worker_attention_total_s", "worker_mlp_total_s",
                                "worker_nonlinear_total_s", "worker_lm_head_s")),
            9)
        assert audit_worker_timing_no_secrets(wt)["ok"] is True
        assert audit_worker_timing_no_secrets(wt)["ok"] is True


def test_ifeval_emits_worker_timing(tmp_path) -> None:
    mod = _load("ifeval_wt", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Write a poem."}) + "\n",
                  encoding="utf-8")
    rep = tmp_path / "rep.json"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend", "folded_remote",
                     "--mock-runtime", "--max-new-tokens", "6",
                     "--precompute-obfuscation-schedule", "--trace-decode-steps",
                     "--trace-worker-timings",
                     "--trace-output-jsonl", str(tmp_path / "tr.jsonl"),
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep)])
    assert rc == 0
    r = json.loads(rep.read_text())
    assert r["worker_timing_requested"] is True
    assert r["avg_worker_backend_forward_s_per_token"] is not None
    assert r["worker_bottleneck_stage"] is not None
    # resident status is surfaced in the report (mock has no real worker -> False)
    assert r["resident_folded_weights"] is False
