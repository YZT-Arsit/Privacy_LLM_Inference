"""Cross-machine (HTTP) TEE <-> GPU protocol tests.

Runs the untrusted GPU worker as a real stdlib HTTP server on localhost and
drives it through the boundary client / orchestrator. numpy + stdlib only; the
mock backend needs no GPU. The qwen7b backend is constructed without importing
torch (lazy), so its health/init can be checked here too.

Run: python -m pytest tests/test_tee_gpu_protocol_remote.py -q
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pllo.protocol import (
    GpuWorkerServer,
    RemoteGpuWorker,
    assert_no_gpu_visible_plaintext,
    assert_no_mask_secret_leak,
    boundary_manifest_metadata,
    boundary_runtime_hash,
    run_protocol,
)
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest

PROMPT = "Explain why privacy matters in private LLM inference systems."
REAL_MR_TD = ("e0199499baacb2e4f4bc73046f25bedf674d42defbe4e854242bd6554a9d155e"
              "df7f3bff8e6202e63ed230e59ab2568a")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def mock_server():
    srv = GpuWorkerServer("127.0.0.1", 0, "mock", audit=True)
    srv.start_background()
    try:
        yield f"http://127.0.0.1:{srv.port}"
    finally:
        srv.shutdown()


@pytest.fixture()
def qwen_server():
    srv = GpuWorkerServer("127.0.0.1", 0, "qwen7b",
                          {"model_path": "/fake", "device": "cpu",
                           "dtype": "float32"}, audit=True)
    srv.start_background()
    try:
        yield f"http://127.0.0.1:{srv.port}", srv
    finally:
        srv.shutdown()


def _post_raw(url: str, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url + path, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return urllib.request.urlopen(req, timeout=10)


# 1.
def test_gpu_worker_server_rejects_plaintext_fields(mock_server) -> None:
    url = mock_server
    # health works
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        assert json.loads(r.read())["tee_used_on_gpu"] is False
    # a body carrying a forbidden plaintext field is rejected with HTTP 400
    for bad_field in ("raw_prompt", "input_ids", "generated_token_ids",
                      "recovered_logits", "mask_secret", "tokenizer_output"):
        payload = {"__msgtype__": "MaskedPrefillRequest", "session_id": "s",
                   bad_field: "leak"}
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post_raw(url, "/prefill", payload)
        assert ei.value.code == 400
        detail = json.loads(ei.value.read())
        assert detail["error"] == "forbidden_field"
        assert any(bad_field in f for f in detail["fields"])
        assert detail["tee_used_on_gpu"] is False


# 2.
def test_boundary_client_sends_no_plaintext(mock_server) -> None:
    out = run_protocol(PROMPT, boundary_backend="simulated", gpu_backend="mock",
                       max_new_tokens=4, hidden_size=64, vocab_size=500,
                       seq_len=8, seed=99991, gpu_worker_url=mock_server)
    trace = out["trace"]
    assert trace.gpu_inbound_messages                    # traffic actually flowed
    # no raw prompt / input_ids / generated tokens on the wire to the GPU
    assert assert_no_gpu_visible_plaintext(
        trace, raw_prompt=PROMPT, input_ids=out["input_ids"],
        generated_token_ids=out["generated_token_ids"], raise_on_fail=False) == []
    assert assert_no_mask_secret_leak(
        trace, out["handles"], raise_on_fail=False) == []


# 3.
def test_cross_machine_mock_protocol_audit_passes(mock_server) -> None:
    out = run_protocol(PROMPT, boundary_backend="simulated", gpu_backend="mock",
                       max_new_tokens=4, hidden_size=64, vocab_size=500,
                       seq_len=8, seed=99991, gpu_worker_url=mock_server)
    trace = out["trace"]
    assert out["gpu_worker_remote"] is True
    assert trace.tee_used_on_gpu is False
    # recovered tokens match the trusted plaintext reference over HTTP
    assert out["tokens_match_reference"] is True
    # remote calls recorded
    assert trace.gpu_calls.get("BoundaryInitRequest") == 1
    assert trace.gpu_calls.get("MaskedPrefillRequest") == 1
    assert trace.gpu_calls.get("MaskedDecodeRequest") == 4 - 1
    audit_passed = (
        not assert_no_gpu_visible_plaintext(
            trace, raw_prompt=PROMPT, input_ids=out["input_ids"],
            generated_token_ids=out["generated_token_ids"], raise_on_fail=False)
        and not assert_no_mask_secret_leak(trace, out["handles"],
                                           raise_on_fail=False)
        and not trace.tee_used_on_gpu)
    assert audit_passed


# 4.
def test_attestation_fields_preserved_in_boundary_client_report(
        mock_server, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "rtgpd", REPO_ROOT / "scripts" / "run_tee_gpu_protocol_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    report = demo.build_report(
        PROMPT, "simulated", "mock", 4, True, hidden_size=64, vocab_size=500,
        seq_len=8, seed=99991, gpu_worker_url=mock_server)
    assert report["gpu_worker_remote"] is True
    assert report["gpu_worker_url"] == mock_server
    assert report["tee_used_on_gpu"] is False

    # craft evidence bound to the runtime hash for THIS metadata
    md = boundary_manifest_metadata("simulated", "mock", REAL_MR_TD)
    rh = boundary_runtime_hash(metadata=md)
    ev = {"tee": "tdx", "tdx": {"td_attributes": {"debug": False}},
          "mr_td": REAL_MR_TD, "report_data": rh, "jwt": "a.b.c"}
    ev_path = tmp_path / "evidence.json"
    ev_path.write_text(json.dumps(ev), encoding="utf-8")

    demo.attach_attestation(report, evidence=str(ev_path),
                            expected_mr_td=REAL_MR_TD)
    assert report["boundary_tee_type"] == "tdx"
    assert report["boundary_attested"] is True
    assert report["runtime_hash_bound"] is True
    assert report["expected_runtime_hash"] == rh
    assert report["mr_td"] == REAL_MR_TD


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "rtgpd", REPO_ROOT / "scripts" / "run_tee_gpu_protocol_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


# 5.
def test_qwen7b_worker_reports_tee_used_false(qwen_server) -> None:
    url, srv = qwen_server
    assert srv.tee_used_on_gpu is False
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        h = json.loads(r.read())
    assert h["gpu_backend"] == "qwen7b"
    assert h["tee_used_on_gpu"] is False
    # init handshake over HTTP also reports tee_used_on_gpu=False (no torch)
    client = RemoteGpuWorker(url, "qwen7b")
    resp = client.init(BoundaryInitRequest(
        session_id="s", hidden_size=3584, vocab_size=152064, num_layers=28,
        dtype="bfloat16", gpu_backend="qwen7b"))
    assert resp.tee_used_on_gpu is False
    assert resp.gpu_backend == "qwen7b"
    client.close()


# 6. cross-machine mock end-to-end report carries the three-way scope fields
def test_boundary_client_mock_report_has_scope_fields(mock_server) -> None:
    demo = _load_demo()
    report = demo.build_report(
        PROMPT, "simulated", "mock", 4, True, hidden_size=64, vocab_size=500,
        seq_len=8, seed=99991, gpu_worker_url=mock_server)
    demo._annotate_cross_machine_scope(
        report, model_name="mock-identity", cross_machine_compute="end_to_end",
        limitations="none")
    assert report["cross_machine_compute"] == "end_to_end"
    assert report["gpu_worker_remote"] is True
    assert report["tee_used_on_gpu"] is False
    # required schema keys present (even if None for the mock identity decoder)
    for k in ("model_name", "compute_correctness_source",
              "security_boundary_source", "connectivity_note", "limitations",
              "teacher_forced_top1_match_rate", "plain_vs_masked_token_match_rate",
              "latency_s", "trusted_bytes", "gpu_bytes", "boundary_calls"):
        assert k in report


# 7. cross-machine qwen7b init/audit probe against a REMOTE worker (no torch)
def test_boundary_client_qwen7b_remote_probe(qwen_server, tmp_path) -> None:
    url, srv = qwen_server
    demo = _load_demo()
    report = demo.build_remote_qwen7b_probe_report(
        PROMPT, "process", 64, True, url, model_name="Qwen2.5-7B-Instruct",
        seq_len=128, num_layers=28, dtype="bfloat16")
    assert report["mode"] == "boundary_client"
    assert report["qwen7b_probe_only"] is True
    assert report["gpu_worker_remote"] is True
    assert report["gpu_worker_url"] == url
    assert report["tee_used_on_gpu"] is False
    assert report["gpu_backend_server_reported"] == "qwen7b"
    # init traffic crossed (no arrays -> gpu_bytes==0) but carried no secret
    assert report["gpu_inbound_message_count"] >= 1
    assert report["gpu_calls"].get("BoundaryInitRequest") == 1
    assert report["gpu_visible_plaintext_fields"] == []
    assert report["leaked_secret_fields"] == []
    assert report["audit_passed"] is True
    assert report["tokens_match_plaintext_reference"] is None  # probe, not compute
    # full scope annotation + attestation fold-in (TDX evidence bound)
    demo._annotate_cross_machine_scope(
        report, model_name="Qwen2.5-7B-Instruct",
        cross_machine_compute="probe_only", limitations="folded-weight shipping")
    md = boundary_manifest_metadata("process", "qwen7b", REAL_MR_TD)
    rh = boundary_runtime_hash(metadata=md)
    ev = {"tee": "tdx", "tdx": {"td_attributes": {"debug": False}},
          "mr_td": REAL_MR_TD, "report_data": rh, "jwt": "a.b.c"}
    ev_path = tmp_path / "evidence.json"
    ev_path.write_text(json.dumps(ev), encoding="utf-8")
    demo.attach_attestation(report, evidence=str(ev_path),
                            expected_mr_td=REAL_MR_TD)
    assert report["boundary_tee_type"] == "tdx"
    assert report["boundary_attested"] is True
    assert report["runtime_hash_bound"] is True
    assert report["model_name"] == "Qwen2.5-7B-Instruct"
    assert report["cross_machine_compute"] == "probe_only"
