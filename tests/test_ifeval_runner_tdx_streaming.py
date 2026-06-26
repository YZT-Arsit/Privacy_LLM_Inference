"""Tests for run_ifeval_generation.py: progress, streaming responses, the
online-deterministic schedule proof (no cuda secret tensors), and the TDX
boundary-client mode. All run on the deterministic mock path (no model/CUDA).

Run:
    PYTHONPATH=$PWD/src pytest tests/test_ifeval_runner_tdx_streaming.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_R = _load("ifeval_runner", "scripts/run_ifeval_generation.py")


def _write_prompts(tmp_path, n=2):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": "ex%d" % i, "prompt": "Question %d?" % i})
        for i in range(n)) + "\n", encoding="utf-8")
    return p


def _run(tmp_path, extra):
    """Run main() on the mock folded path; return (rc, report, resp_lines)."""
    inp = _write_prompts(tmp_path)
    rj = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    argv = ["x", "--input-jsonl", str(inp), "--backend", "folded_remote",
            "--mock-runtime", "--max-new-tokens", "4",
            "--output-response-jsonl", str(rj),
            "--output-report-json", str(rep)] + extra
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    report = json.loads(rep.read_text()) if rep.exists() else None
    lines = (rj.read_text().splitlines() if rj.exists() else [])
    return rc, report, lines


def test_progress_prints_example_i_of_n(tmp_path, capsys) -> None:
    rc, _report, _lines = _run(tmp_path, ["--progress", "--progress-every", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[ifeval] example 1/2" in out
    assert "[ifeval] example 2/2" in out
    assert "phase=generate_start" in out and "phase=done" in out
    assert "eta=" in out


def test_streaming_writes_one_line_per_example(tmp_path) -> None:
    # default streaming ON -> the JSONL has one line per example, parseable
    rc, _report, lines = _run(tmp_path, [])
    assert rc == 0
    assert len(lines) == 2
    ids = [json.loads(ln)["id"] for ln in lines]
    assert ids == ["ex0", "ex1"]


def test_online_deterministic_does_not_materialize_secret_tensors(tmp_path):
    rc, report, _lines = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16",
        "--schedule-proof-mode", "online_deterministic"])
    assert rc == 0
    assert report["schedule_proof_mode"] == "online_deterministic"
    assert report["secret_tensor_precompute_performed"] is False
    assert report["schedule_materialized_on_gpu"] is False
    assert report["schedule_precompute_device"] == "cpu"
    assert report["schedule_secret_derivation"] == "online_deterministic"


def test_schedule_full_coverage_verified(tmp_path) -> None:
    rc, report, _lines = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16"])
    assert rc == 0
    assert report["schedule_full_coverage_verified"] is True
    assert (report["schedule_slots_required_total"]
            == report["schedule_slots_consumed_total"])
    # every example proves slots_consumed == generated_tokens, no gpu leak
    cov = report["schedule_coverage_per_example"]
    assert len(cov) == 2
    for r in cov:
        assert r["slots_consumed_matches_generated_tokens"] is True
        assert r["slots_required"] == r["generated_tokens"] == r["slots_consumed"]
        assert r["schedule_secret_leaked_to_gpu"] is False
        assert r["schedule_materialized_on_gpu"] is False
        assert r["schedule_seed_commitment"]            # a non-empty commitment
    assert report["schedule_secret_leaked_to_gpu"] is False
    assert report["online_remask_still_performed"] is True


def test_legacy_precompute_flag_no_default_cuda_materialize(tmp_path) -> None:
    # the OLD command (just --precompute-obfuscation-schedule, device cuda) must
    # no longer materialize secret tensors -- it falls into the fast metadata path
    rc, report, lines = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16",
        "--device", "cuda"])
    assert rc == 0
    assert report["precompute_obfuscation_schedule"] is True
    assert report["secret_tensor_precompute_performed"] is False
    assert report["schedule_materialized_on_gpu"] is False
    assert report["schedule_proof_mode"] == "online_deterministic"
    assert len(lines) == 2                       # still streamed + completed


def test_precompute_secret_tensors_needs_strong_confirm(tmp_path, capsys) -> None:
    # asking for the heavy mode WITHOUT confirmation warns + falls back (and on
    # the dry/mock path never materializes anyway)
    rc, report, _lines = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "8",
        "--schedule-proof-mode", "precompute_secret_tensors"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "precompute_secret_tensors" in err and "falling back" in err
    assert report["secret_tensor_precompute_performed"] is False


def test_tdx_boundary_client_no_full_weights(tmp_path) -> None:
    rc, report, lines = _run(tmp_path, [
        "--tdx-boundary-client", "--trusted-runtime", "real_tdx",
        "--gpu-worker-url", "http://127.0.0.1:18082"])
    assert rc == 0
    assert report["tdx_boundary_client"] is True
    assert report["tee_mode"] == "real_tdx"
    assert report["trusted_runtime"] == "tdx_guest"
    # the mock folded path loads NO full 7B weights
    assert report["full_model_weights_loaded_in_trusted_runtime"] is False
    assert report["h800_worker_url"] == "http://127.0.0.1:18082"
    # honest: a dry/mock run with no attestation evidence is NOT claim-ready
    assert report["tdx_claim_ready"] is False
    assert len(lines) == 2


def test_tdx_boundary_client_rejects_plaintext_backend(tmp_path) -> None:
    inp = _write_prompts(tmp_path)
    argv = ["x", "--input-jsonl", str(inp), "--backend", "plaintext_local",
            "--mock-runtime", "--tdx-boundary-client", "--max-new-tokens", "4",
            "--output-response-jsonl", str(tmp_path / "r.jsonl"),
            "--output-report-json", str(tmp_path / "r.json")]
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    assert rc == 3                                  # refused


def test_tdx_claim_ready_with_evidence_on_real_only(tmp_path) -> None:
    # evidence attached but still a MOCK run -> claim not ready (honest)
    ev = tmp_path / "evidence.json"
    ev.write_text(json.dumps({"quote": "deadbeef", "mr_td": "abc"}))
    rc, report, _lines = _run(tmp_path, [
        "--tdx-boundary-client", "--gpu-worker-url", "http://127.0.0.1:18082",
        "--attestation-evidence-json", str(ev)])
    assert rc == 0
    assert report["attestation_evidence_attached"] is True
    # dry_run True -> not claim ready despite evidence
    assert report["dry_run"] is True
    assert report["tdx_claim_ready"] is False


def test_proof_mode_none_disables_schedule(tmp_path) -> None:
    rc, report, _lines = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-proof-mode", "none"])
    assert rc == 0
    assert report["precompute_obfuscation_schedule"] is False
    assert report["schedule_slots_required_total"] == 0
    assert report["schedule_full_coverage_verified"] is False
