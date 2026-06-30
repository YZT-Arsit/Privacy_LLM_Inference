"""AAAI long-run hardening: resume + raw-prompt protection + worker retry +
MT-Bench/HumanEval/sensitive preservation + Alibaba TDX quote wrapper.

Run: python -m pytest tests/test_aaai_resume_preservation_alibaba.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
_IFEVAL = REPO_ROOT / "scripts" / "run_ifeval_generation.py"


def _load_script(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ifeval_cmd(inp, resp, rep, *extra, dataset="ifeval"):
    return ([sys.executable, str(_IFEVAL), "--input-jsonl", str(inp),
             "--backend", "plaintext_local", "--mock-runtime", "--dataset",
             dataset, "--max-new-tokens", "4",
             "--output-response-jsonl", str(resp),
             "--output-report-json", str(rep)] + list(extra))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. resume: completed ids skipped, append (not overwrite), status + heartbeat
# ---------------------------------------------------------------------------

def test_resume_skips_completed_and_appends(tmp_path) -> None:
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    status = tmp_path / "s.json"
    hb = tmp_path / "h.json"
    r1 = subprocess.run(
        _ifeval_cmd(inp, resp, rep, "--status-json", str(status),
                    "--heartbeat-json", str(hb)),
        capture_output=True, env=_ENV)
    assert r1.returncode == 0
    first = resp.read_text(encoding="utf-8")
    assert len(first.strip().splitlines()) == 2
    # status + heartbeat content
    s = json.loads(status.read_text())
    h = json.loads(hb.read_text())
    assert s["completed_examples"] == 2 and s["failed_examples"] == 0
    assert s["nonlinear_backend"] and s["paper_ready_so_far"] is True
    assert h["pid"] and h["hostname"] and h["alive"] is False
    assert "elapsed_s" in h and "timestamp" in h
    # resume: both ids already done -> skipped, file NOT overwritten/grown
    r2 = subprocess.run(_ifeval_cmd(inp, resp, rep, "--resume"),
                        capture_output=True, env=_ENV)
    assert r2.returncode == 0
    assert resp.read_text(encoding="utf-8") == first       # unchanged (append-only)
    rep2 = json.loads(rep.read_text())
    assert rep2["skipped_existing_examples"] == 2
    assert rep2["completed_this_run"] == 0


def test_resume_retries_failed_record(tmp_path) -> None:
    # seed an existing output with one 'ok' and one 'failed' record
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"}])
    resp = tmp_path / "resp.jsonl"
    _write_jsonl(resp, [
        {"id": "a", "status": "ok", "response": "x", "num_tokens": 1},
        {"id": "b", "status": "failed", "error_type": "Timeout"}])
    rep = tmp_path / "rep.json"
    # default: failed 'b' is retried (not skipped)
    r = subprocess.run(_ifeval_cmd(inp, resp, rep, "--resume"),
                       capture_output=True, env=_ENV)
    assert r.returncode == 0
    rows = [json.loads(x) for x in resp.read_text().strip().splitlines()]
    # a appended record for 'b' with status ok now exists
    assert any(x["id"] == "b" and x.get("status") == "ok" for x in rows)
    # --skip-failed-existing: 'b' is treated as terminal -> skipped
    resp2 = tmp_path / "resp2.jsonl"
    _write_jsonl(resp2, [
        {"id": "a", "status": "ok", "response": "x", "num_tokens": 1},
        {"id": "b", "status": "failed", "error_type": "Timeout"}])
    rep2 = tmp_path / "rep2.json"
    r2 = subprocess.run(
        _ifeval_cmd(inp, resp2, rep2, "--resume", "--skip-failed-existing"),
        capture_output=True, env=_ENV)
    assert r2.returncode == 0
    assert json.loads(rep2.read_text())["skipped_existing_examples"] == 2


# ---------------------------------------------------------------------------
# 2. raw-prompt protection
# ---------------------------------------------------------------------------

def test_response_jsonl_has_no_raw_prompt_by_default(tmp_path) -> None:
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "TOP SECRET prompt body"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    assert subprocess.run(_ifeval_cmd(inp, resp, rep),
                          capture_output=True, env=_ENV).returncode == 0
    rec = json.loads(resp.read_text().strip())
    # the raw-prompt FIELD is never persisted; only a sha256 is. (The mock stub's
    # `response` echoes the prompt -- that is the trusted-side model output the GPU
    # never sees, not a stored raw prompt.)
    assert "prompt" not in rec and rec["prompt_sha256"]
    rep_json = json.loads(rep.read_text())
    assert rep_json["response_jsonl_contains_raw_prompt"] is False
    assert rep_json["prompt_sha256_only"] is True


def test_save_raw_prompts_opt_in_writes_prompt(tmp_path) -> None:
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "openly logged"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    assert subprocess.run(
        _ifeval_cmd(inp, resp, rep, "--save-raw-prompts"),
        capture_output=True, env=_ENV).returncode == 0
    rec = json.loads(resp.read_text().strip())
    assert rec.get("prompt") == "openly logged"


def test_save_raw_prompts_forbidden_under_paper_facing(tmp_path) -> None:
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "p"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    r = subprocess.run(
        _ifeval_cmd(inp, resp, rep, "--save-raw-prompts",
                    "--paper-facing-no-raw-prompts"),
        capture_output=True, env=_ENV)
    assert r.returncode == 3


def test_sensitive_dataset_never_saves_raw_prompt_or_span(tmp_path) -> None:
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "s1", "prompt": "acct SECRET-12345 summarize",
                        "sensitive_spans": ["SECRET-12345"]}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    # even with --save-raw-prompts, a sensitive dataset must redact
    assert subprocess.run(
        _ifeval_cmd(inp, resp, rep, "--save-raw-prompts",
                    dataset="sensitive_prompt_1024"),
        capture_output=True, env=_ENV).returncode == 0
    txt = resp.read_text()
    assert "SECRET-12345" not in txt
    rec = json.loads(txt.strip())
    assert "prompt" not in rec
    r = json.loads(rep.read_text())
    assert r["sensitive_dataset"] is True
    assert r["response_jsonl_contains_raw_prompt"] is False


def test_sanitize_error_strips_prompt_and_spans() -> None:
    run = _load_script("run_ifeval", "scripts/run_ifeval_generation.py")
    exc = RuntimeError("failed near SECRET-12345 and prompt body XYZ")
    et, msg = run._sanitize_error(exc, sensitive_spans=["SECRET-12345"],
                                  raw_prompt="prompt body XYZ")
    assert et == "RuntimeError"
    assert "SECRET-12345" not in msg and "prompt body XYZ" not in msg


# ---------------------------------------------------------------------------
# 3. worker retry / reconnect (ResilientRemoteGpuWorker)
# ---------------------------------------------------------------------------

class _FlakyClient:
    """Fails `fail_times` with a retriable error, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0
        self.closed = 0

    def decode(self, req):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("connection refused")
        return {"ok": True, "n": self.calls}

    def close(self):
        self.closed += 1


def test_resilient_retries_then_succeeds() -> None:
    from pllo.protocol.resilient_remote import ResilientRemoteGpuWorker
    server = _FlakyClient(fail_times=2)
    slept = []
    w = ResilientRemoteGpuWorker(
        "http://x", max_retries=5, sleep_fn=slept.append,
        client_factory=lambda: server)
    out = w.decode({"step": 1})
    assert out["ok"] is True
    assert w.retry_count == 2 and w.reconnect_count == 2
    assert len(slept) == 2                         # backed off twice


def test_resilient_exhausts_and_raises() -> None:
    from pllo.protocol.resilient_remote import (
        ResilientRemoteGpuWorker, WorkerUnavailable)
    server = _FlakyClient(fail_times=99)
    w = ResilientRemoteGpuWorker(
        "http://x", max_retries=2, sleep_fn=lambda s: None,
        client_factory=lambda: server)
    with pytest.raises(WorkerUnavailable):
        w.decode({"step": 1})
    assert w.retry_count == 2                       # 1 try + 2 retries = 3 attempts


def test_runner_records_failed_without_crashing(tmp_path, monkeypatch) -> None:
    # a per-example failure must NOT crash the whole run (no --fail-fast) and must
    # be recorded as a failed record (no raw prompt) with paper_ready=False.
    run = _load_script("run_ifeval2", "scripts/run_ifeval_generation.py")
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"

    real_stub = run.stub_generate
    calls = {"n": 0}

    def _flaky_stub(ex, n):
        calls["n"] += 1
        if ex.get("id") == "b":
            raise ConnectionError("connection refused (mock)")
        return real_stub(ex, n)

    monkeypatch.setattr(run, "stub_generate", _flaky_stub)
    argv = ["run_ifeval_generation.py", "--input-jsonl", str(inp),
            "--backend", "plaintext_local", "--mock-runtime", "--dataset",
            "ifeval", "--max-new-tokens", "4",
            "--output-response-jsonl", str(resp),
            "--output-report-json", str(rep)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = run.main()
    assert rc == 0                                  # did not crash
    r = json.loads(rep.read_text())
    assert r["failed_examples"] == 1 and r["paper_ready"] is False
    rows = [json.loads(x) for x in resp.read_text().strip().splitlines()]
    failed = [x for x in rows if x.get("status") == "failed"]
    assert failed and failed[0]["id"] == "b"
    assert "prompt" not in failed[0]                # no raw prompt in failure record


# ---------------------------------------------------------------------------
# 4. MT-Bench preservation (turn1/turn2 separate; missing turn -> fail)
# ---------------------------------------------------------------------------

def _preserv(args):
    env = _ENV
    cmd = [sys.executable, str(REPO_ROOT / "scripts"
                               / "evaluate_generation_preservation.py")] + args
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_mt_bench_preservation_two_turn(tmp_path) -> None:
    ds = tmp_path / "mt.jsonl"
    _write_jsonl(ds, [
        {"question_id": "q1", "category": "math", "turns": ["t1a", "t2a"]},
        {"question_id": "q2", "category": "writing", "turns": ["t1b", "t2b"]}])
    _write_jsonl(tmp_path / "p1.jsonl", [
        {"id": "q1", "response": "A"}, {"id": "q2", "response": "B"}])
    _write_jsonl(tmp_path / "o1.jsonl", [
        {"id": "q1", "response": "A"}, {"id": "q2", "response": "B"}])
    _write_jsonl(tmp_path / "p2.jsonl", [
        {"id": "q1", "response": "C"}, {"id": "q2", "response": "D"}])
    _write_jsonl(tmp_path / "o2.jsonl", [
        {"id": "q1", "response": "C"}, {"id": "q2", "response": "D"}])
    out = tmp_path / "pres.json"
    r = _preserv(["--dataset", "mt_bench", "--dataset-jsonl", str(ds),
                  "--plaintext-turn1", str(tmp_path / "p1.jsonl"),
                  "--plaintext-turn2", str(tmp_path / "p2.jsonl"),
                  "--ours-turn1", str(tmp_path / "o1.jsonl"),
                  "--ours-turn2", str(tmp_path / "o2.jsonl"),
                  "--judge-jsonl", str(tmp_path / "judge.jsonl"),
                  "--output-json", str(out)])
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    assert rep["turn1"]["aggregate"]["exact_response_match_rate"] == 1.0
    assert rep["turn2"]["aggregate"]["exact_response_match_rate"] == 1.0
    assert rep["turn2_complete"] is True
    # FastChat-compatible judge file, no external API
    jrows = [json.loads(x)
             for x in (tmp_path / "judge.jsonl").read_text().strip().splitlines()]
    assert jrows[0]["judge"] == "offline_no_external_api"
    assert "plaintext_gpu" in jrows[0]["answers"]


def test_mt_bench_missing_turn2_fails(tmp_path) -> None:
    ds = tmp_path / "mt.jsonl"
    _write_jsonl(ds, [{"question_id": "q1", "turns": ["t1", "t2"]}])
    _write_jsonl(tmp_path / "p1.jsonl", [{"id": "q1", "response": "A"}])
    _write_jsonl(tmp_path / "o1.jsonl", [{"id": "q1", "response": "A"}])
    out = tmp_path / "pres.json"
    r = _preserv(["--dataset", "mt_bench", "--dataset-jsonl", str(ds),
                  "--plaintext-turn1", str(tmp_path / "p1.jsonl"),
                  "--ours-turn1", str(tmp_path / "o1.jsonl"),
                  "--output-json", str(out)])
    assert r.returncode == 1                        # missing turn-2 -> failure
    assert json.loads(out.read_text())["turn2_complete"] is False


def test_gsm8k_preservation_reports_plaintext_ours_delta(tmp_path) -> None:
    ds = tmp_path / "g.jsonl"
    _write_jsonl(ds, [{"id": "1", "question": "1+1?", "answer": "#### 2"},
                      {"id": "2", "question": "2+2?", "answer": "#### 4"}])
    _write_jsonl(tmp_path / "plain.jsonl", [
        {"id": "1", "response": "#### 2"}, {"id": "2", "response": "#### 4"}])
    _write_jsonl(tmp_path / "ours.jsonl", [
        {"id": "1", "response": "#### 2"}, {"id": "2", "response": "#### 5"}])
    out = tmp_path / "pres.json"
    r = _preserv(["--dataset", "gsm8k", "--dataset-jsonl", str(ds),
                  "--plaintext-responses", str(tmp_path / "plain.jsonl"),
                  "--ours-responses", str(tmp_path / "ours.jsonl"),
                  "--output-json", str(out)])
    assert r.returncode == 0, r.stderr
    g = json.loads(out.read_text())["gsm8k"]
    assert g["plaintext_exact_match"] == 1.0
    assert g["ours_exact_match"] == 0.5
    assert g["exact_match_delta"] == 0.5


# ---------------------------------------------------------------------------
# 5. HumanEval pass@1 evaluator
# ---------------------------------------------------------------------------

def test_humaneval_pass1_evaluator(tmp_path) -> None:
    ds = tmp_path / "he.jsonl"
    _write_jsonl(ds, [{"task_id": "t1", "prompt": "def add(a,b):\n",
                       "entry_point": "add",
                       "test": "def check(f):\n    assert f(1,2)==3"}])
    # a correct completion, and a wrong one
    _write_jsonl(tmp_path / "plain.jsonl",
                 [{"id": "t1", "response": "def add(a,b):\n    return a+b\n"}])
    _write_jsonl(tmp_path / "ours.jsonl",
                 [{"id": "t1", "response": "def add(a,b):\n    return a-b\n"}])
    out = tmp_path / "p1.json"
    cmd = [sys.executable,
           str(REPO_ROOT / "scripts" / "evaluate_humaneval_pass1.py"),
           "--dataset-jsonl", str(ds),
           "--plaintext-responses", str(tmp_path / "plain.jsonl"),
           "--ours-responses", str(tmp_path / "ours.jsonl"),
           "--timeout", "10", "--output-json", str(out)]
    assert subprocess.run(cmd, capture_output=True, env=_ENV).returncode == 0
    rep = json.loads(out.read_text())
    assert rep["pass@1_plaintext"] == 1.0
    assert rep["pass@1_ours"] == 0.0
    assert rep["pass@1_delta"] == 1.0


# ---------------------------------------------------------------------------
# 6. SensitivePrompt leakage scan (script-level)
# ---------------------------------------------------------------------------

def test_sensitive_security_script_flags_leak_and_passes_clean(tmp_path) -> None:
    ds = tmp_path / "s.jsonl"
    _write_jsonl(ds, [{"id": "s1", "prompt": "acct SECRET-9 summarize",
                       "sensitive_spans": ["SECRET-9"]}])
    # leaking transcript on a GPU-visible direction
    _write_jsonl(tmp_path / "tx_leak.jsonl", [
        {"direction": "boundary_to_worker", "payload": {"x": "SECRET-9"}}])
    out = tmp_path / "sec.json"
    cmd = [sys.executable,
           str(REPO_ROOT / "scripts" / "evaluate_sensitive_prompt_security.py"),
           "--dataset-jsonl", str(ds),
           "--transcript-jsonl", str(tmp_path / "tx_leak.jsonl"),
           "--output-json", str(out)]
    r = subprocess.run(cmd, capture_output=True, env=_ENV)
    assert r.returncode == 1                        # leak -> nonzero
    assert json.loads(out.read_text())["leakage_pass"] is False
    # clean transcript (masked only) + prompt_sha256 allowed
    _write_jsonl(tmp_path / "tx_ok.jsonl", [
        {"direction": "boundary_to_worker", "x_tilde": "m",
         "prompt_sha256": "deadbeef"}])
    out2 = tmp_path / "sec2.json"
    cmd2 = [sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_sensitive_prompt_security.py"),
            "--dataset-jsonl", str(ds),
            "--transcript-jsonl", str(tmp_path / "tx_ok.jsonl"),
            "--output-json", str(out2)]
    assert subprocess.run(cmd2, capture_output=True, env=_ENV).returncode == 0
    assert json.loads(out2.read_text())["leakage_pass"] is True


# ---------------------------------------------------------------------------
# 7. Alibaba TDX quote wrapper
# ---------------------------------------------------------------------------

def _alibaba():
    return _load_script("alibaba_tdx",
                        "scripts/generate_alibaba_tdx_quote_evidence.py")


def test_alibaba_quote_command_receives_report_data_hex(tmp_path) -> None:
    mod = _alibaba()
    rd = "ab" * 64
    marker = tmp_path / "seen.txt"
    # the quote command must receive {report_data_hex}; echo it to a file + write
    # a quote so the wrapper can proceed
    cmd = ("printf '%s' '{report_data_hex}' > " + str(marker)
           + " && printf 'Q' > {quote_out}")
    q = mod.generate_quote_alibaba(rd, tmp_path, quote_command=cmd)
    assert Path(q).exists()
    assert marker.read_text() == rd


def test_alibaba_stale_report_data_fails_binding(tmp_path) -> None:
    mod = _alibaba()
    # appraisal reports a DIFFERENT (stale) reportdata than the current runtime hash
    appraisal = {"overall_appraisal_result": "PASS",
                 "tdx_reportdata": "00" * 64, "mr_td": "ff" * 6, "debug": False}
    ev = mod.build_evidence(
        runtime_hash_hex="ab" * 64, report_data_hex="ab" * 64,
        nonlinear_backend="A_rightmul", nonlinear_design_metadata_hash="h",
        appraisal=appraisal, expected_mr_td="ff" * 6)
    verdict = mod.verify_bindings(ev, runtime_hash_hex="ab" * 64,
                                  report_data_hex="ab" * 64,
                                  expected_mr_td="ff" * 6)
    assert verdict["tdx_reportdata_binds_runtime_hash"] is False
    assert verdict["all_bindings_ok"] is False


def test_alibaba_simulated_evidence_not_paper_facing(tmp_path) -> None:
    mod = _alibaba()
    out = tmp_path / "ev.json"
    argv = ["x", "--simulate", "--skip-preflight", "--expected-mr-td", "ff" * 6,
            "--output-dir", str(tmp_path / "art"),
            "--output-evidence", str(out)]
    import sys as _sys
    old = _sys.argv
    try:
        _sys.argv = argv
        rc = mod.main()
    finally:
        _sys.argv = old
    assert rc == 0
    ev = json.loads(out.read_text())
    assert ev["simulated_unsigned"] is True
    assert ev["paper_facing"] is False
    # the report_data still binds (plumbing works), but it is unsigned
    assert ev["tdx_reportdata_binds_runtime_hash"] is True


def test_alibaba_reallike_mock_binds_and_paper_facing(tmp_path) -> None:
    mod = _alibaba()
    # compute the runtime hash the wrapper would compute, then feed a verifier
    # that returns exactly that reportdata + debug=false + PASS
    from pllo.protocol.attestation import (
        boundary_manifest_metadata, build_trusted_boundary_manifest,
        compute_runtime_hash_from_manifest, runtime_report_data_hex)
    from pllo.experiments.nonlinear_designs import (
        nonlinear_design_metadata_hash, normalize_nonlinear_backend)
    nb = normalize_nonlinear_backend("A_rightmul")
    md = boundary_manifest_metadata(
        "process", "qwen7b_folded_package", "ff" * 6, protocol_version="8.5",
        nonlinear_backend=nb,
        nonlinear_design_metadata_hash=nonlinear_design_metadata_hash(nb))
    rh = compute_runtime_hash_from_manifest(build_trusted_boundary_manifest(
        metadata=md))
    rd = runtime_report_data_hex(bytes.fromhex(rh))
    verifier_json = json.dumps({"overall_appraisal_result": "PASS",
                                "tdx_reportdata": rd, "mr_td": "ff" * 6,
                                "debug": False})
    out = tmp_path / "ev.json"
    quote_cmd = "printf 'Q' > {quote_out}"
    verify_cmd = "printf '%s' '" + verifier_json + "'"
    argv = ["x", "--skip-preflight", "--nonlinear-backend", "A_rightmul",
            "--expected-mr-td", "ff" * 6, "--quote-command", quote_cmd,
            "--verify-command", verify_cmd, "--output-dir", str(tmp_path / "art"),
            "--output-evidence", str(out)]
    import sys as _sys
    old = _sys.argv
    try:
        _sys.argv = argv
        rc = mod.main()
    finally:
        _sys.argv = old
    assert rc == 0, out.read_text() if out.exists() else "no evidence"
    ev = json.loads(out.read_text())
    assert ev["runtime_hash_binds_nonlinear_backend"] is True
    assert ev["nonlinear_backend"] == "A_rightmul"
    assert ev["tdx_reportdata_binds_runtime_hash"] is True
    assert ev["paper_facing"] is True
    assert ev["quote_source"] == "alibaba_tdx_quote_generation_sample"


def test_alibaba_preflight_reports_missing_without_crash(tmp_path) -> None:
    mod = _alibaba()
    pf = mod.run_preflight(qgen_app=str(tmp_path / "nope_app"),
                           qverify_dir=str(tmp_path / "nope_dir"))
    assert pf["qgen_app_present"] is False
    assert pf["all_ok"] is False
    assert any(m["item"] == "qgen_app" for m in pf["missing"])
