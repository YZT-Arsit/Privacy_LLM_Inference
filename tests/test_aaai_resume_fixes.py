"""AAAI audit fixes: resume completed_total, failed-record sanitization, staged
matrix gating, Alibaba quote wrapper (quote.dat copy / debug / provenance).

Run: python -m pytest tests/test_aaai_resume_fixes.py -q
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
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows),
                          encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. resume completed_total accounting
# ---------------------------------------------------------------------------

def test_recount_single_turn_latest_status(tmp_path) -> None:
    from pllo.benchmarks.run_state import recount_status_from_jsonl
    p = tmp_path / "r.jsonl"
    _write_jsonl(p, [
        {"id": "a", "status": "ok"},
        {"id": "b", "status": "failed"},
        {"id": "b", "status": "ok"},          # later success clears the failure
        {"id": "c", "status": "ok"},
        {"id": "d", "status": "skipped"}])     # skipped is neither ok nor failed
    rc = recount_status_from_jsonl(p)
    assert rc["completed_total"] == 3 and rc["failed_total"] == 0
    assert set(rc["ok_ids"]) == {"a", "b", "c"}


def test_recount_mt_bench_requires_all_turns(tmp_path) -> None:
    from pllo.benchmarks.run_state import recount_status_from_jsonl
    p = tmp_path / "r.jsonl"
    _write_jsonl(p, [
        {"id": "q1", "turn_index": 0, "status": "ok"},
        {"id": "q1", "turn_index": 1, "status": "ok"},     # q1 complete (2 turns)
        {"id": "q2", "turn_index": 0, "status": "ok"}])    # q2 missing turn 1
    rc = recount_status_from_jsonl(p, mt_bench=True,
                                   required_turns={"q1": 2, "q2": 2})
    assert rc["completed_total"] == 1            # only q1
    assert rc["turn_completed_total"] == 3       # 3 ok turn records


def test_resume_report_completed_total(tmp_path) -> None:
    run = _load("aaai_run", "scripts/run_aaai_generation_benchmark.py")
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"},
                       {"id": "c", "prompt": "pc"}])
    resp = tmp_path / "resp.jsonl"
    _write_jsonl(resp, [{"id": "a", "status": "ok", "num_tokens": 1,
                         "turn_index": 0},
                        {"id": "b", "status": "ok", "num_tokens": 1,
                         "turn_index": 0}])
    rep = tmp_path / "rep.json"
    argv = ["x", "--dataset", "ifeval", "--dataset-jsonl", str(inp),
            "--backend", "plaintext_local", "--mock-runtime", "--seq-len", "32",
            "--max-new-tokens", "8", "--resume",
            "--output-response-jsonl", str(resp),
            "--output-report-json", str(rep)]
    old = sys.argv
    try:
        sys.argv = argv
        assert run.main() == 0
    finally:
        sys.argv = old
    d = json.loads(rep.read_text())
    assert d["completed_this_run"] == 1
    assert d["skipped_existing_examples"] == 2
    assert d["completed_total"] == 3            # 1 new + 2 prior
    assert d["failed_total"] == 0
    assert d["response_total_records"] == 3


def test_validator_completed_total_resume_aware(tmp_path) -> None:
    val = _load("aaai_val", "scripts/validate_aaai_generation_results.py")
    rows = [{"id": x, "status": "ok", "turn_index": 0} for x in ("a", "b", "c")]
    # report carries only the per-run completed_examples=1, but completed_total=3
    rep = {"completed_examples": 1, "completed_total": 3}
    assert val._completed_total(rep, rows, dataset="ifeval") == 3
    # no completed_total in report -> recount from rows
    assert val._completed_total({"completed_examples": 1}, rows,
                                dataset="ifeval") == 3
    # the completed==expected check passes with the resume-aware total
    checks = val._general_checks(rep, "ours", 3, completed_total=3)
    ce = [c for c in checks if c["check"].endswith("completed==expected")]
    assert ce and ce[0]["ok"] is True


def test_validator_mt_bench_failed_total_cleared_by_later_ok(tmp_path) -> None:
    val = _load("aaai_val2", "scripts/validate_aaai_generation_results.py")
    rows = [{"id": "q1", "turn_index": 0, "status": "failed"},
            {"id": "q1", "turn_index": 0, "status": "ok"},
            {"id": "q1", "turn_index": 1, "status": "ok"}]
    rc = val._recount_rows(rows, dataset="mt_bench")
    assert rc["completed_total"] == 1 and rc["failed_total"] == 0


# ---------------------------------------------------------------------------
# 2. failed-record / error-log sanitization
# ---------------------------------------------------------------------------

def test_aaai_sanitize_error_strips_prompt_and_span() -> None:
    run = _load("aaai_run2", "scripts/run_aaai_generation_benchmark.py")
    et, msg = run._sanitize_error(
        RuntimeError("crash near SECRET-9 and this whole prompt body line here"),
        raw_prompt="this whole prompt body line here",
        sensitive_spans=["SECRET-9"])
    assert et == "RuntimeError"
    assert "SECRET-9" not in msg and "this whole prompt body line here" not in msg


def test_aaai_failed_record_is_sanitized(tmp_path) -> None:
    run = _load("aaai_run3", "scripts/run_aaai_generation_benchmark.py")
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "s1", "prompt": "acct SECRET-12345 please summarize",
                        "sensitive_spans": ["SECRET-12345"]}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"

    # force the generation path to fail with an exception containing the secret
    def _fake_gen(predictor, prompt, args, is_dry):
        return None, 1, RuntimeError(
            "backend died with input 'acct SECRET-12345 please summarize'")
    run._gen = _fake_gen
    argv = ["x", "--dataset", "sensitive_prompt_1024", "--dataset-jsonl",
            str(inp), "--backend", "plaintext_local", "--mock-runtime",
            "--seq-len", "32", "--max-new-tokens", "8",
            "--max-retries-per-example", "0",
            "--output-response-jsonl", str(resp),
            "--output-report-json", str(rep)]
    old = sys.argv
    try:
        sys.argv = argv
        run.main()
    finally:
        sys.argv = old
    txt = resp.read_text()
    assert "SECRET-12345" not in txt
    rec = json.loads(txt.strip())
    assert rec["status"] == "failed"
    assert "error_message_sanitized" in rec
    assert "SECRET-12345" not in json.dumps(rec)
    assert "prompt" not in rec                      # no raw prompt in failure


def test_sensitive_security_passes_on_sanitized_failed_record(tmp_path) -> None:
    from pllo.security.sensitive_scan import scan_sensitive_leakage
    ds = [{"sensitive_spans": ["SECRET-12345"]}]
    failed_log = ["{\"status\": \"failed\", \"error_message_sanitized\": "
                  "\"backend died with input '<redacted>'\"}"]
    rep = scan_sensitive_leakage(dataset_rows=ds, error_logs=failed_log)
    assert rep["leakage_pass"] is True


# ---------------------------------------------------------------------------
# 3. staged backend gated out of the default matrix
# ---------------------------------------------------------------------------

def _matrix_args(**over):
    import argparse
    ns = argparse.Namespace(
        models=["gpt2"], datasets=["ifeval"], dataset_dir="x",
        qwen_path=None, llama_path=None, gpt2_path=None, gpu_worker_url=None,
        embedding_path=None, attestation_evidence_json=None, expected_mr_td=None,
        gpu_staged_schedule_dir=None, paper_facing_aaai=False,
        include_staged=False, staged_is_experimental=False,
        require_staged_schedule=False, output_dir="outputs/aaai")
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_matrix_default_excludes_staged() -> None:
    mtx = _load("aaai_mtx", "scripts/run_aaai_experiment_matrix.py")
    plan = mtx.build_plan(_matrix_args())
    backends = {g["backend"] for g in plan["generation"]}
    assert "folded_remote_staged" not in backends
    assert plan["staged_included"] is False


def test_matrix_include_staged_marks_experimental() -> None:
    mtx = _load("aaai_mtx2", "scripts/run_aaai_experiment_matrix.py")
    plan = mtx.build_plan(_matrix_args(include_staged=True))
    staged = [g for g in plan["generation"]
              if g["backend"] == "folded_remote_staged"]
    assert staged and all(g["experimental"] is True for g in staged)
    assert all(g["latency_claim_allowed"] is False for g in staged)
    assert plan["staged_latency_claim_allowed"] is False


def test_staged_report_fields_forbid_latency_claim() -> None:
    from pllo.runtime.gpu_staged_schedule import staged_schedule_report_fields
    f = staged_schedule_report_fields({"schedule_id": "s", "num_slots": 4},
                                      {"staged_schedule_no_secret_audit_passed":
                                       True}, slots_consumed=4)
    assert f["do_not_use_as_latency_claim"] is True
    assert f["online_remask_still_performed"] is True
    assert f["staged_schedule_metadata_only"] is True


# ---------------------------------------------------------------------------
# 4. Alibaba quote wrapper extras
# ---------------------------------------------------------------------------

def _alibaba():
    return _load("alibaba2", "scripts/generate_alibaba_tdx_quote_evidence.py")


def test_alibaba_copies_quote_dat_to_quote_out(tmp_path) -> None:
    mod = _alibaba()
    rd = "cd" * 64
    # the command writes quote.dat (NOT {quote_out}); the wrapper must copy it
    cmd = "printf 'QUOTEBYTES' > quote.dat"
    q = mod.generate_quote_alibaba(rd, tmp_path, quote_command=cmd)
    assert Path(q).name == "td_quote.dat" and Path(q).read_text() == "QUOTEBYTES"


def test_alibaba_debug_true_fails_binding() -> None:
    mod = _alibaba()
    appraisal = {"overall_appraisal_result": "PASS",
                 "tdx_reportdata": "ab" * 64, "mr_td": "ff" * 6, "debug": True}
    ev = mod.build_evidence(
        runtime_hash_hex="ab" * 64, report_data_hex="ab" * 64,
        nonlinear_backend="A_rightmul", nonlinear_design_metadata_hash="h",
        appraisal=appraisal, expected_mr_td="ff" * 6)
    v = mod.verify_bindings(ev, runtime_hash_hex="ab" * 64,
                            report_data_hex="ab" * 64, expected_mr_td="ff" * 6)
    assert v["debug_false"] is False and v["all_bindings_ok"] is False
    assert ev["tdx"]["td_attributes"]["debug"] is True


def test_alibaba_real_like_has_command_provenance(tmp_path) -> None:
    mod = _alibaba()
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
    rh = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=md))
    rd = runtime_report_data_hex(bytes.fromhex(rh))
    vjson = json.dumps({"overall_appraisal_result": "PASS", "tdx_reportdata": rd,
                        "mr_td": "ff" * 6, "debug": False})
    out = tmp_path / "ev.json"
    argv = ["x", "--skip-preflight", "--nonlinear-backend", "A_rightmul",
            "--expected-mr-td", "ff" * 6,
            "--quote-command", "printf 'Q' > {quote_out}",
            "--verify-command", "printf '%s' '" + vjson + "'",
            "--output-dir", str(tmp_path / "art"), "--output-evidence", str(out)]
    old = sys.argv
    try:
        sys.argv = argv
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc == 0
    ev = json.loads(out.read_text())
    assert ev["paper_facing"] is True
    assert ev["command_provenance"]["quote_command"] == "printf 'Q' > {quote_out}"
    assert ev["command_provenance"]["simulate"] is False
    assert ev["quote_source"] == "alibaba_tdx_quote_generation_sample"
