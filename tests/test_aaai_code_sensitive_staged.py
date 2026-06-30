"""AAAI code-gen / sensitive / long-prompt / GPU-staged-schedule pipeline.

Run: python -m pytest tests/test_aaai_code_sensitive_staged.py -q
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


def _load_script(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 1. datasets
# ---------------------------------------------------------------------------

def test_humaneval_conversion(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_humaneval
    p = tmp_path / "he.jsonl"
    p.write_text('{"task_id":"HumanEval/0","prompt":"def f():\\n",'
                 '"entry_point":"f","test":"def check(c): assert True",'
                 '"canonical_solution":"    return 1"}\n', encoding="utf-8")
    rows = load_humaneval(p)
    assert rows[0]["id"] == "HumanEval/0" and rows[0]["entry_point"] == "f"
    assert rows[0]["test"] and rows[0]["canonical_solution"]


def test_mbpp_conversion(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_mbpp
    p = tmp_path / "mb.jsonl"
    p.write_text('{"task_id":7,"text":"Write add","test_list":["assert add(1,2)==3"],'
                 '"code":"def add(a,b): return a+b"}\n', encoding="utf-8")
    rows = load_mbpp(p)
    assert rows[0]["id"] == "7" and rows[0]["meta"]["test_list"]


def test_longbench_lite_marks_not_official(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_longbench_1024_lite
    p = tmp_path / "lb.jsonl"
    p.write_text('{"id":"x","prompt":"long ...","answer":"a","task":"qa"}\n',
                 encoding="utf-8")
    rows = load_longbench_1024_lite(p)
    assert rows[0]["dataset"] == "longbench_1024_lite"
    assert rows[0]["meta"]["not_official_longbench_score"] is True


def test_sensitive_synthetic_spans_generated() -> None:
    from pllo.benchmarks.sensitive_prompts import build_sensitive_prompt_set
    rows = build_sensitive_prompt_set(num_per_bucket=2, buckets=(128, 512),
                                      seed=1)
    assert len(rows) == 4
    for r in rows:
        assert r["dataset"] == "sensitive_prompt_1024"
        assert r["sensitive_spans"]                    # non-empty fabricated spans
        assert r["meta"]["contains_real_pii"] is False
        # each span actually appears in the prompt
        assert all(s in r["prompt"] for s in r["sensitive_spans"])
        assert len(r["prompt"].split()) <= r["length_bucket"]


# ---------------------------------------------------------------------------
# 2. code eval
# ---------------------------------------------------------------------------

def test_extractor_codeblock_and_plain() -> None:
    from pllo.benchmarks.code_eval import extract_code
    assert extract_code("pre\n```python\nx=1\n```\npost") == "x=1"
    assert extract_code("def f():\n    return 1") == "def f():\n    return 1"


def test_passk_evaluator_pass_and_fail() -> None:
    from pllo.benchmarks.code_eval import evaluate_humaneval_example
    ok = evaluate_humaneval_example(
        prompt="def add(a,b):\n", completion="    return a+b\n",
        test="def check(c):\n    assert c(1,2)==3", entry_point="add", timeout=8)
    assert ok["passed"] is True
    bad = evaluate_humaneval_example(
        prompt="def add(a,b):\n", completion="    return a-b\n",
        test="def check(c):\n    assert c(1,2)==3", entry_point="add", timeout=8)
    assert bad["passed"] is False


def test_passk_evaluator_timeout() -> None:
    from pllo.benchmarks.code_eval import evaluate_humaneval_example
    r = evaluate_humaneval_example(
        prompt="def f():\n", completion="    while True:\n        pass\n",
        test="def check(c):\n    c()", entry_point="f", timeout=1)
    assert r["passed"] is False and r["error_type"] == "timeout"


def test_failed_code_does_not_crash_eval() -> None:
    from pllo.benchmarks.code_eval import evaluate_mbpp_example, pass_at_1
    r = evaluate_mbpp_example(completion="def add(:\n  syntax error",
                             test_list=["assert add(1,2)==3"], timeout=5)
    assert r["passed"] is False
    agg = pass_at_1([{"id": "a", "passed": True}, {"id": "b", "passed": False}])
    assert agg["pass@1"] == 0.5 and agg["failed_cases"] == ["b"]


# ---------------------------------------------------------------------------
# 3. staged schedule
# ---------------------------------------------------------------------------

def _manifest():
    from pllo.runtime.gpu_staged_schedule import build_staged_schedule
    return build_staged_schedule(schedule_id="s", nonlinear_backend="A_rightmul",
                                 seq_len=1024, max_new_tokens=512, num_layers=1)


@pytest.mark.parametrize("inject", [
    {"n_inv": [[1.0]]}, {"n": [[1.0]]}, {"raw_pad": [0.1]},
    {"token_ids": [1, 2, 3]}, {"recovery_matrix": [[1.0]]},
    {"plaintext_embedding": [0.0]},
])
def test_audit_rejects_raw_secrets(inject) -> None:
    from pllo.runtime.gpu_staged_schedule import (
        StagedScheduleSecretLeak, audit_gpu_staged_schedule_no_secrets)
    m = _manifest()
    m["slots"][0].update(inject)
    with pytest.raises(StagedScheduleSecretLeak):
        audit_gpu_staged_schedule_no_secrets(m)


def test_audit_rejects_flag_and_accepts_clean() -> None:
    from pllo.runtime.gpu_staged_schedule import (
        StagedScheduleSecretLeak, audit_gpu_staged_schedule_no_secrets)
    m = _manifest()
    m["contains_raw_pad"] = True
    with pytest.raises(StagedScheduleSecretLeak):
        audit_gpu_staged_schedule_no_secrets(m)
    audit = audit_gpu_staged_schedule_no_secrets(_manifest())
    assert audit["staged_schedule_no_secret_audit_passed"] is True
    assert audit["gpu_staged_schedule_contains_raw_masks"] is False


def test_audit_accepts_xpad_cpad_tilde() -> None:
    from pllo.runtime.gpu_staged_schedule import (
        audit_gpu_staged_schedule_no_secrets)
    m = _manifest()
    m["slots"][0]["xpad_tilde_ref"] = "layer_000:q_proj:xpad_tilde"
    m["slots"][0]["cpad_tilde_ref"] = "layer_000:q_proj:cpad_tilde"
    audit = audit_gpu_staged_schedule_no_secrets(m)
    assert audit["staged_schedule_no_secret_audit_passed"] is True


def test_staged_report_fields_no_secrets() -> None:
    from pllo.runtime.gpu_staged_schedule import (
        audit_gpu_staged_schedule_no_secrets, staged_schedule_report_fields)
    m = _manifest()
    audit = audit_gpu_staged_schedule_no_secrets(m)
    rf = staged_schedule_report_fields(m, audit, slots_consumed=m["num_slots"])
    assert rf["staged_schedule_full_coverage_verified"] is True
    assert rf["raw_input_protected"] is True
    assert rf["plaintext_logits_on_gpu"] is False
    assert rf["sampled_token_on_gpu"] is False


def test_prestage_cli_paper_facing_requires_a_rightmul(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    cmd = [sys.executable, str(REPO_ROOT / "scripts"
                               / "prestage_gpu_obfuscation_schedule.py"),
           "--folded-package-path", "/x", "--nonlinear-backend", "current",
           "--num-layers", "1", "--output-dir", str(tmp_path / "s"),
           "--paper-facing-aaai"]
    assert subprocess.run(cmd, capture_output=True, env=env).returncode == 3


# ---------------------------------------------------------------------------
# 4. runner: resume + sensitive transcript scan + paper-facing staged gate
# ---------------------------------------------------------------------------

def test_resume_humaneval(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    ds = tmp_path / "he.jsonl"
    ds.write_text('{"task_id":"HumanEval/0","prompt":"def f():\\n","entry_point":"f"}\n'
                  '{"task_id":"HumanEval/1","prompt":"def g():\\n","entry_point":"g"}\n',
                  encoding="utf-8")
    base = [sys.executable, str(REPO_ROOT / "scripts"
                                / "run_aaai_generation_benchmark.py"),
            "--dataset", "humaneval", "--dataset-jsonl", str(ds),
            "--backend", "folded_remote", "--mock-runtime", "--seq-len", "32",
            "--max-new-tokens", "8", "--run-id", "he",
            "--output-response-jsonl", str(tmp_path / "r.jsonl"),
            "--output-report-json", str(tmp_path / "rep.json")]
    assert subprocess.run(base, capture_output=True, env=env).returncode == 0
    p = subprocess.run(base + ["--resume"], capture_output=True, text=True, env=env)
    assert p.returncode == 0 and "skipped=2" in p.stdout


def test_sensitive_scan_catches_leak_and_passes_clean() -> None:
    from pllo.security.sensitive_scan import scan_sensitive_leakage
    ds = [{"sensitive_spans": ["sk-FAKE-XYZ", "Jordan Quill"]}]
    leak = scan_sensitive_leakage(
        dataset_rows=ds,
        transcript_entries=[{"direction": "boundary_to_worker",
                             "payload": {"x": "leak sk-FAKE-XYZ here"}}])
    assert leak["leakage_pass"] is False and leak["gpu_visible_span_leaks"] >= 1
    clean = scan_sensitive_leakage(
        dataset_rows=ds,
        transcript_entries=[{"direction": "boundary_to_worker",
                             "x_tilde": "masked", "xpad_tilde": "ok"}])
    assert clean["leakage_pass"] is True


def test_sensitive_scan_ignores_trusted_side_response() -> None:
    # a span appearing on a NON-gpu-visible direction (trusted output) is not a leak
    from pllo.security.sensitive_scan import scan_sensitive_leakage
    ds = [{"sensitive_spans": ["sk-FAKE-XYZ"]}]
    r = scan_sensitive_leakage(
        dataset_rows=ds,
        transcript_entries=[{"direction": "trusted_output",
                             "response": "sk-FAKE-XYZ"}])
    assert r["leakage_pass"] is True


def test_paper_facing_gate_requires_staged_when_required() -> None:
    from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations
    base = {"dataset": "humaneval", "backend": "folded_remote",
            "nonlinear_backend": "A_rightmul", "seq_len": 1024,
            "max_new_tokens": 512, "stop_on_eos": True, "decoding": "greedy",
            "dry_run": False, "mock_runtime": False, "tdx_boundary_client": True,
            "full_model_weights_loaded_in_trusted_runtime": False,
            "attestation_evidence_attached": True,
            "h800_worker_health": {"ok": True, "nonlinear_backend": "A_rightmul"},
            "h800_worker_tee_used_on_gpu": False, "nonlinear_trusted_calls": 0,
            "trusted_nonlinear_ops_count": 0,
            "nonlinear_single_tee_entry_exit": True,
            "compatible_masks_verified": True,
            "base_linear_pad_all_modules_covered": True,
            "require_staged_schedule": True}
    ev = {"tee": "tdx", "runtime_hash_binds_nonlinear_backend": True,
          "nonlinear_backend": "A_rightmul", "report_data": "a",
          "runtime_hash": "a", "tdx": {"td_attributes": {"debug": False}}}
    # missing staged audit -> fail
    v = aaai_generation_violations(base, evidence=ev)
    assert any("staged_schedule" in x for x in v)
    # with staged audit passed -> ok
    ok = dict(base, staged_schedule_used=True,
              staged_schedule_no_secret_audit_passed=True)
    assert aaai_generation_violations(ok, evidence=ev) == []


# ---------------------------------------------------------------------------
# 5. validator: staged requirement + raw-mask flag
# ---------------------------------------------------------------------------

def test_validator_requires_staged_audit_and_rejects_raw_mask() -> None:
    from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations
    rep = {"dataset": "humaneval", "backend": "folded_remote",
           "nonlinear_backend": "A_rightmul", "seq_len": 1024,
           "max_new_tokens": 512, "stop_on_eos": True, "decoding": "greedy",
           "dry_run": False, "tdx_boundary_client": True,
           "attestation_evidence_attached": True,
           "h800_worker_health": {"nonlinear_backend": "A_rightmul"},
           "h800_worker_tee_used_on_gpu": False, "nonlinear_trusted_calls": 0,
           "trusted_nonlinear_ops_count": 0,
           "nonlinear_single_tee_entry_exit": True,
           "compatible_masks_verified": True,
           "base_linear_pad_all_modules_covered": True,
           "require_staged_schedule": True, "staged_schedule_used": True,
           "staged_schedule_no_secret_audit_passed": True,
           "gpu_staged_schedule_contains_raw_masks": True}    # leak!
    ev = {"tee": "tdx", "runtime_hash_binds_nonlinear_backend": True,
          "nonlinear_backend": "A_rightmul", "report_data": "a",
          "runtime_hash": "a", "tdx": {"td_attributes": {"debug": False}}}
    v = aaai_generation_violations(rep, evidence=ev)
    assert any("contains_raw_masks" in x for x in v)


# ---------------------------------------------------------------------------
# 6. cleanup: new stale rules
# ---------------------------------------------------------------------------

def test_cleanup_flags_non_aaai_backend_and_staged_claims(tmp_path) -> None:
    cu = _load_script("cleanup", "scripts/cleanup_stale_experiments.py")
    d = tmp_path / "outputs" / "aaai" / "run"
    d.mkdir(parents=True)
    # non-AAAI backend under AAAI dir
    (d / "amulet_report.json").write_text(
        '{"backend":"folded_remote_amulet_secure_R","paper_ready":true}',
        encoding="utf-8")
    # claims staged but audit not passed
    (d / "staged_report.json").write_text(
        '{"staged_schedule_used":true,"staged_schedule_no_secret_audit_passed":false,'
        '"paper_ready":true}', encoding="utf-8")
    # raw mask flag
    (d / "leak_report.json").write_text(
        '{"gpu_staged_schedule_contains_raw_masks":true,"paper_ready":true}',
        encoding="utf-8")
    cls = cu.classify([str(tmp_path / "outputs")], allow_package_cleanup=False)
    stale_paths = " ".join(s["path"] for s in cls["stale"])
    assert "amulet_report.json" in stale_paths
    assert "staged_report.json" in stale_paths
    assert "leak_report.json" in stale_paths


def test_compare_staged_unstaged_emits_delta(tmp_path) -> None:
    cmp = _load_script("cmp", "scripts/compare_staged_vs_unstaged_latency.py")
    un = tmp_path / "un.json"
    st = tmp_path / "st.json"
    un.write_text('{"end_to_end_latency_s": 10.0, "boundary_calls": 100}',
                  encoding="utf-8")
    st.write_text('{"end_to_end_latency_s": 7.0, "boundary_calls": 40,'
                  '"staged_schedule_no_secret_audit_passed": true,'
                  '"staged_schedule_used": true}', encoding="utf-8")
    out = tmp_path / "cmp.json"
    rc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts"
                             / "compare_staged_vs_unstaged_latency.py"),
         "--unstaged-report", str(un), "--staged-report", str(st),
         "--output-json", str(out)],
        capture_output=True, env={**os.environ,
                                  "PYTHONPATH": str(REPO_ROOT / "src")})
    assert rc.returncode == 0
    r = json.loads(out.read_text())
    assert r["deltas"]["end_to_end_latency_s"]["delta"] == -3.0
    assert r["staged_no_secret_audit_passed"] is True
    assert r["raw_input_protected"] is True
