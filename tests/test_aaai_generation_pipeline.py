"""AAAI generation pipeline: datasets, resume, paper-facing gate, resilient
remote, validator, cleanup.

Run: python -m pytest tests/test_aaai_generation_pipeline.py -q
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
# 1. dataset preparation
# ---------------------------------------------------------------------------

def test_ifeval_normalization_preserves_meta(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_ifeval
    p = tmp_path / "ife.jsonl"
    p.write_text('{"key":"k1","prompt":"Write a haiku.",'
                 '"instruction_id_list":["punctuation"],"kwargs":[{"n":1}]}\n',
                 encoding="utf-8")
    rows = load_ifeval(p)
    assert rows[0]["id"] == "k1"
    assert rows[0]["dataset"] == "ifeval"
    assert rows[0]["meta"]["instruction_id_list"] == ["punctuation"]
    assert rows[0]["meta"]["kwargs"] == [{"n": 1}]


def test_gsm8k_answer_extraction_and_loader(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import (
        extract_gsm8k_answer, gsm8k_exact_match, load_gsm8k)
    assert extract_gsm8k_answer("blah #### 1,024") == "1024"
    assert extract_gsm8k_answer("the result is 3.5 and then 42") == "42"
    assert gsm8k_exact_match("... #### 7", "7") is True
    p = tmp_path / "g.jsonl"
    p.write_text('{"question":"q?","answer":"work\\n#### 12"}\n', encoding="utf-8")
    zero = load_gsm8k(p, prompt_style="zero_shot")
    assert zero[0]["final_answer"] == "12" and "####" in zero[0]["prompt"]
    few = load_gsm8k(p, prompt_style="few_shot_cot")
    assert "Question:" in few[0]["prompt"] and few[0]["final_answer"] == "12"


def test_mt_bench_two_turn_conversion(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_mt_bench
    p = tmp_path / "m.jsonl"
    p.write_text('{"question_id":"81","category":"writing",'
                 '"turns":["t1","t2"]}\n', encoding="utf-8")
    rows = load_mt_bench(p)
    assert rows[0]["turns"] == ["t1", "t2"]
    assert rows[0]["category"] == "writing"
    assert rows[0]["meta"]["num_turns"] == 2


def test_optional_code_dataset_interfaces(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import (OPTIONAL_DATASETS,
                                                     load_dataset)
    assert OPTIONAL_DATASETS == ("humaneval", "mbpp")
    he = tmp_path / "he.jsonl"
    he.write_text('{"task_id":"HumanEval/0","prompt":"def f():",'
                  '"entry_point":"f"}\n', encoding="utf-8")
    rows = load_dataset("humaneval", he)
    assert rows[0]["dataset"] == "humaneval" and rows[0]["id"] == "HumanEval/0"
    mb = tmp_path / "mb.jsonl"
    mb.write_text('{"task_id":3,"text":"Write x","test_list":["assert"]}\n',
                  encoding="utf-8")
    rows2 = load_dataset("mbpp", mb)
    assert rows2[0]["dataset"] == "mbpp" and "test_list" in rows2[0]["meta"]


def test_dataset_card_has_shas(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import (build_dataset_card,
                                                     load_gsm8k)
    src = tmp_path / "g.jsonl"
    src.write_text('{"question":"q?","answer":"#### 5"}\n', encoding="utf-8")
    out = tmp_path / "gsm8k.jsonl"
    rows = load_gsm8k(src)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    card = build_dataset_card(dataset_name="gsm8k", split="test", rows=rows,
                              source_path=src, output_path=out, now=0)
    assert card["num_examples"] == 1
    assert card["input_sha256"] and card["output_sha256"]
    assert card["max_prompt_tokens_estimate"] > 0


# ---------------------------------------------------------------------------
# 2. resume / status / heartbeat
# ---------------------------------------------------------------------------

def test_completed_ids_and_plan(tmp_path) -> None:
    from pllo.benchmarks.run_state import (completed_ids_from_jsonl,
                                           failed_ids_from_jsonl, plan_examples)
    p = tmp_path / "resp.jsonl"
    p.write_text('{"id":"a","status":"ok"}\n'
                 '{"id":"b","status":"failed"}\n'
                 '{"id":"c","status":"ok"}\n', encoding="utf-8")
    done = completed_ids_from_jsonl(p)
    assert done == {"a", "c"}
    assert failed_ids_from_jsonl(p) == {"b"}
    examples = [{"id": x} for x in ("a", "b", "c", "d")]
    to_run, skipped = plan_examples(examples, done, resume=True)
    assert skipped == ["a", "c"]
    assert [e["id"] for e in to_run] == ["b", "d"]


def test_run_state_status_and_heartbeat(tmp_path) -> None:
    from pllo.benchmarks.run_state import RunState
    clock = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    st = RunState("r1", dataset="gsm8k", backend="folded_remote",
                  total_examples=2, status_json=str(tmp_path / "s.json"),
                  heartbeat_json=str(tmp_path / "h.json"),
                  clock=lambda: next(clock))
    st.begin_example("a")
    st.record_completed("a", tokens=5)
    st.record_failed("b", error_type="Timeout", error_message="boom", retries=3)
    st.finish()
    s = json.loads((tmp_path / "s.json").read_text())
    h = json.loads((tmp_path / "h.json").read_text())
    assert s["completed_examples"] == 1 and s["failed_examples"] == 1
    assert s["generated_tokens_total"] == 5 and s["last_completed_id"] == "a"
    assert s["failed"][0]["error_type"] == "Timeout"
    assert s["end_time"] is not None and h["alive"] is False


def test_append_jsonl_record_flushes(tmp_path) -> None:
    from pllo.benchmarks.run_state import append_jsonl_record
    p = tmp_path / "r.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        append_jsonl_record(fh, {"id": "x", "status": "ok"})
        # readable immediately by another handle (flush happened)
        assert "x" in Path(p).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. paper-facing AAAI gate
# ---------------------------------------------------------------------------

def _good_report(**over):
    r = {"dataset": "gsm8k", "backend": "folded_remote",
         "nonlinear_backend": "A_rightmul", "seq_len": 1024,
         "max_new_tokens": 512, "stop_on_eos": True, "decoding": "greedy",
         "dry_run": False, "mock_runtime": False, "tdx_boundary_client": True,
         "full_model_weights_loaded_in_trusted_runtime": False,
         "attestation_evidence_attached": True,
         "h800_worker_health": {"ok": True, "nonlinear_backend": "A_rightmul"},
         "h800_worker_tee_used_on_gpu": False, "nonlinear_trusted_calls": 0,
         "trusted_nonlinear_ops_count": 0, "nonlinear_single_tee_entry_exit": True,
         "compatible_masks_verified": True,
         "residual_mask_is_signed_permutation": True,
         "attention_qk_scores_preserved": True,
         "swiglu_shared_channel_permutation": True,
         "arbitrary_dense_mask_rejected": True,
         "base_linear_pad_all_modules_covered": True}
    r.update(over)
    return r


def _good_evidence(**over):
    ev = {"tee": "tdx", "runtime_hash_binds_nonlinear_backend": True,
          "nonlinear_backend": "A_rightmul", "report_data": "ab",
          "runtime_hash": "ab", "mr_td": "MRTD",
          "tdx": {"td_attributes": {"debug": False}}}
    ev.update(over)
    return ev


def test_aaai_gate_full_report_passes() -> None:
    from pllo.benchmarks.aaai_paper_facing import (aaai_generation_violations,
                                                   is_aaai_paper_facing)
    assert aaai_generation_violations(_good_report(),
                                      evidence=_good_evidence(),
                                      expected_mr_td="MRTD") == []
    assert is_aaai_paper_facing(_good_report(), evidence=_good_evidence())


@pytest.mark.parametrize("over,needle", [
    ({"max_new_tokens": 256}, "max_new_tokens"),
    ({"seq_len": 512}, "seq_len"),
    ({"stop_on_eos": False}, "stop_on_eos"),
    ({"nonlinear_backend": "current"}, "nonlinear_backend"),
    ({"nonlinear_backend": "amulet_secure_R"}, "nonlinear_backend"),
    ({"dry_run": True}, "dry_run"),
    ({"mock_runtime": True}, "mock_runtime"),
    ({"nonlinear_trusted_calls": 1}, "nonlinear_trusted_calls"),
    ({"compatible_masks_verified": False}, "compatible_masks_verified"),
    ({"base_linear_pad_all_modules_covered": False}, "base_linear_pad"),
])
def test_aaai_gate_rejects(over, needle) -> None:
    from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations
    viol = aaai_generation_violations(_good_report(**over),
                                      evidence=_good_evidence())
    assert any(needle in v for v in viol), viol


def test_aaai_gate_rejects_missing_evidence_and_simulated() -> None:
    from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations
    # missing evidence
    v1 = aaai_generation_violations(_good_report(), evidence=None)
    assert any("attestation" in v for v in v1)
    # simulated evidence
    v2 = aaai_generation_violations(
        _good_report(), evidence=_good_evidence(simulated_unsigned=True))
    assert any("simulated_unsigned" in v for v in v2)
    # binding false
    v3 = aaai_generation_violations(
        _good_report(),
        evidence=_good_evidence(runtime_hash_binds_nonlinear_backend=False))
    assert any("runtime_hash_binds_nonlinear_backend" in v for v in v3)


def test_runner_static_gate_rejects(tmp_path) -> None:
    """run_aaai_generation_benchmark.py fails fast on static violations."""
    ds = tmp_path / "g.jsonl"
    ds.write_text('{"id":"a","prompt":"hi","dataset":"gsm8k"}\n', encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    base = [sys.executable, str(REPO_ROOT / "scripts"
                                / "run_aaai_generation_benchmark.py"),
            "--dataset", "gsm8k", "--dataset-jsonl", str(ds),
            "--backend", "folded_remote", "--nonlinear-backend", "A_rightmul",
            "--seq-len", "1024", "--max-new-tokens", "512", "--require-real",
            "--tdx-boundary-client", "--attestation-evidence-json", "/x.json",
            "--paper-facing-aaai",
            "--output-response-jsonl", str(tmp_path / "r.jsonl"),
            "--output-report-json", str(tmp_path / "rep.json")]
    # bad max_new_tokens
    bad = [a if a != "512" else "256" for a in base]
    assert subprocess.run(bad, capture_output=True, env=env).returncode == 3
    # disable eos
    assert subprocess.run(base + ["--disable-eos-stop"], capture_output=True,
                          env=env).returncode == 3
    # mock runtime forbidden
    assert subprocess.run(base + ["--mock-runtime"], capture_output=True,
                          env=env).returncode == 3
    # current design
    cur = [("current" if a == "A_rightmul" else a) for a in base]
    assert subprocess.run(cur, capture_output=True, env=env).returncode == 3


# ---------------------------------------------------------------------------
# 4. resilient remote
# ---------------------------------------------------------------------------

class _FlakyClient:
    def __init__(self, fail_times, exc=ConnectionError("connection refused")):
        self.fail_times = fail_times
        self.calls = 0
        self.closed = 0
        self._exc = exc

    def health(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self._exc
        return {"ok": True, "after_calls": self.calls}

    def close(self):
        self.closed += 1


def test_resilient_retries_then_succeeds() -> None:
    from pllo.protocol.resilient_remote import ResilientRemoteGpuWorker
    # one persistent "server" (same instance across reconnects): first two calls
    # are refused, the third succeeds.
    server = _FlakyClient(fail_times=2)
    w = ResilientRemoteGpuWorker("http://x", max_retries=5,
                                 sleep_fn=lambda s: None,
                                 client_factory=lambda: server)
    out = w.health()
    assert out["ok"] is True
    assert w.retry_count == 2
    assert w.reconnect_count == 2
    assert server.closed >= 2            # dropped/redialed on each failure


def test_resilient_exhausts_and_raises() -> None:
    from pllo.protocol.resilient_remote import (ResilientRemoteGpuWorker,
                                                WorkerUnavailable)
    w = ResilientRemoteGpuWorker("http://x", max_retries=2,
                                 sleep_fn=lambda s: None,
                                 client_factory=lambda: _FlakyClient(99))
    with pytest.raises(WorkerUnavailable):
        w.health()
    assert w.last_error is not None


def test_resilient_does_not_retry_non_retriable() -> None:
    from pllo.protocol.resilient_remote import (ResilientRemoteGpuWorker,
                                                WorkerUnavailable)

    class _Bad:
        def __init__(self):
            self.calls = 0

        def health(self):
            self.calls += 1
            raise RuntimeError("GPU worker HTTP 400: bad request")

        def close(self):
            pass
    bad = _Bad()
    w = ResilientRemoteGpuWorker("http://x", max_retries=5,
                                 sleep_fn=lambda s: None,
                                 client_factory=lambda: bad)
    with pytest.raises(WorkerUnavailable):
        w.health()
    assert bad.calls == 1            # 4xx is not retried
    assert w.retry_count == 0


# ---------------------------------------------------------------------------
# 5. validator
# ---------------------------------------------------------------------------

def _val():
    return _load_script("aaai_val",
                        "scripts/validate_aaai_generation_results.py")


def _resp_rows(ids, text="answer", **extra):
    return [{"id": i, "status": "ok", "turn_index": 0, "response": text,
             "finish_reason": "eos", **extra} for i in ids]


def test_validator_passes_minimal_valid() -> None:
    val = _val()
    plaintext_rep = {"seq_len": 1024, "max_new_tokens": 512, "stop_on_eos": True,
                     "decoding": "greedy", "dry_run": False, "mock_runtime": False,
                     "completed_examples": 2, "failed_examples": 0,
                     "backend": "plaintext_local"}
    ours_rep = _good_report(completed_examples=2, failed_examples=0)
    rep = val.validate(plaintext_rep, ours_rep,
                       _resp_rows(["a", "b"]), _resp_rows(["a", "b"]),
                       dataset="gsm8k", card={"num_examples": 2},
                       evidence=_good_evidence(), expected_mr_td="MRTD",
                       allow_failed=False)
    assert rep["passed"] is True
    assert rep["utility_preservation"]["exact_response_match_rate"] == 1.0


def test_validator_fails_on_id_mismatch() -> None:
    val = _val()
    rep = val.validate(
        {"seq_len": 1024, "max_new_tokens": 512, "stop_on_eos": True,
         "completed_examples": 2, "backend": "plaintext_local"},
        _good_report(completed_examples=2),
        _resp_rows(["a", "b"]), _resp_rows(["a", "c"]),
        dataset="gsm8k", card={"num_examples": 2}, evidence=_good_evidence(),
        expected_mr_td="MRTD", allow_failed=True)
    assert rep["passed"] is False
    assert any("ids_align" in c["check"] for c in rep["failed_checks"])


def test_validator_fails_missing_compatible_masks_and_trusted_calls() -> None:
    val = _val()
    base_plain = {"seq_len": 1024, "max_new_tokens": 512, "stop_on_eos": True,
                  "completed_examples": 1, "backend": "plaintext_local"}
    # missing compatible_masks_verified
    r1 = val.validate(base_plain, _good_report(completed_examples=1,
                                              compatible_masks_verified=False),
                      _resp_rows(["a"]), _resp_rows(["a"]), dataset="gsm8k",
                      card={"num_examples": 1}, evidence=_good_evidence(),
                      expected_mr_td="MRTD", allow_failed=True)
    assert r1["passed"] is False
    # trusted calls > 0
    r2 = val.validate(base_plain, _good_report(completed_examples=1,
                                              nonlinear_trusted_calls=3),
                      _resp_rows(["a"]), _resp_rows(["a"]), dataset="gsm8k",
                      card={"num_examples": 1}, evidence=_good_evidence(),
                      expected_mr_td="MRTD", allow_failed=True)
    assert r2["passed"] is False


def test_validator_fails_missing_worker_health() -> None:
    val = _val()
    rep = val.validate(
        {"seq_len": 1024, "max_new_tokens": 512, "stop_on_eos": True,
         "completed_examples": 1, "backend": "plaintext_local"},
        _good_report(completed_examples=1, h800_worker_health=None),
        _resp_rows(["a"]), _resp_rows(["a"]), dataset="gsm8k",
        card={"num_examples": 1}, evidence=_good_evidence(),
        expected_mr_td="MRTD", allow_failed=True)
    assert rep["passed"] is False


# ---------------------------------------------------------------------------
# 6. cleanup
# ---------------------------------------------------------------------------

def _cleanup():
    return _load_script("cleanup", "scripts/cleanup_stale_experiments.py")


def test_cleanup_dry_run_moves_nothing(tmp_path) -> None:
    cu = _cleanup()
    d = tmp_path / "outputs" / "aaai" / "run"
    d.mkdir(parents=True)
    rpt = d / "report.json"
    rpt.write_text('{"dry_run": true, "paper_ready": false}', encoding="utf-8")
    rep = cu.run_cleanup([str(tmp_path / "outputs")],
                         archive_root=tmp_path / "arch", execute=False,
                         allow_package_cleanup=False)
    assert rep["num_stale"] == 1 and rep["num_moved"] == 0
    assert rpt.exists()                       # nothing moved on dry-run


def test_cleanup_execute_archives_stale(tmp_path) -> None:
    cu = _cleanup()
    d = tmp_path / "outputs" / "aaai" / "run"
    d.mkdir(parents=True)
    rpt = d / "report.json"
    rpt.write_text('{"dry_run": true, "paper_ready": false}', encoding="utf-8")
    (d / "responses.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    rep = cu.run_cleanup([str(tmp_path / "outputs")],
                         archive_root=tmp_path / "arch", execute=True,
                         allow_package_cleanup=False)
    assert rep["num_moved"] >= 1
    assert not rpt.exists()                   # moved to archive
    assert list((tmp_path / "arch").rglob("report.json"))


def test_cleanup_keeps_model_package_raw_dirs(tmp_path) -> None:
    cu = _cleanup()
    # a folded package dir (manifest with security_claim) is protected
    pkg = tmp_path / "outputs" / "packages" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(
        '{"security_claim":"gpu_receives_folded_weights_without_mask_secrets",'
        '"dry_run":true}', encoding="utf-8")
    # a raw dataset dir is protected by path
    raw = tmp_path / "outputs" / "raw" / "x"
    raw.mkdir(parents=True)
    (raw / "data.json").write_text('{"dry_run":true,"paper_ready":false}',
                                   encoding="utf-8")
    rep = cu.run_cleanup([str(tmp_path / "outputs")],
                         archive_root=tmp_path / "arch", execute=True,
                         allow_package_cleanup=False)
    assert (pkg / "manifest.json").exists()
    assert (raw / "data.json").exists()
    assert rep["num_skipped_protected"] >= 1
