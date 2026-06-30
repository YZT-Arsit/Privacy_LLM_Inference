"""AAAI A_rightmul audit round: compatible-mask sub-gate, logits-parity diagnostic,
finish_reason propagation, degeneration gate, TEE-claim wording, gen-config flags.

Run: python -m pytest tests/test_aaai_audit_round3.py -q
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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
# P2: logits-parity diagnostic core
# ---------------------------------------------------------------------------

def test_compare_step_top1_top5_rank() -> None:
    from pllo.benchmarks.logits_parity import compare_step
    # identical -> top1 agree, zero error, rank 0
    m = compare_step([0.0, 5.0, 1.0, 2.0], [0.0, 5.0, 1.0, 2.0])
    assert m["top1_agree"] and m["max_abs_error"] == 0.0 and m["rank_error"] == 0
    assert m["top5_overlap"] == 1.0
    # ours demotes the right token (id1) to rank 2 -> top1 disagree, rank_error>0
    m2 = compare_step([0.0, 5.0, 1.0, 2.0], [9.0, 1.0, 0.5, 3.0])
    assert m2["top1_agree"] is False
    assert m2["plain_top1"] == 1 and m2["ours_top1"] == 0
    assert m2["rank_error"] >= 1 and m2["max_abs_error"] > 0


def test_compare_run_localises_prefill_vs_decode() -> None:
    from pllo.benchmarks.logits_parity import compare_run
    ok = [([0.0, 1.0], [0.0, 1.0]) for _ in range(4)]
    r = compare_run(ok)
    assert r["passed"] and r["first_divergence_step"] is None
    assert r["divergence_phase"] is None and r["top1_agreement_rate"] == 1.0
    # divergence at prefill (step 0)
    pre = [([1.0, 0.0], [0.0, 1.0])] + [([1.0, 0.0], [1.0, 0.0]) for _ in range(3)]
    rp = compare_run(pre)
    assert rp["first_divergence_step"] == 0 and rp["divergence_phase"] == "prefill"
    # divergence first at a decode step
    dec = [([1.0, 0.0], [1.0, 0.0]), ([1.0, 0.0], [1.0, 0.0]),
           ([1.0, 0.0], [0.0, 1.0])]
    rd = compare_run(dec)
    assert rd["first_divergence_step"] == 2 and rd["divergence_phase"] == "decode"


def test_diagnose_script_self_test(tmp_path) -> None:
    out = tmp_path / "parity.json"
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    import os
    env = {**os.environ, **env}
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "diagnose_logits_parity.py"),
         "--self-test", "--max-steps", "5", "--output-json", str(out)],
        capture_output=True, env=env)
    # prompt 1 diverges -> overall sanity fails -> exit 1
    assert r.returncode == 1
    d = json.loads(out.read_text())
    assert d["per_prompt"][0]["passed"] is True
    assert d["per_prompt"][1]["divergence_phase"] == "decode"
    assert d["logits_parity_sanity_passed"] is False


# ---------------------------------------------------------------------------
# P1: compatible-mask sub-conditions gate (fail hard on each)
# ---------------------------------------------------------------------------

def _good_report(**over):
    from tests.test_aaai_generation_pipeline import _good_report as g  # reuse
    return g(**over)


def _gate(report, **kw):
    from pllo.benchmarks.aaai_paper_facing import aaai_generation_violations
    return aaai_generation_violations(report, **kw)


def _good_evidence():
    return {"tee": "tdx", "runtime_hash_binds_nonlinear_backend": True,
            "nonlinear_backend": "A_rightmul", "report_data": "ab",
            "runtime_hash": "ab", "mr_td": "MRTD",
            "tdx": {"td_attributes": {"debug": False}}, "paper_facing": True}


import pytest


@pytest.mark.parametrize("cond", [
    "residual_mask_is_signed_permutation",
    "attention_qk_scores_preserved",
    "swiglu_shared_channel_permutation",
    "arbitrary_dense_mask_rejected"])
def test_gate_fails_on_each_compatible_mask_condition(cond) -> None:
    rep = _good_report(**{cond: False})
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    viol = _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD")
    assert any(cond in x for x in viol), viol


def test_gate_passes_with_all_mask_conditions() -> None:
    rep = _good_report()
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    assert _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD") == []


# ---------------------------------------------------------------------------
# P7: degeneration / finish_reason / parity gate
# ---------------------------------------------------------------------------

def test_gate_fails_on_finish_reason_null() -> None:
    rep = _good_report(any_finish_reason_null=True,
                       finish_reason_null_ids=["1005"])
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    viol = _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD")
    assert any("any_finish_reason_null" in x for x in viol)


def test_gate_fails_on_degenerate_response() -> None:
    rep = _good_report(degenerate_response_count=1,
                       degenerate_responses=[{"id": "1005"}],
                       repeat_ratio_threshold=0.5)
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    viol = _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD")
    assert any("degenerate_response_count" in x for x in viol)


def test_gate_fails_on_parity_sanity_false() -> None:
    rep = _good_report(logits_parity_sanity_passed=False)
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    viol = _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD")
    assert any("logits_parity_sanity_passed" in x for x in viol)


def test_gate_ok_when_parity_passed_true() -> None:
    rep = _good_report(logits_parity_sanity_passed=True)
    rep["h800_worker_health"] = {"ok": True, "nonlinear_backend": "A_rightmul"}
    assert _gate(rep, evidence=_good_evidence(), expected_mr_td="MRTD") == []


# ---------------------------------------------------------------------------
# runner: degeneration scan + report fields (P5/P6/P7 wiring, mock backend)
# ---------------------------------------------------------------------------

def test_scan_degeneration_catches_repeat_and_null() -> None:
    run = _load("aaai_deg", "scripts/run_aaai_generation_benchmark.py")
    import tempfile
    p = Path(tempfile.mkdtemp()) / "r.jsonl"
    _write_jsonl(p, [
        {"id": "1005", "status": "ok", "num_tokens": 512, "finish_reason": None,
         "token_ids": [7] * 512},
        {"id": "ok", "status": "ok", "num_tokens": 5, "finish_reason": "eos",
         "token_ids": [1, 2, 3, 4, 5]}])
    sc = run.scan_degeneration(p, max_new_tokens=512)
    assert sc["any_finish_reason_null"] is True
    assert sc["finish_reason_null_ids"] == ["1005"]
    assert sc["degenerate_response_count"] == 1
    assert sc["max_repeat_ratio"] == 1.0


def test_runner_report_has_finish_reason_and_tee_claim_fields(tmp_path) -> None:
    run = _load("aaai_run_r3", "scripts/run_aaai_generation_benchmark.py")
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, [{"id": "a", "prompt": "pa"}])
    resp = tmp_path / "resp.jsonl"
    rep = tmp_path / "rep.json"
    argv = ["x", "--dataset", "ifeval", "--dataset-jsonl", str(inp),
            "--backend", "plaintext_local", "--mock-runtime", "--seq-len", "32",
            "--max-new-tokens", "8", "--align-generation-config",
            "--output-response-jsonl", str(resp),
            "--output-report-json", str(rep)]
    old = sys.argv
    try:
        sys.argv = argv
        run.main()
    finally:
        sys.argv = old
    d = json.loads(rep.read_text())
    # P6 wording
    assert d["tee_claim"] == "single trusted runtime session, zero nonlinear TEE crossings"
    assert d["trusted_runtime_session"] == "single"
    assert "mask_token_embedding" in d["per_decode_trusted_ops"]
    # P7 aggregates present
    assert d["any_finish_reason_null"] is False
    assert d["degenerate_response_count"] == 0
    assert d["truncation_policy"] == "keep_last_seq_len"
    assert d["align_generation_config"] is True
    # P5: the response record carries a finish_reason (mock stub -> "length")
    rec = json.loads(resp.read_text().strip())
    assert rec["finish_reason"] == "length"
