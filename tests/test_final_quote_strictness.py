"""Final strictness checks for Alibaba TDX evidence and failed-record redaction."""

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


def _ev(appraisal, expected="ff" * 6):
    mod = _load("alibaba_strict", "scripts/generate_alibaba_tdx_quote_evidence.py")
    return mod.build_evidence(
        runtime_hash_hex="ab" * 64,
        report_data_hex="ab" * 64,
        nonlinear_backend="A_rightmul",
        nonlinear_design_metadata_hash="h",
        appraisal=appraisal,
        expected_mr_td=expected,
    )


def _verdict(appraisal, expected="ff" * 6):
    mod = _load("alibaba_strict2", "scripts/generate_alibaba_tdx_quote_evidence.py")
    ev = _ev(appraisal, expected=expected)
    return mod.verify_bindings(
        ev, runtime_hash_hex="ab" * 64,
        report_data_hex="ab" * 64,
        expected_mr_td=expected,
    )


def test_missing_debug_fails_closed():
    v = _verdict({
        "overall_appraisal_result": "PASS",
        "tdx_reportdata": "ab" * 64,
        "mr_td": "ff" * 6,
        "verifier_returncode": 0,
    })
    assert v["debug_present"] is False
    assert v["all_bindings_ok"] is False


def test_missing_mr_td_fails_when_expected_is_provided():
    ev = _ev({
        "overall_appraisal_result": "PASS",
        "tdx_reportdata": "ab" * 64,
        "debug": False,
        "verifier_returncode": 0,
    })
    assert ev["mr_td"] is None
    assert ev["expected_mr_td"] == "ff" * 6
    v = _verdict({
        "overall_appraisal_result": "PASS",
        "tdx_reportdata": "ab" * 64,
        "debug": False,
        "verifier_returncode": 0,
    })
    assert v["mr_td_present"] is False
    assert v["all_bindings_ok"] is False


def test_verifier_returncode_nonzero_fails():
    v = _verdict({
        "overall_appraisal_result": "PASS",
        "tdx_reportdata": "ab" * 64,
        "mr_td": "ff" * 6,
        "debug": False,
        "verifier_returncode": 2,
    })
    assert v["verifier_returncode_ok"] is False
    assert v["all_bindings_ok"] is False


def test_unknown_and_simulated_pass_do_not_count_as_appraisal_ok():
    for result in ("UNKNOWN", "SIMULATED_PASS", "FAILED", "ERROR", ""):
        v = _verdict({
            "overall_appraisal_result": result,
            "tdx_reportdata": "ab" * 64,
            "mr_td": "ff" * 6,
            "debug": False,
            "verifier_returncode": 0,
        })
        assert v["appraisal_ok"] is False
        assert v["all_bindings_ok"] is False


def test_real_like_passes_and_stale_reportdata_fails():
    good = {
        "overall_appraisal_result": "PASS",
        "tdx_reportdata": "ab" * 64,
        "mr_td": "ff" * 6,
        "debug": False,
        "verifier_returncode": 0,
    }
    assert _verdict(good)["all_bindings_ok"] is True
    stale = dict(good, tdx_reportdata="cd" * 64)
    assert _verdict(stale)["tdx_reportdata_binds_runtime_hash"] is False
    assert _verdict(stale)["all_bindings_ok"] is False


def test_parse_verifier_output_supports_aliases():
    mod = _load("alibaba_parse", "scripts/generate_alibaba_tdx_quote_evidence.py")
    parsed = mod.parse_verifier_output(
        "overall_appraisal_result: PASS\n"
        "tdx_report_data: 0x" + ("ab" * 64) + "\n"
        "tdx_mr_td: " + ("ff" * 6) + "\n"
        "td_attributes.debug: false\n"
        "verifier_returncode: 0\n"
    )
    assert parsed["tdx_reportdata"] == "ab" * 64
    assert parsed["mr_td"] == "ff" * 6
    assert parsed["debug"] is False
    assert parsed["verifier_returncode"] == 0


def test_sanitize_error_redacts_technical_payload_and_sensitive_span():
    run = _load("aaai_run_sanitize", "scripts/run_aaai_generation_benchmark.py")
    et, msg = run._sanitize_error(
        RuntimeError(
            "boom token_ids=[1,2,3] raw_mask=tensor([[1,2]]) "
            "N_inv=array([[9]]) plaintext_logits=[0.1] SECRET-77"
        ),
        raw_prompt="please summarize SECRET-77",
        sensitive_spans=["SECRET-77"],
    )
    assert et == "RuntimeError"
    dumped = json.dumps({"msg": msg})
    assert "SECRET-77" not in dumped
    assert "token_ids" not in dumped
    assert "raw_mask" not in dumped
    assert "N_inv" not in dumped
    assert "plaintext_logits" not in dumped
    assert "<redacted_technical_payload>" in msg
