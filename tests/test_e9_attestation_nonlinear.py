"""Regression tests: E9 attested-remote attestation binds the right backend +
the selected nonlinear design (not the generic ``qwen7b`` boundary).

No model / worker / torch needed -- the runtime-hash recipe and the gate logic
are stdlib. Run: python -m pytest tests/test_e9_attestation_nonlinear.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.real_predictors import (  # noqa: E402
    RealBackendUnavailable,
    expected_remote_runtime_hash,
)
from pllo.protocol.attestation import (  # noqa: E402
    boundary_manifest_metadata,
    build_trusted_boundary_manifest,
    compute_runtime_hash_from_manifest,
    verify_evidence,
)


def _hash(gpu_backend, nonlinear_backend=None, mrtd="mrtd"):
    md = boundary_manifest_metadata("process", gpu_backend, mrtd,
                                    nonlinear_backend=nonlinear_backend)
    return compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=md))


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


# 1. E9 attested runtime hash uses qwen7b_folded_package, NOT qwen7b.
def test_e9_runtime_hash_uses_folded_package_backend() -> None:
    got = expected_remote_runtime_hash("current", "mrtd")
    assert got == _hash("qwen7b_folded_package", "current")
    assert got != _hash("qwen7b", "current")           # must NOT be the qwen7b hash


# 2. runtime hash changes across current vs trusted_shortcut.
def test_e9_runtime_hash_differs_by_design() -> None:
    a = expected_remote_runtime_hash("current", "mrtd")
    b = expected_remote_runtime_hash("trusted_shortcut", "mrtd")
    assert a != b
    assert b == expected_remote_runtime_hash("amulet_migrated", "mrtd")   # alias


# 3. evidence generated for current FAILS for trusted_shortcut.
def test_e9_current_evidence_fails_for_trusted_shortcut() -> None:
    rh_cur = expected_remote_runtime_hash("current", "MRTD")
    rh_ts = expected_remote_runtime_hash("trusted_shortcut", "MRTD")
    evidence = {"tee": "tdx", "tdx": {"td_attributes": {"debug": False}},
                "jwt": "a.b.c", "report_data": rh_cur, "mr_td": "MRTD"}
    res_cur = verify_evidence(evidence, bytes.fromhex(rh_cur),
                              expected_mr_td="MRTD")
    assert res_cur.runtime_hash_bound is True and res_cur.verified is True
    res_ts = verify_evidence(evidence, bytes.fromhex(rh_ts),
                             expected_mr_td="MRTD")
    assert res_ts.runtime_hash_bound is not True
    assert res_ts.verified is False


# 4. run_e9 --nonlinear-backend trusted_shortcut --require-real passes that
#    backend into the predictor constructor.
def test_e9_script_passes_nonlinear_backend_to_predictor(tmp_path,
                                                         monkeypatch) -> None:
    ds = tmp_path / "mmlu.jsonl"
    ds.write_text(json.dumps({
        "id": "1", "dataset": "mmlu", "task_type": "multiple_choice",
        "metric": "accuracy", "question": "q?",
        "choices": ["A. a", "B. b", "C. c", "D. d"], "answer": "C"}) + "\n",
        encoding="utf-8")

    runners = importlib.import_module("pllo.benchmarks.runners")
    captured = {}

    def _fake_build_predictor(backend, **kw):
        captured["backend"] = backend
        captured["nonlinear_backend"] = kw.get("nonlinear_backend")
        raise RealBackendUnavailable("stubbed (capture only)")

    monkeypatch.setattr(runners, "build_predictor", _fake_build_predictor)

    mod = _load("e9", "scripts/run_e9_task_utility_benchmark.py")
    rc = _main(mod, [
        "x", "--dataset-jsonl", str(ds), "--task-type", "multiple_choice",
        "--backend", "tdx_attested_remote", "--require-real",
        "--nonlinear-backend", "trusted_shortcut",
        "--model-path", "/nonexistent", "--gpu-worker-url", "http://127.0.0.1:9",
        "--embedding-path", str(tmp_path),
        "--attestation-evidence", str(tmp_path / "ev.json"),
        "--output-json", str(tmp_path / "out.json")])
    # require-real + unavailable backend -> exit 3, but we captured the kwarg
    assert rc == 3
    assert captured.get("backend") == "tdx_attested_remote"
    assert captured.get("nonlinear_backend") == "trusted_shortcut"


# 5. final_submission_gate fails + warns on a formal-security claim for
#    trusted_shortcut, but allows it for current (established).
def test_gate_formal_security_trusted_shortcut_fails() -> None:
    gate = _load("gate", "scripts/final_submission_gate.py")
    rep = gate.build_gate_report(
        [], required_claims=["formal_security[trusted_shortcut]"])
    assert rep["gate_passed"] is False
    names = {c["name"]: c["ok"] for c in rep["checks"]}
    assert names.get("formal_security_claim[trusted_shortcut]") is False
    assert any("trusted_shortcut_cannot_support_formal_security_claim" in w
               for w in rep["warnings"])


def test_gate_formal_security_current_check_passes() -> None:
    gate = _load("gate2", "scripts/final_submission_gate.py")
    rep = gate.build_gate_report(
        [], required_claims=["formal_security[current]"])
    names = {c["name"]: c["ok"] for c in rep["checks"]}
    # the formal-security check itself passes for the established design
    assert names.get("formal_security_claim[current]") is True


def test_gate_warns_design_B_not_formally_claimed() -> None:
    gate = _load("gate3", "scripts/final_submission_gate.py")
    rep = gate.build_gate_report(
        [], nonlinear_backends=["current", "trusted_shortcut"])
    assert any("design_B_security_not_formally_claimed" in w
               for w in rep["warnings"])
