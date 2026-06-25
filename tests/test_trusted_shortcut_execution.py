"""Guards on the trusted_shortcut design's honesty schema.

trusted_shortcut is now WIRED into the real folded path (it executes the Amulet
lift -- see tests/test_trusted_shortcut_real_path_wiring.py for the measured
execution proof). These tests pin the surrounding honesty schema:
* the op-backend mapping must reach the Amulet backend;
* the Amulet backend's GELU must actually lift onto the untrusted accelerator;
* the current backend's GELU must run in the trusted boundary;
* an EXECUTION-bearing trusted_shortcut report that lacks lift evidence (tag-only)
  must still fail claim validation, the final gate, and E15 comparison;
* a non-execution report (a build) is design-independent and is NOT flagged.

Backend op tests use CPU torch only (no CUDA / model / H800). Run:
    python -m pytest tests/test_trusted_shortcut_execution.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments import nonlinear_designs as nd  # noqa: E402


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


# ---- mapping + backend execution -----------------------------------------

def test_op_backend_for_design_maps_trusted_shortcut() -> None:
    assert nd.op_backend_for_design("trusted_shortcut") == "amulet_migrated"
    assert nd.op_backend_for_design("current") == "current"


def test_amulet_backend_gelu_executes_lift() -> None:
    torch = pytest.importorskip("torch")
    from pllo.nonlinear.registry import make_nonlinear_backend
    b = make_nonlinear_backend(nd.op_backend_for_design("trusted_shortcut"))
    r = b.gelu(torch.randn(3, 8))
    assert r.extra.get("location") == "untrusted_accelerator"
    assert int(r.extra.get("lift_k", 0)) >= 2
    assert r.trusted_calls == 0
    assert r.gpu_bytes > 0


def test_current_backend_gelu_trusted() -> None:
    torch = pytest.importorskip("torch")
    from pllo.nonlinear.registry import make_nonlinear_backend
    b = make_nonlinear_backend(nd.op_backend_for_design("current"))
    r = b.gelu(torch.randn(3, 8))
    assert r.extra.get("location") == "trusted_boundary"
    assert r.trusted_calls == 1
    assert r.gpu_bytes == 0


# ---- real path IS wired now (executes the Amulet lift) --------------------

def test_real_path_execution_status() -> None:
    assert nd.real_path_executes("current") is True
    # trusted_shortcut is now WIRED into the real folded path (executes the lift).
    assert nd.real_path_executes("trusted_shortcut") is True
    f = nd.nonlinear_design_report_fields("trusted_shortcut")
    assert f["nonlinear_op_backend"] == "amulet_migrated"
    assert f["nonlinear_real_path_executed"] is True
    # the design-fields stamp is only a CAPABILITY annotation: it does NOT by
    # itself assert the lift ran -- an execution-bearing run must override
    # amulet_lift_executed with measured counters.
    assert f["amulet_lift_executed"] is False
    assert f["nonlinear_execution_status"] == "lifted_on_accelerator"


def test_paper_facing_trusted_shortcut_no_longer_refused() -> None:
    # after wiring, a paper-facing (non-dry-run) trusted_shortcut run is allowed
    # because it genuinely executes the lift (no NonlinearDesignNotWired).
    assert nd.assert_real_path_execution(
        "trusted_shortcut", dry_run=False) == "trusted_shortcut"


# ---- claim validation refuses tag-only trusted_shortcut -------------------

def _ts_report(executed):
    r = {"stage": "e9_pairwise_utility_preservation",
         "nonlinear_backend": "trusted_shortcut", "utility_preserved": True,
         "paper_ready": True, "dry_run": False, "dataset": "mmlu",
         "delta_abs": 0.0}
    if executed:
        r.update({"nonlinear_op_backend": "amulet_migrated",
                  "amulet_lift_executed": True, "lifted_nonlinear_ops_count": 56,
                  "lift_k": 4, "lifted_gpu_bytes": 99999})
    return r


def test_real_report_requires_backend_execution() -> None:
    from pllo.experiments.claim_validator import build_claim_report
    tag_only = build_claim_report(
        [{"file": "t.json", "report": _ts_report(executed=False)}],
        required_claims=["public_benchmark_utility_preserved[trusted_shortcut]"])
    assert tag_only["all_required_supported"] is False
    assert tag_only["trusted_shortcut_executed_in_real_path"] is False

    executed = build_claim_report(
        [{"file": "t.json", "report": _ts_report(executed=True)}],
        required_claims=["public_benchmark_utility_preserved[trusted_shortcut]"])
    assert executed["all_required_supported"] is True
    assert "public_benchmark_utility_preserved[trusted_shortcut]" in \
        executed["backend_tagged_supported"]


def test_final_gate_blocks_tag_only_trusted_shortcut() -> None:
    gate = _load("gate_ts", "scripts/final_submission_gate.py")
    rep = gate.build_gate_report(
        [{"file": "t.json", "report": _ts_report(executed=False)}])
    assert rep["gate_passed"] is False
    names = {c["name"]: c["ok"] for c in rep["checks"]}
    assert names.get("trusted_shortcut_not_executed_in_real_path") is False


# ---- E15 refuses tag-only trusted_shortcut --------------------------------

def test_e15_refuses_tag_only_trusted_shortcut() -> None:
    from pllo.experiments.nonlinear_design_comparison import build_comparison
    rbb = {
        "current": [{"stage": "tee_gpu_protocol_demo", "nonlinear_backend":
                     "current", "tokens_exact_match": True, "decode_latency": 1.0,
                     "trusted_bytes": 5000, "gpu_bytes": 1000, "boundary_calls": 10,
                     "paper_ready": True, "dry_run": False}],
        "trusted_shortcut": [{"stage": "tee_gpu_protocol_demo",
                              "nonlinear_backend": "trusted_shortcut",
                              "tokens_exact_match": True, "decode_latency": 0.5,
                              "trusted_bytes": 4000, "gpu_bytes": 2000,
                              "boundary_calls": 10, "paper_ready": True,
                              "dry_run": False}],  # tag-only: no amulet evidence
    }
    rep = build_comparison(rbb)
    assert rep["trusted_shortcut_tag_only"] is True
    assert rep["recommendation"]["recommendation_status"] == "insufficient_evidence"
    assert rep["recommendation"].get("final_recommendation") is None
    assert any("trusted_shortcut_not_executed_in_real_path" in m
               for m in rep["recommendation"]["missing_evidence"])


# ---- build is design-independent; a build report is not execution-bearing ---

def test_build_dry_run_trusted_shortcut_now_executed_capability(tmp_path) -> None:
    build = _load("buildts2", "scripts/build_qwen7b_folded_package.py")
    rc = _main(build, ["x", "--dry-run", "--num-layers", "1",
                       "--output-dir", str(tmp_path / "pkg"),
                       "--nonlinear-backend", "trusted_shortcut",
                       "--output-json", str(tmp_path / "b.json")])
    assert rc == 0
    rep = json.loads((tmp_path / "b.json").read_text())
    assert rep["nonlinear_backend"] == "trusted_shortcut"
    # the design is wired (capability), so the build report annotates executed...
    assert rep["nonlinear_real_path_executed"] is True
    assert rep["nonlinear_execution_status"] == "lifted_on_accelerator"
    # ...but a BUILD never runs a nonlinearity, so it is NOT execution-bearing and
    # must NOT be flagged tag-only (it legitimately carries no lift counters).
    assert nd.trusted_shortcut_tag_only(rep) is False
