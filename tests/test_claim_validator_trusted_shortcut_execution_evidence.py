"""Claim validator / gate honesty around trusted_shortcut execution evidence.

Now that trusted_shortcut is wired, the validator must:
* SUPPORT a [trusted_shortcut] claim backed by a report with real Amulet-lift
  execution evidence;
* still REFUSE an execution-bearing trusted_shortcut report that lacks that
  evidence (tag-only -- it ran the current path under a design tag);
* NOT flag a non-execution (build) report as tag-only;
* still REFUSE any FORMAL security claim for trusted_shortcut (security is not
  formally claimed regardless of execution).

stdlib only. Run:
    python -m pytest tests/test_claim_validator_trusted_shortcut_execution_evidence.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.claim_validator import build_claim_report  # noqa: E402
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    report_has_amulet_execution,
    trusted_shortcut_tag_only,
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ts_pairwise(executed):
    r = {"stage": "e9_pairwise_utility_preservation",
         "nonlinear_backend": "trusted_shortcut", "utility_preserved": True,
         "paper_ready": True, "dry_run": False, "dataset": "mmlu",
         "delta_abs": 0.0}
    if executed:
        r.update({"nonlinear_op_backend": "amulet_migrated",
                  "amulet_lift_executed": True, "lifted_nonlinear_ops_count": 56,
                  "lift_k": 4, "lifted_gpu_bytes": 222222})
    return r


def _ts_build():
    # a folded-package build is design-independent + not execution-bearing
    return {"stage": "folded_package_build", "nonlinear_backend":
            "trusted_shortcut", "nonlinear_op_backend": "amulet_migrated",
            "nonlinear_real_path_executed": True, "amulet_lift_executed": False,
            "num_shards": 30, "folded_package_loaded": True}


# ---- report_has_amulet_execution truth table ------------------------------

def test_report_has_amulet_execution_true_for_real_wired_report() -> None:
    assert report_has_amulet_execution(_ts_pairwise(executed=True)) is True


def test_report_has_amulet_execution_false_for_tag_only_report() -> None:
    assert report_has_amulet_execution(_ts_pairwise(executed=False)) is False


# ---- claim validator ------------------------------------------------------

def test_executed_trusted_shortcut_supports_claim() -> None:
    rep = build_claim_report(
        [{"file": "ts.json", "report": _ts_pairwise(executed=True)}],
        required_claims=["public_benchmark_utility_preserved[trusted_shortcut]"])
    assert rep["all_required_supported"] is True
    assert rep["trusted_shortcut_executed_in_real_path"] is True
    assert "public_benchmark_utility_preserved[trusted_shortcut]" in \
        rep["backend_tagged_supported"]


def test_tag_only_trusted_shortcut_refused() -> None:
    rep = build_claim_report(
        [{"file": "ts.json", "report": _ts_pairwise(executed=False)}],
        required_claims=["public_benchmark_utility_preserved[trusted_shortcut]"])
    assert rep["all_required_supported"] is False
    assert rep["trusted_shortcut_tag_only_files"] == ["ts.json"]


def test_build_report_not_flagged_tag_only() -> None:
    b = _ts_build()
    assert trusted_shortcut_tag_only(b) is False
    rep = build_claim_report([{"file": "b.json", "report": b}])
    # a build report must not contaminate the executed-in-real-path signal
    assert rep["trusted_shortcut_tag_only_files"] == []


# ---- final gate: formal security still refused for trusted_shortcut --------

def test_final_gate_still_rejects_formal_security_for_trusted_shortcut() -> None:
    gate = _load("gate_fs", "scripts/final_submission_gate.py")
    rep = gate.build_gate_report(
        [{"file": "ts.json", "report": _ts_pairwise(executed=True)}],
        required_claims=["formal_security[trusted_shortcut]"],
        nonlinear_backends=["trusted_shortcut"])
    names = {c["name"]: c["ok"] for c in rep["checks"]}
    assert names.get("formal_security_claim[trusted_shortcut]") is False
    assert any("cannot_support_formal_security_claim" in w
               for w in rep["warnings"])
