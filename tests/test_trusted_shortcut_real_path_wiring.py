"""Proof that trusted_shortcut is GENUINELY executed in the real folded path.

Builds a tiny dry-run folded package, then runs the package-backed decode probe
(the real worker path: MaskedQwenSession boundary + Qwen7BFoldedPackageGpuBackend
+ folded_worker + the FoldedNonlinearRunner) under BOTH designs and inspects the
report the worker stamped from MEASURED NonlinearOpResult counters:

* trusted_shortcut -> nonlinear_op_backend=amulet_migrated, amulet_lift_executed
  True, lifted_nonlinear_ops_count>0, lift_k>=2, lifted_gpu_bytes>0, NOT tag-only;
  AND the generated tokens still match the trusted in-process reference (the lift
  is correctness-preserving).
* current -> nonlinear_op_backend=current, amulet_lift_executed False,
  trusted_nonlinear_ops_count>0.

Needs transformers for the tiny random Qwen2 (no CUDA / no checkpoint / no H800).
Run: python -m pytest tests/test_trusted_shortcut_real_path_wiring.py -q
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


@pytest.fixture(scope="module")
def tiny_pkg(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    d = tmp_path_factory.mktemp("ts_wire_pkg")
    build = _load("build_wire", "scripts/build_qwen7b_folded_package.py")
    rc = _main(build, ["x", "--dry-run", "--num-layers", "2",
                       "--output-dir", str(d / "pkg"),
                       "--nonlinear-backend", "current",
                       "--output-json", str(d / "b.json")])
    assert rc == 0
    return str(d / "pkg")


def _run_decode(tmp_path, pkg, backend):
    probe = _load("decode_wire_%s" % backend,
                  "scripts/run_qwen7b_folded_package_decode_probe.py")
    out = tmp_path / ("decode_%s.json" % backend)
    rc = _main(probe, ["x", "--dry-run", "--folded-package-path", pkg,
                       "--max-new-tokens", "4", "--seq-len", "8",
                       "--nonlinear-backend", backend,
                       "--output-json", str(out), "--output-md", "/dev/null"])
    assert rc == 0, "decode probe failed for %s" % backend
    return json.loads(out.read_text())


def test_trusted_shortcut_real_path_not_tag_only(tiny_pkg, tmp_path) -> None:
    rep = _run_decode(tmp_path, tiny_pkg, "trusted_shortcut")
    # correctness preserved: package tokens match the trusted in-process ref
    assert rep["tokens_exact_match"] is True
    # MEASURED amulet-lift execution evidence (not a metadata tag)
    assert rep["nonlinear_op_backend"] == "amulet_migrated"
    assert rep["amulet_lift_executed"] is True
    assert rep["lifted_nonlinear_ops_count"] > 0
    assert rep["lift_k"] >= 2
    assert rep["lifted_gpu_bytes"] > 0
    assert rep["nonlinear_execution_status"] == "lifted_on_accelerator"
    assert "silu" in rep["migrated_ops_by_type"]
    # honesty predicates agree it really executed
    assert nd.report_has_amulet_execution(rep) is True
    assert nd.trusted_shortcut_tag_only(rep) is False


def test_current_real_path_trusted(tiny_pkg, tmp_path) -> None:
    rep = _run_decode(tmp_path, tiny_pkg, "current")
    assert rep["tokens_exact_match"] is True
    assert rep["nonlinear_op_backend"] == "current"
    assert rep["amulet_lift_executed"] is False
    assert rep["lifted_nonlinear_ops_count"] == 0
    assert rep["trusted_nonlinear_ops_count"] > 0
    assert nd.report_has_amulet_execution(rep) is False


def test_both_designs_same_tokens(tiny_pkg, tmp_path) -> None:
    cur = _run_decode(tmp_path, tiny_pkg, "current")
    ts = _run_decode(tmp_path, tiny_pkg, "trusted_shortcut")
    # the lift is correctness-preserving: both designs decode identically
    assert cur["package_token_ids"] == ts["package_token_ids"]
