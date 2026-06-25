"""Tests for the non-invasive H800/TDX support patches.

Covers: embedding-artifact CLI flag, E9 fixture/tiny hard-fail under
--require-real, the H800 stage checker (fail on missing / pass on minimal
synthetic), the E3/dual-matrix command templates carrying --folded-package-path,
and the TDX minimum runbook mentioning per-design runtime hash + evidence.

stdlib only; no torch / GPU / model / network. Run:
    python -m pytest tests/test_h800_support_patches.py -q
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


# ---- embedding-artifact CLI ----------------------------------------------

def test_embedding_artifact_help_has_nonlinear_backend() -> None:
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts"
                             / "build_qwen7b_embedding_artifact.py"), "--help"],
        capture_output=True, text=True, timeout=120)
    assert "--nonlinear-backend" in out.stdout


# ---- E9 fixture/tiny hard-fail under --require-real ------------------------

def _e9(tmp, ds_name, extra):
    ds = tmp / ds_name
    ds.write_text(json.dumps({
        "id": "1", "dataset": "mmlu", "task_type": "multiple_choice",
        "metric": "accuracy", "question": "q?",
        "choices": ["A. a", "B. b", "C. c", "D. d"], "answer": "C"}) + "\n",
        encoding="utf-8")
    mod = _load("e9p", "scripts/run_e9_task_utility_benchmark.py")
    return _main(mod, ["x", "--dataset-jsonl", str(ds),
                       "--task-type", "multiple_choice",
                       "--output-json", str(tmp / "out.json")] + extra)


def test_e9_require_real_rejects_tiny_path(tmp_path) -> None:
    rc = _e9(tmp_path, "mmlu_tiny.jsonl", ["--require-real",
                                           "--backend", "plaintext_local"])
    assert rc == 3


def test_e9_require_real_rejects_fixture_path(tmp_path) -> None:
    sub = tmp_path / "fixture"
    sub.mkdir()
    rc = _e9(sub, "mmlu.jsonl", ["--require-real", "--backend", "plaintext_local"])
    assert rc == 3


def test_e9_stub_allows_tiny_without_require_real(tmp_path) -> None:
    # without --require-real the tiny path is allowed (stub/dry-run report)
    rc = _e9(tmp_path, "mmlu_tiny.jsonl", ["--backend", "plaintext_local"])
    assert rc == 0
    rep = json.loads((tmp_path / "out.json").read_text())
    assert rep["dry_run"] is True and rep["paper_ready"] is not True


# ---- H800 stage checker ---------------------------------------------------

def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _synth_h800(results, pkg_root):
    designs = ["current", "trusted_shortcut"]
    datasets = ["mmlu", "gsm8k", "boolq", "ag_news"]
    hashes = {"current": "a" * 64, "trusted_shortcut": "b" * 64}
    for d in designs:
        _write(results / ("folded_verify_%s.json" % d),
               {"stage": "folded_package_verify", "package_valid": True,
                "nonlinear_backend_ok": True, "nonlinear_backend": d,
                "manifest_hash": hashes[d], "num_shards": 29})
        # boundary artifact dir
        bm = pkg_root / ("qwen7b_boundary_artifact_%s" % d) / "boundary_meta.json"
        _write(bm, {"artifact_type": "trusted_boundary_embedding",
                    "nonlinear_backend": d})
        # E3 with tokens 1,4,8,16 + fields
        rows = [{"seq_len": 128, "max_new_tokens": t, "tokens_exact_match": True,
                 "audit_passed": True} for t in (1, 4, 8, 16)]
        _write(results / ("e3_%s.json" % d),
               {"stage": "remote_package_decode_scaling", "nonlinear_backend": d,
                "rows": rows})
        _write(results / ("e9_aggregate_%s.json" % d),
               {"stage": "e9_aggregate_utility_preservation",
                "nonlinear_backend": d, "utility_preserved": True,
                "paper_ready": True, "dry_run": False})
        for ds in datasets:
            _write(results / ("e9_%s_%s.json" % (ds, d)),
                   {"stage": "e9_task_utility_benchmark",
                    "backend": "tdx_attested_remote", "nonlinear_backend": d,
                    "dataset": ds, "paper_ready": True, "dry_run": False})
    for ds in datasets:
        _write(results / ("e9_%s_plaintext.json" % ds),
               {"stage": "e9_task_utility_benchmark", "backend": "plaintext_local",
                "dataset": ds, "paper_ready": True, "dry_run": False})
    _write(results / "security_negative_tests.json",
           {"stage": "security_negative_tests", "all_passed": True})


def test_h800_checker_fails_on_empty(tmp_path) -> None:
    chk = _load("h800a", "scripts/check_h800_stage_outputs.py")
    rep = chk.run_check({"results_dir": str(tmp_path / "empty"),
                         "package_root": str(tmp_path / "empty")})
    assert rep["passed"] is False
    assert rep["num_required_failed"] > 0


def test_h800_checker_passes_on_minimal_synthetic(tmp_path) -> None:
    results = tmp_path / "outputs"
    pkg = tmp_path / "pkgs"
    _synth_h800(results, pkg)
    chk = _load("h800b", "scripts/check_h800_stage_outputs.py")
    rep = chk.run_check({"results_dir": str(results), "package_root": str(pkg)})
    assert rep["passed"] is True, rep["blockers"]


def test_h800_checker_detects_fixture_data(tmp_path) -> None:
    results = tmp_path / "outputs"
    pkg = tmp_path / "pkgs"
    _synth_h800(results, pkg)
    # poison one E9 report with a fixture dataset path
    _write(results / "e9_bad.json",
           {"stage": "e9_task_utility_benchmark", "backend": "plaintext_local",
            "dataset": "mmlu", "dataset_jsonl": "tests/fixtures/benchmarks/x.jsonl",
            "paper_ready": True, "dry_run": False})
    chk = _load("h800c", "scripts/check_h800_stage_outputs.py")
    rep = chk.run_check({"results_dir": str(results), "package_root": str(pkg)})
    assert rep["passed"] is False
    assert any("no_fixture_or_tiny" in b for b in rep["blockers"])


# ---- E3 / dual-matrix command template carries --folded-package-path -------

def test_dual_matrix_e3_command_has_folded_package_path() -> None:
    dm = importlib.import_module("pllo.experiments.dual_nonlinear_matrix")
    plan = dm.build_matrix_plan(
        nonlinear_backends=["current", "trusted_shortcut"], model_path="/m",
        model_name="Q", base_output_root="/r", outputs_dir="outputs/dm",
        seq_len=128, max_new_tokens_list=[1, 4, 8, 16], run_mode="plan",
        include={k: True for k in ("build", "local_probes", "remote_decode",
                                   "tdx_lite", "tdx_attested", "lora",
                                   "public_benchmarks", "latency", "security")},
        gpu_worker_url="http://127.0.0.1:18082")
    cmds = [s["command"] for b in plan["per_backend"].values()
            for s in b["steps"]]
    e3 = [c for c in cmds if "run_e3_remote_decode_scaling" in c]
    assert e3
    for c in e3:
        assert "--folded-package-path" in c
        assert "--gpu-worker-url" in c
    # no invalid --package-path leaks (verify uses the accepted alias)
    stray = [c for c in cmds if "--package-path " in c
             and "verify_folded_package" not in c]
    assert not stray, stray
    # the real TDX gpu-backend is used, never a nonexistent "boundary" backend
    assert all("--gpu-backend boundary" not in c for c in cmds)


def test_e3_script_guard_requires_folded_package_path(tmp_path) -> None:
    e3 = _load("e3g", "scripts/run_e3_remote_decode_scaling.py")
    try:
        rc = _main(e3, ["x", "--gpu-worker-url", "http://127.0.0.1:9",
                        "--max-new-tokens-list", "1,4",
                        "--output-json", str(tmp_path / "e3.json")])
        assert rc not in (0, None)
    except SystemExit as exc:                               # argparse error -> exit 2
        assert exc.code not in (0, None)


# ---- TDX minimum runbook mentions per-design runtime hash + evidence -------

def test_tdx_minimum_runbook_per_design_evidence() -> None:
    txt = (REPO_ROOT / "docs" / "runbooks"
           / "REAL_TDX_MINIMUM_DEPLOYMENT_EVIDENCE_RUNBOOK.md").read_text(
        encoding="utf-8")
    low = txt.lower()
    assert "current" in low and "trusted_shortcut" in low
    assert "runtime hash" in low
    assert "per design" in low or "separately per design" in low
    # the cross-design negative control must be documented
    assert "fails for trusted_shortcut" in low or "negative control" in low
    assert "ss" not in low.split("##")[0] or "not installed" in low
