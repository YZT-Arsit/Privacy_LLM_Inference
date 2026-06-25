"""Fixture-only tests for the two-design experiment matrix (Task 3).

No execution / torch / model / GPU / network. Run:
    python -m pytest tests/test_dual_nonlinear_matrix.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.dual_nonlinear_matrix import (  # noqa: E402
    build_matrix_plan,
    iter_commands,
    render_md,
)


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


_ALL_ON = {
    "build": True, "local_probes": True, "remote_decode": True,
    "tdx_lite": True, "tdx_attested": True, "lora": True,
    "public_benchmarks": True, "latency": True, "security": True,
}


def _plan(include=None, backends="current,trusted_shortcut"):
    return build_matrix_plan(
        nonlinear_backends=backends, model_path="/models/qwen",
        model_name="Qwen2.5-7B-Instruct", base_output_root="packages",
        outputs_dir="outputs/dual_nonlinear", seq_len=128,
        max_new_tokens_list=[1, 4, 8, 16], run_mode="plan",
        include=include if include is not None else dict(_ALL_ON))


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------


def test_plan_includes_both_backends() -> None:
    plan = _plan()
    assert plan["nonlinear_backends"] == ["current", "trusted_shortcut"]
    assert set(plan["per_backend"]) == {"current", "trusted_shortcut"}
    assert plan["stage"] == "dual_nonlinear_experiment_matrix"
    assert plan["total_step_count"] > 0


def test_paths_namespaced_by_backend() -> None:
    plan = _plan()
    blob = json.dumps(plan)
    assert "packages/qwen7b_folded_full_current" in blob
    assert "packages/qwen7b_folded_full_trusted_shortcut" in blob
    for backend in ("current", "trusted_shortcut"):
        out = plan["per_backend"][backend]["namespaced_paths"]["outputs_dir"]
        assert out == "outputs/dual_nonlinear/%s" % backend
        # each step's outputs live under that backend's dir
        for step in plan["per_backend"][backend]["steps"]:
            for f in step["expected_output_files"]:
                assert ("/%s/" % backend) in f or backend in f


def test_tdx_attested_steps_flagged() -> None:
    plan = _plan()
    found = 0
    for backend, step in iter_commands(plan):
        if step["id"] == "tdx_attested_decode":
            found += 1
            assert step["tdx_evidence_must_be_regenerated"] is True
            assert step["required_server_state"] == "tdx_quote_bound"
            assert step["side"] == "tdx"
    assert found == 2  # one per backend


def test_commands_carry_nonlinear_backend_flag() -> None:
    plan = _plan()
    cur = " ".join(s["command"] for _, s in iter_commands(plan)
                   if s["backend"] == "current"
                   and "--nonlinear-backend" in s["command"])
    ts = " ".join(s["command"] for _, s in iter_commands(plan)
                  if s["backend"] == "trusted_shortcut"
                  and "--nonlinear-backend" in s["command"])
    assert "--nonlinear-backend current" in cur
    assert "--nonlinear-backend current" not in ts
    assert "--nonlinear-backend trusted_shortcut" in ts
    assert "--nonlinear-backend trusted_shortcut" not in cur


def test_runtime_hash_precedes_attested_decode() -> None:
    plan = _plan()
    ids = [s["id"] for _, s in iter_commands(plan)
           if s["backend"] == "current"]
    assert "runtime_hash" in ids
    assert ids.index("runtime_hash") < ids.index("tdx_attested_decode")


def test_disabling_lora_drops_lora_steps() -> None:
    inc = dict(_ALL_ON)
    inc["lora"] = False
    plan = _plan(include=inc)
    lora_ids = {"lora_build", "lora_verify", "lora_local_probe",
                "lora_remote_probe", "e10_lora_utility"}
    seen = {s["id"] for _, s in iter_commands(plan)}
    assert not (lora_ids & seen)
    # full plan does have them
    full = {s["id"] for _, s in iter_commands(_plan())}
    assert lora_ids <= full


def test_always_steps_present_even_when_all_off() -> None:
    plan = _plan(include={})  # all flags off
    for backend in ("current", "trusted_shortcut"):
        ids = [s["id"] for s in plan["per_backend"][backend]["steps"]]
        assert ids == ["deployment_truth", "claim_validator",
                       "package_final_artifacts"]


def test_claim_validator_backend_tagged() -> None:
    plan = _plan()
    for backend in ("current", "trusted_shortcut"):
        cmd = next(s["command"] for s in plan["per_backend"][backend]["steps"]
                   if s["id"] == "claim_validator")
        assert "public_benchmark_utility_preserved[%s]" % backend in cmd


def test_render_md_and_notes() -> None:
    plan = _plan()
    md = render_md(plan)
    assert "current" in md and "trusted_shortcut" in md
    assert md.count("```") >= plan["total_step_count"] * 2
    assert plan["notes"] and plan["limitations"]


# --------------------------------------------------------------------------
# runner script
# --------------------------------------------------------------------------


def _runner():
    return _load("dnm", "scripts/run_dual_nonlinear_experiment_matrix.py")


def test_script_plan_mode_writes_json_and_md(tmp_path) -> None:
    mod = _runner()
    oj = tmp_path / "plan.json"
    om = tmp_path / "plan.md"
    rc = _main(mod, [
        "x", "--nonlinear-backends", "current,trusted_shortcut",
        "--model-path", "/models/qwen", "--model-name", "Qwen2.5-7B-Instruct",
        "--base-output-root", str(tmp_path / "packages"),
        "--outputs-dir", "outputs/dual_nonlinear", "--seq-len", "128",
        "--max-new-tokens-list", "1,4,8,16", "--run-mode", "plan",
        "--include-build", "true", "--include-local-probes", "true",
        "--include-remote-decode", "true", "--include-tdx-lite", "true",
        "--include-tdx-attested", "true", "--include-lora", "true",
        "--include-public-benchmarks", "true", "--include-latency", "true",
        "--include-security", "true",
        "--output-json", str(oj), "--output-md", str(om)])
    assert rc == 0
    assert oj.exists() and om.exists()
    r = json.loads(oj.read_text())
    assert r["total_step_count"] > 0
    assert set(r["per_backend"]) == {"current", "trusted_shortcut"}


def test_script_disable_lora_flag(tmp_path) -> None:
    mod = _runner()
    oj = tmp_path / "plan.json"
    rc = _main(mod, [
        "x", "--nonlinear-backends", "current,trusted_shortcut",
        "--model-path", "/models/qwen",
        "--base-output-root", str(tmp_path / "packages"),
        "--outputs-dir", "outputs/dual_nonlinear", "--run-mode", "plan",
        "--include-lora", "false", "--output-json", str(oj)])
    assert rc == 0
    blob = oj.read_text()
    assert "lora_build" not in blob
    assert "e10_lora_utility" not in blob


def test_script_verify_only_lists_missing_and_no_subprocess(tmp_path,
                                                            monkeypatch) -> None:
    mod = _runner()

    def _boom(*a, **k):
        raise AssertionError("subprocess must not run in verify-only")

    monkeypatch.setattr(mod.subprocess, "run", _boom)
    oj = tmp_path / "plan.json"
    rc = _main(mod, [
        "x", "--nonlinear-backends", "current,trusted_shortcut",
        "--model-path", "/models/qwen",
        "--base-output-root", str(tmp_path / "packages"),
        "--outputs-dir", str(tmp_path / "out"), "--run-mode", "verify-only",
        "--include-build", "true", "--output-json", str(oj)])
    assert rc == 0
    # verify report written alongside; lists missing files
    vr = json.loads((tmp_path / "plan.json.verify.json").read_text())
    assert vr["any_missing"] is True
    assert any(r["missing_files"] for r in vr["results"])
