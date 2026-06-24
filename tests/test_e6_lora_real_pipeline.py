"""Dry-run tests for the E6 real-run orchestration + helper scripts (Task 8):
pipeline command plan, validate_lora_effect, prepare_tdx_lora_lite_inputs,
check_tdx_measurement_coverage, and package_final_artifacts. stdlib only -- no
H800 / TDX / CUDA / full Qwen / torch.

Run: python -m pytest tests/test_e6_lora_real_pipeline.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
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


# ---------------------------------------------------------------------------
# Task 1: E6 real pipeline command plan
# ---------------------------------------------------------------------------


def test_e6_pipeline_plan(tmp_path) -> None:
    mod = _load("e6pipe", "scripts/run_e6_lora_real_h800_pipeline.py")
    plan_json = tmp_path / "plan.json"
    rc = _main(mod, ["x",
                     "--model-path", "/root/Qwen2.5-7B",
                     "--base-folded-package-path", "/root/base",
                     "--embedding-artifact-path", "/root/art",
                     "--output-lora-package", "/root/lora",
                     "--lora-mode", "synthetic", "--lora-rank", "4",
                     "--lora-alpha", "8", "--target-modules", "q_proj,v_proj",
                     "--gpu-worker-url", "http://127.0.0.1:18083",
                     "--listen-port", "18083", "--plan-only",
                     "--output-json", str(plan_json)])
    assert rc == 0
    plan = json.loads(plan_json.read_text())
    names = [s["name"] for s in plan["steps"]]
    # required ordered backbone + worker step
    for expected in ("build_lora_package", "verify_lora_package",
                     "local_lora_probe", "worker_check_or_start",
                     "remote_lora_decode"):
        assert expected in names
    assert names.index("build_lora_package") < names.index("verify_lora_package")
    assert names.index("worker_check_or_start") < names.index(
        "remote_lora_decode")
    by = {s["name"]: s for s in plan["steps"]}
    assert "build_qwen7b_lora_folded_package.py" in by["build_lora_package"][
        "command"]
    assert "--output-dir /root/lora" in by["build_lora_package"]["command"]
    assert "run_qwen7b_lora_folded_remote_decode_probe.py" in by[
        "remote_lora_decode"]["command"]
    assert "--gpu-worker-url http://127.0.0.1:18083" in by["remote_lora_decode"][
        "command"]
    # TDX-lite prep emitted by default
    assert "prepare_tdx_lite_inputs" in names


def test_e6_pipeline_hf_mode_requires_adapter(tmp_path) -> None:
    mod = _load("e6pipe2", "scripts/run_e6_lora_real_h800_pipeline.py")
    try:
        _main(mod, ["x", "--base-folded-package-path", "/b",
                    "--embedding-artifact-path", "/a",
                    "--output-lora-package", "/l", "--lora-mode", "hf",
                    "--plan-only"])
        assert False, "expected SystemExit for hf mode without adapter"
    except SystemExit as exc:
        assert exc.code != 0


def test_e6_pipeline_hf_plan_uses_adapter(tmp_path) -> None:
    mod = _load("e6pipe3", "scripts/run_e6_lora_real_h800_pipeline.py")
    plan_json = tmp_path / "plan.json"
    rc = _main(mod, ["x", "--model-path", "/m",
                     "--base-folded-package-path", "/b",
                     "--embedding-artifact-path", "/a",
                     "--output-lora-package", "/l", "--lora-mode", "hf",
                     "--raw-lora-adapter-path", "/adapter",
                     "--adapter-format", "hf_peft", "--plan-only",
                     "--output-json", str(plan_json)])
    assert rc == 0
    plan = json.loads(plan_json.read_text())
    build = next(s for s in plan["steps"] if s["name"] == "build_lora_package")
    assert "--raw-lora-adapter-path /adapter" in build["command"]
    assert "--adapter-format hf_peft" in build["command"]


# ---------------------------------------------------------------------------
# Task 3: validate_lora_effect
# ---------------------------------------------------------------------------


def test_validate_lora_effect_detects_change(tmp_path) -> None:
    mod = _load("vle", "scripts/validate_lora_effect.py")
    nl = tmp_path / "nl.json"
    ll = tmp_path / "ll.json"
    nl.write_text(json.dumps({"package_token_ids": [10, 20, 30, 40]}))
    ll.write_text(json.dumps({"package_token_ids": [10, 99, 30, 41]}))
    out = tmp_path / "v.json"
    rc = _main(mod, ["x", "--no-lora-json", str(nl), "--lora-json", str(ll),
                     "--require-effect", "true", "--output-json", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["tokens_differ"] is True
    assert r["token_diff_positions"] == [1, 3]
    assert r["top1_changed"] is False         # first token identical (10==10)
    assert r["lora_has_effect"] is True
    assert r["warning"] is None


def test_validate_lora_effect_warns_when_identical(tmp_path) -> None:
    mod = _load("vle2", "scripts/validate_lora_effect.py")
    nl = tmp_path / "nl.json"
    ll = tmp_path / "ll.json"
    same = {"package_token_ids": [1, 2, 3, 4]}
    nl.write_text(json.dumps(same))
    ll.write_text(json.dumps(same))
    out = tmp_path / "v.json"
    rc = _main(mod, ["x", "--no-lora-json", str(nl), "--lora-json", str(ll),
                     "--require-effect", "true", "--output-json", str(out)])
    assert rc == 1                            # require-effect + no effect
    r = json.loads(out.read_text())
    assert r["tokens_differ"] is False
    assert r["lora_has_effect"] is False
    assert "IDENTICAL" in r["warning"]


# ---------------------------------------------------------------------------
# Task 4: prepare_tdx_lora_lite_inputs
# ---------------------------------------------------------------------------


def test_prepare_tdx_lite_inputs_command(tmp_path) -> None:
    mod = _load("ptdx", "scripts/prepare_tdx_lora_lite_inputs.py")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({
        "stage": "qwen7b_lora_folded_remote_decode_probe",
        "input_ids": [5, 6, 7, 8],
        "package_token_ids": [11, 12, 13, 14]}))
    out = tmp_path / "out"
    rc = _main(mod, ["x", "--reference-json", str(ref),
                     "--embedding-path", "/tdx/art",
                     "--gpu-worker-url", "http://10.0.0.1:18083",
                     "--output-dir", str(out)])
    assert rc == 0
    replay = json.loads((out / "tdx_lora_replay.json").read_text())
    assert replay["input_ids"] == [5, 6, 7, 8]
    assert replay["expected_token_ids"] == [11, 12, 13, 14]
    assert replay["tdx_constraints"]["no_full_model_on_tdx"] is True
    ids = json.loads((out / "tdx_lora_input_ids.json").read_text())
    assert ids["input_ids"] == [5, 6, 7, 8]
    exp = json.loads((out / "tdx_lora_expected_tokens.json").read_text())
    assert exp["expected_token_ids"] == [11, 12, 13, 14]
    sh = (out / "run_tdx_lora_lite_decode.sh").read_text()
    assert "--skip-reference true" in sh
    assert "--embedding-path /tdx/art" in sh
    assert "--input-ids-file" in sh
    assert "--expected-token-ids 11,12,13,14" in sh
    # TDX command must NOT carry full model / base package / raw LoRA
    assert "--model-path" not in sh
    assert "--folded-package-path" not in sh
    assert "--raw-lora" not in sh


# ---------------------------------------------------------------------------
# Task 5: check_tdx_measurement_coverage
# ---------------------------------------------------------------------------


def test_measurement_coverage_ok() -> None:
    mod = _load("cov", "scripts/check_tdx_measurement_coverage.py")
    cov = mod.compute_coverage()
    assert cov["unmeasured_boundary_imports"] == []
    # the lite boundary surface must be in the closure
    closure = set(cov["closure"])
    for f in ("src/pllo/experiments/folded_probe_common.py",
              "src/pllo/deployment/embedding_artifact.py",
              "src/pllo/ops/causal_lm_boundaries.py",
              "src/pllo/protocol/remote.py"):
        assert f in closure


def test_measurement_coverage_catches_missing(monkeypatch) -> None:
    mod = _load("cov2", "scripts/check_tdx_measurement_coverage.py")
    # drop a genuine boundary file from the measured set -> must be flagged
    reduced = tuple(p for p in mod.DEFAULT_TRUSTED_BOUNDARY_PATHS
                    if "embedding_artifact.py" not in p)
    monkeypatch.setattr(mod, "DEFAULT_TRUSTED_BOUNDARY_PATHS", reduced)
    cov = mod.compute_coverage()
    assert "src/pllo/deployment/embedding_artifact.py" in cov[
        "unmeasured_boundary_imports"]
    assert _main(mod, ["x"]) == 1


# ---------------------------------------------------------------------------
# Task 6: package_final_artifacts
# ---------------------------------------------------------------------------


def test_package_final_artifacts_handles_missing(tmp_path) -> None:
    mod = _load("pkg", "scripts/package_final_artifacts.py")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # one real file present; everything else missing
    (outputs / "e8_lora_final_report.json").write_text(json.dumps({"x": 1}))
    tar = tmp_path / "final.tar.gz"
    rc = _main(mod, ["x", "--outputs-dir", str(outputs),
                     "--tee-artifacts-dir", str(tmp_path / "nope"),
                     "--extra-file", str(tmp_path / "missing.sha256"),
                     "--base-folded-package-path", str(tmp_path / "nobase"),
                     "--output-tar", str(tar)])
    assert rc == 0
    assert tar.is_file()
    with tarfile.open(tar, "r:gz") as t:
        members = t.getnames()
        assert "MANIFEST.json" in members
        assert "outputs/e8_lora_final_report.json" in members
        manifest = json.loads(t.extractfile("MANIFEST.json").read())
    assert manifest["num_files"] >= 1
    assert "e8_lora_report" in manifest["categories_present"]
    # missing optional categories are recorded, not fatal
    assert manifest["missing"]
    assert any("tee_artifacts" in k for k in manifest["missing"])
    assert any("extra_files" in k for k in manifest["missing"])
