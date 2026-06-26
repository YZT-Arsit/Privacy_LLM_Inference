"""preflight_tdx_h800_bridge.py pure helpers + assembly (no network/subprocess).

Run:
    PYTHONPATH=$PWD/src pytest tests/test_tdx_h800_preflight.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_P = _load("preflight", "scripts/preflight_tdx_h800_bridge.py")

_HEALTH_OK = {"status": "ok", "gpu_backend": "qwen7b_folded_package",
              "tee_used_on_gpu": False,
              "resident_status": {"resident_folded_weights": True,
                                  "resident_cache_active": True}}


def test_check_worker_health_ok() -> None:
    w = _P.check_worker_health(_HEALTH_OK)
    assert w["reachable"] and w["status_ok"]
    assert w["tee_used_on_gpu"] is False and w["tee_used_on_gpu_is_false"] is True
    assert w["gpu_backend"] == "qwen7b_folded_package"
    assert w["resident_folded_weights"] is True


def test_check_worker_health_unreachable_or_tee_on() -> None:
    assert _P.check_worker_health(None)["reachable"] is False
    bad = dict(_HEALTH_OK, tee_used_on_gpu=True)
    assert _P.check_worker_health(bad)["tee_used_on_gpu_is_false"] is False


def test_fetch_worker_health_with_injected_fetcher() -> None:
    health = _P.fetch_worker_health(
        "http://x:18082", fetcher=lambda u: json.dumps(_HEALTH_OK))
    assert health == _HEALTH_OK
    # a failing fetcher -> None (never raises)
    def boom(_u):
        raise RuntimeError("down")
    assert _P.fetch_worker_health("http://x:18082", fetcher=boom) is None


def test_ssh_reachable_with_injected_runner() -> None:
    assert _P.ssh_reachable("h800-new", runner=lambda cmd: 0) is True
    assert _P.ssh_reachable("h800-new", runner=lambda cmd: 255) is False
    assert _P.ssh_reachable("", runner=lambda cmd: 0) is False


def test_check_measurement_coverage(tmp_path) -> None:
    ok = tmp_path / "ok.log"
    ok.write_text("blah\nTDX MEASUREMENT COVERAGE: OK\n")
    assert _P.check_measurement_coverage(ok)["coverage_ok"] is True
    bad = tmp_path / "bad.log"
    bad.write_text("some failure\n")
    assert _P.check_measurement_coverage(bad)["coverage_ok"] is False
    assert _P.check_measurement_coverage(None)["coverage_ok"] is None
    assert _P.check_measurement_coverage(tmp_path / "missing.log") == {
        "present": False, "coverage_ok": False}


def test_check_model_meta_tokenizer_only(tmp_path) -> None:
    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    meta = _P.check_model_meta(base)
    assert meta["config.json"] is True
    assert meta["tokenizer_any"] is True
    assert meta["config_and_tokenizer_present"] is True
    assert meta["full_weights_present"] is False        # no safetensors


def test_check_model_meta_flags_full_weights(tmp_path) -> None:
    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    (base / "model-00001-of-00004.safetensors").write_text("x")
    meta = _P.check_model_meta(base)
    assert meta["full_weights_present"] is True          # informational


def test_build_preflight_pass(tmp_path) -> None:
    rep = _P.build_preflight(
        local={"hostname": "tdx-guest", "uname": "Linux", "tdx_cpu_flag": True,
               "tdx_device_present": True},
        ssh_reachable=True,
        worker=_P.check_worker_health(_HEALTH_OK),
        files={"input_jsonl": True, "embedding_path": True},
        meta={"config_and_tokenizer_present": True},
        coverage={"present": True, "coverage_ok": True})
    assert rep["preflight_passed"] is True
    assert rep["tdx_host"] == "tdx-guest"
    assert rep["h800_worker_tee_used_on_gpu"] is False
    assert rep["tdx_measurement_coverage_ok"] is True


def test_build_preflight_fail_when_tee_on_or_missing() -> None:
    base = dict(
        local={"hostname": "h", "uname": "L", "tdx_cpu_flag": False,
               "tdx_device_present": False},
        ssh_reachable=True,
        worker=_P.check_worker_health(_HEALTH_OK),
        files={"input_jsonl": True, "embedding_path": True},
        meta={"config_and_tokenizer_present": True},
        coverage={"present": True, "coverage_ok": True})
    # worker TEE on -> fail
    bad = dict(base, worker=_P.check_worker_health(
        dict(_HEALTH_OK, tee_used_on_gpu=True)))
    assert _P.build_preflight(**bad)["preflight_passed"] is False
    # missing input -> fail
    bad2 = dict(base, files={"input_jsonl": False, "embedding_path": True})
    assert _P.build_preflight(**bad2)["preflight_passed"] is False
    # coverage explicitly false -> fail
    bad3 = dict(base, coverage={"present": True, "coverage_ok": False})
    assert _P.build_preflight(**bad3)["preflight_passed"] is False


def test_preflight_main_writes_outputs(tmp_path, monkeypatch) -> None:
    inp = tmp_path / "in.jsonl"
    inp.write_text("{}\n")
    emb = tmp_path / "emb"
    emb.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    cov = tmp_path / "cov.log"
    cov.write_text("TDX MEASUREMENT COVERAGE: OK\n")
    # avoid real network/ssh
    monkeypatch.setattr(_P, "fetch_worker_health", lambda url: _HEALTH_OK)
    monkeypatch.setattr(_P, "ssh_reachable", lambda alias: True)
    outj, outm = tmp_path / "pf.json", tmp_path / "pf.md"
    argv = ["x", "--gpu-worker-url", "http://127.0.0.1:18082",
            "--h800-worker-ssh-alias", "h800-new",
            "--input-jsonl", str(inp), "--embedding-path", str(emb),
            "--model-meta-path", str(model), "--tdx-measurement-log", str(cov),
            "--output-json", str(outj), "--output-md", str(outm)]
    old = sys.argv
    try:
        sys.argv = argv
        rc = _P.main()
    finally:
        sys.argv = old
    assert rc == 0
    rep = json.loads(outj.read_text())
    assert rep["preflight_passed"] is True
    assert outm.read_text().startswith("# TDX -> H800 bridge preflight")
