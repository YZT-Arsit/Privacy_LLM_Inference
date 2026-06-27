"""run_generation_utility_benchmark.py folded_remote = boundary-client path.

Proves the generation utility benchmark reuses the validated folded_remote
boundary-client predictor (tokenizer/artifact only, no full 7B on the trusted
side) and threads the chat-template / generation-config / EOS alignment. Uses a
FAKE build_predictor so no model/worker is needed.

Run:
    PYTHONPATH=$PWD/src pytest tests/test_generation_benchmark_boundary_client.py -q
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


_R = _load("gen_util_runner", "scripts/run_generation_utility_benchmark.py")


class _FakeFolded:
    """Boundary-client predictor: NO _model attribute (no full weights)."""
    def generate(self, prompt):
        return {"text": "reasoning 5 then 7 ... 24/8=3\n#### 3",
                "token_ids": list(range(8))}
    def stats(self):
        return {"audit_passed": True, "tee_used_on_gpu": False,
                "worker_has_mask_secrets": False}
    def close(self):
        pass


def _dataset(tmp_path):
    p = tmp_path / "gsm.jsonl"
    p.write_text(json.dumps({
        "id": "g1", "prompt": "Natalia sold 24 clips?", "answer": "#### 3",
        "task_type": "generation_exact", "dataset_name": "gsm8k"}) + "\n",
        encoding="utf-8")
    return p


def _run(tmp_path, extra, fake_build=None):
    ds = _dataset(tmp_path)
    oj = tmp_path / "out.json"
    argv = ["x", "--dataset-jsonl", str(ds), "--backend", "folded_remote",
            "--model-path", str(tmp_path / "model"),
            "--gpu-worker-url", "http://127.0.0.1:18082",
            "--embedding-path", str(tmp_path / "art"),
            "--require-real", "--output-json", str(oj)] + extra
    old_argv, old_build = sys.argv, _R.build_predictor
    try:
        sys.argv = argv
        if fake_build is not None:
            _R.build_predictor = fake_build
        rc = _R.main()
    finally:
        sys.argv = old_argv
        _R.build_predictor = old_build
    report = json.loads(oj.read_text()) if oj.exists() else None
    return rc, report


def test_folded_remote_threads_boundary_client_params(tmp_path) -> None:
    captured = {}

    def fake_build(backend, **kw):
        captured.update(kw)
        captured["backend"] = backend
        return _FakeFolded()

    rc, report = _run(tmp_path, [
        "--tdx-boundary-client", "--trusted-runtime", "tdx_guest",
        "--tee-mode", "real_tdx", "--h800-worker-ssh-alias", "h800-new",
        "--use-chat-template", "--align-generation-config",
        "--repetition-penalty", "1.05"], fake_build=fake_build)
    assert rc == 0
    # the validated boundary-client params are threaded into build_predictor
    assert captured["backend"] == "folded_remote"
    assert captured["use_chat_template"] is True
    assert captured["align_generation_config"] is True
    assert captured["repetition_penalty"] == 1.05
    assert captured["stop_on_eos"] is True
    # NO full 7B weights loaded on the trusted side
    assert report["full_model_weights_loaded_in_trusted_runtime"] is False
    assert report["tdx_boundary_client"] is True
    assert report["tee_mode"] == "real_tdx"
    assert report["trusted_runtime"] == "tdx_guest"
    assert report["h800_worker_url"] == "http://127.0.0.1:18082"
    assert report["h800_worker_ssh_alias"] == "h800-new"
    # GSM8K marker extraction works through the runner (the fix from before)
    gen = report["generations"][0]
    assert gen["extracted_number"] == "3"
    assert gen["numeric_exact_match"] is True


def test_disable_eos_stop_threads_false(tmp_path) -> None:
    captured = {}

    def fake_build(backend, **kw):
        captured.update(kw)
        return _FakeFolded()

    rc, _report = _run(tmp_path, ["--disable-eos-stop"], fake_build=fake_build)
    assert rc == 0
    assert captured["stop_on_eos"] is False


def test_plaintext_plus_tdx_refused(tmp_path) -> None:
    ds = _dataset(tmp_path)
    argv = ["x", "--dataset-jsonl", str(ds), "--backend", "plaintext_local",
            "--tdx-boundary-client", "--model-path", str(tmp_path / "m"),
            "--output-json", str(tmp_path / "o.json")]
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    assert rc == 3


def test_full_weights_guard_trips_if_model_loaded(tmp_path) -> None:
    # a (mis)built predictor that DID load weights -> guard must refuse
    class _BadPredictor:
        _model = object()                      # signals full weights present
        def generate(self, p):
            return {"text": "#### 3", "token_ids": [1]}
        def stats(self):
            return {}
        def close(self):
            pass

    rc, _report = _run(tmp_path, ["--tdx-boundary-client"],
                       fake_build=lambda backend, **kw: _BadPredictor())
    assert rc == 3
