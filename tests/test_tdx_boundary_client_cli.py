"""run_ifeval_generation.py TDX boundary-client mode (mock path, no model/CUDA).

Run:
    PYTHONPATH=$PWD/src pytest tests/test_tdx_boundary_client_cli.py -q
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


_R = _load("ifeval_runner_tdx", "scripts/run_ifeval_generation.py")


def _prompts(tmp_path, n=2):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": "ex%d" % i, "prompt": "Q%d?" % i})
        for i in range(n)) + "\n", encoding="utf-8")
    return p


def _run(tmp_path, extra, backend="folded_remote"):
    inp = _prompts(tmp_path)
    rj, rep = tmp_path / "r.jsonl", tmp_path / "r.json"
    argv = ["x", "--input-jsonl", str(inp), "--backend", backend,
            "--mock-runtime", "--max-new-tokens", "4",
            "--output-response-jsonl", str(rj),
            "--output-report-json", str(rep)] + extra
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    report = json.loads(rep.read_text()) if rep.exists() else None
    return rc, report


def test_tdx_cli_parses_and_reports(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--tdx-boundary-client", "--tee-mode", "real_tdx",
        "--trusted-runtime", "tdx_guest",
        "--h800-worker-ssh-alias", "h800-new",
        "--gpu-worker-url", "http://127.0.0.1:18082"])
    assert rc == 0
    assert report["tdx_boundary_client"] is True
    assert report["tee_mode"] == "real_tdx"
    assert report["trusted_runtime"] == "tdx_guest"
    assert report["h800_worker_ssh_alias"] == "h800-new"
    assert report["h800_worker_url"] == "http://127.0.0.1:18082"
    assert report["tdx_host"]                       # hostname captured


def test_folded_tdx_client_loads_no_full_weights(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--tdx-boundary-client", "--gpu-worker-url", "http://127.0.0.1:18082"])
    assert rc == 0
    assert report["full_model_weights_loaded_in_trusted_runtime"] is False


def test_tdx_client_refuses_plaintext_backend(tmp_path) -> None:
    rc, _report = _run(tmp_path, ["--tdx-boundary-client"],
                       backend="plaintext_local")
    assert rc == 3                                   # refused (loads weights)


def test_tdx_claim_ready_is_honest_on_mock(tmp_path) -> None:
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"quote": "ab", "mr_td": "cd"}))
    rc, report = _run(tmp_path, [
        "--tdx-boundary-client", "--gpu-worker-url", "http://127.0.0.1:18082",
        "--attestation-evidence-json", str(ev)])
    assert rc == 0
    assert report["attestation_evidence_attached"] is True
    assert report["dry_run"] is True                # mock -> not a real run
    assert report["tdx_claim_ready"] is False       # honest


def test_tee_mode_explicit_overrides_derivation(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--tdx-boundary-client", "--tee-mode", "real_tdx_custom",
        "--gpu-worker-url", "http://127.0.0.1:18082"])
    assert rc == 0
    assert report["tee_mode"] == "real_tdx_custom"


def test_non_tdx_run_defaults_process(tmp_path) -> None:
    rc, report = _run(tmp_path, ["--gpu-worker-url", "http://127.0.0.1:18082"])
    assert rc == 0
    assert report["tdx_boundary_client"] is False
    assert report["trusted_runtime"] == "process"
    assert report["tee_mode"] == "process_boundary"
    assert report["tdx_claim_ready"] is False
