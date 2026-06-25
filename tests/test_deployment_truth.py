"""Deployment truth checker -- pure parsing, stdlib only.

Run: python -m pytest tests/test_deployment_truth.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.deployment_truth import (  # noqa: E402
    allowed_and_forbidden_claims,
    deployment_truth_report,
    infer_deployment_truth,
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


def test_mock_backend() -> None:
    rep = {"stage": "demo", "gpu_backend": "mock", "gpu_worker_remote": False,
           "dry_run": False}
    truth = infer_deployment_truth(rep)
    assert truth["mock_backend"] is True
    assert truth["gpu_real"] is False
    claims = allowed_and_forbidden_claims(truth)
    assert "production_ready_secure_serving" in claims["forbidden_claims"]


def test_dry_run_folded_remote() -> None:
    rep = {"stage": "demo", "gpu_backend": "qwen7b_folded_package",
           "gpu_worker_remote": True, "dry_run": True,
           "folded_package_loaded": True, "package_backed_decode": True}
    truth = infer_deployment_truth(rep)
    assert truth["gpu_real"] is False
    assert truth["folded_package_real"] is False


def test_real_tdx_attested_no_lora() -> None:
    rep = {
        "stage": "tdx_attested_qwen7b_folded_remote_decode",
        "gpu_backend": "qwen7b_folded_package", "gpu_worker_remote": True,
        "dry_run": False, "folded_package_loaded": True,
        "folded_package_valid": True, "package_backed_decode": True,
        "package_backed_prefill": True, "lora_enabled": False,
        "boundary_mode": "lite", "boundary_attested": True,
        "runtime_hash_bound": True,
        "server_health": {"gpu_name": "NVIDIA H800"},
        "attestation": {"tee_type": "tdx", "verified": True, "available": True,
                        "mr_td_match": True},
    }
    truth = infer_deployment_truth(rep)
    assert truth["tee_real"] is True
    assert truth["tee_type"] == "tdx"
    assert truth["attestation_verified"] is True
    assert truth["mr_td_verified"] is True
    assert truth["runtime_hash_bound"] is True
    assert truth["gpu_type"] == "h800"
    assert truth["boundary_holds_full_model"] is False
    claims = allowed_and_forbidden_claims(truth)
    assert "real_tdx_remote_h800_package_backed_decode" in claims[
        "allowed_claims"]
    assert "no_lora_package_backed_decode" in claims["allowed_claims"]


def test_synthetic_lora_forbidden() -> None:
    rep = {"stage": "lora_demo", "gpu_backend": "qwen7b_folded_package",
           "gpu_worker_remote": True, "dry_run": False, "lora_enabled": True,
           "lora_mode": "synthetic", "folded_lora_loaded": True,
           "package_backed_decode": True}
    truth = infer_deployment_truth(rep)
    assert truth["lora_synthetic"] is True
    assert truth["lora_real_hf_adapter"] is False
    claims = allowed_and_forbidden_claims(truth)
    assert "real_lora_tdx_attested" in claims["forbidden_claims"]
    assert "folded_lora_package_backed_decode" in claims["allowed_claims"]
    assert any("synthetic" in w for w in claims["warnings"])


def test_deployment_truth_report_shape() -> None:
    res = deployment_truth_report({"stage": "x", "gpu_backend": "mock",
                                   "dry_run": True})
    assert res["stage"] == "deployment_truth"
    assert res["source_stage"] == "x"
    assert "truth" in res and "allowed_claims" in res
    # full key contract
    expected = {
        "dry_run", "mock_backend", "gpu_real", "gpu_type", "gpu_worker_remote",
        "tee_real", "tee_type", "attestation_evidence_present",
        "attestation_verified", "runtime_hash_bound", "mr_td_verified",
        "folded_package_real", "folded_package_loaded", "folded_package_valid",
        "package_backed_prefill", "package_backed_decode", "boundary_mode",
        "boundary_holds_full_model", "boundary_holds_full_folded_package",
        "lora_enabled", "lora_synthetic", "lora_real_hf_adapter",
        "folded_lora_loaded", "folded_lora_valid", "production_transport",
        "research_prototype_transport",
    }
    assert set(res["truth"].keys()) == expected


def test_check_deployment_truth_script(tmp_path) -> None:
    r1 = {"stage": "a", "gpu_backend": "mock", "dry_run": False}
    r2 = {"stage": "b", "gpu_backend": "qwen7b_folded_package",
          "gpu_worker_remote": True, "dry_run": False,
          "folded_package_loaded": True, "package_backed_decode": True,
          "lora_enabled": False}
    p1 = tmp_path / "r1.json"
    p2 = tmp_path / "r2.json"
    p1.write_text(json.dumps(r1))
    p2.write_text(json.dumps(r2))
    mod = _load("checkdt", "scripts/check_deployment_truth.py")
    oj = tmp_path / "out.json"
    rc = _main(mod, ["x", "--result-json", str(p1), "--result-json", str(p2),
                     "--output-json", str(oj),
                     "--output-md", str(tmp_path / "out.md")])
    assert rc == 0
    out = json.loads(oj.read_text())
    assert out["stage"] == "deployment_truth"
    assert len(out["results"]) == 2
