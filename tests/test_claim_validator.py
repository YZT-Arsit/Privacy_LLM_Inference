"""Task G: paper claim validator must refuse overclaims. stdlib only.

Run: python -m pytest tests/test_claim_validator.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.claim_validator import build_claim_report  # noqa: E402

_SEC = {"gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "audit_passed": True}


def _attested_no_lora(dry=False):
    return dict(_SEC, stage="qwen7b_folded_remote_package_decode",
                gpu_backend="qwen7b_folded_package", gpu_worker_remote=True,
                dry_run=dry, boundary_mode="lite", folded_package_loaded=True,
                folded_package_valid=True, package_backed_prefill=True,
                package_backed_decode=True, tokens_exact_match=True,
                lora_enabled=False, boundary_attested=True,
                runtime_hash_bound=True,
                attestation={"tee_type": "tdx", "verified": True,
                             "available": True, "mr_td_match": True})


def _lora_dry():
    return dict(_SEC, stage="qwen7b_lora_folded_remote_decode_probe",
                gpu_backend="qwen7b_folded_package", gpu_worker_remote=True,
                dry_run=True, boundary_mode="lite", lora_enabled=True,
                folded_lora_loaded=True, folded_lora_valid=True,
                package_backed_decode=True, tokens_exact_match=True,
                worker_has_raw_lora=False, lora_mode="synthetic")


def _wrap(reports):
    return [{"file": "r%d.json" % i, "report": r}
            for i, r in enumerate(reports)]


def test_attested_no_lora_supports_attested_claim() -> None:
    rep = build_claim_report(_wrap([_attested_no_lora()]))
    assert "no_lora_tdx_attested_remote_package_decode" in rep["supported_claims"]
    assert "no_lora_h800_remote_package_decode" in rep["supported_claims"]
    assert "real_tdx_attestation_bound_to_runtime_hash" in rep["supported_claims"]
    # no LoRA evidence -> LoRA real claims unsupported
    assert "folded_lora_h800_real_validated" in rep["unsupported_claims"]
    # production always unsupported here
    assert "production_ready_serving" in rep["unsupported_claims"]


def test_dry_run_lora_only_supports_dry_run_claim() -> None:
    rep = build_claim_report(_wrap([_lora_dry()]))
    assert "folded_lora_dry_run_validated" in rep["supported_claims"]
    # dry-run must NOT support real/attested LoRA claims
    assert "folded_lora_h800_real_validated" in rep["unsupported_claims"]
    assert "folded_lora_tdx_attested_validated" in rep["unsupported_claims"]
    # overclaim risk recorded for the real shape with dry_run reason
    risks = {(o["claim"]) for o in rep["overclaim_risks"]}
    assert "folded_lora_h800_real_validated" in risks


def test_remote_non_attested_does_not_support_attested() -> None:
    r = _attested_no_lora()
    r.pop("attestation")
    r["boundary_attested"] = False
    r["runtime_hash_bound"] = False
    rep = build_claim_report(_wrap([r]))
    assert "no_lora_h800_remote_package_decode" in rep["supported_claims"]
    assert ("no_lora_tdx_attested_remote_package_decode"
            in rep["unsupported_claims"])


def test_required_claims_gate() -> None:
    rep = build_claim_report(
        _wrap([_lora_dry()]),
        required_claims=["folded_lora_tdx_attested_validated"])
    assert rep["all_required_supported"] is False
    assert any("REQUIRED" in w for w in rep["warnings"])


def test_negative_tests_and_training_claims() -> None:
    neg = {"stage": "security_negative_tests", "all_passed": True}
    train = {"stage": "private_lora_training_probe", "loss_decreased": True,
             "audit_passed": True}
    rep = build_claim_report(_wrap([neg, train]))
    assert "security_negative_tests_passed" in rep["supported_claims"]
    assert "private_lora_training_tiny_prototype" in rep["supported_claims"]


def test_script(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "vpc", REPO_ROOT / "scripts" / "validate_paper_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f1 = tmp_path / "att.json"
    f1.write_text(json.dumps(_attested_no_lora()))
    oj = tmp_path / "claims.json"
    old = sys.argv
    try:
        sys.argv = ["x", "--result-json", str(f1), "--output-json", str(oj),
                    "--output-md", str(tmp_path / "claims.md")]
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc == 0
    r = json.loads(oj.read_text())
    assert "no_lora_tdx_attested_remote_package_decode" in r["supported_claims"]
