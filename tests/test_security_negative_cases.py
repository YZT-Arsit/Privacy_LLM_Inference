"""Task E: security negative cases must all be DETECTED. stdlib + numpy only.

Run: python -m pytest tests/test_security_negative_cases.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security.negative_cases import (  # noqa: E402
    audit_decode_report,
    check_lora_meta_flags,
    run_all_negative_cases,
    scan_lora_package_names,
)

EXPECTED = {
    "worker_has_mask_secrets_true",
    "gpu_visible_plaintext_contains_input_ids",
    "leaked_secret_fields_contains_mask_seed",
    "raw_lora_tensor_name_in_package",
    "optimizer_state_in_package",
    "training_data_in_package",
    "boundary_artifact_hash_mismatch",
    "base_folded_manifest_hash_mismatch",
    "folded_lora_base_manifest_hash_mismatch",
    "attestation_report_data_mismatch",
    "attestation_mr_td_mismatch",
    "stale_attestation_evidence_changed_runtime_hash",
    "transcript_labels_in_boundary_to_worker",
    "transcript_recovered_logits_in_worker_to_boundary",
}


def test_all_14_negative_cases_detected() -> None:
    rep = run_all_negative_cases()
    assert rep["num_cases"] == 14
    names = {c["negative_test_name"] for c in rep["cases"]}
    assert names == EXPECTED
    failing = [c for c in rep["cases"] if not c["pass"]]
    assert failing == [], "undetected negative cases: %s" % failing
    assert rep["all_passed"] is True


def test_clean_inputs_do_not_trip_detectors() -> None:
    # the guards must NOT fire on legitimate inputs (no false positives)
    clean = {"worker_has_mask_secrets": False, "worker_has_raw_lora": False,
             "tee_used_on_gpu": False, "gpu_visible_plaintext_fields": [],
             "leaked_secret_fields": []}
    assert audit_decode_report(clean)["audit_passed"] is True
    assert scan_lora_package_names(
        ["wq_tilde", "q_proj_lora_a_tilde", "q_proj_lora_b_tilde"])["clean"]
    assert check_lora_meta_flags(
        {"contains_raw_lora": False, "contains_optimizer_state": False,
         "contains_training_data": False, "contains_mask_secrets": False})[
        "clean"]


def test_negative_tests_script(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "snt", REPO_ROOT / "scripts" / "run_security_negative_tests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    js = tmp_path / "neg.json"
    old = sys.argv
    try:
        sys.argv = ["x", "--output-json", str(js),
                    "--output-md", str(tmp_path / "neg.md")]
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc == 0
    rep = json.loads(js.read_text())
    assert rep["all_passed"] is True
    assert rep["num_pass"] == 14
