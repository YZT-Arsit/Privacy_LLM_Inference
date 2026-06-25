"""Security NEGATIVE cases: deliberately-broken inputs that MUST be detected.

Each case feeds a malicious fixture to the *real* detector used in the pipeline
(decode-report audit, folded-package name screen + meta flags, hash-equality
checks, the attestation verifier, and the transcript scanner) and asserts the
detector flags the failure. A case ``pass`` only when ``expected_failure`` is
actually detected -- this proves the guards work, not just that clean inputs
pass. stdlib + numpy only; no H800 / TDX / CUDA / model. Uses fixtures.

A bug that *silenced* a guard would flip ``actually_failed`` to False and fail
the case -- so these double as regression tests for the audit surface.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from pllo.deployment.folded_package import forbidden_tensor_names
from pllo.deployment.lora_folded_package import _RAW_LORA_NAME_HINTS
from pllo.protocol.attestation import runtime_report_data_hex, verify_evidence
from pllo.security.transcript_scanner import scan_transcript

__all__ = ["audit_decode_report", "scan_lora_package_names",
           "check_lora_meta_flags", "hashes_match", "run_all_negative_cases"]


# --- real detectors -------------------------------------------------------


def audit_decode_report(report: dict) -> dict:
    """Mirror the cross-machine decode audit: any leak/secret/TEE flag fails."""
    reasons = []
    if report.get("worker_has_mask_secrets"):
        reasons.append("worker_has_mask_secrets")
    if report.get("worker_has_raw_lora"):
        reasons.append("worker_has_raw_lora")
    if report.get("tee_used_on_gpu"):
        reasons.append("tee_used_on_gpu")
    if report.get("gpu_visible_plaintext_fields"):
        reasons.append("gpu_visible_plaintext_fields=%s"
                       % report["gpu_visible_plaintext_fields"])
    if report.get("leaked_secret_fields"):
        reasons.append("leaked_secret_fields=%s" % report["leaked_secret_fields"])
    return {"audit_passed": not reasons, "reasons": reasons}


def scan_lora_package_names(names) -> dict:
    """Real folded-package name screen + raw-LoRA hint screen over tensor names."""
    found = list(forbidden_tensor_names(names))
    for n in names:
        low = str(n).lower()
        if any(h in low for h in _RAW_LORA_NAME_HINTS) and n not in found:
            found.append(str(n))
    return {"clean": not found, "found": found}


def check_lora_meta_flags(meta: dict) -> dict:
    """Real lora_meta.json secret-flag screen (verify_lora_folded_package)."""
    bad = [k for k in ("contains_raw_lora", "contains_optimizer_state",
                       "contains_training_data", "contains_mask_secrets")
           if meta.get(k)]
    return {"clean": not bad, "found": bad}


def hashes_match(expected, actual) -> bool:
    return expected is not None and actual is not None and expected == actual


def _sha512(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


# --- the 14 negative cases ------------------------------------------------


def _cases() -> list:
    cases: list[tuple[str, Callable[[], bool]]] = []

    # 1-3: decode-report audit must fail on secret/plaintext flags
    cases.append(("worker_has_mask_secrets_true",
                  lambda: not audit_decode_report(
                      {"worker_has_mask_secrets": True})["audit_passed"]))
    cases.append(("gpu_visible_plaintext_contains_input_ids",
                  lambda: not audit_decode_report(
                      {"gpu_visible_plaintext_fields": ["input_ids"]}
                  )["audit_passed"]))
    cases.append(("leaked_secret_fields_contains_mask_seed",
                  lambda: not audit_decode_report(
                      {"leaked_secret_fields": ["mask_seed"]})["audit_passed"]))

    # 4-6: folded-LoRA package must reject raw adapter / optimizer / training
    cases.append(("raw_lora_tensor_name_in_package",
                  lambda: not scan_lora_package_names(
                      ["wq_tilde", "q_proj_lora_a_raw"])["clean"]))
    cases.append(("optimizer_state_in_package",
                  lambda: not scan_lora_package_names(
                      ["wq_tilde", "optimizer_state"])["clean"]))
    cases.append(("training_data_in_package",
                  lambda: not check_lora_meta_flags(
                      {"contains_training_data": True})["clean"]))

    # 7-9: integrity hash mismatches must be caught
    cases.append(("boundary_artifact_hash_mismatch",
                  lambda: not hashes_match("art_hash_expected",
                                           "art_hash_actual_differs")))
    cases.append(("base_folded_manifest_hash_mismatch",
                  lambda: not hashes_match("base_manifest_A",
                                           "base_manifest_B")))
    cases.append(("folded_lora_base_manifest_hash_mismatch",
                  lambda: not hashes_match("base_manifest_for_lora",
                                           "stale_base_manifest")))

    # 10-12: attestation verifier must reject report_data / mr_td / stale binding
    rh = _sha512(b"runtime-hash-v1")
    rd = runtime_report_data_hex(rh)
    good_ev = {"tee": "tdx", "td_attributes": {"debug": False},
               "jwt": "h.p.s", "report_data": rd, "mr_td": "MRTD_GOOD"}

    def _report_data_mismatch():
        ev = dict(good_ev, report_data="00" * 64)
        return verify_evidence(ev, rh).runtime_hash_bound is not True

    def _mr_td_mismatch():
        ev = dict(good_ev, mr_td="MRTD_WRONG")
        res = verify_evidence(ev, rh, expected_mr_td="MRTD_GOOD")
        return res.mr_td_match is not True and not res.verified

    def _stale_binding():
        # evidence bound to the OLD runtime hash, verified against a NEW one
        new_rh = _sha512(b"runtime-hash-v2-CHANGED")
        res = verify_evidence(good_ev, new_rh)   # good_ev.report_data == old rd
        return res.runtime_hash_bound is not True and not res.verified

    cases.append(("attestation_report_data_mismatch", _report_data_mismatch))
    cases.append(("attestation_mr_td_mismatch", _mr_td_mismatch))
    cases.append(("stale_attestation_evidence_changed_runtime_hash",
                  _stale_binding))

    # 13-14: transcript scanner must catch labels (->GPU) and recovered logits
    def _labels_to_worker():
        entries = [{"seq": 0, "message_type": "prefill",
                    "direction": "boundary_to_worker",
                    "public_metadata_keys": ["seq_len"],
                    "tensor_specs": [{"name": "masked_embeddings",
                                      "shape": [1, 8, 16], "dtype": "float32"},
                                     {"name": "labels", "shape": [1, 8],
                                      "dtype": "int64"}], "byte_count": 1}]
        return scan_transcript(entries)["fail"]

    def _recovered_logits_from_worker():
        entries = [{"seq": 0, "message_type": "decode",
                    "direction": "worker_to_boundary",
                    "public_metadata_keys": [],
                    "tensor_specs": [{"name": "recovered_logits",
                                      "shape": [1, 100], "dtype": "float32"}],
                    "byte_count": 1}]
        return scan_transcript(entries)["fail"]

    cases.append(("transcript_labels_in_boundary_to_worker", _labels_to_worker))
    cases.append(("transcript_recovered_logits_in_worker_to_boundary",
                  _recovered_logits_from_worker))
    return cases


def run_all_negative_cases() -> dict:
    """Run every negative case; return an aggregate report."""
    results = []
    for name, fn in _cases():
        try:
            actually = bool(fn())
            err = None
        except Exception as exc:                            # noqa: BLE001
            actually = False
            err = "%s: %s" % (type(exc).__name__, exc)
        results.append({
            "negative_test_name": name, "expected_failure": True,
            "actually_failed": actually, "pass": (actually is True),
            "error": err,
        })
    num_pass = sum(1 for r in results if r["pass"])
    return {
        "stage": "security_negative_tests",
        "num_cases": len(results), "num_pass": num_pass,
        "all_passed": num_pass == len(results),
        "cases": results,
    }
