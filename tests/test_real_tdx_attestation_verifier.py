"""Gate 1.5-B — external challenge-response attestation verifier (fail-closed).

Tests the BINDING logic locally (report_data / nonce / runtime-hash / field bindings).
The cryptographic quote signature+measurement check is delegated to a platform
`chain_verifier`; without a real quote it stays False, so `attestation_verified` is
False off-guest (honest). Real quote generation + chain verification is Gate 2.
"""

from __future__ import annotations

import pytest

from pllo.experiments.real_tdx_attestation import (
    AttestationFailClosed,
    ExternalAttestationVerifier,
    compute_report_data,
)


def _challenge_and_evidence(runtime_hash="ab" * 32, config_digest="cd" * 32,
                            run_id="run1", model_id="Qwen2.5-0.5B"):
    v = ExternalAttestationVerifier()
    ch = v.issue_challenge(run_id=run_id, model_id=model_id, config_digest=config_digest,
                           expected_runtime_hash=runtime_hash)
    rd = compute_report_data(runtime_hash_hex=runtime_hash, config_digest=config_digest,
                             run_id=run_id, model_id=model_id, verifier_nonce=ch.verifier_nonce)
    ev = {"report_data_hex": rd, "runtime_hash_hex": runtime_hash, "mr_td": "mrtd",
          "quote_bytes": b"...", "nonce": ch.verifier_nonce, "run_id": run_id,
          "model_id": model_id, "config_digest": config_digest}
    return v, ch, ev


def test_binding_passes_but_unverified_without_chain():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch)                 # no chain_verifier
    assert res["report_data_match"] and res["runtime_hash_bound"] and res["nonce_fresh"]
    assert res["attestation_verified"] is False               # honest: chain not checked
    assert res["tee_type"] == "unverified"


def test_full_verification_with_chain_verifier():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
    assert res["attestation_verified"] is True and res["tee_type"] == "tdx"
    assert res["quote_chain_verified"] is True


def test_nonce_replay_rejected():
    v, ch, ev = _challenge_and_evidence()
    v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
    with pytest.raises(AttestationFailClosed, match="replay"):
        v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)


def test_wrong_nonce_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["nonce"] = "00" * 32
    with pytest.raises(AttestationFailClosed, match="nonce"):
        v.verify(evidence=ev, challenge=ch)


def test_report_data_not_bound_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["report_data_hex"] = "ff" * 32          # not bound to runtime_hash+nonce
    with pytest.raises(AttestationFailClosed, match="report_data"):
        v.verify(evidence=ev, challenge=ch)


def test_runtime_hash_mismatch_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["runtime_hash_hex"] = "00" * 32
    with pytest.raises(AttestationFailClosed, match="runtime hash"):
        v.verify(evidence=ev, challenge=ch)


def test_bound_field_mismatch_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["model_id"] = "some-other-model"        # attacker swaps the model id
    with pytest.raises(AttestationFailClosed, match="bound field"):
        v.verify(evidence=ev, challenge=ch)


def test_chain_verifier_failure_leaves_unverified():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: False)
    assert res["attestation_verified"] is False
