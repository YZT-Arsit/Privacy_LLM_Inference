"""Gate 1.5-B / Gate 2 §3 — external challenge-response attestation verifier.

Tests the BINDING + measurement-policy logic locally (report_data / nonce /
runtime-hash / gradient_convention / optimizer_mode / guest-ECDH-pubkey bindings
and the mr_td allowlist + debug policy). The cryptographic quote signature+chain
check is delegated to a platform ``chain_verifier``; without it, attestation stays
False off-guest (honest). Real quote generation + DCAP chain verification is Gate 2.
"""

from __future__ import annotations

import pytest

from pllo.experiments.real_tdx_attestation import (
    AttestationFailClosed,
    ExternalAttestationVerifier,
    MeasurementPolicy,
    compute_report_data,
)

_MR = "aa11" * 24                      # 96 hex placeholder mr_td
_GUEST_PUB = "bb" * 32


def _challenge_and_evidence(runtime_hash="ab" * 32, config_digest="cd" * 32,
                            run_id="run1", model_id="Qwen2.5-0.5B",
                            gradient_convention="nout_dual",
                            optimizer_mode="gpu_masked_sgd", guest_pub=_GUEST_PUB,
                            mr_td=_MR, debug=False, policy=None):
    v = ExternalAttestationVerifier(policy=policy or MeasurementPolicy(
        allowed_mr_td=frozenset({_MR})))
    ch = v.issue_challenge(run_id=run_id, model_id=model_id, config_digest=config_digest,
                           expected_runtime_hash=runtime_hash,
                           gradient_convention=gradient_convention,
                           optimizer_mode=optimizer_mode)
    rd = compute_report_data(runtime_hash_hex=runtime_hash, config_digest=config_digest,
                             run_id=run_id, model_id=model_id,
                             gradient_convention=gradient_convention,
                             optimizer_mode=optimizer_mode,
                             guest_ephemeral_public_hex=guest_pub,
                             verifier_nonce=ch.verifier_nonce)
    ev = {"report_data_hex": rd, "runtime_hash_hex": runtime_hash, "mr_td": mr_td,
          "debug": debug, "quote_bytes": b"...", "quote_hash": "qh" * 32,
          "nonce": ch.verifier_nonce, "run_id": run_id, "model_id": model_id,
          "config_digest": config_digest, "gradient_convention": gradient_convention,
          "optimizer_mode": optimizer_mode, "guest_ephemeral_public_hex": guest_pub}
    return v, ch, ev


def test_report_data_is_64_bytes():
    rd = compute_report_data(runtime_hash_hex="ab" * 32, config_digest="cd" * 32,
                             run_id="r", model_id="m", gradient_convention="nout_dual",
                             optimizer_mode="gpu_masked_sgd",
                             guest_ephemeral_public_hex=_GUEST_PUB, verifier_nonce="ff" * 32)
    assert len(bytes.fromhex(rd)) == 64          # exact TDX report_data width


def test_binding_passes_but_unverified_without_chain():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch)                 # no chain_verifier
    assert res["report_data_match"] and res["runtime_hash_bound"] and res["nonce_fresh"]
    assert res["measurement_policy_passed"] is True
    assert res["attestation_verified"] is False               # honest: chain not checked
    assert res["tee_type"] == "unverified"


def test_full_verification_with_chain_verifier():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
    assert res["attestation_verified"] is True and res["tee_type"] == "tdx"
    assert res["quote_chain_verified"] is True and res["measurement_policy_passed"]


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
    ev["report_data_hex"] = "ff" * 64          # not bound to runtime_hash+nonce+ecdh
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


def test_optimizer_mode_swap_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["optimizer_mode"] = "trusted_adamw"     # swap the frozen mode
    with pytest.raises(AttestationFailClosed, match="bound field"):
        v.verify(evidence=ev, challenge=ch)


def test_gradient_convention_swap_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev["gradient_convention"] = "independent_mout"
    with pytest.raises(AttestationFailClosed, match="bound field"):
        v.verify(evidence=ev, challenge=ch)


def test_missing_guest_pubkey_rejected():
    v, ch, ev = _challenge_and_evidence()
    ev.pop("guest_ephemeral_public_hex")
    with pytest.raises(AttestationFailClosed, match="guest ephemeral"):
        v.verify(evidence=ev, challenge=ch)


def test_swapped_guest_pubkey_breaks_binding():
    # a MITM swapping the guest ECDH pubkey breaks report_data binding
    v, ch, ev = _challenge_and_evidence()
    ev["guest_ephemeral_public_hex"] = "cc" * 32
    with pytest.raises(AttestationFailClosed, match="report_data"):
        v.verify(evidence=ev, challenge=ch)


def test_debug_mode_denied_by_policy():
    v, ch, ev = _challenge_and_evidence(debug=True)
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
    assert res["measurement_policy_passed"] is False
    assert res["attestation_verified"] is False and res["debug_mode"] is True


def test_mr_td_not_in_allowlist_denied():
    v, ch, ev = _challenge_and_evidence(mr_td="dead" * 24)
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
    assert res["measurement_policy_passed"] is False
    assert res["attestation_verified"] is False


def test_chain_verifier_failure_leaves_unverified():
    v, ch, ev = _challenge_and_evidence()
    res = v.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: False)
    assert res["attestation_verified"] is False
