"""Tests for the non-measured length-hiding GPU-channel audit wrapper.

Confirms the extended forbidden-name enforcement (token_ids / plaintext_logits /
dummy_token_id / eos_decision / finish_reason / pad / inverse / prg_seed / ...)
catches leaks the canonical substring set misses, while real masked GPU requests
(masked_embedding + public metadata) and folded ``*_tilde`` tensors still pass --
in BOTH default and strict length-hiding modes. The canonical scanner is unchanged
(attestation hash preserved); this wrapper only ADDS names.

Run: python -m pytest tests/test_length_hiding_audit.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security.length_hiding_audit import (  # noqa: E402
    EXTENDED_FORBIDDEN_GPU_VISIBLE,
    audit_gpu_request_payloads,
    forbidden_names_in_payload,
    scan_length_hiding_transcript,
)
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    MaskedDecodeRequest, MaskedPrefillRequest)
from pllo.protocol.wire import encode_message  # noqa: E402


def _entry(direction, *, tensors=None, meta_keys=None, seq=0):
    return {"direction": direction, "seq": seq, "message_type": "MaskedDecode",
            "tensor_specs": [{"name": n} for n in (tensors or [])],
            "public_metadata_keys": list(meta_keys or [])}


# ---- extended forbidden names not covered by the canonical set --------------

def test_extended_set_adds_length_hiding_names() -> None:
    for name in ("token_ids", "plaintext_logits", "dummy_token_id",
                 "eos_decision", "finish_reason", "pad", "inverse", "prg_seed",
                 "generated_token_history"):
        assert name in EXTENDED_FORBIDDEN_GPU_VISIBLE


def test_payload_names_catch_new_forbidden() -> None:
    for bad in ("token_ids", "plaintext_logits", "dummy_token_id", "eos_decision",
                "finish_reason", "generated_token_history", "prg_seed", "inverse"):
        got = forbidden_names_in_payload({bad: 1, "session_id": "x"})
        assert got, "%s should be flagged" % bad


def test_payload_clean_masked_request_passes() -> None:
    payload = {"session_id": "e9", "masked_embedding": "b64...",
               "position": 7, "step": 3, "batch_size": 1, "seq_len": 16}
    assert forbidden_names_in_payload(payload) == []


def test_payload_allows_tilde_tensors() -> None:
    assert forbidden_names_in_payload({"a_tilde": 1, "wq_tilde": 2}) == []


def test_real_encoded_requests_are_clean() -> None:
    import numpy as np
    pre = MaskedPrefillRequest(session_id="e9",
                               masked_embeddings=np.zeros((1, 4, 4), "float32"),
                               positions=[0, 1, 2, 3], batch_size=1, seq_len=4)
    dec = MaskedDecodeRequest(session_id="e9",
                              masked_embedding=np.zeros((1, 1, 4), "float32"),
                              position=4, step=1)
    rep = audit_gpu_request_payloads([encode_message(pre), encode_message(dec)])
    assert rep["fail"] is False
    assert rep["forbidden_fields_found"] == []


def test_malicious_payload_fails_audit() -> None:
    rep = audit_gpu_request_payloads([
        {"session_id": "e9", "masked_embedding": "b64"},          # clean
        {"session_id": "e9", "token_ids": [1, 2, 3]},             # leak
    ])
    assert rep["fail"] is True
    assert "token_ids" in rep["forbidden_fields_found"]
    assert rep["per_payload"][0]["index"] == 1


# ---- transcript scanning (canonical + extra) ------------------------------

def test_clean_length_hiding_transcript_passes() -> None:
    entries = [
        _entry("boundary_to_worker", tensors=["masked_embedding"],
               meta_keys=["session_id", "position", "step"]),
        _entry("worker_to_boundary", tensors=["masked_logits"],
               meta_keys=["session_id", "step"]),
    ]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is False
    assert rep["forbidden_fields_found"] == []


def test_extra_forbidden_tensor_is_caught() -> None:
    entries = [_entry("boundary_to_worker", tensors=["token_ids"])]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is True
    assert "token_ids" in rep["forbidden_fields_found"]
    assert rep["extra_leaks"] and rep["extra_leaks"][0]["field"] == "token_ids"


def test_dummy_token_id_in_transcript_is_caught() -> None:
    entries = [_entry("boundary_to_worker", meta_keys=["dummy_token_id"])]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is True
    assert "dummy_token_id" in rep["forbidden_fields_found"]


def test_canonical_leak_still_caught_via_wrapper() -> None:
    entries = [_entry("boundary_to_worker", tensors=["input_ids"])]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is True
    assert any(l["matched_forbidden"] == "input_ids"
               for l in rep["canonical_leaks"])


def test_trusted_side_directions_not_scanned() -> None:
    # a forbidden name on a NON-GPU-visible direction is out of scope
    entries = [_entry("trusted_internal", tensors=["plaintext_logits"])]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is False


def test_tilde_tensor_allowed_in_transcript() -> None:
    entries = [_entry("boundary_to_worker", tensors=["a_tilde", "wq_tilde"])]
    rep = scan_length_hiding_transcript(entries)
    assert rep["fail"] is False
