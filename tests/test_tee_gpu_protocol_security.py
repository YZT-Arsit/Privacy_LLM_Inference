"""Security tests for the TEE-boundary <-> untrusted-GPU-worker protocol.

Asserts the confidentiality contract against the *exact* messages recorded as
crossing to the untrusted GPU worker. numpy only; the GPU worker runs in a
separate spawn process, the boundary runtime in-process (simulated) for speed.

Run: python -m pytest tests/test_tee_gpu_protocol_security.py -q
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pllo.protocol import (
    BoundaryInitRequest,
    MaskedDecodeRequest,
    MaskedPrefillRequest,
    Qwen7BGpuBackend,
    assert_no_gpu_visible_plaintext,
    assert_no_mask_secret_leak,
    assert_wrong_mask_recovery_fails,
    run_protocol,
)
from pllo.protocol.tee_gpu_messages import (
    GPU_INBOUND_TYPES,
    GPU_OUTBOUND_TYPES,
)

PROMPT = "Explain why privacy matters in private LLM inference systems."
SEED = 99991


@pytest.fixture(scope="module")
def run():
    return run_protocol(
        PROMPT, boundary_backend="simulated", gpu_backend="mock",
        max_new_tokens=5, hidden_size=64, vocab_size=500, seq_len=8, seed=SEED)


# 1.
def test_gpu_messages_contain_no_raw_prompt(run) -> None:
    trace = run["trace"]
    msgs = trace.gpu_inbound_messages + trace.gpu_outbound_messages
    assert msgs                                            # something crossed
    for m in msgs:
        for f in dataclasses.fields(m):
            v = getattr(m, f.name)
            if isinstance(v, str):
                assert PROMPT not in v
            if isinstance(v, dict):
                assert PROMPT not in repr(v)
    findings = assert_no_gpu_visible_plaintext(
        trace, raw_prompt=PROMPT, raise_on_fail=False)
    assert findings == [], findings


# 2.
def test_gpu_messages_contain_no_input_ids(run) -> None:
    findings = assert_no_gpu_visible_plaintext(
        run["trace"], input_ids=run["input_ids"], raise_on_fail=False)
    assert findings == [], findings


# 3.
def test_gpu_messages_contain_no_generated_token_ids(run) -> None:
    gen = run["generated_token_ids"]
    assert gen.size > 0
    findings = assert_no_gpu_visible_plaintext(
        run["trace"], generated_token_ids=gen, raise_on_fail=False)
    assert findings == [], findings


# 4.
def test_gpu_messages_contain_no_recovered_logits(run) -> None:
    # recovered logits are produced trusted-side and never put on the channel.
    rec0 = run["plaintext_logits_first"]                  # same shape as recovered
    findings = assert_no_gpu_visible_plaintext(
        run["trace"], recovered_logits=rec0, raise_on_fail=False)
    assert findings == [], findings
    # structurally: every GPU message is an allowed (masked/public) type
    for m in run["trace"].gpu_inbound_messages:
        assert isinstance(m, GPU_INBOUND_TYPES)
    for m in run["trace"].gpu_outbound_messages:
        assert isinstance(m, GPU_OUTBOUND_TYPES)


# 5.
def test_gpu_messages_contain_no_mask_secrets(run) -> None:
    findings = assert_no_mask_secret_leak(
        run["trace"], run["handles"], raise_on_fail=False)
    assert findings == [], findings


# 6.
def test_wrong_mask_recovery_fails(run) -> None:
    m = assert_wrong_mask_recovery_fails(
        run["masked_logits_first"], run["handles"], run["wrong_handles"],
        run["plaintext_logits_first"], raise_on_fail=False)
    assert m["correct_recovers_plaintext"] is True
    assert m["wrong_mask_diverges"] is True
    assert m["wrong_token_match"] is False
    assert m["findings"] == []
    assert m["wrong_max_abs_err"] > m["correct_max_abs_err"]


# 7.
def test_boundary_call_counts_reported(run) -> None:
    bc = run["trace"].boundary_calls
    # prefill embeds once + one embed per decoded token after the first
    assert bc["embed_and_mask"] == 1 + (5 - 1)
    assert bc["recover_logits"] == 5
    assert bc["sample"] == 5
    gc = run["trace"].gpu_calls
    assert gc["BoundaryInitRequest"] == 1
    assert gc["MaskedPrefillRequest"] == 1
    assert gc["MaskedDecodeRequest"] == 5 - 1
    assert run["trace"].gpu_bytes > 0
    assert run["trace"].trusted_bytes > 0


# 8.
def test_qwen7b_backend_reports_tee_used_false() -> None:
    be = Qwen7BGpuBackend(model_path="/fake/path", device="cuda",
                          dtype="bfloat16")
    assert be.tee_used is False
    assert be.describe()["tee_used"] is False
    resp = be.init(BoundaryInitRequest(
        session_id="s", hidden_size=3584, vocab_size=152064, num_layers=28,
        dtype="bfloat16", gpu_backend="qwen7b"))
    assert resp.tee_used_on_gpu is False
    assert resp.gpu_backend == "qwen7b"


# --- extra: correctness + the audit actually catches injected leaks --------


def test_recovered_tokens_match_plaintext_reference(run) -> None:
    assert run["tokens_match_reference"] is True
    assert run["trace"].tee_used_on_gpu is False


def test_audit_detects_injected_plaintext_leak(run) -> None:
    trace = run["trace"]
    tampered = dataclasses.replace(trace)
    leaky = MaskedPrefillRequest(
        session_id="s", masked_embeddings=run["input_ids"].astype(np.float32),
        positions=[0], batch_size=1, seq_len=int(run["input_ids"].shape[1]))
    # smuggle the real input_ids onto the channel
    leaky = dataclasses.replace(leaky, masked_embeddings=run["input_ids"])
    tampered.gpu_inbound_messages = list(trace.gpu_inbound_messages) + [leaky]
    findings = assert_no_gpu_visible_plaintext(
        tampered, input_ids=run["input_ids"], raise_on_fail=False)
    assert findings, "audit should flag smuggled input_ids"
    with pytest.raises(AssertionError):
        assert_no_gpu_visible_plaintext(
            tampered, input_ids=run["input_ids"], raise_on_fail=True)


def test_audit_detects_injected_secret_leak(run) -> None:
    trace = run["trace"]
    handles = run["handles"]
    tampered = dataclasses.replace(trace)
    leaky = MaskedDecodeRequest(
        session_id="s", masked_embedding=handles.vocab_perm.astype(np.float64),
        position=0, step=0)
    leaky = dataclasses.replace(leaky, masked_embedding=handles.residual_perm)
    tampered.gpu_inbound_messages = list(trace.gpu_inbound_messages) + [leaky]
    findings = assert_no_mask_secret_leak(
        tampered, handles, raise_on_fail=False)
    assert findings, "audit should flag smuggled residual_perm"
