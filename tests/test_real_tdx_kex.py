"""Gate 2 §4 — attested ECDH key exchange + AEAD session channel."""

from __future__ import annotations

import pytest

from pllo.experiments.real_tdx_kex import (
    AeadReceiver,
    AeadSender,
    GuestKeyExchange,
    KexError,
    VerifierKeyExchange,
    quote_hash_hex,
)

_CTX = dict(run_id="run1", config_digest="cd" * 32, optimizer_mode="gpu_masked_sgd",
            gradient_convention="nout_dual")


def _establish(qh="00" * 32):
    guest = GuestKeyExchange()
    verifier = VerifierKeyExchange()
    g = guest.complete(verifier_public_hex=verifier.public_hex, quote_hash_hex=qh, **_CTX)
    v = verifier.complete(guest_public_hex=guest.public_hex, quote_hash_hex=qh, **_CTX)
    return g, v


def test_both_sides_derive_identical_keys():
    g, v = _establish()
    assert g.c2s_key == v.c2s_key and g.s2c_key == v.s2c_key
    assert g.c2s_key != g.s2c_key                 # distinct directional keys
    assert len(g.c2s_key) == 32


def test_context_binding_changes_keys():
    # a different quote hash (different attested TD) yields unrelated keys
    g1, _ = _establish(qh="00" * 32)
    g2, _ = _establish(qh="11" * 32)
    assert g1.c2s_key != g2.c2s_key


def test_aead_roundtrip_client_to_service():
    g, v = _establish()
    tx = AeadSender(key=v.c2s_key, run_id="run1", direction=1)      # client sends
    rx = AeadReceiver(key=g.c2s_key, run_id="run1", direction=1)    # service recvs
    seq, ct = tx.seal("/train/logits_loss", b"payload")
    out = rx.open(endpoint="/train/logits_loss", seq=seq, ciphertext=ct,
                  plaintext_len=len(b"payload"), now=0.0)
    assert out == b"payload"


def test_tampered_ciphertext_rejected():
    g, v = _establish()
    tx = AeadSender(key=v.c2s_key, run_id="run1", direction=1)
    rx = AeadReceiver(key=g.c2s_key, run_id="run1", direction=1)
    seq, ct = tx.seal("/train/init", b"hello")
    bad = bytearray(ct); bad[0] ^= 0x01
    with pytest.raises(KexError, match="AEAD"):
        rx.open(endpoint="/train/init", seq=seq, ciphertext=bytes(bad),
                plaintext_len=5, now=0.0)


def test_wrong_endpoint_aad_rejected():
    g, v = _establish()
    tx = AeadSender(key=v.c2s_key, run_id="run1", direction=1)
    rx = AeadReceiver(key=g.c2s_key, run_id="run1", direction=1)
    seq, ct = tx.seal("/train/init", b"hello")
    with pytest.raises(KexError, match="AEAD"):    # AAD endpoint mismatch
        rx.open(endpoint="/train/packed_update", seq=seq, ciphertext=ct,
                plaintext_len=5, now=0.0)


def test_replay_and_reorder_rejected():
    g, v = _establish()
    tx = AeadSender(key=v.c2s_key, run_id="run1", direction=1)
    rx = AeadReceiver(key=g.c2s_key, run_id="run1", direction=1)
    s1, c1 = tx.seal("/train/init", b"a")
    s2, c2 = tx.seal("/train/init", b"b")
    rx.open(endpoint="/train/init", seq=s2, ciphertext=c2, plaintext_len=1, now=0.0)
    with pytest.raises(KexError, match="monotonic|replay"):
        rx.open(endpoint="/train/init", seq=s1, ciphertext=c1, plaintext_len=1, now=1.0)


def test_session_timeout_enforced():
    g, v = _establish()
    tx = AeadSender(key=v.c2s_key, run_id="run1", direction=1)
    rx = AeadReceiver(key=g.c2s_key, run_id="run1", direction=1, timeout_s=10.0)
    s1, c1 = tx.seal("/train/init", b"a")
    rx.open(endpoint="/train/init", seq=s1, ciphertext=c1, plaintext_len=1, now=0.0)
    s2, c2 = tx.seal("/train/init", b"b")
    with pytest.raises(KexError, match="expired"):
        rx.open(endpoint="/train/init", seq=s2, ciphertext=c2, plaintext_len=1, now=11.0)


def test_bad_pubkey_rejected():
    guest = GuestKeyExchange()
    with pytest.raises(KexError, match="public key"):
        guest.complete(verifier_public_hex="00" * 4, quote_hash_hex="00" * 32, **_CTX)


def test_quote_hash_stable():
    assert quote_hash_hex(b"abc") == quote_hash_hex(b"abc")
    assert quote_hash_hex(b"abc") != quote_hash_hex(b"abd")
