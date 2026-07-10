"""Gate 1.5-C — session MAC / monotonic sequence / timeout tests."""

from __future__ import annotations

import pytest

from pllo.experiments.real_tdx_session import (
    SessionError,
    SessionSigner,
    SessionValidator,
    new_session_key,
    request_mac,
)


def _pair(timeout=300.0):
    key = new_session_key()
    return (SessionSigner(key=key, run_id="r"),
            SessionValidator(key=key, run_id="r", timeout_s=timeout))


def test_valid_signed_requests_accepted_in_order():
    s, v = _pair()
    t = 0.0
    for body in (b"a", b"bb", b"ccc"):
        seq, mac = s.sign(body)
        v.check(seq=seq, body=body, mac=mac, now=t); t += 1.0


def test_bad_mac_rejected():
    s, v = _pair()
    seq, mac = s.sign(b"body")
    with pytest.raises(SessionError, match="MAC"):
        v.check(seq=seq, body=b"tampered", mac=mac, now=0.0)


def test_wrong_key_rejected():
    s, v = _pair()
    seq, _ = s.sign(b"body")
    forged = request_mac(new_session_key(), "r", seq, b"body")
    with pytest.raises(SessionError, match="MAC"):
        v.check(seq=seq, body=b"body", mac=forged, now=0.0)


def test_replayed_or_reordered_sequence_rejected():
    s, v = _pair()
    seq1, mac1 = s.sign(b"one")
    seq2, mac2 = s.sign(b"two")
    v.check(seq=seq2, body=b"two", mac=mac2, now=0.0)          # accept seq 2
    with pytest.raises(SessionError, match="monotonic|replay"):
        v.check(seq=seq1, body=b"one", mac=mac1, now=1.0)      # seq 1 now stale
    with pytest.raises(SessionError, match="monotonic|replay"):
        v.check(seq=seq2, body=b"two", mac=mac2, now=1.0)      # replay seq 2


def test_session_timeout_enforced():
    s, v = _pair(timeout=10.0)
    seq, mac = s.sign(b"x")
    v.check(seq=seq, body=b"x", mac=mac, now=0.0)
    seq2, mac2 = s.sign(b"y")
    with pytest.raises(SessionError, match="expired"):
        v.check(seq=seq2, body=b"y", mac=mac2, now=11.0)


def test_close_destroys_state():
    s, v = _pair()
    seq, mac = s.sign(b"x"); v.check(seq=seq, body=b"x", mac=mac, now=0.0)
    v.close()
    assert v.key == b"" and v._last_seq == 0
