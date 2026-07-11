"""Gate 2 §4 — attestation-bound ephemeral key exchange + AEAD session channel.

The Gate 1.5-C session used a *hard-coded* per-session HMAC key handed over some
trusted side-channel. Gate 2 replaces it with an X25519 ECDH exchange whose GUEST
ephemeral public key is bound into the TD Quote's ``report_data``. Because the quote
is attested externally, the verifier knows the ECDH public key it is talking to was
generated inside the measured TD -- so the derived session keys are *attestation
bound*, not merely confidential.

Flow:
  1. verifier issues a fresh nonce (challenge);
  2. guest generates an ephemeral X25519 keypair INSIDE the TD; its public key is
     folded into report_data and the real quote is produced;
  3. verifier cryptographically verifies the quote (DCAP QVL) + all bound fields;
  4. verifier sends ITS ephemeral public key; both sides ECDH -> shared secret;
  5. HKDF derives DISTINCT client->service and service->client keys, salted by the
     quote hash and bound (HKDF info) to run_id, config_digest, protocol version,
     optimizer_mode, gradient_convention;
  6. application messages use AEAD (ChaCha20-Poly1305); the sequence number is the
     AEAD nonce and (run_id, endpoint, seq, length) is authenticated as AAD.

Masked tensors do NOT rely on transport confidentiality for their security (they
are already masked); the AEAD gives message authenticity + anti-replay + defence in
depth, and the attested ECDH gives *identity* (keys provably belong to the TD).

Session keys / shared secrets are NEVER logged or exported. Fail-closed throughout.
"""

from __future__ import annotations

import hmac
import struct
from dataclasses import dataclass, field
from hashlib import sha256

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

PROTOCOL_VERSION = "pllo.train.v1"
_C2S = b"pllo-train-v1 client->service"
_S2C = b"pllo-train-v1 service->client"


class KexError(Exception):
    """Key-exchange / AEAD failure. Fail-closed; the request is refused."""


def _pub_hex(pub: X25519PublicKey) -> str:
    from cryptography.hazmat.primitives import serialization
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _pub_from_hex(hex_str: str) -> X25519PublicKey:
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise KexError(f"bad public key hex: {exc}") from exc
    if len(raw) != 32:
        raise KexError(f"X25519 public key must be 32 bytes, got {len(raw)}")
    return X25519PublicKey.from_public_bytes(raw)


def _kex_context(*, run_id: str, config_digest: str, optimizer_mode: str,
                 gradient_convention: str, quote_hash_hex: str) -> bytes:
    """HKDF `info` binding the derived keys to the exact attested session identity."""
    parts = (PROTOCOL_VERSION, run_id, config_digest, optimizer_mode,
             gradient_convention, quote_hash_hex)
    out = bytearray()
    for p in parts:
        b = p.encode()
        out += struct.pack("<I", len(b)) + b
    return bytes(out)


def _derive_keys(shared_secret: bytes, *, context: bytes, quote_hash_hex: str
                 ) -> tuple[bytes, bytes]:
    """HKDF-SHA256 -> (c2s_key, s2c_key), 32 bytes each, distinct info labels.

    Salt = quote hash so keys are bound to THIS attested quote; a different quote
    (different report_data / different TD) yields unrelated keys."""
    salt = bytes.fromhex(quote_hash_hex) if quote_hash_hex else b""

    def hk(label: bytes) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
                    info=label + b"\x00" + context).derive(shared_secret)

    return hk(_C2S), hk(_S2C)


@dataclass
class KeyExchangeResult:
    c2s_key: bytes
    s2c_key: bytes
    peer_public_hex: str
    quote_hash_hex: str


class GuestKeyExchange:
    """Runs INSIDE the TD. Ephemeral X25519 keypair; the public key is bound into
    report_data before the quote is produced. After the verifier's public key
    arrives, derives the two directional session keys."""

    def __init__(self):
        self._priv = X25519PrivateKey.generate()
        self.public_hex = _pub_hex(self._priv.public_key())

    def complete(self, *, verifier_public_hex: str, run_id: str, config_digest: str,
                 optimizer_mode: str, gradient_convention: str,
                 quote_hash_hex: str) -> KeyExchangeResult:
        peer = _pub_from_hex(verifier_public_hex)
        shared = self._priv.exchange(peer)
        ctx = _kex_context(run_id=run_id, config_digest=config_digest,
                           optimizer_mode=optimizer_mode,
                           gradient_convention=gradient_convention,
                           quote_hash_hex=quote_hash_hex)
        c2s, s2c = _derive_keys(shared, context=ctx, quote_hash_hex=quote_hash_hex)
        return KeyExchangeResult(c2s, s2c, verifier_public_hex, quote_hash_hex)


class VerifierKeyExchange:
    """Runs OUTSIDE the guest. Only proceeds after the guest's public key has been
    attested (bound in a verified quote's report_data)."""

    def __init__(self):
        self._priv = X25519PrivateKey.generate()
        self.public_hex = _pub_hex(self._priv.public_key())

    def complete(self, *, guest_public_hex: str, run_id: str, config_digest: str,
                 optimizer_mode: str, gradient_convention: str,
                 quote_hash_hex: str) -> KeyExchangeResult:
        peer = _pub_from_hex(guest_public_hex)
        shared = self._priv.exchange(peer)
        ctx = _kex_context(run_id=run_id, config_digest=config_digest,
                           optimizer_mode=optimizer_mode,
                           gradient_convention=gradient_convention,
                           quote_hash_hex=quote_hash_hex)
        c2s, s2c = _derive_keys(shared, context=ctx, quote_hash_hex=quote_hash_hex)
        return KeyExchangeResult(c2s, s2c, guest_public_hex, quote_hash_hex)


# ---------------------------------------------------------------------------
# AEAD application channel (ChaCha20-Poly1305) with monotonic sequence + timeout
# ---------------------------------------------------------------------------
def _nonce(seq: int, direction: int) -> bytes:
    if seq < 0 or seq > 0xFFFFFFFFFFFFFFFF:
        raise KexError("sequence out of range")
    return struct.pack("<I", direction) + struct.pack("<Q", seq)  # 4 + 8 = 12 bytes


def _aad(run_id: str, endpoint: str, seq: int, length: int) -> bytes:
    return (run_id.encode() + b"\x00" + endpoint.encode() + b"\x00"
            + struct.pack("<Q", seq) + struct.pack("<Q", length))


@dataclass
class AeadSender:
    """Client or service SEND side. Monotonic sequence; ChaCha20-Poly1305 AEAD."""
    key: bytes
    run_id: str
    direction: int          # 1 = c2s, 2 = s2c
    _seq: int = 0

    def seal(self, endpoint: str, plaintext: bytes) -> tuple[int, bytes]:
        self._seq += 1
        aead = ChaCha20Poly1305(self.key)
        ct = aead.encrypt(_nonce(self._seq, self.direction), plaintext,
                          _aad(self.run_id, endpoint, self._seq, len(plaintext)))
        return self._seq, ct


@dataclass
class AeadReceiver:
    """Service or client RECEIVE side. Verifies tag + monotonic seq + timeout."""
    key: bytes
    run_id: str
    direction: int
    timeout_s: float = 300.0
    _last_seq: int = 0
    _started_at: float | None = None

    def open(self, *, endpoint: str, seq: int, ciphertext: bytes, plaintext_len: int,
             now: float) -> bytes:
        if self._started_at is None:
            self._started_at = now
        if now - self._started_at > self.timeout_s:
            raise KexError("session expired")
        if not isinstance(seq, int) or seq <= self._last_seq:
            raise KexError(f"non-monotonic/replayed sequence {seq} (last {self._last_seq})")
        aead = ChaCha20Poly1305(self.key)
        try:
            pt = aead.decrypt(_nonce(seq, self.direction), ciphertext,
                              _aad(self.run_id, endpoint, seq, plaintext_len))
        except Exception as exc:                       # invalid tag / tamper
            raise KexError("AEAD authentication failed") from exc
        self._last_seq = seq
        return pt

    def close(self) -> None:
        self.key = b""
        self._last_seq = 0


def quote_hash_hex(quote_bytes: bytes) -> str:
    """Stable hash of the raw TD Quote used as the ECDH salt + session identity."""
    return sha256(quote_bytes).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a), str(b))
