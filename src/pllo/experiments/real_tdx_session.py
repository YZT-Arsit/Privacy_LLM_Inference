"""Gate 1.5-C — application-layer session protection for the training boundary.

Defense-in-depth ON TOP of the SSH tunnel. After external attestation verifies, the
guest establishes a per-session key; every request carries a monotonic sequence number
and an HMAC-SHA256 MAC over (run_id || seq || body). The validator enforces: correct
MAC, strictly increasing sequence (replay/reorder rejected), single-use sequence, and a
per-session timeout. Bind localhost by default; remote exposure only via explicit tunnel.
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass, field
from hashlib import sha256


class SessionError(Exception):
    """Session-auth failure. Fail-closed; the request is refused."""


def new_session_key() -> bytes:
    """Guest-generated per-session key (established AFTER verified attestation)."""
    return secrets.token_bytes(32)


def request_mac(key: bytes, run_id: str, seq: int, body: bytes) -> str:
    msg = run_id.encode() + b"\x00" + str(int(seq)).encode() + b"\x00" + body
    return hmac.new(key, msg, sha256).hexdigest()


@dataclass
class SessionSigner:
    """Client side: signs each request with a monotonic sequence."""
    key: bytes
    run_id: str
    _seq: int = 0

    def sign(self, body: bytes) -> tuple[int, str]:
        self._seq += 1
        return self._seq, request_mac(self.key, self.run_id, self._seq, body)


@dataclass
class SessionValidator:
    """Guest side: verifies MAC + monotonic sequence + timeout. Uses a monotonic
    clock supplied by the caller (no wall-clock cross-machine subtraction)."""
    key: bytes
    run_id: str
    timeout_s: float = 300.0
    _last_seq: int = 0
    _started_at: float | None = None
    _seen: set = field(default_factory=set)

    def check(self, *, seq: int, body: bytes, mac: str, now: float) -> None:
        if self._started_at is None:
            self._started_at = now
        if now - self._started_at > self.timeout_s:
            raise SessionError("session expired")
        if not isinstance(seq, int) or seq <= self._last_seq:
            raise SessionError(f"non-monotonic/replayed sequence {seq} (last {self._last_seq})")
        expected = request_mac(self.key, self.run_id, seq, body)
        if not hmac.compare_digest(expected, str(mac)):
            raise SessionError("bad request MAC")
        self._last_seq = seq
        self._seen.add(seq)

    def close(self) -> None:
        """Destroy transient session state."""
        self.key = b""
        self._seen.clear()
        self._last_seq = 0
