"""Small authenticated tensor RPC used only by the adapted baseline."""

from __future__ import annotations

import hashlib
import hmac
import io
import socket
import struct
import threading
import time
from typing import Any

import torch

MAX_FRAME = 8 * 1024 * 1024 * 1024


def _encode(payload: dict[str, Any], secret: bytes) -> bytes:
    buffer = io.BytesIO(); torch.save(payload, buffer)
    body = buffer.getvalue()
    tag = hmac.new(secret, body, hashlib.sha256).digest()
    return tag + body


def _decode(frame: bytes, secret: bytes) -> dict[str, Any]:
    if len(frame) < 32:
        raise ValueError("short authenticated frame")
    tag, body = frame[:32], frame[32:]
    if not hmac.compare_digest(tag, hmac.new(secret, body, hashlib.sha256).digest()):
        raise ValueError("RPC authentication failed")
    value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("RPC payload must be a dictionary")
    return value


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 16 * 1024 * 1024))
        if not chunk:
            raise EOFError("RPC peer closed connection")
        chunks.append(chunk); remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, payload: dict[str, Any], secret: bytes) -> int:
    frame = _encode(payload, secret)
    if len(frame) > MAX_FRAME:
        raise ValueError("RPC frame exceeds safety limit")
    sock.sendall(struct.pack("!Q", len(frame)) + frame)
    return len(frame) + 8


def receive_frame(sock: socket.socket, secret: bytes) -> tuple[dict[str, Any], int]:
    length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if length > MAX_FRAME:
        raise ValueError("RPC frame exceeds safety limit")
    return _decode(_read_exact(sock, length), secret), length + 8


class RPCClient:
    def __init__(self, host: str, port: int, secret_hex: str, *, timeout: float = 300):
        self.secret = bytes.fromhex(secret_hex)
        if len(self.secret) != 32:
            raise ValueError("RPC secret must be 32 bytes")
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)
        self.lock = threading.Lock()
        self.request_id = 0
        self.bytes_sent = 0; self.bytes_received = 0; self.round_trips = 0; self.transport_ns = 0

    def call(self, method: str, **arguments):
        with self.lock:
            self.request_id += 1
            started = time.perf_counter_ns()
            self.bytes_sent += send_frame(self.sock, {"id": self.request_id, "method": method, "args": arguments}, self.secret)
            response, received = receive_frame(self.sock, self.secret)
            self.bytes_received += received; self.round_trips += 1
            self.transport_ns += time.perf_counter_ns() - started
        if response.get("id") != self.request_id:
            raise RuntimeError("RPC response id mismatch")
        if not response.get("ok"):
            raise RuntimeError(f"TDX RPC error: {response.get('error')}")
        return response.get("result")

    def close(self) -> None:
        try: self.call("shutdown_connection")
        except Exception: pass
        self.sock.close()
