#!/usr/bin/env python3
"""One-shot private-VPC receiver for an already AEAD-encrypted staging blob."""
from __future__ import annotations

import argparse
import hashlib
import socket
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-bytes", type=int, required=True)
    ap.add_argument("--expected-sha256", required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite ciphertext receiver output")
    digest, total = hashlib.sha256(), 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        conn, _ = server.accept()
        with conn, args.output.open("xb") as handle:
            while True:
                block = conn.recv(1 << 20)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                total += len(block)
    actual = digest.hexdigest()
    print(f"RECEIVED bytes={total} sha256={actual}", flush=True)
    if total != args.expected_bytes or actual != args.expected_sha256:
        raise RuntimeError("received ciphertext integrity mismatch")


if __name__ == "__main__":
    main()
