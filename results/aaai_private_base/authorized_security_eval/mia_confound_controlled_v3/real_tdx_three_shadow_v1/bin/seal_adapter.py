#!/usr/bin/env python3
"""Seal/unseal a frozen adapter for ciphertext-only A10 staging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

MAGIC = b"MIA3ADAPTERv1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seal", "unseal"])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--aad", required=True)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite output")
    aad = args.aad.encode()
    if args.mode == "seal":
        if args.key.exists():
            raise RuntimeError("refusing to overwrite key")
        key, nonce = os.urandom(32), os.urandom(12)
        plaintext = args.input.read_bytes()
        args.key.write_bytes(key)
        args.key.chmod(0o600)
        args.output.write_bytes(MAGIC + nonce + ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad))
        report = {"schema": "mia3_adapter_chacha20poly1305_v1", "status": "PASS",
                  "aad": args.aad, "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
                  "ciphertext_sha256": sha(args.output), "plaintext_bytes": len(plaintext),
                  "ciphertext_bytes": args.output.stat().st_size}
        if args.manifest:
            if args.manifest.exists():
                raise RuntimeError("refusing to overwrite manifest")
            args.manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    else:
        blob = args.input.read_bytes()
        if not blob.startswith(MAGIC):
            raise RuntimeError("sealed adapter magic mismatch")
        nonce, ciphertext = blob[len(MAGIC):len(MAGIC)+12], blob[len(MAGIC)+12:]
        plaintext = ChaCha20Poly1305(args.key.read_bytes()).decrypt(nonce, ciphertext, aad)
        args.output.write_bytes(plaintext)
        print(json.dumps({"status": "PASS", "plaintext_sha256": sha(args.output),
                          "ciphertext_sha256": sha(args.input)}))


if __name__ == "__main__":
    main()
