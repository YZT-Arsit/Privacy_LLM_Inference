"""Minimal TDX-side transport benchmark service (NO torch / gamma / attestation).

Isolates PURE transport of the direct channel: same binary framing + optional HMAC as the real
service, but the payload is opaque bytes (no tensor serialization, no model). Ops:
  handshake        -> ack
  source {nbytes}  -> returns an nbytes opaque payload (measures TDX->H800 download)
  sink             -> reads payload, returns tiny ack (measures H800->TDX upload)
  close            -> exit
Reads with large buffered reads; writes each frame with ONE write + ONE flush. Run as an SSH
forced/loose command; H800 drives it over one persistent channel. Benchmark-only, not security path.
"""
from __future__ import annotations
import hashlib, hmac, json, os, struct, sys, time

MAX_MSG = 256 * 1024 * 1024
READ_BUF = 4 * 1024 * 1024        # 4 MB buffered reads


def _read(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(min(n - len(buf), READ_BUF))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def read_frame(f):
    (hlen,) = struct.unpack(">I", _read(f, 4))
    header = json.loads(_read(f, hlen).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8))
    payload = _read(f, plen) if plen else b""
    return header, payload


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    f.write(struct.pack(">I", len(hb)) + hb + struct.pack(">Q", len(payload)))
    if payload:
        f.write(payload)
    f.flush()


def mac(key, payload):
    return hmac.new(key, hashlib.sha256(payload).digest(), hashlib.sha256).hexdigest()


def main():
    key = bytes.fromhex(os.environ.get("BENCH_KEY_HEX", "00" * 32))
    fin = sys.stdin.buffer; fout = sys.stdout.buffer
    ZERO = b"\0" * READ_BUF
    while True:
        try:
            header, payload = read_frame(fin)
        except EOFError:
            break
        op = header.get("op")
        if op == "close":
            break
        if op == "handshake":
            write_frame(fout, {"op": "handshake_ack", "service": "tdx_bench"})
            continue
        if op == "source":
            n = int(header.get("nbytes", 0)); use_hmac = bool(header.get("hmac_on"))
            # build payload without per-chunk python overhead
            body = (ZERO * (n // READ_BUF)) + b"\0" * (n % READ_BUF) if n else b""
            t0 = time.time(); h = {"op": "source_ack"}
            if use_hmac:
                h["hmac"] = mac(key, body)
            h["hmac_us"] = round((time.time() - t0) * 1e6, 1)
            write_frame(fout, h, body)
            continue
        if op == "sink":
            use_hmac = bool(header.get("hmac_on")); ok = True; t0 = time.time()
            if use_hmac:
                ok = (header.get("hmac") == mac(key, payload))
            write_frame(fout, {"op": "sink_ack", "bytes_in": len(payload), "hmac_ok": ok,
                               "hmac_us": round((time.time() - t0) * 1e6, 1)})
            continue
        write_frame(fout, {"op": "reject", "reason": "unknown_op"})


if __name__ == "__main__":
    main()
