"""H800-side direct-channel transport benchmark (runs ON H800).

Opens ONE persistent SSH channel to the TDX bench service and measures framed / HMAC throughput
for 1/8/16/32 MB payloads in both directions, with read-path instrumentation (syscall count,
chunk sizes, blocked-read time, framing time, HMAC time). No per-message process restart, no Mac
relay. Emits framed_throughput_results.csv + transport_instrumentation.json.

    python3 scripts/transport_bench.py --tdx root@IP --key /root/.ssh/h800_to_tdx \
        --service-cmd "BENCH_KEY_HEX=<hex> python3 /root/privacy_llm_obfuscation/scripts/tdx_bench_service.py" \
        --out <dir>
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, os, select, struct, subprocess, sys, time
from pathlib import Path

READ_BUF = 4 * 1024 * 1024
SIZES_MB = [1, 8, 16, 32]


class Counter:
    def __init__(self): self.reads = 0; self.bytes = 0; self.blocked = 0.0; self.sizes = []


def make_reader(f, ctr):
    def _read(n):
        buf = bytearray()
        while len(buf) < n:
            t0 = time.time()
            r, _, _ = select.select([f.fileno()], [], [], 60.0)
            ctr.blocked += time.time() - t0
            if not r:
                raise TimeoutError("bench_read_timeout")
            chunk = f.read(min(n - len(buf), READ_BUF))
            if not chunk:
                raise EOFError
            ctr.reads += 1; ctr.bytes += len(chunk); ctr.sizes.append(len(chunk))
            buf += chunk
        return bytes(buf)
    return _read


def read_frame(reader):
    (hlen,) = struct.unpack(">I", reader(4))
    header = json.loads(reader(hlen).decode())
    (plen,) = struct.unpack(">Q", reader(8))
    payload = reader(plen) if plen else b""
    return header, payload


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    t0 = time.time()
    f.write(struct.pack(">I", len(hb)) + hb + struct.pack(">Q", len(payload)))
    if payload:
        f.write(payload)
    f.flush()
    return time.time() - t0


def macf(key, payload):
    return hmac.new(key, hashlib.sha256(payload).digest(), hashlib.sha256).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdx", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--key-hex", default="00" * 32)
    ap.add_argument("--out", default="results/aaai_private_base/direct_transport")
    args = ap.parse_args()
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    key = bytes.fromhex(args.key_hex)

    p = subprocess.Popen(["ssh", "-i", args.key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                          "-o", "ServerAliveInterval=15", args.tdx, args.service_cmd],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    ctr = Counter(); reader = make_reader(p.stdout, ctr)
    write_frame(p.stdin, {"op": "handshake"}); hs, _ = read_frame(reader)
    assert hs.get("op") == "handshake_ack", hs

    rows = []
    for mb in SIZES_MB:
        n = mb * 1024 * 1024
        for use_hmac in (False, True):
            # ---- download (TDX -> H800): source ----
            c0 = Counter(); rd = make_reader(p.stdout, c0)
            t0 = time.time()
            fr_us = write_frame(p.stdin, {"op": "source", "nbytes": n, "hmac_on": use_hmac})
            h, payload = read_frame(rd)
            dl_wall = time.time() - t0
            hmac_us = 0.0
            if use_hmac:
                th = time.time(); ok = (h.get("hmac") == macf(key, payload)); hmac_us = (time.time() - th) * 1e6
            assert len(payload) == n, (len(payload), n)
            rows.append({"dir": "download_TDX_to_H800", "MB": mb, "hmac": use_hmac,
                         "wall_s": round(dl_wall, 4), "throughput_MBps": round(mb / dl_wall, 3),
                         "reads": c0.reads, "mean_read_KB": round((sum(c0.sizes)/len(c0.sizes))/1024, 1) if c0.sizes else 0,
                         "max_read_KB": round(max(c0.sizes)/1024, 1) if c0.sizes else 0,
                         "blocked_read_s": round(c0.blocked, 4),
                         "client_hmac_us": round(hmac_us, 1), "server_hmac_us": h.get("hmac_us")})
            # ---- upload (H800 -> TDX): sink ----
            body = b"\0" * n
            hh = {"op": "sink", "hmac_on": use_hmac}
            th = time.time()
            if use_hmac: hh["hmac"] = macf(key, body)
            client_mac_us = (time.time() - th) * 1e6
            t0 = time.time(); fr_us = write_frame(p.stdin, hh, body)
            ack, _ = read_frame(reader); ul_wall = time.time() - t0
            assert ack.get("op") == "sink_ack" and ack.get("bytes_in") == n, ack
            rows.append({"dir": "upload_H800_to_TDX", "MB": mb, "hmac": use_hmac,
                         "wall_s": round(ul_wall, 4), "throughput_MBps": round(mb / ul_wall, 3),
                         "reads": 0, "mean_read_KB": 0, "max_read_KB": 0, "blocked_read_s": 0,
                         "client_hmac_us": round(client_mac_us, 1) if use_hmac else 0,
                         "server_hmac_ok_us": ack.get("hmac_us")})
    write_frame(p.stdin, {"op": "close"}); p.stdin.close()
    try: p.wait(timeout=10)
    except Exception: p.kill()

    import csv
    with open(OUT / "framed_throughput_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    inst = {"read_buf_bytes": READ_BUF, "framing": "binary [4B hlen][hdr][8B plen][payload], one flush/frame",
            "rows": rows}
    (OUT / "transport_instrumentation.json").write_text(json.dumps(inst, indent=2))
    print(json.dumps({"framed_throughput": [{k: r[k] for k in ("dir", "MB", "hmac", "throughput_MBps", "blocked_read_s")} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
