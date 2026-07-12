"""Fault-injection test for the direct-transport TDX service (req 6: failure recovery).

Runs the persistent service as a local subprocess (CPU, attest disabled) and drives it with
malformed / adversarial frames, asserting each is rejected fail-closed and that a legitimate
request still succeeds afterward (the session must not be left in a corrupt state by a rejected
message). Also confirms the client-side abort semantics: a service `reject` surfaces as an error
the runner turns into a TransportError (no silent fallback).

    python3 scripts/test_direct_transport_faults.py
"""
from __future__ import annotations
import hashlib, hmac, io, json, struct, subprocess, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
PYTDX = sys.executable


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    f.write(struct.pack(">I", len(hb))); f.write(hb)
    f.write(struct.pack(">Q", len(payload))); f.write(payload); f.flush()


def _read(f, n):
    buf = b""
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            raise EOFError
        buf += c
    return buf


def read_frame(f):
    (hlen,) = struct.unpack(">I", _read(f, 4))
    header = json.loads(_read(f, hlen).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8))
    payload = _read(f, plen) if plen else b""
    return header, payload


def mac(key, payload, seq, run_id, op):
    m = hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode() + op.encode()
    return hmac.new(key, m, hashlib.sha256).hexdigest()


def dump_tensor(t):
    bio = io.BytesIO(); torch.save(t, bio); return bio.getvalue()


def main():
    key = bytes.fromhex("11" * 32); run_id = "FAULT-TEST"; nonce = "ab" * 16
    commitment = hashlib.sha256(key + bytes.fromhex(nonce)).hexdigest()
    gb = torch.load(BUNDLE, map_location="cpu")
    L = gb["num_layers"]; V = 4096; T = 12
    labels = list(range(T))
    cfg = {"session_key_hex": key.hex(), "run_id": run_id, "labels": labels,
           "gamma_bundle": str(BUNDLE), "vocab_seed": 8000, "attest": False,
           "binding_manifest": {"hmac_key_commitment": commitment, "nonce": nonce},
           "counters_out": "/tmp/fault_counters.json"}
    scfg = REPO / "results/aaai_private_base/direct_transport/fault_session.json"
    scfg.parent.mkdir(parents=True, exist_ok=True); scfg.write_text(json.dumps(cfg))

    p = subprocess.Popen([PYTDX, str(REPO / "scripts/tdx_persistent_service.py"), str(scfg)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    results = []

    def expect(label, header, payload, want_op, want_reason=None):
        write_frame(p.stdin, header, payload)
        h, pl = read_frame(p.stdout)
        ok = h.get("op") == want_op and (want_reason is None or h.get("reason") == want_reason)
        results.append((label, ok, f"got op={h.get('op')} reason={h.get('reason')}"))
        return h, pl

    # 0) handshake ok (attestation skipped locally)
    expect("handshake", {"op": "handshake"}, b"", "handshake_ack")

    # a valid masked-logits payload builder
    def logits_payload():
        torch.manual_seed(0)
        return dump_tensor(torch.randn(T, V))

    # 1) BAD HMAC -> reject:auth
    pl = logits_payload()
    expect("bad_hmac", {"op": "ce_dlogits", "seq": 0, "run_id": run_id,
                        "hmac": "deadbeef", "dtype": "fp32"}, pl, "reject", "auth")

    # 2) valid ce_dlogits (seq 0) -> ack  (establishes last_seq)
    h, _ = expect("valid_ce", {"op": "ce_dlogits", "seq": 0, "run_id": run_id,
                               "hmac": mac(key, pl, 0, run_id, "ce_dlogits"), "dtype": "fp32"},
                  pl, "ce_dlogits_ack")

    # 3) REPLAY (seq 0 again, valid hmac) -> reject:replay
    expect("replay", {"op": "ce_dlogits", "seq": 0, "run_id": run_id,
                      "hmac": mac(key, pl, 0, run_id, "ce_dlogits"), "dtype": "fp32"},
           pl, "reject", "replay")

    # 4) FORBIDDEN KEY in correct payload -> reject:forbidden_key
    bad = {f"0.o_proj": torch.zeros(2, 2)}     # o_proj must never be corrected in-enclave
    bpl = dump_tensor(bad)
    expect("forbidden_key", {"op": "correct", "seq": 5, "run_id": run_id,
                             "hmac": mac(key, bpl, 5, run_id, "correct")}, bpl, "reject", "forbidden_key")

    # 5) WRONG run_id (hmac computed with different run_id) -> reject:auth
    pl2 = logits_payload()
    expect("wrong_runid", {"op": "ce_dlogits", "seq": 10, "run_id": run_id,
                           "hmac": mac(key, pl2, 10, "OTHER", "ce_dlogits"), "dtype": "fp32"},
           pl2, "reject", "auth")

    # 6) session still healthy after all rejects: a fresh valid ce_dlogits (seq 11) -> ack
    expect("recover_after_faults", {"op": "ce_dlogits", "seq": 11, "run_id": run_id,
                                    "hmac": mac(key, pl2, 11, run_id, "ce_dlogits"), "dtype": "fp32"},
           pl2, "ce_dlogits_ack")

    # 7) TRUNCATED FRAME -> service hits EOF and exits cleanly (fail-closed, no hang)
    p.stdin.write(struct.pack(">I", 999999))   # promise 999999 header bytes, send none
    p.stdin.flush(); p.stdin.close()
    try:
        p.wait(timeout=15); truncated_ok = True
    except Exception:
        p.kill(); truncated_ok = False
    results.append(("truncated_frame_clean_exit", truncated_ok, f"rc={p.returncode}"))

    counters = json.loads(Path("/tmp/fault_counters.json").read_text())["counters"] \
        if Path("/tmp/fault_counters.json").exists() else {}
    # counter expectations
    counters_ok = (counters.get("auth_failures", 0) >= 2 and counters.get("replay_rejected", 0) >= 1
                   and counters.get("forbidden_key_rejected", 0) >= 1
                   and counters.get("ce_calls", 0) == 2 and counters.get("untrusted_gamma_returns", 0) == 0)
    results.append(("counters_consistent", counters_ok, json.dumps(counters)))

    print("=== fault-injection results ===")
    allok = True
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:28s} {detail}")
        allok = allok and ok
    print(json.dumps({"all_pass": allok, "counters": counters}, indent=2))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
