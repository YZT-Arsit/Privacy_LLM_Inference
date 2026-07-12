"""Persistent attested TDX boundary service (direct transport).

Runs INSIDE the Intel TDX guest as an SSH FORCED COMMAND: the untrusted H800 opens ONE
persistent SSH channel (port 22 -- the only reachable port) and this service reads framed
requests from stdin and writes framed responses to stdout, looping across MANY training
steps. No per-step process restart, no Mac in the data path, no temp-file ferry.

Ops (one verified session reused across steps):
  handshake     -> fresh attestation evidence + service info (attestation ONCE, not per step)
  ce_dlogits    -> masked logits in  -> CE + masked dlogits out (in-enclave, labels private)
  correct       -> packed masked A-grads in -> gram_inv-corrected grads out (in-enclave)
  close         -> exit

Security: HMAC-SHA256 over each message + monotonic seq (replay reject); bounded message
size; fail-closed on bad tag / seq / oversize / malformed. Holds labels, vocab perm (seed
8000), and the gamma bundle (gram_inv rebuilt in-enclave). Never logs plaintext; never
returns gamma / gram_inv. Session key provisioned by the Mac control-plane at setup.
"""
from __future__ import annotations
import hashlib, hmac, io, json, struct, sys, time
from pathlib import Path

import torch

MAX_MSG = 200 * 1024 * 1024        # 200 MB bound
ATTN = ("q_proj", "k_proj", "v_proj"); MLP = ("gate_proj", "up_proj")
ALLOWED = set(ATTN) | set(MLP)
FORBIDDEN = ("o_proj", "down_proj", "N_inv", "Nr", "gamma", "pad", "token", "input_ids", "label")


def orthogonal_signed_perm(n, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype); N[torch.arange(n), perm] = signs
    return N


def vocab_perm(V, seed=8000):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(V, generator=g)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(V)
    return perm, inv


def build_gram_inv(gb):
    H, L = gb["hidden"], gb["num_layers"]
    Nr = orthogonal_signed_perm(H, gb["nr_seed"], torch.float64)
    attn = {l: Nr.T @ torch.diag(gb["ga"][l].double() ** 2) @ Nr for l in range(L)}
    mlp = {l: Nr.T @ torch.diag(gb["gm"][l].double() ** 2) @ Nr for l in range(L)}
    return attn, mlp


# ---- framing over binary stdin/stdout ----
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
    if hlen > 1 << 20:
        raise ValueError("header_too_large")
    header = json.loads(_read(f, hlen).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8))
    if plen > MAX_MSG:
        raise ValueError("payload_too_large")
    payload = _read(f, plen) if plen else b""
    return header, payload


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    f.write(struct.pack(">I", len(hb))); f.write(hb)
    f.write(struct.pack(">Q", len(payload))); f.write(payload)
    f.flush()


def load_tensor(b):
    return torch.load(io.BytesIO(b), map_location="cpu")


def dump_tensor(t):
    bio = io.BytesIO(); torch.save(t, bio); return bio.getvalue()


def mac(key, payload, seq, run_id, op):
    m = hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode() + op.encode()
    return hmac.new(key, m, hashlib.sha256).hexdigest()


def do_attestation(cfg, out_dir):
    """Fresh attestation at session setup (once). Reuses gate0_d4_attestation."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gate0_d4_attestation import verify_quote_jwt, generate_quote_alibaba, d4_report_data_hex
        manifest = cfg["binding_manifest"]
        rd = d4_report_data_hex(manifest)
        outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
        quote = generate_quote_alibaba(rd, outp, quote_out_name="session_quote.dat")
        parsed = verify_quote_jwt(quote)
        rdx = (parsed.get("tdx_reportdata") or "").lower().removeprefix("0x")
        verified = bool(parsed.get("overall_appraisal_result") == "SUCCESS"
                        and rdx == rd.lower() and parsed.get("debug") is False)
        ev = {"attestation_verified": verified,
              "overall_appraisal_result": parsed.get("overall_appraisal_result"),
              "reportdata_bound": rdx == rd.lower(), "debug_false": parsed.get("debug") is False,
              "mr_td": parsed.get("mr_td"), "report_data": rd, "session_setup_once": True}
        (outp / "session_attestation.json").write_text(json.dumps(ev, indent=2))
        return ev
    except Exception as e:
        return {"attestation_verified": False, "error": repr(e)[:300]}


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else \
        json.loads(Path("/tmp/direct_session.json").read_text())
    key = bytes.fromhex(cfg["session_key_hex"]); run_id = cfg["run_id"]
    labels = torch.tensor(cfg["labels"])
    gb = torch.load(cfg["gamma_bundle"], map_location="cpu")
    gi_attn, gi_mlp = build_gram_inv(gb)
    perm = perm_inv = None
    last_seq = -1
    counters = {"ce_calls": 0, "correct_calls": 0, "auth_failures": 0, "replay_rejected": 0,
                "malformed_rejected": 0, "forbidden_key_rejected": 0,
                "untrusted_gamma_returns": 0, "correction_missing_targets": -1}
    fin = sys.stdin.buffer; fout = sys.stdout.buffer
    att = do_attestation(cfg, cfg.get("attest_out", "/tmp/direct_attest")) if cfg.get("attest") else {"attestation_skipped": True}

    while True:
        try:
            header, payload = read_frame(fin)
        except EOFError:
            break
        op = header.get("op")
        if op == "close":
            break
        if op == "handshake":
            write_frame(fout, {"op": "handshake_ack", "attestation": att, "counters": counters,
                               "service": "tdx_persistent", "run_id": run_id})
            continue
        # authenticated ops
        seq = header.get("seq", -1)
        try:
            if header.get("hmac") != mac(key, payload, seq, run_id, op):
                counters["auth_failures"] += 1
                write_frame(fout, {"op": "reject", "reason": "auth", "seq": seq}); continue
            if seq <= last_seq:
                counters["replay_rejected"] += 1
                write_frame(fout, {"op": "reject", "reason": "replay", "seq": seq}); continue
        except Exception:
            counters["malformed_rejected"] += 1
            write_frame(fout, {"op": "reject", "reason": "malformed"}); continue

        if op == "ce_dlogits":
            t0 = time.time()
            masked = load_tensor(payload).float()
            V = masked.shape[1]
            if perm is None:
                perm, perm_inv = vocab_perm(V, cfg.get("vocab_seed", 8000))
            plain = masked[:, perm]
            lg = plain[:-1]; tg = labels[1:len(plain)]
            loss = torch.nn.functional.cross_entropy(lg, tg)
            probs = torch.softmax(lg, -1); dp = probs.clone()
            dp[torch.arange(tg.shape[0]), tg] -= 1.0; dp /= tg.shape[0]
            dpl = torch.zeros_like(plain); dpl[:-1] = dp
            dmask = dpl[:, perm_inv]
            counters["ce_calls"] += 1; last_seq = seq
            out = dump_tensor(dmask.to(masked.dtype) if header.get("dtype") == "bf16"
                              else dmask)
            resp_seq = seq + 1
            write_frame(fout, {"op": "ce_dlogits_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "ce_dlogits_ack"),
                               "ce_loss": float(loss), "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)

        elif op == "correct":
            t0 = time.time()
            grads = load_tensor(payload)
            seen = set(); corrected = {}
            bad = False
            for k, gA in grads.items():
                l_str, proj = k.split(".", 1); l = int(l_str)
                if proj not in ALLOWED or any(s in k for s in FORBIDDEN):
                    counters["forbidden_key_rejected"] += 1; bad = True; break
                gi = gi_attn[l] if proj in ATTN else gi_mlp[l]
                corrected[k] = (gA.double() @ gi).to(gA.dtype)
                seen.add((l, proj))
            if bad:
                write_frame(fout, {"op": "reject", "reason": "forbidden_key", "seq": seq}); continue
            expected = {(l, p) for l in range(gb["num_layers"]) for p in ALLOWED}
            counters["correction_missing_targets"] = len(expected - seen)
            if seen != expected:
                write_frame(fout, {"op": "reject", "reason": "incomplete", "seq": seq}); continue
            counters["correct_calls"] += 1; last_seq = seq
            out = dump_tensor(corrected)
            resp_seq = seq + 1
            write_frame(fout, {"op": "correct_ack", "seq": resp_seq,
                               "hmac": mac(key, out, resp_seq, run_id, "correct_ack"),
                               "corrected": len(corrected), "missing": counters["correction_missing_targets"],
                               "compute_sec": time.time() - t0,
                               "bytes_in": len(payload), "bytes_out": len(out)}, out)
        else:
            counters["malformed_rejected"] += 1
            write_frame(fout, {"op": "reject", "reason": "unknown_op"})
    # final counters to a file for the Mac control-plane to record
    Path(cfg.get("counters_out", "/tmp/direct_session_counters.json")).write_text(
        json.dumps({"counters": counters, "attestation": att}, indent=2))


if __name__ == "__main__":
    main()
