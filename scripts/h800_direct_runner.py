"""Persistent H800 direct-transport runner (runs ON H800).

Loads the package + MaskedQwen + masked LoRA ONCE, opens ONE persistent SSH channel to the
TDX forced-command service, and runs the whole O1-C training loop in memory:

  forward (keep graph) -> send masked logits -> recv masked dlogits
    -> autograd.grad(logits, params, dlogits)  (NO recompute; graph retained)
    -> apply o/down A + all B on GPU ; send q/k/v/gate/up A-grads -> recv corrected
    -> apply corrected A-updates (+ in-memory momentum buffers)

No temp files, no shell polling, no Mac in the data path, no per-step process restart.
Records per-step timing decomposition (serialize / network / enclave / H800 compute) + bytes.
"""
from __future__ import annotations
import argparse, hashlib, hmac, io, json, os, struct, subprocess, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (PackageNativeLoader, MaskedQwen, fail_closed_checks,
                                 compute_root_hash, new_counters, LORA_TARGETS, PKG, CKPT,
                                 EXPECTED_ROOT_HASH)
from h800_d4_worker import rank_masked_init, GPU_EXACT_A, CORR_A, bf16_numerics

MAX_MSG = 200 * 1024 * 1024


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


def write_frame(f, header, payload=b""):
    hb = json.dumps(header).encode()
    f.write(struct.pack(">I", len(hb))); f.write(hb)
    f.write(struct.pack(">Q", len(payload))); f.write(payload); f.flush()


def dump_tensor(t):
    bio = io.BytesIO(); torch.save(t, bio); return bio.getvalue()


def load_tensor(b):
    return torch.load(io.BytesIO(b), map_location="cpu")


def mac(key, payload, seq, run_id, op):
    m = hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode() + op.encode()
    return hmac.new(key, m, hashlib.sha256).hexdigest()


class TDXChannel:
    """One persistent SSH channel to the TDX forced-command service."""
    def __init__(self, key, tdx, service_cmd):
        self.p = subprocess.Popen(
            ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=30", tdx, service_cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def request(self, header, payload=b""):
        t0 = time.time()
        write_frame(self.p.stdin, header, payload)
        h, pl = read_frame(self.p.stdout)
        return h, pl, time.time() - t0

    def close(self):
        try:
            write_frame(self.p.stdin, {"op": "close"}); self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)     # session config json (session key, run_id)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--input-ids", required=True)
    ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    sess = json.loads(Path(args.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]

    device = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text())
    root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
    loader = PackageNativeLoader(PKG, device, DTYPE); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, device, DTYPE)
    # masked LoRA (in memory, persistent across steps)
    state = {}
    for l in range(model.L):
        for proj in LORA_TARGETS:
            W = loader.tensors[f"L{l}.{proj}.w"]
            A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0])
            model.lora[(l, proj)] = (A.to(device, DTYPE).requires_grad_(True),
                                     B.to(device, DTYPE).requires_grad_(True))
    buf = {k: (torch.zeros_like(A), torch.zeros_like(B)) for k, (A, B) in
           {f"{l}.{p}": v for (l, p), v in model.lora.items()}.items()} if args.momentum > 0 else None

    ids = torch.tensor(json.loads(Path(args.input_ids).read_text())["input_ids"][:args.seq_len]).to(device)

    ch = TDXChannel(args.key, args.tdx, args.service_cmd)
    hs, _, _ = ch.request({"op": "handshake"})
    attestation = hs.get("attestation", {})

    steps = []; last_seq = -1
    for step in range(args.steps):
        tstep = time.time(); counters = new_counters()
        # forward (keep graph for backward)
        tf = time.time(); logits = model.forward(ids, counters); torch.cuda.synchronize()
        fwd_sec = time.time() - tf
        # ce_dlogits over the channel
        ts = time.time(); payload = dump_tensor(logits.detach().to(torch.float32 if args.dtype=="fp32" else torch.bfloat16).cpu())
        ser_sec = time.time() - ts
        seq = last_seq + 1
        hdr = {"op": "ce_dlogits", "seq": seq, "run_id": run_id, "dtype": args.dtype,
               "hmac": mac(key, payload, seq, run_id, "ce_dlogits")}
        rh, rpl, net1 = ch.request(hdr, payload)
        assert rh.get("op") == "ce_dlogits_ack", rh
        assert rh["hmac"] == mac(key, rpl, rh["seq"], run_id, "ce_dlogits_ack"), "dlogits hmac"
        ce = rh["ce_loss"]; dlog = load_tensor(rpl).to(device, DTYPE); last_seq = rh["seq"]
        # backward via retained graph (no recompute)
        tb = time.time()
        params, keys = [], []
        for (l, proj), (A, B) in model.lora.items():
            params += [A, B]; keys += [(l, proj)]
        grads = torch.autograd.grad(logits, params, grad_outputs=dlog, allow_unused=True)
        gmap = {keys[i]: (grads[2*i], grads[2*i+1]) for i in range(len(keys))}
        corr = {}; gpu_applied = 0
        with torch.no_grad():
            for (l, proj), (A, B) in model.lora.items():
                gA, gB = gmap[(l, proj)]
                bA, bB = (buf[f"{l}.{proj}"] if buf is not None else (None, None))
                if gB is not None:
                    if buf is not None: bB = args.momentum*bB + gB; B -= args.lr*bB
                    else: B -= args.lr*gB
                if proj in GPU_EXACT_A:
                    if gA is not None:
                        if buf is not None: bA = args.momentum*bA + gA; A -= args.lr*bA
                        else: A -= args.lr*gA
                        gpu_applied += 1
                else:
                    corr[f"{l}.{proj}"] = gA.detach().float().cpu()
                if buf is not None: buf[f"{l}.{proj}"] = (bA, bB)
        torch.cuda.synchronize(); bwd_sec = time.time() - tb
        # correct over the channel
        cpayload = dump_tensor(corr)
        cseq = last_seq + 1
        chdr = {"op": "correct", "seq": cseq, "run_id": run_id,
                "hmac": mac(key, cpayload, cseq, run_id, "correct")}
        crh, crpl, net2 = ch.request(chdr, cpayload)
        assert crh.get("op") == "correct_ack", crh
        assert crh["hmac"] == mac(key, crpl, crh["seq"], run_id, "correct_ack"), "corr hmac"
        corrected = load_tensor(crpl); last_seq = crh["seq"]
        # apply corrected A (+ momentum buffer) in memory
        ta = time.time()
        with torch.no_grad():
            for (l, proj), (A, B) in model.lora.items():
                if proj not in CORR_A: continue
                cg = corrected[f"{l}.{proj}"].to(device, DTYPE)
                if buf is not None:
                    bA, bB = buf[f"{l}.{proj}"]; bA = args.momentum*bA + cg; A -= args.lr*bA
                    buf[f"{l}.{proj}"] = (bA, bB)
                else:
                    A -= args.lr*cg
        torch.cuda.synchronize(); apply_sec = time.time() - ta
        finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A,B) in model.lora.values())
        steps.append({"step": step, "ce": ce, "gpu_exact_A": gpu_applied, "tdx_corrected": crh["corrected"],
                      "missing": crh["missing"], "finite": finite,
                      "logical_invocations": 2,  # ce_dlogits + correct (session/attest once)
                      "bytes": {"logits_up": rh["bytes_in"], "dlogits_down": rh["bytes_out"],
                                "corr_up": crh["bytes_in"], "corrected_down": crh["bytes_out"]},
                      "timing": {"fwd_sec": fwd_sec, "serialize_sec": ser_sec, "bwd_sec": bwd_sec,
                                 "apply_sec": apply_sec,
                                 "net_ce_sec": net1, "net_correct_sec": net2,
                                 "enclave_ce_sec": rh["compute_sec"], "enclave_correct_sec": crh["compute_sec"],
                                 "step_wall_sec": time.time()-tstep}})
        print(json.dumps({"step": step, "ce": round(ce,5), "gpuA": gpu_applied,
                          "tdxA": crh["corrected"], "miss": crh["missing"], "finite": finite,
                          "net_ce": round(net1,2), "net_corr": round(net2,2),
                          "wall": round(time.time()-tstep,2)}), flush=True)
    ch.close()
    result = {"run_id": run_id, "profile": "L11_O1C_momentum" if args.momentum>0 else "L10_O1C_sgd",
              "dtype": args.dtype, "transport_profile": "direct_h800_tdx", "steps_run": len(steps),
              "attestation": attestation, "fail_closed_all_pass": all(t["passed"] for t in fc),
              "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "trajectory": steps,
              "final_lora_state_saved": True}
    Path(args.out).write_text(json.dumps(result, indent=2))
    # persist final adapter for deployment/equiv
    torch.save({f"{l}.{p}": (A.detach().cpu(), B.detach().cpu()) for (l,p),(A,B) in model.lora.items()},
               str(Path(args.out).with_suffix(".adapter.pt")))
    print(json.dumps({"done": True, "steps": len(steps),
                      "ce_first_last": [steps[0]["ce"], steps[-1]["ce"]] if steps else None}))


if __name__ == "__main__":
    main()
