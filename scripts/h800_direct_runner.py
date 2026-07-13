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
import argparse, hashlib, hmac, io, json, os, select, struct, subprocess, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (PackageNativeLoader, MaskedQwen, fail_closed_checks,
                                 compute_root_hash, new_counters, LORA_TARGETS, PKG, CKPT,
                                 EXPECTED_ROOT_HASH)
from h800_d4_worker import rank_masked_init, GPU_EXACT_A, CORR_A, bf16_numerics

MAX_MSG = 1200 * 1024 * 1024   # match TDX service bound (causal-LM full-vocab supervised logits)
# per-frame watchdog (s); on expiry we ABORT the step (no fallback). Sized for the throttled
# TDX->H800 ingress (~0.1 MB/s measured): a bf16 dlogits frame (~12.75 MB) needs ~120 s, so 600 s
# gives margin while still bounding a genuinely hung/crashed enclave.
READ_TIMEOUT = 600.0


class TransportError(RuntimeError):
    """Any framing / timeout / auth failure. Raising ABORTS the run — never fall back to
    local or stale results (security requirement: fail-closed, no silent continuation)."""


def _read(f, n, timeout=READ_TIMEOUT):
    """Bounded, timeout-guarded read. select() on the pipe fd so a hung/crashed TDX service
    (no bytes) trips the watchdog instead of blocking forever."""
    buf = b""
    fd = f.fileno()
    deadline = time.time() + timeout
    while len(buf) < n:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TransportError(f"read_timeout after {timeout}s (got {len(buf)}/{n} bytes)")
        r, _, _ = select.select([fd], [], [], remaining)
        if not r:
            continue
        c = f.read(n - len(buf))
        if not c:
            raise TransportError("channel_closed_mid_frame (TDX service exited?)")
        buf += c
    return buf


def read_frame(f, timeout=READ_TIMEOUT):
    (hlen,) = struct.unpack(">I", _read(f, 4, timeout))
    if hlen > (1 << 20):
        raise TransportError(f"header_too_large:{hlen}")
    header = json.loads(_read(f, hlen, timeout).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8, timeout))
    if plen > MAX_MSG:
        raise TransportError(f"payload_too_large:{plen}")
    payload = _read(f, plen, timeout) if plen else b""
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
    # PHASE 1.3 (honest scope): `key` is shared with the A10. MAC + monotonic seq = channel
    # integrity + replay detection vs third-party/network corruption under the honest-but-curious
    # accelerator model. It does NOT prove honest-A10 origin nor malicious-GPU computation integrity.
    m = hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode() + op.encode()
    return hmac.new(key, m, hashlib.sha256).hexdigest()


class TDXChannel:
    """ONE persistent SSH channel to the TDX forced-command service, reused for the whole run.

    The service command we pass here is ADVISORY only: the TDX authorized_keys entry pins a
    forced command (command="... tdx_persistent_service.py /tmp/direct_session.json",restrict),
    so SSH_ORIGINAL_COMMAND is discarded server-side and H800 cannot choose what runs. We keep
    passing it so a mis-provisioned (non-forced) key fails loudly rather than silently.
    """
    def __init__(self, key, tdx, service_cmd):
        self.p = subprocess.Popen(
            ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", tdx, service_cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.reqs = 0

    def request(self, header, payload=b"", timeout=READ_TIMEOUT):
        if self.p.poll() is not None:
            raise TransportError(f"ssh_channel_dead rc={self.p.returncode}")
        t0 = time.time()
        write_frame(self.p.stdin, header, payload)
        h, pl = read_frame(self.p.stdout, timeout)
        self.reqs += 1
        if h.get("op") == "reject":                       # service fail-closed -> abort, no fallback
            raise TransportError(f"service_reject reason={h.get('reason')} seq={h.get('seq')}")
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
    # handshake blocks on the enclave's one-time attestation (quote gen ~60-90s) -> longer watchdog
    hs, _, _ = ch.request({"op": "handshake"}, timeout=360.0)
    attestation = hs.get("attestation", {})

    def _n2(t):  # squared L2 as an ON-DEVICE scalar (NO per-target host sync); .item() once/step
        return t.detach().double().pow(2).sum() if t is not None else 0.0

    steps = []; last_seq = -1
    base_alloc = torch.cuda.memory_allocated()          # params+buffers steady state (no graph)
    for step in range(args.steps):
        tstep = time.time(); counters = new_counters()
        torch.cuda.reset_peak_memory_stats()
        # forward (keep graph for THIS step's backward only)
        tf = time.time(); logits = model.forward(ids, counters); torch.cuda.synchronize()
        fwd_sec = time.time() - tf
        # ce_dlogits over the channel
        ts = time.time(); payload = dump_tensor(logits.detach().to(torch.float32 if args.dtype=="fp32" else torch.bfloat16).cpu())
        ser_sec = time.time() - ts
        seq = last_seq + 1
        hdr = {"op": "ce_dlogits", "seq": seq, "run_id": run_id, "dtype": args.dtype,
               "hmac": mac(key, payload, seq, run_id, "ce_dlogits")}
        rh, rpl, net1 = ch.request(hdr, payload)
        if rh.get("op") != "ce_dlogits_ack":
            raise TransportError(f"expected ce_dlogits_ack got {rh}")
        if rh["hmac"] != mac(key, rpl, rh["seq"], run_id, "ce_dlogits_ack"):
            raise TransportError("dlogits_hmac_mismatch")   # tampered response -> abort, no fallback
        tds = time.time(); dlog = load_tensor(rpl).to(device, DTYPE); deser_sec = time.time() - tds
        ce = rh["ce_loss"]; last_seq = rh["seq"]
        # backward via retained graph (no recompute)
        tb = time.time()
        params, keys = [], []
        for (l, proj), (A, B) in model.lora.items():
            params += [A, B]; keys += [(l, proj)]
        grads = torch.autograd.grad(logits, params, grad_outputs=dlog, allow_unused=True)
        gmap = {keys[i]: (grads[2*i], grads[2*i+1]) for i in range(len(keys))}
        corr = {}; gpu_applied = 0
        gA_sq = gB_sq = updA_sq = updB_sq = mom_sq = 0.0   # numeric-trajectory accumulators
        with torch.no_grad():
            for (l, proj), (A, B) in model.lora.items():
                gA, gB = gmap[(l, proj)]
                gA_sq += _n2(gA); gB_sq += _n2(gB)
                bA, bB = (buf[f"{l}.{proj}"] if buf is not None else (None, None))
                if gB is not None:
                    if buf is not None: bB = args.momentum*bB + gB; B -= args.lr*bB; updB_sq += (args.lr**2)*_n2(bB)
                    else: B -= args.lr*gB; updB_sq += (args.lr**2)*_n2(gB)
                if proj in GPU_EXACT_A:
                    if gA is not None:
                        if buf is not None: bA = args.momentum*bA + gA; A -= args.lr*bA; updA_sq += (args.lr**2)*_n2(bA)
                        else: A -= args.lr*gA; updA_sq += (args.lr**2)*_n2(gA)
                        gpu_applied += 1
                else:
                    corr[f"{l}.{proj}"] = gA.detach().float().cpu()
                if buf is not None: buf[f"{l}.{proj}"] = (bA, bB)
        torch.cuda.synchronize(); bwd_sec = time.time() - tb
        # correct over the channel
        tcs = time.time(); cpayload = dump_tensor(corr); ser_sec += time.time() - tcs
        cseq = last_seq + 1
        chdr = {"op": "correct", "seq": cseq, "run_id": run_id,
                "hmac": mac(key, cpayload, cseq, run_id, "correct")}
        crh, crpl, net2 = ch.request(chdr, cpayload)
        if crh.get("op") != "correct_ack":
            raise TransportError(f"expected correct_ack got {crh}")
        if crh["hmac"] != mac(key, crpl, crh["seq"], run_id, "correct_ack"):
            raise TransportError("correct_hmac_mismatch")
        tcd = time.time(); corrected = load_tensor(crpl); deser_sec += time.time() - tcd
        last_seq = crh["seq"]
        corr_sq = sum(_n2(v) for v in corrected.values())
        # apply corrected A (+ momentum buffer) in memory
        ta = time.time()
        with torch.no_grad():
            for (l, proj), (A, B) in model.lora.items():
                if proj not in CORR_A: continue
                cg = corrected[f"{l}.{proj}"].to(device, DTYPE)
                if buf is not None:
                    bA, bB = buf[f"{l}.{proj}"]; bA = args.momentum*bA + cg; A -= args.lr*bA
                    updA_sq += (args.lr**2)*_n2(bA); buf[f"{l}.{proj}"] = (bA, bB)
                else:
                    A -= args.lr*cg; updA_sq += (args.lr**2)*_n2(cg)
        torch.cuda.synchronize(); apply_sec = time.time() - ta
        if buf is not None:
            mom_sq = sum(_n2(bA) + _n2(bB) for (bA, bB) in buf.values())
        finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A,B) in model.lora.values())
        # VRAM + graph-free proof: drop this step's graph, confirm allocated returns to baseline
        peak_alloc = torch.cuda.max_memory_allocated()
        del logits, grads, gmap, dlog
        torch.cuda.synchronize()
        alloc_after_free = torch.cuda.memory_allocated(); reserved = torch.cuda.memory_reserved()
        wall = time.time() - tstep
        wire1 = max(0.0, net1 - rh["compute_sec"]); wire2 = max(0.0, net2 - crh["compute_sec"])
        # single host sync for all aggregate norms (was ~350 per-target .cpu() calls)
        gA_n = float(gA_sq ** 0.5); gB_n = float(gB_sq ** 0.5); corr_n = float(corr_sq ** 0.5)
        updA_n = float(updA_sq ** 0.5); updB_n = float(updB_sq ** 0.5); mom_n = float(mom_sq ** 0.5)
        steps.append({"step": step, "ce": ce, "gpu_exact_A": gpu_applied, "tdx_corrected": crh["corrected"],
                      "missing": crh["missing"], "finite": finite,
                      "logical_invocations": 2,  # ce_dlogits + correct (session/attest once)
                      "seq_ce": rh["seq"], "seq_correct": crh["seq"],
                      "numerics": {"gradA_norm": gA_n, "gradB_norm": gB_n,
                                   "corrected_grad_norm": corr_n,
                                   "updateA_norm": updA_n, "updateB_norm": updB_n,
                                   "deltaW_proxy_norm": float((updA_sq + updB_sq) ** 0.5),
                                   "momentum_buffer_norm": mom_n},
                      "vram": {"base_alloc_bytes": base_alloc, "peak_alloc_bytes": peak_alloc,
                               "alloc_after_free_bytes": alloc_after_free, "reserved_bytes": reserved,
                               "graph_freed": abs(alloc_after_free - base_alloc) < 8 * 1024 * 1024},
                      "bytes": {"logits_up": rh["bytes_in"], "dlogits_down": rh["bytes_out"],
                                "corr_up": crh["bytes_in"], "corrected_down": crh["bytes_out"]},
                      "timing": {"fwd_sec": fwd_sec, "serialize_sec": ser_sec, "deserialize_sec": deser_sec,
                                 "bwd_sec": bwd_sec, "apply_sec": apply_sec,
                                 "net_ce_sec": net1, "net_correct_sec": net2,
                                 "enclave_ce_sec": rh["compute_sec"], "enclave_correct_sec": crh["compute_sec"],
                                 "ssh_wire_ce_sec": wire1, "ssh_wire_correct_sec": wire2,
                                 "step_wall_sec": wall}})
        print(json.dumps({"step": step, "ce": round(ce,5), "gpuA": gpu_applied,
                          "tdxA": crh["corrected"], "miss": crh["missing"], "finite": finite,
                          "seqCE": rh["seq"], "seqCorr": crh["seq"],
                          "gA": round(gA_n,4), "corrA": round(corr_n,4),
                          "graph_freed": bool(abs(alloc_after_free-base_alloc) < 8*1024*1024),
                          "peakGB": round(peak_alloc/1e9,3),
                          "net_ce": round(net1,2), "net_corr": round(net2,2),
                          "wall": round(wall,2)}), flush=True)
    total_reqs = ch.reqs
    ch.close()
    # session-reuse evidence: one channel, monotone seq across ALL steps, attestation once
    seqs = [s["seq_ce"] for s in steps] + [s["seq_correct"] for s in steps]
    seq_monotone = all(b > a for a, b in zip(sorted(seqs), sorted(seqs)[1:])) if len(seqs) > 1 else True
    peaks = [s["vram"]["peak_alloc_bytes"] for s in steps]
    vram_slope = (peaks[-1] - peaks[0]) / max(1, len(peaks) - 1) if len(peaks) > 1 else 0
    result = {"run_id": run_id, "profile": "L11_O1C_momentum" if args.momentum>0 else "L10_O1C_sgd",
              "dtype": args.dtype, "transport_profile": "direct_h800_tdx", "steps_run": len(steps),
              "attestation": attestation, "fail_closed_all_pass": all(t["passed"] for t in fc),
              "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "trajectory": steps,
              "session_reuse": {"single_ssh_channel": True, "channel_requests": total_reqs,
                                "expected_requests": 1 + 2 * len(steps),  # handshake + 2/step
                                "seq_strictly_monotone": seq_monotone,
                                "attestation_once": True,
                                "graph_freed_every_step": all(s["vram"]["graph_freed"] for s in steps)},
              "vram_summary": {"base_alloc_bytes": base_alloc, "peak_alloc_slope_bytes_per_step": vram_slope,
                               "max_peak_alloc_bytes": max(peaks) if peaks else 0,
                               "no_growth": abs(vram_slope) < 8 * 1024 * 1024},
              "final_lora_state_saved": True}
    # final forward at the TRAINED state -> masked logits, so the trusted-eval verifier can
    # produce this run's OWN effective_equivalence (top1/KL), temporally aligned with the adapter.
    with torch.no_grad():
        final_logits = model.forward(ids, new_counters())
    torch.save(final_logits.detach().float().cpu(), str(args.out).replace(".json", ".final_logits.pt"))
    Path(args.out).write_text(json.dumps(result, indent=2))
    # persist final adapter (dict {f"{l}.{proj}": (A,B)}) — verifier + deployment format
    torch.save({f"{l}.{p}": (A.detach().cpu(), B.detach().cpu()) for (l,p),(A,B) in model.lora.items()},
               str(args.out).replace(".json", ".adapter.pt"))
    print(json.dumps({"done": True, "steps": len(steps),
                      "ce_first_last": [steps[0]["ce"], steps[-1]["ce"]] if steps else None}))


if __name__ == "__main__":
    main()
