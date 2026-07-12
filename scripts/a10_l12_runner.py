"""A10 L12 (protected AdamW) persistent runner — runs ON A10, data plane A10<->TDX private.

GPU-exact AdamW (monomial transforms) for A of {o,down} and B of {o,down,v,gate,up}; TRUSTED enclave
AdamW (m,v,step held in TDX) for A of {q,k,v,gate,up} and B of {q,k}. No m/v for trusted factors on
A10; no gamma; no plaintext gradient (A10 sends masked grads only); fail-closed on stale enclave state.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (PackageNativeLoader, MaskedQwen, fail_closed_checks, compute_root_hash,
                                 new_counters, LORA_TARGETS, PKG, CKPT, EXPECTED_ROOT_HASH)
from h800_d4_worker import rank_masked_init
from h800_direct_runner import TDXChannel, mac, dump_tensor, load_tensor, TransportError

TRUSTED_A = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
TRUSTED_B = {"q_proj", "k_proj"}
GPU_A = {"o_proj", "down_proj"}                                  # A GPU-exact
GPU_B = {"o_proj", "down_proj", "v_proj", "gate_proj", "up_proj"}   # B GPU-exact


def adamw_gpu(theta, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta = theta - lr * (mhat / (vhat.sqrt() + eps) + wd * theta)
    return theta, m, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--input-ids", required=True); ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DT = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    sess = json.loads(Path(a.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]
    dev = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text()); root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
    loader = PackageNativeLoader(PKG, dev, DT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, DT)
    for l in range(model.L):
        for proj in LORA_TARGETS:
            W = loader.tensors[f"L{l}.{proj}.w"]
            A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0])
            model.lora[(l, proj)] = (A.to(dev, DT).requires_grad_(True), B.to(dev, DT).requires_grad_(True))
    # GPU-exact Adam moments (trusted factors' moments live ONLY in TDX)
    gm_ = {}
    for (l, proj), (A, B) in model.lora.items():
        if proj in GPU_A: gm_[f"A.{l}.{proj}"] = [torch.zeros_like(A), torch.zeros_like(A)]
        if proj in GPU_B: gm_[f"B.{l}.{proj}"] = [torch.zeros_like(B), torch.zeros_like(B)]
    ids = torch.tensor(json.loads(Path(a.input_ids).read_text())["input_ids"][:a.seq_len]).to(dev)

    ch = TDXChannel(a.key, a.tdx, a.service_cmd)
    hs, _, _ = ch.request({"op": "handshake"}, timeout=360.0)
    attestation = hs.get("attestation", {})
    # init trusted AdamW state in enclave (send INITIAL masked trusted factors)
    initA = {f"{l}.{p}": A.detach().float().cpu() for (l, p), (A, B) in model.lora.items() if p in TRUSTED_A}
    initB = {f"{l}.{p}": B.detach().float().cpu() for (l, p), (A, B) in model.lora.items() if p in TRUSTED_B}
    ipayload = dump_tensor({"A": initA, "B": initB})
    ih = {"op": "init_adamw", "seq": 0, "run_id": run_id,
          "hmac": mac(key, ipayload, 0, run_id, "init_adamw"),
          "hparams": {"lr": a.lr, "b1": b1, "b2": b2, "eps": eps, "wd": wd}}
    rh, _, _ = ch.request(ih, ipayload)
    if rh.get("op") != "init_adamw_ack" or not rh.get("trusted_adamw_state_present"):
        raise TransportError(f"init_adamw failed: {rh}")
    last_seq = rh["seq"]; t = 0; steps = []
    for step in range(a.steps):
        ts = time.time(); counters = new_counters()
        logits = model.forward(ids, counters); torch.cuda.synchronize()
        payload = dump_tensor(logits.detach().to(torch.bfloat16 if a.dtype == "bf16" else torch.float32).cpu())
        seq = last_seq + 1
        rh, rpl, net1 = ch.request({"op": "ce_dlogits", "seq": seq, "run_id": run_id, "dtype": a.dtype,
                                    "hmac": mac(key, payload, seq, run_id, "ce_dlogits")}, payload)
        if rh.get("op") != "ce_dlogits_ack" or rh["hmac"] != mac(key, rpl, rh["seq"], run_id, "ce_dlogits_ack"):
            raise TransportError("ce_dlogits fail")
        ce = rh["ce_loss"]; dlog = load_tensor(rpl).to(dev, DT); last_seq = rh["seq"]; t += 1
        params, keys = [], []
        for (l, proj), (A, B) in model.lora.items(): params += [A, B]; keys += [(l, proj)]
        grads = torch.autograd.grad(logits, params, grad_outputs=dlog, allow_unused=True)
        gmap = {keys[i]: (grads[2 * i], grads[2 * i + 1]) for i in range(len(keys))}
        gA_tr, gB_tr = {}, {}
        with torch.no_grad():
            for (l, proj), (A, B) in model.lora.items():
                gA, gB = gmap[(l, proj)]
                if proj in GPU_A and gA is not None:
                    m, v = gm_[f"A.{l}.{proj}"]; A2, m, v = adamw_gpu(A, gA, m, v, t, a.lr, b1, b2, eps, wd)
                    A.copy_(A2); gm_[f"A.{l}.{proj}"] = [m, v]
                elif proj in TRUSTED_A and gA is not None:
                    gA_tr[f"{l}.{proj}"] = gA.detach().float().cpu()
                if proj in GPU_B and gB is not None:
                    m, v = gm_[f"B.{l}.{proj}"]; B2, m, v = adamw_gpu(B, gB, m, v, t, a.lr, b1, b2, eps, wd)
                    B.copy_(B2); gm_[f"B.{l}.{proj}"] = [m, v]
                elif proj in TRUSTED_B and gB is not None:
                    gB_tr[f"{l}.{proj}"] = gB.detach().float().cpu()
        torch.cuda.synchronize()
        # trusted AdamW in enclave -> updated masked factors
        spayload = dump_tensor({"gA": gA_tr, "gB": gB_tr}); sseq = last_seq + 1
        srh, srpl, net2 = ch.request({"op": "adamw_step", "seq": sseq, "run_id": run_id,
                                      "hmac": mac(key, spayload, sseq, run_id, "adamw_step")}, spayload)
        if srh.get("op") != "adamw_step_ack" or srh["hmac"] != mac(key, srpl, srh["seq"], run_id, "adamw_step_ack"):
            raise TransportError(f"adamw_step fail: {srh}")
        upd = load_tensor(srpl); last_seq = srh["seq"]
        with torch.no_grad():
            for k, v in upd.get("A", {}).items():
                l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; model.lora[(l, proj)][0].copy_(v.to(dev, DT))
            for k, v in upd.get("B", {}).items():
                l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; model.lora[(l, proj)][1].copy_(v.to(dev, DT))
        finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A, B) in model.lora.values())
        steps.append({"step": step, "ce": ce, "gpu_adamw_A": sum(1 for (l, p) in model.lora if p in GPU_A),
                      "trusted_A": len(gA_tr), "trusted_B": len(gB_tr), "adamw_updated_A": srh["updated_A"],
                      "adamw_updated_B": srh["updated_B"], "adam_t": srh["adam_t"], "missing": srh["missing"],
                      "finite": finite, "net_ce": net1, "net_adamw": net2, "wall": time.time() - ts})
        print(json.dumps({"step": step, "ce": round(ce, 5), "trA": len(gA_tr), "trB": len(gB_tr),
                          "updA": srh["updated_A"], "updB": srh["updated_B"], "t": srh["adam_t"],
                          "miss": srh["missing"], "finite": finite, "wall": round(time.time() - ts, 2)}), flush=True)
    # final forward for equivalence
    with torch.no_grad():
        final_logits = model.forward(ids, new_counters())
    torch.save(final_logits.detach().float().cpu(), str(a.out).replace(".json", ".final_logits.pt"))
    torch.save({f"{l}.{p}": (A.detach().cpu(), B.detach().cpu()) for (l, p), (A, B) in model.lora.items()},
               str(a.out).replace(".json", ".adapter.pt"))
    ch.close()
    Path(a.out).write_text(json.dumps({"run_id": run_id, "profile": "L12_O1C_adamw_protected",
        "dtype": a.dtype, "transport_profile": "direct_a10_tdx_private", "steps_run": len(steps),
        "attestation": attestation, "fail_closed_all_pass": all(x["passed"] for x in fc),
        "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "trajectory": steps}, indent=2))
    print(json.dumps({"done": True, "steps": len(steps),
                      "ce_first_last": [steps[0]["ce"], steps[-1]["ce"]] if steps else None}))


if __name__ == "__main__":
    main()
