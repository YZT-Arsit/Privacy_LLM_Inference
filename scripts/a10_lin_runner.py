"""A10 SGD/momentum LoRA runner (linear optimizers) — O1-C hybrid vs O1-A naive, on real A10+TDX.

Same mixed precision as L12: BF16 fwd/bwd + BF16 runtime copies regenerated from FP32 master; FP32 master
+ (momentum buffer) authoritative. Linear optimizers are metric-equivariant, so:
  * O1-C (L10 sgd / L11 momentum): gamma-fed A{q,k,v,gate,up}+B{q,k} -> TRUSTED enclave plaintext-domain
    optimizer (reuses the verified deployed-convention transforms); o/down A + o/down/v/gate/up B are
    orthogonal/monomial -> GPU-exact masked optimizer. Effective trajectory == plaintext (L1/L3).
  * O1-A (L2 sgd / L4 momentum): naive masked optimizer on ALL factors, NO correction/enclave. Exact for
    orthogonal factors but DIVERGES on the gamma-fed A factors (that divergence is the point; not hidden).

Logs per-step masked grads + refolds so the plaintext (L1/L3) reference oracle can reconstruct from REAL
gradients. --profile o1c|o1a ; --optimizer sgd|momentum ; --mom.
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
GPU_A = {"o_proj", "down_proj"}
GPU_B = {"o_proj", "down_proj", "v_proj", "gate_proj", "up_proj"}


def gpu_step(theta, g, buf, opt, lr, mom):
    if opt == "momentum":
        buf = mom * buf + g
        return theta - lr * buf, buf
    return theta - lr * g, buf   # sgd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=42); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum"])
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--profile", default="o1c", choices=["o1c", "o1a"])
    ap.add_argument("--input-ids", required=True); ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--adapter-id", default="")
    ap.add_argument("--log-transport", default="")
    ap.add_argument("--log-layers", default="")   # e.g. "0,11,23" -> log only these layers (small tlog)
    a = ap.parse_args()
    log_layers = set(int(x) for x in a.log_layers.split(",")) if a.log_layers else None
    CDT = torch.bfloat16; MDT = torch.float32
    sess = json.loads(Path(a.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]
    adapter_id = a.adapter_id or run_id
    dev = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text()); root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
    loader = PackageNativeLoader(PKG, dev, CDT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, CDT)
    ENC = a.profile == "o1c"

    master = {}; buf = {}
    seed_base = 7000 + a.seed
    for l in range(model.L):
        for proj in LORA_TARGETS:
            W = loader.tensors[f"L{l}.{proj}.w"]
            A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0], seed_base=seed_base)
            master[(l, proj)] = [A.to(dev, MDT), B.to(dev, MDT)]
            buf[f"A.{l}.{proj}"] = torch.zeros_like(master[(l, proj)][0])
            buf[f"B.{l}.{proj}"] = torch.zeros_like(master[(l, proj)][1])

    def regen_runtime():
        for (l, proj), (A, B) in master.items():
            model.lora[(l, proj)] = (A.detach().to(CDT).requires_grad_(True),
                                     B.detach().to(CDT).requires_grad_(True))
    regen_runtime()
    ids = torch.tensor(json.loads(Path(a.input_ids).read_text())["input_ids"][:a.seq_len]).to(dev)

    ch = TDXChannel(a.key, a.tdx, a.service_cmd)
    hs, _, _ = ch.request({"op": "handshake"}, timeout=360.0)
    attestation = hs.get("attestation", {}); last_seq = -1

    if ENC:
        initA = {f"{l}.{p}": A.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_A}
        initB = {f"{l}.{p}": B.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_B}
        ipayload = dump_tensor({"A": initA, "B": initB})
        ih = {"op": "init_adamw", "seq": 0, "run_id": run_id,
              "hmac": mac(key, ipayload, 0, run_id, "init_adamw"),
              "hparams": {"lr": a.lr, "state_dtype": "fp32", "adapter_id": adapter_id,
                          "optimizer": a.optimizer, "mom": a.mom,
                          "optimizer_profile": f"L{'10' if a.optimizer=='sgd' else '11'}_{a.optimizer}"}}
        rh, _, _ = ch.request(ih, ipayload)
        if rh.get("op") != "init_adamw_ack" or not rh.get("trusted_adamw_state_present"):
            raise TransportError(f"init failed: {rh}")
        last_seq = rh["seq"]

    steps = []; tlog = []
    for step in range(a.steps):
        ts = time.time(); regen_runtime()
        logits = model.forward(ids, new_counters()); torch.cuda.synchronize()
        payload = dump_tensor(logits.detach().to(torch.bfloat16).cpu()); seq = last_seq + 1
        rh, rpl, net1 = ch.request({"op": "ce_dlogits", "seq": seq, "run_id": run_id, "dtype": "bf16",
                                    "hmac": mac(key, payload, seq, run_id, "ce_dlogits")}, payload)
        if rh.get("op") != "ce_dlogits_ack": raise TransportError("ce_dlogits fail")
        ce = rh["ce_loss"]; dlog = load_tensor(rpl).to(dev, CDT); last_seq = rh["seq"]
        params, keys = [], []
        for (l, proj) in master: A, B = model.lora[(l, proj)]; params += [A, B]; keys += [(l, proj)]
        grads = torch.autograd.grad(logits, params, grad_outputs=dlog, allow_unused=True)
        gmap = {keys[i]: (grads[2 * i], grads[2 * i + 1]) for i in range(len(keys))}
        gA_tr, gB_tr = {}, {}; slog = {"step": step, "gA_tilde": {}, "gB_tilde": {}, "refoldA": {}, "refoldB": {}}
        with torch.no_grad():
            for (l, proj) in master:
                A, B = master[(l, proj)]; gA, gB = gmap[(l, proj)]
                a_enc = ENC and proj in TRUSTED_A; b_enc = ENC and proj in TRUSTED_B
                if gA is not None and not a_enc:                        # GPU masked step (raw)
                    A2, bA = gpu_step(A, gA.to(MDT), buf[f"A.{l}.{proj}"], a.optimizer, a.lr, a.mom)
                    master[(l, proj)][0] = A2; buf[f"A.{l}.{proj}"] = bA
                elif gA is not None and a_enc:
                    gt = gA.to(MDT).cpu(); gA_tr[f"{l}.{proj}"] = gt
                    if a.log_transport and (log_layers is None or l in log_layers): slog["gA_tilde"][f"{l}.{proj}"] = gt.clone()
                if gB is not None and not b_enc:
                    B2, bB = gpu_step(B, gB.to(MDT), buf[f"B.{l}.{proj}"], a.optimizer, a.lr, a.mom)
                    master[(l, proj)][1] = B2; buf[f"B.{l}.{proj}"] = bB
                elif gB is not None and b_enc:
                    gt = gB.to(MDT).cpu(); gB_tr[f"{l}.{proj}"] = gt
                    if a.log_transport and (log_layers is None or l in log_layers): slog["gB_tilde"][f"{l}.{proj}"] = gt.clone()
        torch.cuda.synchronize()
        net2 = 0.0
        if ENC:
            spayload = dump_tensor({"gA": gA_tr, "gB": gB_tr}); sseq = last_seq + 1
            srh, srpl, net2 = ch.request({"op": "adamw_step", "seq": sseq, "run_id": run_id,
                                          "hmac": mac(key, spayload, sseq, run_id, "adamw_step")}, spayload)
            if srh.get("op") != "adamw_step_ack": raise TransportError(f"step fail: {srh}")
            upd = load_tensor(srpl); last_seq = srh["seq"]
            with torch.no_grad():
                for k, val in upd.get("A", {}).items():
                    l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][0] = val.to(dev, MDT)
                    if a.log_transport and (log_layers is None or l in log_layers): slog["refoldA"][k] = val.clone()
                for k, val in upd.get("B", {}).items():
                    l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][1] = val.to(dev, MDT)
                    if a.log_transport and (log_layers is None or l in log_layers): slog["refoldB"][k] = val.clone()
            ver = srh.get("state_version")
        else:
            ver = None
        finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A, B) in master.values())
        steps.append({"step": step, "ce": ce, "state_version": ver, "finite": finite,
                      "net_ce": net1, "net_step": net2, "wall": time.time() - ts})
        if a.log_transport: tlog.append(slog)
        print(json.dumps({"step": step, "ce": round(ce, 5), "prof": a.profile, "opt": a.optimizer,
                          "ver": ver, "finite": finite, "wall": round(time.time() - ts, 2)}), flush=True)

    regen_runtime()
    with torch.no_grad():
        final_logits = model.forward(ids, new_counters())
    torch.save(final_logits.detach().float().cpu(), str(a.out).replace(".json", ".final_logits.pt"))
    torch.save({f"{l}.{p}": (A.detach().float().cpu(), B.detach().float().cpu())
                for (l, p), (A, B) in master.items()}, str(a.out).replace(".json", ".adapter.pt"))
    if a.log_transport and tlog: torch.save(tlog, a.log_transport)
    ch.close()
    Path(a.out).write_text(json.dumps({"run_id": run_id, "profile": a.profile, "optimizer": a.optimizer,
        "mom": a.mom if a.optimizer == "momentum" else None, "seed": a.seed, "compute_dtype": "bf16",
        "master_dtype": "fp32", "transport_profile": "direct_a10_tdx_private", "steps_run": len(steps),
        "attestation": attestation, "fail_closed_all_pass": all(x["passed"] for x in fc),
        "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "trajectory": steps}, indent=2))
    print(json.dumps({"done": True, "steps": len(steps),
                      "ce_first_last": [steps[0]["ce"], steps[-1]["ce"]] if steps else None}))


if __name__ == "__main__":
    main()
