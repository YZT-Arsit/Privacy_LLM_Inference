"""A10 L12 MIXED-PRECISION runner (BF16 compute / FP32 master) — runs ON A10, data plane A10<->TDX.

Frozen L12 design (results/.../l12_mixed_precision_design_freeze.json):
  * model forward/backward activations: BF16
  * GPU runtime LoRA copies: BF16 (regenerated from FP32 master each step)
  * authoritative LoRA master (trusted factors): FP32 inside TDX; m,v,step FP32 inside TDX
  * GPU-exact monomial factors: FP32 master + FP32 m,v on GPU; BF16 runtime copy; no BF16 authoritative state
  * updates computed from FP32 master; no BF16 tensor is ever the authoritative AdamW state

Trusted enclave AdamW (m,v,step in TDX) for A of {q,k,v,gate,up} + B of {q,k}; GPU-exact monomial
AdamW for A of {o,down} + B of {o,down,v,gate,up}. A10 never sees trusted m/v/gamma/plaintext-grad.

Logs, per step and per trusted factor: the masked grad SENT (gA_tilde) and the enclave's re-folded
factor RETURNED, so an offline verification oracle can reconstruct the L5 plaintext-AdamW reference
from the REAL gradients (masked_grad @ M^T) and confirm the enclave output un-folds to it. The L12
arm itself is real hardware (real GPU fwd/bwd + real TDX enclave AdamW) — the oracle is a checker,
never a substitute.

Optional restart-continuity: --checkpoint-at K sends checkpoint_adamw after step K; --restore-first
(with --ckpt-path/--expected-binding) restores enclave optimizer state before step 0.
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
    ap.add_argument("--seed", type=int, default=1234)   # -> lora init seed_base = 7000 + seed
    ap.add_argument("--input-ids", required=True); ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-at", type=int, default=-1)      # send checkpoint_adamw after this step index
    ap.add_argument("--restore-first", action="store_true")      # restore enclave state before step 0
    ap.add_argument("--ckpt-path", default=""); ap.add_argument("--expected-binding", default="")
    ap.add_argument("--adapter-id", default="")
    ap.add_argument("--gpu-state-path", default="")   # local durable state for GPU-exact factors (restart)
    ap.add_argument("--log-transport", default="")               # per-step masked-grad + refold log for the oracle
    a = ap.parse_args()
    # MIXED PRECISION: compute dtype = bf16 (activations, runtime copies); master dtype = fp32
    CDT = torch.bfloat16; MDT = torch.float32
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    sess = json.loads(Path(a.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]
    adapter_id = a.adapter_id or run_id
    dev = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text()); root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
    loader = PackageNativeLoader(PKG, dev, CDT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, CDT)

    # FP32 master + BF16 runtime for ALL factors. Runtime copies (requires_grad) drive the bf16 forward.
    master = {}     # (l,proj) -> [A_fp32, B_fp32]  (authoritative for GPU-exact factors on A10)
    gm_ = {}        # GPU-exact fp32 moments (trusted factors' m/v live ONLY in TDX)
    seed_base = 7000 + a.seed
    for l in range(model.L):
        for proj in LORA_TARGETS:
            W = loader.tensors[f"L{l}.{proj}.w"]
            A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0], seed_base=seed_base)
            A = A.to(dev, MDT); B = B.to(dev, MDT)
            master[(l, proj)] = [A, B]
            if proj in GPU_A: gm_[f"A.{l}.{proj}"] = [torch.zeros_like(A), torch.zeros_like(A)]
            if proj in GPU_B: gm_[f"B.{l}.{proj}"] = [torch.zeros_like(B), torch.zeros_like(B)]

    def regen_runtime():
        # BF16 runtime copies regenerated from FP32 master; these are the ONLY tensors in forward.
        for (l, proj), (A, B) in master.items():
            model.lora[(l, proj)] = (A.detach().to(CDT).requires_grad_(True),
                                     B.detach().to(CDT).requires_grad_(True))
    regen_runtime()
    ids = torch.tensor(json.loads(Path(a.input_ids).read_text())["input_ids"][:a.seq_len]).to(dev)

    ch = TDXChannel(a.key, a.tdx, a.service_cmd)
    hs, _, _ = ch.request({"op": "handshake"}, timeout=360.0)
    attestation = hs.get("attestation", {})
    last_seq = -1

    t_resume = 0
    if a.restore_first and a.gpu_state_path and Path(a.gpu_state_path).exists():
        # resume GPU-exact factor master + moments (enclave restores trusted factors separately)
        gs = torch.load(a.gpu_state_path, map_location=dev, weights_only=False)
        for k, val in gs["master"].items():
            l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)] = [val[0].to(dev, MDT), val[1].to(dev, MDT)]
        gm_ = {k: [v[0].to(dev, MDT), v[1].to(dev, MDT)] for k, v in gs["gm"].items()}
        t_resume = int(gs.get("t", 0))
        regen_runtime()

    if a.restore_first:
        exp = json.loads(a.expected_binding)
        rseq = last_seq + 1
        rh, _, _ = ch.request({"op": "restore_adamw", "seq": rseq, "run_id": run_id,
                               "hmac": mac(key, b"", rseq, run_id, "restore_adamw"),
                               "ckpt_path": a.ckpt_path, "expected_binding": exp,
                               "min_version": int(exp.get("version", 0)),
                               "hparams": {"lr": a.lr, "b1": b1, "b2": b2, "eps": eps, "wd": wd,
                                           "state_dtype": "fp32", "adapter_id": adapter_id,
                                           "optimizer_profile": exp.get("optimizer_profile", "L12_adamw")}})
        if rh.get("op") != "restore_adamw_ack" or not rh.get("trusted_adamw_state_present"):
            raise TransportError(f"restore_adamw failed: {rh}")
        last_seq = rh["seq"]
    else:
        # init trusted AdamW state in enclave from INITIAL masked trusted factors (FP32 master).
        initA = {f"{l}.{p}": A.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_A}
        initB = {f"{l}.{p}": B.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_B}
        ipayload = dump_tensor({"A": initA, "B": initB})
        ih = {"op": "init_adamw", "seq": 0, "run_id": run_id,
              "hmac": mac(key, ipayload, 0, run_id, "init_adamw"),
              "hparams": {"lr": a.lr, "b1": b1, "b2": b2, "eps": eps, "wd": wd,
                          "state_dtype": "fp32", "adapter_id": adapter_id,
                          "optimizer_profile": "L12_adamw"}}
        rh, _, _ = ch.request(ih, ipayload)
        if rh.get("op") != "init_adamw_ack" or not rh.get("trusted_adamw_state_present"):
            raise TransportError(f"init_adamw failed: {rh}")
        last_seq = rh["seq"]

    t = t_resume; steps = []; tlog = []
    for step in range(a.steps):
        ts = time.time(); counters = new_counters()
        regen_runtime()
        logits = model.forward(ids, counters); torch.cuda.synchronize()
        payload = dump_tensor(logits.detach().to(torch.bfloat16).cpu())
        seq = last_seq + 1
        rh, rpl, net1 = ch.request({"op": "ce_dlogits", "seq": seq, "run_id": run_id, "dtype": "bf16",
                                    "hmac": mac(key, payload, seq, run_id, "ce_dlogits")}, payload)
        if rh.get("op") != "ce_dlogits_ack" or rh["hmac"] != mac(key, rpl, rh["seq"], run_id, "ce_dlogits_ack"):
            raise TransportError("ce_dlogits fail")
        ce = rh["ce_loss"]; dlog = load_tensor(rpl).to(dev, CDT); last_seq = rh["seq"]; t += 1
        params, keys = [], []
        for (l, proj) in master: A, B = model.lora[(l, proj)]; params += [A, B]; keys += [(l, proj)]
        grads = torch.autograd.grad(logits, params, grad_outputs=dlog, allow_unused=True)
        gmap = {keys[i]: (grads[2 * i], grads[2 * i + 1]) for i in range(len(keys))}
        gA_tr, gB_tr = {}, {}; slog = {"step": step, "gA_tilde": {}, "gpu_master_norms": {}}
        with torch.no_grad():
            for (l, proj) in master:
                A, B = master[(l, proj)]; gA, gB = gmap[(l, proj)]
                # GPU-exact monomial factors: FP32 master AdamW on GPU; BF16 grad cast up to FP32.
                if proj in GPU_A and gA is not None:
                    m, v = gm_[f"A.{l}.{proj}"]; A2, m, v = adamw_gpu(A, gA.to(MDT), m, v, t, a.lr, b1, b2, eps, wd)
                    master[(l, proj)][0] = A2; gm_[f"A.{l}.{proj}"] = [m, v]
                elif proj in TRUSTED_A and gA is not None:
                    gt = gA.to(MDT).cpu(); gA_tr[f"{l}.{proj}"] = gt
                    if a.log_transport: slog["gA_tilde"][f"{l}.{proj}"] = gt.clone()
                if proj in GPU_B and gB is not None:
                    m, v = gm_[f"B.{l}.{proj}"]; B2, m, v = adamw_gpu(B, gB.to(MDT), m, v, t, a.lr, b1, b2, eps, wd)
                    master[(l, proj)][1] = B2; gm_[f"B.{l}.{proj}"] = [m, v]
                elif proj in TRUSTED_B and gB is not None:
                    gt = gB.to(MDT).cpu(); gB_tr[f"{l}.{proj}"] = gt
                    if a.log_transport: slog.setdefault("gB_tilde", {})[f"{l}.{proj}"] = gt.clone()
        torch.cuda.synchronize()
        # trusted AdamW in enclave (FP32 master/m/v in TDX) -> re-folded FP32 masked factors
        spayload = dump_tensor({"gA": gA_tr, "gB": gB_tr}); sseq = last_seq + 1
        srh, srpl, net2 = ch.request({"op": "adamw_step", "seq": sseq, "run_id": run_id,
                                      "hmac": mac(key, spayload, sseq, run_id, "adamw_step")}, spayload)
        if srh.get("op") != "adamw_step_ack" or srh["hmac"] != mac(key, srpl, srh["seq"], run_id, "adamw_step_ack"):
            raise TransportError(f"adamw_step fail: {srh}")
        upd = load_tensor(srpl); last_seq = srh["seq"]
        with torch.no_grad():
            for k, val in upd.get("A", {}).items():
                l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][0] = val.to(dev, MDT)
                if a.log_transport: slog.setdefault("refoldA", {})[k] = val.clone()
            for k, val in upd.get("B", {}).items():
                l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][1] = val.to(dev, MDT)
                if a.log_transport: slog.setdefault("refoldB", {})[k] = val.clone()
        finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A, B) in master.values())
        steps.append({"step": step, "ce": ce, "trusted_A": len(gA_tr), "trusted_B": len(gB_tr),
                      "adamw_updated_A": srh["updated_A"], "adamw_updated_B": srh["updated_B"],
                      "adam_t": srh["adam_t"], "state_version": srh.get("state_version"),
                      "runtime_copy_matches_master_transform": srh.get("runtime_copy_matches_master_transform"),
                      "missing": srh["missing"], "finite": finite, "net_ce": net1, "net_adamw": net2,
                      "wall": time.time() - ts})
        if a.log_transport: tlog.append(slog)
        print(json.dumps({"step": step, "ce": round(ce, 5), "trA": len(gA_tr), "trB": len(gB_tr),
                          "ver": srh.get("state_version"), "rt_match": srh.get("runtime_copy_matches_master_transform"),
                          "miss": srh["missing"], "finite": finite, "wall": round(time.time() - ts, 2)}), flush=True)
        # optional durable encrypted checkpoint after step `checkpoint_at`
        if a.checkpoint_at == step:
            cseq = last_seq + 1
            crh, _, _ = ch.request({"op": "checkpoint_adamw", "seq": cseq, "run_id": run_id,
                                    "hmac": mac(key, b"", cseq, run_id, "checkpoint_adamw")})
            if crh.get("op") != "checkpoint_adamw_ack":
                raise TransportError(f"checkpoint failed: {crh}")
            last_seq = crh["seq"]
            steps[-1]["checkpoint"] = {"path": crh["ckpt_path"], "version": crh["state_version"],
                                       "checkpoint_seq": crh.get("checkpoint_seq"), "binding": crh.get("binding"),
                                       "aead": crh.get("aead"),
                                       "bytes": crh["ckpt_bytes"], "durable_plaintext": crh["durable_plaintext"]}
            if a.gpu_state_path:   # persist GPU-exact factor state locally (enclave state is durable in TDX)
                torch.save({"master": {f"{l}.{p}": [A.detach().cpu(), B.detach().cpu()]
                                       for (l, p), (A, B) in master.items()},
                            "gm": {k: [x.detach().cpu() for x in v] for k, v in gm_.items()}, "t": t},
                           a.gpu_state_path)
            print(json.dumps({"checkpoint": crh["ckpt_path"], "version": crh["state_version"]}), flush=True)

    # regenerate final runtime from master, final forward for equivalence
    regen_runtime()
    with torch.no_grad():
        final_logits = model.forward(ids, new_counters())
    torch.save(final_logits.detach().float().cpu(), str(a.out).replace(".json", ".final_logits.pt"))
    torch.save({f"{l}.{p}": (A.detach().float().cpu(), B.detach().float().cpu())
                for (l, p), (A, B) in master.items()}, str(a.out).replace(".json", ".adapter.pt"))
    if a.log_transport and tlog:
        torch.save(tlog, a.log_transport)
    ch.close()
    Path(a.out).write_text(json.dumps({"run_id": run_id, "profile": "L12_mixed_bf16compute_fp32master",
        "seed": a.seed, "compute_dtype": "bf16", "master_dtype": "fp32",
        "transport_profile": "direct_a10_tdx_private", "steps_run": len(steps),
        "attestation": attestation, "fail_closed_all_pass": all(x["passed"] for x in fc),
        "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "trajectory": steps}, indent=2))
    print(json.dumps({"done": True, "steps": len(steps),
                      "ce_first_last": [steps[0]["ce"], steps[-1]["ce"]] if steps else None}))


if __name__ == "__main__":
    main()
