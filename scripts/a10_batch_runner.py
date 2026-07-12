"""A10 batched utility runner (runs ON A10; data plane A10<->TDX private VPC).

Trains/evals a masked LoRA adapter on the IMMUTABLE transformed package (root bfd578b8) over
official SST-2 (classification) or GSM8K (causal-LM) with a per-batch authenticated protocol:
  * forward each example of the microbatch (single-seq masked forward), collect ONLY the
    supervised-position logits (cls: 1 position/example; clm: answer positions/example);
  * send them + a signed batch descriptor to TDX (ce_batch) -> receive authenticated dlogits;
  * scatter dlogits back to per-example grad_outputs, autograd wrt LoRA factors, accumulate;
  * optimizer step: GPU-exact monomial AdamW for {o,down}A + {o,down,v,gate,up}B (FP32 master),
    enclave FP32 trusted AdamW for {q,k,v,gate,up}A + {q,k}B (identical to a10_mixed_runner).

Profiles (pre-registered arms, see datasets/tokenized/deviations.json):
  L0  = base floor: no LoRA effect, eval only.
  L5  = fp32-compute exact plaintext-equivalent reference (fp32 forward + fp32 master).
  L12 = bf16-compute protected deployment (bf16 forward + fp32 master).
A10 holds NO labels/targets/loss; TDX holds the private label table + monotonic batch ledger.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (PackageNativeLoader, MaskedQwen, fail_closed_checks, compute_root_hash,
                                 new_counters, LORA_TARGETS, PKG, CKPT, EXPECTED_ROOT_HASH)
from h800_d4_worker import rank_masked_init
from h800_direct_runner import TDXChannel, mac, dump_tensor, load_tensor, TransportError
from batch_dataplane import batch_mac

TRUSTED_A = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
TRUSTED_B = {"q_proj", "k_proj"}
GPU_A = {"o_proj", "down_proj"}
GPU_B = {"o_proj", "down_proj", "v_proj", "gate_proj", "up_proj"}


def adamw_gpu(theta, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    return theta - lr * (mhat / (vhat.sqrt() + eps) + wd * theta), m, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--tdx", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--service-cmd", required=True)
    ap.add_argument("--profile", required=True, choices=["L0", "L5", "L12"])
    ap.add_argument("--task", required=True, choices=["cls", "clm"])
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--train-data", default=""); ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-data", default=""); ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--schedule", default="")
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-max", type=int, default=0)     # cap eval examples (0 = all)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--template-hash", required=True); ap.add_argument("--tokenizer-hash", required=True)
    ap.add_argument("--label-schema-hash", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--adapter-id", default="")
    ap.add_argument("--checkpoint-at", type=int, default=-1)
    ap.add_argument("--ledger-checkpoint-at", type=int, default=-1)
    ap.add_argument("--gpu-state-path", default="")
    ap.add_argument("--start-step", type=int, default=0)   # resume: skip schedule entries < start-step
    ap.add_argument("--restore-first", action="store_true")
    ap.add_argument("--ckpt-path", default=""); ap.add_argument("--expected-binding", default="")
    ap.add_argument("--ledger-path", default="")
    ap.add_argument("--max-seq", type=int, default=512)
    a = ap.parse_args()
    CDT = torch.float32 if a.profile == "L5" else torch.bfloat16
    MDT = torch.float32
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    sess = json.loads(Path(a.session).read_text())
    key = bytes.fromhex(sess["session_key_hex"]); run_id = sess["run_id"]
    adapter_id = a.adapter_id or run_id
    dev = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text()); root = compute_root_hash(PKG)
    fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
    loader = PackageNativeLoader(PKG, dev, CDT); loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, dev, CDT)
    pkg_hash = EXPECTED_ROOT_HASH

    # ---- LoRA master (fp32) + runtime (compute dtype) ----
    master, gm_ = {}, {}
    seed_base = 7000 + a.seed
    train_lora = a.profile in ("L5", "L12")
    if train_lora:
        for l in range(model.L):
            for proj in LORA_TARGETS:
                W = loader.tensors[f"L{l}.{proj}.w"]
                A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0], seed_base=seed_base)
                master[(l, proj)] = [A.to(dev, MDT), B.to(dev, MDT)]
                if proj in GPU_A: gm_[f"A.{l}.{proj}"] = [torch.zeros_like(master[(l, proj)][0]),
                                                          torch.zeros_like(master[(l, proj)][0])]
                if proj in GPU_B: gm_[f"B.{l}.{proj}"] = [torch.zeros_like(master[(l, proj)][1]),
                                                          torch.zeros_like(master[(l, proj)][1])]

    def regen_runtime():
        model.lora = {}
        for (l, proj), (A, B) in master.items():
            model.lora[(l, proj)] = (A.detach().to(CDT).requires_grad_(True),
                                     B.detach().to(CDT).requires_grad_(True))
    if train_lora:
        regen_runtime()

    a10 = torch.load(a.train_data, map_location="cpu", weights_only=False) if a.train_data else []
    a10_by_id = {int(d["sample_id"]): d for d in a10}
    sched = json.loads(Path(a.schedule).read_text())["schedule"] if a.schedule else []
    schedule_hash = hashlib.sha256(json.dumps({"schedule": sched}).encode()).hexdigest() if False else \
        (json.loads(Path(a.schedule).read_text()).get("hash", "") if a.schedule else "")

    ch = TDXChannel(a.key, a.tdx, a.service_cmd)
    hs, _, _ = ch.request({"op": "handshake"}, timeout=360.0)
    attestation = hs.get("attestation", {})
    last_seq = -1

    def desc_for(split, task, epoch, gbi, opt_step, micro, sample_ids, sup_counts, seq_shape, aw):
        return {"run_id": run_id, "dataset_id": a.dataset_id, "split": split, "task": task,
                "epoch": epoch, "optimizer_step": opt_step, "microbatch_index": micro,
                "global_batch_index": gbi, "sample_ids": sample_ids, "seq_shape": seq_shape,
                "sup_counts": sup_counts, "accum_window": aw, "package_hash": pkg_hash,
                "adapter_id": adapter_id, "template_hash": a.template_hash,
                "tokenizer_hash": a.tokenizer_hash, "label_schema_hash": a.label_schema_hash}

    # ---- enclave AdamW init / restore (L5,L12) ----
    if train_lora and not a.restore_first:
        initA = {f"{l}.{p}": A.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_A}
        initB = {f"{l}.{p}": B.detach().float().cpu() for (l, p), (A, B) in master.items() if p in TRUSTED_B}
        ipayload = dump_tensor({"A": initA, "B": initB}); iseq = last_seq + 1
        ih = {"op": "init_adamw", "seq": iseq, "run_id": run_id,
              "hmac": mac(key, ipayload, iseq, run_id, "init_adamw"),
              "hparams": {"lr": a.lr, "b1": b1, "b2": b2, "eps": eps, "wd": wd, "state_dtype": "fp32",
                          "adapter_id": adapter_id, "optimizer_profile": f"{a.profile}_adamw"}}
        rh, _, _ = ch.request(ih, ipayload)
        if rh.get("op") != "init_adamw_ack" or not rh.get("trusted_adamw_state_present"):
            raise TransportError(f"init_adamw failed: {rh}")
        last_seq = rh["seq"]
    t_opt = 0
    if train_lora and a.restore_first:
        if a.gpu_state_path and Path(a.gpu_state_path).exists():
            gs = torch.load(a.gpu_state_path, map_location=dev, weights_only=False)
            for k, val in gs["master"].items():
                l, proj = int(k.split(".")[0]), k.split(".", 1)[1]
                master[(l, proj)] = [val[0].to(dev, MDT), val[1].to(dev, MDT)]
            gm_ = {k: [v[0].to(dev, MDT), v[1].to(dev, MDT)] for k, v in gs["gm"].items()}
            t_opt = int(gs.get("t", 0)); regen_runtime()
        exp = json.loads(a.expected_binding); rseq = last_seq + 1
        rh, _, _ = ch.request({"op": "restore_adamw", "seq": rseq, "run_id": run_id,
                               "hmac": mac(key, b"", rseq, run_id, "restore_adamw"),
                               "ckpt_path": a.ckpt_path, "expected_binding": exp,
                               "min_version": int(exp.get("version", 0)),
                               "hparams": {"lr": a.lr, "b1": b1, "b2": b2, "eps": eps, "wd": wd,
                                           "state_dtype": "fp32", "adapter_id": adapter_id,
                                           "optimizer_profile": exp.get("optimizer_profile", f"{a.profile}_adamw")}})
        if rh.get("op") != "restore_adamw_ack" or not rh.get("trusted_adamw_state_present"):
            raise TransportError(f"restore_adamw failed: {rh}")
        last_seq = rh["seq"]
        if a.ledger_path:      # also restore the batch ledger
            lseq = last_seq + 1
            lh, _, _ = ch.request({"op": "restore_ledger", "seq": lseq, "run_id": run_id,
                                   "hmac": mac(key, b"", lseq, run_id, "restore_ledger"),
                                   "ledger_path": a.ledger_path, "expected_binding": exp})
            if lh.get("op") != "restore_ledger_ack":
                raise TransportError(f"restore_ledger failed: {lh}")
            last_seq = lh["seq"]

    def fwd_sup_logits(rec):
        """forward one example; return (logits_full, sup_positions, sup_logits)."""
        ids = torch.tensor(rec["input_ids"][:a.max_seq]).to(dev)
        logits = model.forward(ids, new_counters())     # [T,V]
        if a.task == "cls":
            pos = [min(rec["sup_pos"], logits.shape[0] - 1)]
        else:
            pos = list(range(rec["sup_start"], logits.shape[0] - 1))
        sup = logits[torch.tensor(pos, device=dev)]     # [k,V]
        return logits, pos, sup

    MAX_ROWS = 256   # cap supervised rows per ce_batch frame so the CPU-only TDX guest keeps up

    def send_ce(cat, desc_base, op="ce_batch"):
        """Send supervised logits (chunked for clm) -> (ce_mean, dlog_full, nll_sum, net). The clm
        gradient uses a full-batch denominator so summing chunk dlogits == the mean-reduction gradient."""
        nonlocal last_seq
        rows = cat.shape[0]
        nchunks = 1 if (a.task == "cls" or rows <= MAX_ROWS) else (rows + MAX_ROWS - 1) // MAX_ROWS
        dlog_full = torch.zeros(rows, cat.shape[1], device=dev, dtype=CDT) if op == "ce_batch" else None
        ce_sum = 0.0; nll_sum = 0.0; net = 0.0
        for c in range(nchunks):
            r0 = c * MAX_ROWS; r1 = min(rows, r0 + MAX_ROWS); sub = cat[r0:r1]
            payload = dump_tensor({"logits": sub.detach().to(torch.bfloat16).cpu()})
            d = dict(desc_base); d.update({"chunk": c, "nchunks": nchunks, "row_start": r0,
                     "sup_denominator": rows, "seq_shape": list(sub.shape)})
            seq = last_seq + 1
            rh, rpl, nt = ch.request({"op": op, "seq": seq, "run_id": run_id, "dtype": "bf16",
                "desc": d, "hmac": mac(key, payload, seq, run_id, op),
                "bmac": batch_mac(key, payload, seq, run_id, op, d)}, payload)
            ack = op + "_ack"
            if rh.get("op") != ack:
                raise TransportError(f"{op} fail: {rh}")
            last_seq = rh["seq"]; net += nt
            if op == "ce_batch":
                if rh["hmac"] != mac(key, rpl, rh["seq"], run_id, ack):
                    raise TransportError("ce_batch hmac")
                dlog_full[r0:r1] = load_tensor(rpl).to(dev, CDT); ce_sum += rh["ce_loss"]
            else:
                nll_sum += rh.get("sum_nll", 0.0)
        ce_mean = ce_sum if a.task == "cls" else (ce_sum / max(rows, 1))
        return ce_mean, dlog_full, nll_sum, net

    # ================= TRAIN =================
    steps = []
    if train_lora and a.max_steps > 0:
        for si, entry in enumerate(sched):
            if entry["global_batch_index"] < a.start_step:
                continue
            if len([s for s in steps]) >= a.max_steps:
                break
            ts = time.time()
            regen_runtime()
            recs = [a10_by_id[int(s)] for s in entry["sample_ids"]]
            fwds, sup_list, pos_list, sup_counts = [], [], [], []
            for rec in recs:
                logits, pos, sup = fwd_sup_logits(rec)
                fwds.append(logits); pos_list.append(pos); sup_list.append(sup); sup_counts.append(len(pos))
            cat = torch.cat(sup_list, 0)                  # cls:[n,V]; clm:[total_sup,V]
            desc = desc_for(entry["split"] if "split" in entry else a.train_split, a.task, entry["epoch"],
                            entry["global_batch_index"], entry["optimizer_step"], entry["microbatch_index"],
                            list(entry["sample_ids"]), sup_counts, list(cat.shape), int(entry.get("size", len(recs))))
            ce, dlog, _, net1 = send_ce(cat, desc, op="ce_batch"); t_opt += 1
            # scatter dlogits back per example, autograd, accumulate
            off = 0; gacc = {k: [torch.zeros_like(master[k][0]), torch.zeros_like(master[k][1])] for k in master}
            for logits, pos, cnt in zip(fwds, pos_list, sup_counts):
                go = torch.zeros_like(logits)
                go[torch.tensor(pos, device=dev)] = dlog[off:off + cnt]
                off += cnt
                params, keys = [], []
                for (l, proj) in master: A, B = model.lora[(l, proj)]; params += [A, B]; keys += [(l, proj)]
                grads = torch.autograd.grad(logits, params, grad_outputs=go, retain_graph=False, allow_unused=True)
                for i, k in enumerate(keys):
                    if grads[2 * i] is not None: gacc[k][0] += grads[2 * i].to(MDT)
                    if grads[2 * i + 1] is not None: gacc[k][1] += grads[2 * i + 1].to(MDT)
            # optimizer step
            gA_tr, gB_tr = {}, {}
            with torch.no_grad():
                for (l, proj) in master:
                    gA, gB = gacc[(l, proj)]
                    if proj in GPU_A:
                        m, v = gm_[f"A.{l}.{proj}"]; A2, m, v = adamw_gpu(master[(l, proj)][0], gA, m, v, t_opt, a.lr, b1, b2, eps, wd)
                        master[(l, proj)][0] = A2; gm_[f"A.{l}.{proj}"] = [m, v]
                    elif proj in TRUSTED_A:
                        gA_tr[f"{l}.{proj}"] = gA.cpu()
                    if proj in GPU_B:
                        m, v = gm_[f"B.{l}.{proj}"]; B2, m, v = adamw_gpu(master[(l, proj)][1], gB, m, v, t_opt, a.lr, b1, b2, eps, wd)
                        master[(l, proj)][1] = B2; gm_[f"B.{l}.{proj}"] = [m, v]
                    elif proj in TRUSTED_B:
                        gB_tr[f"{l}.{proj}"] = gB.cpu()
            spayload = dump_tensor({"gA": gA_tr, "gB": gB_tr}); sseq = last_seq + 1
            srh, srpl, net2 = ch.request({"op": "adamw_step", "seq": sseq, "run_id": run_id,
                                          "hmac": mac(key, spayload, sseq, run_id, "adamw_step")}, spayload)
            if srh.get("op") != "adamw_step_ack" or srh["hmac"] != mac(key, srpl, srh["seq"], run_id, "adamw_step_ack"):
                raise TransportError(f"adamw_step fail: {srh}")
            upd = load_tensor(srpl); last_seq = srh["seq"]
            with torch.no_grad():
                for k, val in upd.get("A", {}).items():
                    l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][0] = val.to(dev, MDT)
                for k, val in upd.get("B", {}).items():
                    l, proj = int(k.split(".")[0]), k.split(".", 1)[1]; master[(l, proj)][1] = val.to(dev, MDT)
            finite = all(bool(torch.isfinite(A).all() and torch.isfinite(B).all()) for (A, B) in master.values())
            steps.append({"gbi": entry["global_batch_index"], "opt_step": entry["optimizer_step"], "ce": ce,
                          "n": len(recs), "sup_total": int(cat.shape[0]), "finite": finite,
                          "ledger_last": rh.get("ledger_last"), "net_ce": net1, "net_adamw": net2,
                          "wall": round(time.time() - ts, 3)})
            print(json.dumps({"gbi": entry["global_batch_index"], "ce": round(ce, 5), "n": len(recs),
                              "finite": finite, "wall": round(time.time() - ts, 2)}), flush=True)
            # optional durable checkpoints for restart test
            if a.checkpoint_at == len(steps) - 1:
                cseq = last_seq + 1
                crh, _, _ = ch.request({"op": "checkpoint_adamw", "seq": cseq, "run_id": run_id,
                                        "hmac": mac(key, b"", cseq, run_id, "checkpoint_adamw")})
                last_seq = crh["seq"]; steps[-1]["checkpoint"] = {"path": crh["ckpt_path"], "binding": crh["binding"],
                    "version": crh["state_version"], "checkpoint_seq": crh.get("checkpoint_seq")}
                if a.gpu_state_path:
                    torch.save({"master": {f"{l}.{p}": [A.detach().cpu(), B.detach().cpu()] for (l, p), (A, B) in master.items()},
                                "gm": {k: [x.detach().cpu() for x in v] for k, v in gm_.items()}, "t": t_opt}, a.gpu_state_path)
            if a.ledger_checkpoint_at == len(steps) - 1:
                lseq = last_seq + 1
                lrh, _, _ = ch.request({"op": "checkpoint_ledger", "seq": lseq, "run_id": run_id,
                                        "hmac": mac(key, b"", lseq, run_id, "checkpoint_ledger"),
                                        "adapter_id": adapter_id})
                last_seq = lrh["seq"]; steps[-1]["ledger_checkpoint"] = {"path": lrh["ledger_path"],
                    "state": lrh["ledger_state"]}

    # ================= EVAL =================
    evalres = None
    if a.eval_data:
        regen_runtime() if train_lora else None
        ev = torch.load(a.eval_data, map_location="cpu", weights_only=False)
        if a.eval_max > 0: ev = ev[:a.eval_max]
        n_correct = 0; n = 0; nll_sum = 0.0; nsup = 0; preds = []
        with torch.no_grad():
            bs = 16
            for b0 in range(0, len(ev), bs):
                chunk = ev[b0:b0 + bs]
                sup_list, pos_list, sup_counts, sids = [], [], [], []
                for rec in chunk:
                    ids = torch.tensor(rec["input_ids"][:a.max_seq]).to(dev)
                    logits = model.forward(ids, new_counters())
                    if a.task == "cls":
                        pos = [min(rec["sup_pos"], logits.shape[0] - 1)]
                    else:
                        pos = list(range(rec["sup_start"], logits.shape[0] - 1))
                    sup_list.append(logits[torch.tensor(pos, device=dev)]); pos_list.append(pos)
                    sup_counts.append(len(pos)); sids.append(int(rec["sample_id"]))
                cat = torch.cat(sup_list, 0)
                desc = desc_for(a.eval_split, a.task, 0, -1, -1, 0, sids, sup_counts, list(cat.shape), len(chunk))
                if a.task == "cls":
                    payload = dump_tensor({"logits": cat.detach().to(torch.bfloat16).cpu()})
                    seq = last_seq + 1
                    rh, rpl, _ = ch.request({"op": "eval_batch", "seq": seq, "run_id": run_id, "desc": desc,
                                             "hmac": mac(key, payload, seq, run_id, "eval_batch"),
                                             "bmac": batch_mac(key, payload, seq, run_id, "eval_batch", desc)}, payload)
                    if rh.get("op") != "eval_batch_ack":
                        raise TransportError(f"eval_batch fail: {rh}")
                    last_seq = rh["seq"]; res = load_tensor(rpl)
                    n_correct += int(res["correct"].sum()); n += len(chunk)
                    nll_sum += float(res["nll"].sum())
                    for sid, p, c in zip(sids, res["pred"].tolist(), res["correct"].tolist()):
                        preds.append({"sample_id": sid, "pred": p, "correct": c})
                else:
                    _, _, nll_s, _ = send_ce(cat, desc, op="eval_batch")
                    nll_sum += nll_s; nsup += int(cat.shape[0]); n += len(chunk)
        evalres = {"split": a.eval_split, "n": n,
                   "accuracy": (n_correct / n) if (a.task == "cls" and n) else None,
                   "mean_nll": (nll_sum / (n if a.task == "cls" else max(nsup, 1))),
                   "n_correct": n_correct if a.task == "cls" else None}
        if preds:
            torch.save(preds, str(a.out).replace(".json", ".preds.pt"))

    # save adapter (masked; NO plaintext reconstruction) for downstream generation/analysis
    if train_lora:
        torch.save({f"{l}.{p}": (A.detach().float().cpu(), B.detach().float().cpu())
                    for (l, p), (A, B) in master.items()}, str(a.out).replace(".json", ".adapter.pt"))
    ch.close()
    Path(a.out).write_text(json.dumps({
        "run_id": run_id, "profile": a.profile, "task": a.task, "dataset_id": a.dataset_id,
        "seed": a.seed, "compute_dtype": str(CDT), "master_dtype": "fp32", "lr": a.lr,
        "schedule_hash": schedule_hash, "steps_run": len(steps), "start_step": a.start_step,
        "attestation": attestation, "fail_closed_all_pass": all(x["passed"] for x in fc),
        "package_root_hash_matches": root == EXPECTED_ROOT_HASH, "eval": evalres,
        "trajectory": steps}, indent=2))
    print(json.dumps({"done": True, "profile": a.profile, "steps": len(steps), "eval": evalres}))


if __name__ == "__main__":
    main()
