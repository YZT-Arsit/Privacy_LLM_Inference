#!/usr/bin/env python
"""Experiment 1: does a PRIVATE (fine-tuned) deployed model defeat the
attention-fingerprint and embedding-norm attacks?

Threat model.  The adversary is the untrusted GPU operator.  Both attacks are
mask-INVARIANT, so what the adversary observes equals the TRUE fingerprint /
norm of the *deployed* model.  To turn an observation into a token id the
adversary must build a *table* over the vocabulary.  In the open-weight setting
the table is built from the PUBLIC weights and matches exactly -> full break
(reconfirmed here).  In the private setting the deployed model is fine-tuned and
its weights (attention proj + embedding) stay secret in the TEE; the adversary
can only build the table from the PUBLIC base -> table != observed.

We fine-tune a REAL LoRA (q/k/v/o) + a trainable embedding on the model and, at
several training checkpoints, run BOTH attacks with the attacker's table fixed
to the pristine public base.  We report recovery vs training steps.

  * fingerprint attack: F(token) in R^{L*Hq}, position-0 diagonal self-logit
    q_i.k_group(i)/sqrt(hd) per (layer, query head).  Table = base, obs = ft.
  * norm attack: ||embed(token)|| (1-D).  Table = base embed row norms,
    obs = ft embed row norms.

Both tables are frozen to the pristine base BEFORE any training.
"""
from __future__ import annotations
import argparse, json, math, os, time
import torch


def _parse_layer_idx(name: str) -> int | None:
    # find ".layers.<i>." in a (possibly peft-wrapped) module name
    parts = name.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None


def collect_fingerprints(model, token_ids, device, cfg, batch=256):
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = cfg.hidden_size // n_q
    grp = n_q // n_kv
    L = cfg.num_hidden_layers
    cap = {}
    handles = []
    for name, mod in model.named_modules():
        li = _parse_layer_idx(name)
        if li is None:
            continue
        if name.endswith("self_attn.q_proj"):
            handles.append(mod.register_forward_hook(
                lambda m, i, o, li=li: cap.__setitem__((li, "q"), o.detach())))
        elif name.endswith("self_attn.k_proj"):
            handles.append(mod.register_forward_hook(
                lambda m, i, o, li=li: cap.__setitem__((li, "k"), o.detach())))
    feats = []
    try:
        for s in range(0, len(token_ids), batch):
            ids = token_ids[s:s + batch].to(device).unsqueeze(1)
            cap.clear()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            B = ids.shape[0]
            fb = torch.empty(B, L * n_q, dtype=torch.float32)
            for i in range(L):
                q = cap[(i, "q")][:, 0, :].reshape(B, n_q, hd).float()
                k = cap[(i, "k")][:, 0, :].reshape(B, n_kv, hd).float()
                kg = k.repeat_interleave(grp, dim=1)
                fb[:, i * n_q:(i + 1) * n_q] = (q * kg).sum(-1) / (hd ** 0.5)
            feats.append(fb)
            print(f"    fp {s + B}/{len(token_ids)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0)


def fp_recovery(F_table, F_query, ks=(1, 10, 100)):
    n = F_table.shape[0]
    Ft = F_table / (F_table.norm(dim=1, keepdim=True) + 1e-9)
    Fq = F_query / (F_query.norm(dim=1, keepdim=True) + 1e-9)
    idx = torch.arange(n)
    hit = {k: 0 for k in ks}
    CH = 1024
    for s in range(0, n, CH):
        sims = Fq[s:s + CH] @ Ft.T
        order = sims.argsort(dim=1, descending=True)
        tgt = idx[s:s + CH].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}_pct": round(hit[k] / n * 100, 3) for k in ks}


def norm_recovery(norm_table, norm_query, ks=(1, 10, 100)):
    # 1-D NN: nearest base-norm token to each observed norm
    n = norm_table.shape[0]
    idx = torch.arange(n)
    hit = {k: 0 for k in ks}
    CH = 2048
    nt = norm_table.view(1, -1)
    for s in range(0, n, CH):
        d = (norm_query[s:s + CH].view(-1, 1) - nt).abs()
        order = d.argsort(dim=1)  # ascending distance
        tgt = idx[s:s + CH].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}_pct": round(hit[k] / n * 100, 3) for k in ks}


def embed_weight(model):
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens") and hasattr(mod, "weight"):
            return mod.weight.detach()
    raise RuntimeError("no embed_tokens found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-tokens", type=int, default=10000)
    ap.add_argument("--checkpoints", default="0,100,300")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/private_weight_defense_result.json")
    args = ap.parse_args()

    ckpts = sorted(int(x) for x in args.checkpoints.split(","))
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    cfg = model.config
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(args.seed)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]

    # ---- attacker tables: frozen to the pristine PUBLIC base ------------------
    print("building PUBLIC-base fingerprint table (attacker's table)...", flush=True)
    F_base = collect_fingerprints(model, token_ids, device, cfg)
    norm_base = embed_weight(model)[token_ids].float().norm(dim=1).cpu()
    base_fp = fp_recovery(F_base, F_base)          # sanity: table vs itself
    print("  base self-match (open-weight ceiling):", base_fp, flush=True)

    results = {"model": args.model, "candidate_tokens": n,
               "fingerprint_dim": int(F_base.shape[1]),
               "random_top1_pct": round(100.0 / n, 4),
               "open_weight_fingerprint": base_fp,
               "open_weight_norm": norm_recovery(norm_base, norm_base),
               "checkpoints": []}

    # ---- build LoRA + trainable embedding ------------------------------------
    from peft import LoraConfig, get_peft_model
    lconf = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # data
    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            txt = ""
            for kk in ("instruction", "prompt", "input", "question"):
                if d.get(kk):
                    txt += str(d[kk]) + "\n"
            for kk in ("response", "output", "answer", "completion"):
                if d.get(kk):
                    txt += str(d[kk])
            if txt.strip():
                rows.append(txt.strip())
    print(f"loaded {len(rows)} training rows", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)

    def measure(step):
        model.eval()
        Fq = collect_fingerprints(model, token_ids, device, cfg)
        nq = embed_weight(model)[token_ids].float().norm(dim=1).cpu()
        rec = {"step": step,
               "fingerprint_public_table": fp_recovery(F_base, Fq),
               "norm_public_table": norm_recovery(norm_base, nq),
               "fp_shift_l2_median": round(float(
                   (Fq - F_base).norm(dim=1).median()), 4),
               "embed_norm_shift_median": round(float(
                   (nq - norm_base).abs().median()), 5)}
        model.train()
        print("  MEASURE", json.dumps(rec), flush=True)
        results["checkpoints"].append(rec)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if 0 in ckpts:
        measure(0)
    step = 0
    ri = 0
    accum = 4
    opt.zero_grad()
    t0 = time.time()
    while step < args.max_steps:
        batch_txt = rows[ri % len(rows)]
        ri += 1
        enc = tok(batch_txt, return_tensors="pt", truncation=True,
                  max_length=256).to(device)
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / accum).backward()
        if ri % accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  step {step}/{args.max_steps} loss={float(out.loss):.4f}"
                      f" ({(time.time()-t0)/step:.1f}s/step)", flush=True)
            if step in ckpts:
                measure(step)
    print("DONE", json.dumps(results, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
