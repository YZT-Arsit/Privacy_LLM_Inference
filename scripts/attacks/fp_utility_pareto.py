#!/usr/bin/env python
"""Fingerprint-vs-utility Pareto: can fine-tuning remove the attention-logit
fingerprint WITHOUT destroying the model?

For each attention learning rate we fine-tune q/k/v/o (the fingerprint's own
projections; dense -> affects every token, unlike sparse embedding grads) for a
fixed number of steps, then measure BOTH:
  * fingerprint top-1 recovery (attacker's PUBLIC-base table), and
  * held-out perplexity (utility).
A defense is only meaningful if it drives recovery toward random (0.01%) while
keeping perplexity near baseline. We show recovery only collapses once
perplexity has already blown up -> fine-tuning cannot defend without wrecking
the model. Fresh model reload per lr so each point is independent.
"""
from __future__ import annotations
import argparse, json, math, time
import torch


def _layer_idx(name):
    parts = name.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None


def collect_fp(model, token_ids, device, cfg, batch=256):
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_q
    grp = n_q // n_kv
    L = cfg.num_hidden_layers
    cap, handles = {}, []
    for name, mod in model.named_modules():
        li = _layer_idx(name)
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
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0)


def top1(F_table, F_query):
    Ft = F_table / (F_table.norm(dim=1, keepdim=True) + 1e-9)
    Fq = F_query / (F_query.norm(dim=1, keepdim=True) + 1e-9)
    n = F_table.shape[0]
    idx = torch.arange(n)
    hit = 0
    for s in range(0, n, 1024):
        pred = (Fq[s:s + 1024] @ Ft.T).argmax(1)
        hit += int((pred == idx[s:s + 1024]).sum())
    return round(hit / n * 100, 3)


def perplexity(model, tok, texts, device):
    tot_loss, tot_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True,
                      max_length=256).to(device)
            out = model(**enc, labels=enc["input_ids"])
            ntok = enc["input_ids"].shape[1] - 1
            tot_loss += float(out.loss) * ntok
            tot_tok += ntok
    return round(math.exp(tot_loss / max(tot_tok, 1)), 3)


def load_model(path, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-tokens", type=int, default=8000)
    ap.add_argument("--lrs", default="0,2e-5,1e-4,5e-4,2e-3")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="/root/fp_utility_pareto_result.json")
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = ""
            for kk in ("instruction", "question", "prompt"):
                if d.get(kk):
                    t += str(d[kk]) + "\n"
            for kk in ("response", "output", "answer"):
                if d.get(kk):
                    t += str(d[kk]); break
            if t.strip():
                rows.append(t.strip())
    heldout = rows[-20:]
    train = rows[:-20]
    print(f"{len(train)} train / {len(heldout)} heldout", flush=True)

    # baseline model -> F_base table + baseline ppl
    model = load_model(args.model, device)
    cfg = model.config
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(0)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]
    F_base = collect_fp(model, token_ids, device, cfg)
    base_ppl = perplexity(model, tok, heldout, device)
    print(f"baseline: fp_top1={top1(F_base, F_base)} ppl={base_ppl}", flush=True)
    del model
    torch.cuda.empty_cache()

    results = {"model": args.model, "candidates": n, "steps": args.steps,
               "random_top1": round(100.0 / n, 4), "baseline_ppl": base_ppl,
               "points": [{"lr": 0.0, "fp_top1": top1(F_base, F_base),
                           "ppl": base_ppl, "train_loss": None}]}

    for lr in [float(x) for x in args.lrs.split(",")]:
        if lr == 0.0:
            continue
        print(f"\n=== lr={lr} ===", flush=True)
        model = load_model(args.model, device)
        for p in model.parameters():
            p.requires_grad_(False)
        tp = []
        for name, p in model.named_parameters():
            if _layer_idx(name) is not None and ".self_attn." in name and any(
                    name.endswith(x + s) for x in
                    ("q_proj", "k_proj", "v_proj", "o_proj")
                    for s in (".weight", ".bias")):
                p.requires_grad_(True); tp.append(p)
        model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(tp, lr=lr)
        model.train(); opt.zero_grad()
        step, ri, accum = 0, 0, 4
        last = 0.0
        t0 = time.time()
        while step < args.steps:
            enc = tok(train[ri % len(train)], return_tensors="pt",
                      truncation=True, max_length=256).to(device)
            ri += 1
            out = model(**enc, labels=enc["input_ids"])
            last = float(out.loss)
            (out.loss / accum).backward()
            if ri % accum == 0:
                opt.step(); opt.zero_grad(); step += 1
        Fq = collect_fp(model, token_ids, device, cfg)
        rec = top1(F_base, Fq)
        ppl = perplexity(model, tok, heldout, device)
        pt = {"lr": lr, "fp_top1": rec, "ppl": ppl,
              "train_loss": round(last, 3),
              "fp_shift_l2_med": round(float((Fq - F_base).norm(dim=1).median()), 1)}
        print("POINT", json.dumps(pt), f"({time.time()-t0:.0f}s)", flush=True)
        results["points"].append(pt)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        del model, opt
        torch.cuda.empty_cache()

    print("DONE", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
