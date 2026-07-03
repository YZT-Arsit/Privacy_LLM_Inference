#!/usr/bin/env python
"""Experiment 1c (rigorous): does FULL fine-tuning of the fingerprint's own
inputs (embedding + q/k/v/o, genuinely trained -- NO peft) defeat:
  (A) the attention-logit fingerprint attack, and
  (B) the embedding-norm attack (CONJFORMER App D.4 claims fine-tuning the
      embedding table removes the norm structure)?

Attacker table frozen to the pristine PUBLIC base. We also report a SELF-table
norm recovery (fine-tuned vs fine-tuned) to test CONJFORMER's actual claim that
fine-tuning destroys the norm *structure* itself (not just shifts it).
Confirms the embedding really trains (embed_norm_shift > 0).
"""
from __future__ import annotations
import argparse, json, time
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


def cos_rec(Ft0, Fq0, ks=(1, 10, 100)):
    n = Ft0.shape[0]
    Ft = Ft0 / (Ft0.norm(dim=1, keepdim=True) + 1e-9)
    Fq = Fq0 / (Fq0.norm(dim=1, keepdim=True) + 1e-9)
    idx = torch.arange(n)
    hit = {k: 0 for k in ks}
    for s in range(0, n, 1024):
        order = (Fq[s:s + 1024] @ Ft.T).argsort(dim=1, descending=True)
        tgt = idx[s:s + 1024].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}": round(hit[k] / n * 100, 3) for k in ks}


def norm_rec(nt, nq, ks=(1, 10, 100)):
    n = nt.shape[0]
    idx = torch.arange(n)
    hit = {k: 0 for k in ks}
    ntv = nt.view(1, -1)
    for s in range(0, n, 2048):
        order = (nq[s:s + 2048].view(-1, 1) - ntv).abs().argsort(dim=1)
        tgt = idx[s:s + 2048].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}": round(hit[k] / n * 100, 3) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-tokens", type=int, default=10000)
    ap.add_argument("--checkpoints", default="0,300,800")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--embed-lr", type=float, default=5e-5)
    ap.add_argument("--attn-lr", type=float, default=2e-4)
    ap.add_argument("--out", default="/root/private_weight_defense_c_result.json")
    args = ap.parse_args()
    ckpts = sorted(int(x) for x in args.checkpoints.split(","))
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    cfg = model.config
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(0)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]

    emb = model.get_input_embeddings().weight
    print("building PUBLIC-base tables...", flush=True)
    model.eval()
    F_base = collect_fp(model, token_ids, device, cfg)
    norm_base = emb.detach()[token_ids].float().norm(dim=1).cpu()
    print("  open-weight fp:", cos_rec(F_base, F_base),
          "| norm self:", norm_rec(norm_base, norm_base), flush=True)

    # ---- freeze all, then unfreeze embedding + attention proj (NO peft) ------
    for p in model.parameters():
        p.requires_grad_(False)
    embed_params, attn_params = [], []
    for name, p in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            p.requires_grad_(True); embed_params.append(p)
        elif _layer_idx(name) is not None and ".self_attn." in name and (
                any(name.endswith(x + suf) for x in
                    ("q_proj", "k_proj", "v_proj", "o_proj")
                    for suf in (".weight", ".bias"))):
            p.requires_grad_(True); attn_params.append(p)
    print(f"trainable: embed={sum(p.numel() for p in embed_params)/1e6:.0f}M "
          f"attn={sum(p.numel() for p in attn_params)/1e6:.0f}M", flush=True)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": embed_params, "lr": args.embed_lr},
                             {"params": attn_params, "lr": args.attn_lr}])

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
    print(f"{len(rows)} rows", flush=True)
    results = {"model": args.model, "candidate_tokens": n,
               "random_top1": round(100.0 / n, 4),
               "open_weight_fp": cos_rec(F_base, F_base),
               "open_weight_norm_self": norm_rec(norm_base, norm_base),
               "checkpoints": []}

    def measure(step):
        model.eval()
        Fq = collect_fp(model, token_ids, device, cfg)
        nq = model.get_input_embeddings().weight.detach()[token_ids].float(
            ).norm(dim=1).cpu()
        rec = {"step": step,
               "fp_public_table": cos_rec(F_base, Fq),
               "norm_public_table": norm_rec(norm_base, nq),
               "norm_self_table_finetuned": norm_rec(nq, nq),
               "fp_shift_l2_med": round(float((Fq - F_base).norm(dim=1).median()), 2),
               "fp_base_l2_med": round(float(F_base.norm(dim=1).median()), 2),
               "embed_norm_shift_med": round(float((nq - norm_base).abs().median()), 4)}
        model.train()
        print("MEASURE", json.dumps(rec), flush=True)
        results["checkpoints"].append(rec)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if 0 in ckpts:
        measure(0)
    step, ri, accum = 0, 0, 4
    model.train(); opt.zero_grad(); t0 = time.time()
    while step < args.max_steps:
        enc = tok(rows[ri % len(rows)], return_tensors="pt",
                  truncation=True, max_length=256).to(device)
        ri += 1
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / accum).backward()
        if ri % accum == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 50 == 0:
                print(f"  step {step} loss={float(out.loss):.4f} "
                      f"({(time.time()-t0)/step:.1f}s/s)", flush=True)
            if step in ckpts:
                measure(step)
    print("DONE", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
