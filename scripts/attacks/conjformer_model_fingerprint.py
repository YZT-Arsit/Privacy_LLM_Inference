#!/usr/bin/env python
"""Confirm the attention-logit fingerprint attack on CONJFORMER's OWN model
families (GPT-2, Llama-3.2-1B). CONJFORMER's conjugation preserves attention
logits (their Fig. 2: Q_hat K_hat^T = Q K^T), so the fingerprint is unchanged by
their rotation; the only open question is whether these smaller models' logits
are distinctive enough to identify tokens. We test exactly that.

Handles GPT-2 (fused c_attn Conv1D) and Llama/Qwen (q_proj/k_proj, GQA).
Isolated single tokens at position 0; per-(layer,head) diagonal self-logit.
"""
from __future__ import annotations
import argparse, json
import torch


def collect_gpt2(model, cfg, token_ids, device, batch=512):
    n_head, n_embd = cfg.n_head, cfg.n_embd
    hd, L = n_embd // n_head, cfg.n_layer
    cap, handles = {}, []
    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.attn.c_attn.register_forward_hook(
            lambda m, inp, o, i=i: cap.__setitem__(i, o.detach())))
    feats = []
    try:
        for s in range(0, len(token_ids), batch):
            ids = token_ids[s:s + batch].to(device).unsqueeze(1)
            cap.clear()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            B = ids.shape[0]
            fb = torch.empty(B, L * n_head, dtype=torch.float32)
            for i in range(L):
                qkv = cap[i][:, 0, :]                       # [B, 3*n_embd]
                q = qkv[:, :n_embd].reshape(B, n_head, hd).float()
                k = qkv[:, n_embd:2 * n_embd].reshape(B, n_head, hd).float()
                fb[:, i * n_head:(i + 1) * n_head] = (q * k).sum(-1) / (hd ** 0.5)
            feats.append(fb)
            print(f"  fp {s+B}/{len(token_ids)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0)


def collect_llama(model, cfg, token_ids, device, batch=512):
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    grp = n_q // n_kv
    L = cfg.num_hidden_layers
    cap, handles = {}, []
    layers = model.model.layers
    for i, lyr in enumerate(layers):
        handles.append(lyr.self_attn.q_proj.register_forward_hook(
            lambda m, inp, o, i=i: cap.__setitem__((i, "q"), o.detach())))
        handles.append(lyr.self_attn.k_proj.register_forward_hook(
            lambda m, inp, o, i=i: cap.__setitem__((i, "k"), o.detach())))
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
            print(f"  fp {s+B}/{len(token_ids)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0)


def recover(F_table, F_query, ks=(1, 10, 100)):
    n = F_table.shape[0]
    Ft = F_table / (F_table.norm(dim=1, keepdim=True) + 1e-9)
    Fq = F_query / (F_query.norm(dim=1, keepdim=True) + 1e-9)
    idx = torch.arange(n)
    hit = {k: 0 for k in ks}
    for s in range(0, n, 1024):
        order = (Fq[s:s + 1024] @ Ft.T).argsort(dim=1, descending=True)
        tgt = idx[s:s + 1024].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}_pct": round(hit[k] / n * 100, 3) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", choices=["gpt2", "llama"], required=True)
    ap.add_argument("--n-tokens", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoModelForCausalLM
    print("loading", args.model, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=device).eval()
    cfg = model.config
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(0)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]
    coll = collect_gpt2 if args.arch == "gpt2" else collect_llama
    F = coll(model, cfg, token_ids, device)
    # table fp32, query = bf16 round-trip (realistic observed-vs-table precision gap)
    F_table = F.float()
    F_query = F.bfloat16().float()
    res = {"model": args.model, "arch": args.arch, "candidate_tokens": n,
           "fingerprint_dim": int(F.shape[1]),
           "random_top1_pct": round(100.0 / n, 4),
           "exact_table_vs_table": recover(F_table, F_table),
           "bf16_query_vs_fp32_table": recover(F_table, F_query)}
    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
