#!/usr/bin/env python
"""Experiment 1b: can *aggressive* fine-tuning (real trainable embedding +
higher-rank attention LoRA, longer) defeat the attention-fingerprint attack?

Attacker table is fixed to the pristine PUBLIC base. We report cosine-NN
recovery vs training steps, AND diagnostics that explain robustness:
  * cosine recovery (standard attack)
  * common-mode-removed cosine recovery (subtract mean fingerprint) -- tests
    whether the fine-tune shift is just a shared/low-rank common-mode offset
    that cosine already ignores.
  * embedding-norm attack recovery (now with a genuinely trainable embedding).
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


def cos_recovery(F_table, F_query, ks=(1, 10, 100)):
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
    return {f"top{k}": round(hit[k] / n * 100, 3) for k in ks}


def norm_recovery(nt, nq, ks=(1, 10, 100)):
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
    ap.add_argument("--checkpoints", default="0,200,600")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--embed-lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/root/private_weight_defense_b_result.json")
    args = ap.parse_args()

    ckpts = sorted(int(x) for x in args.checkpoints.split(","))
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading base...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    cfg = model.config
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(0)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]

    print("building PUBLIC-base tables...", flush=True)
    F_base = collect_fp(model, token_ids, device, cfg)
    norm_base = model.get_input_embeddings().weight.detach()[token_ids].float(
        ).norm(dim=1).cpu()
    Fb_cm = F_base - F_base.mean(0, keepdim=True)   # common-mode removed
    print("  open-weight fp:", cos_recovery(F_base, F_base),
          "| tied_embed:", bool(getattr(cfg, "tie_word_embeddings", False)),
          flush=True)

    results = {"model": args.model, "candidate_tokens": n, "rank": args.rank,
               "random_top1": round(100.0 / n, 4),
               "open_weight_fp": cos_recovery(F_base, F_base),
               "checkpoints": []}

    lconf = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf)
    # make the embedding genuinely trainable (peft froze it)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    rows = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            txt = ""
            for kk in ("instruction", "prompt", "question"):
                if d.get(kk):
                    txt += str(d[kk]) + "\n"
            for kk in ("response", "output", "answer"):
                if d.get(kk):
                    txt += str(d[kk]); break
            if txt.strip():
                rows.append(txt.strip())
    print(f"{len(rows)} rows", flush=True)

    lora_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and "lora" in n_.lower()]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 1e-4},
         {"params": [emb], "lr": args.embed_lr}])

    def measure(step):
        model.eval()
        Fq = collect_fp(model, token_ids, device, cfg)
        nq = model.get_input_embeddings().weight.detach()[token_ids].float(
            ).norm(dim=1).cpu()
        Fq_cm = Fq - Fq.mean(0, keepdim=True)
        rec = {"step": step,
               "fp_cosine": cos_recovery(F_base, Fq),
               "fp_cosine_common_mode_removed": cos_recovery(Fb_cm, Fq_cm),
               "norm_attack": norm_recovery(norm_base, nq),
               "fp_shift_l2_median": round(float(
                   (Fq - F_base).norm(dim=1).median()), 3),
               "fp_base_l2_median": round(float(
                   F_base.norm(dim=1).median()), 3),
               "embed_norm_shift_median": round(float(
                   (nq - norm_base).abs().median()), 5)}
        model.train()
        print("MEASURE", json.dumps(rec), flush=True)
        results["checkpoints"].append(rec)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if 0 in ckpts:
        measure(0)
    step, ri, accum = 0, 0, 4
    opt.zero_grad()
    t0 = time.time()
    while step < args.max_steps:
        enc = tok(rows[ri % len(rows)], return_tensors="pt",
                  truncation=True, max_length=256).to(device)
        ri += 1
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / accum).backward()
        if ri % accum == 0:
            opt.step(); opt.zero_grad(); step += 1
            if step % 40 == 0:
                print(f"  step {step} loss={float(out.loss):.4f} "
                      f"({(time.time()-t0)/step:.1f}s/s)", flush=True)
            if step in ckpts:
                measure(step)
    print("DONE", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
