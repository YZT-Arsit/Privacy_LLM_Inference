#!/usr/bin/env python
"""Task 2b: (1) how many early layers must be relocated into the TEE to drive
position-0 recovery to random, and (2) on REAL chat-formatted prompts, does the
layer-0 guardrail protect the USER-CONTENT tokens (which carry context) even
though the fixed public template prefix still leaks?

Key structural fact: position 0 attends only to itself at every layer, so its
fingerprint is context-free at ALL layers -> relocating early layers cannot zero
it (deep layers still carry a context-free token_0 signature). The practical fix
is that early positions are the fixed chat template (public), and the user's
sensitive tokens come after it with context.
"""
from __future__ import annotations
import argparse, json
import torch


def _layer_idx(name):
    parts = name.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None


def register(model):
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
    return cap, handles


def diag_fp(cap, cfg, B, S):
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_q
    grp = n_q // n_kv
    L = cfg.num_hidden_layers
    out = torch.empty(B, S, L, n_q, dtype=torch.float32)
    for i in range(L):
        q = cap[(i, "q")].reshape(B, S, n_q, hd).float()
        k = cap[(i, "k")].reshape(B, S, n_kv, hd).float()
        kg = k.repeat_interleave(grp, dim=2)
        out[:, :, i, :] = (q * kg).sum(-1) / (hd ** 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-cand", type=int, default=10000)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default="/root/task2b_result.json")
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device).eval()
    cfg = model.config
    V, L = cfg.vocab_size, cfg.num_hidden_layers

    # chat-formatted prompts + template prefix length
    sys_msg = "You are a helpful assistant."
    def render(q):
        s = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False)["input_ids"]
    _pre = tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": ""}],
        add_generation_prompt=True, tokenize=False)
    prefix_len = len(tok(_pre, add_special_tokens=False)["input_ids"])

    raws = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q = str(d.get("instruction") or d.get("question") or "")
            if len(q) > 20:
                raws.append(q)
            if len(raws) >= args.n_seq:
                break
    enc_seqs = [render(q)[:args.max_len] for q in raws]
    print(f"{len(enc_seqs)} chat prompts; template prefix ~{prefix_len} tokens", flush=True)

    appearing = set()
    for ids in enc_seqs:
        appearing.update(ids)
    appearing = list(appearing)
    g = torch.Generator().manual_seed(0)
    fillers = torch.randperm(V, generator=g).tolist()
    cand = list(dict.fromkeys(appearing + fillers))[:max(args.n_cand, len(appearing))]
    cand_t = torch.tensor(cand)
    pos_in_cand = {int(v): i for i, v in enumerate(cand)}
    print(f"candidates = {len(cand)}", flush=True)

    cap, handles = register(model)
    tbl = []
    for s in range(0, len(cand_t), 256):
        ids = cand_t[s:s + 256].to(device).unsqueeze(1)
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        tbl.append(diag_fp(cap, cfg, ids.shape[0], 1)[:, 0])
    TABLE = torch.cat(tbl, 0)                                    # [C,L,Hq]

    obs = []
    for ids in enc_seqs:
        idt = torch.tensor(ids, device=device).unsqueeze(0)
        cap.clear()
        with torch.no_grad():
            model(input_ids=idt, use_cache=False)
        obs.append((diag_fp(cap, cfg, 1, idt.shape[1])[0], ids))
    for h in handles:
        h.remove()

    def recover(layer_mask):
        Tsel = TABLE[:, layer_mask, :].reshape(TABLE.shape[0], -1)
        Tn = Tsel / (Tsel.norm(dim=1, keepdim=True) + 1e-9)
        b = {"pos0": [0, 0], "pos1_4": [0, 0], "template_prefix": [0, 0],
             "USER_CONTENT": [0, 0], "all": [0, 0]}
        for fp, ids in obs:
            F = fp[:, layer_mask, :].reshape(fp.shape[0], -1)
            Fn = F / (F.norm(dim=1, keepdim=True) + 1e-9)
            pred = (Fn @ Tn.T).argmax(1)
            for p, tid in enumerate(ids):
                if tid not in pos_in_cand:
                    continue
                hit = int(pred[p].item() == pos_in_cand[tid])
                keys = ["all"]
                keys.append("pos0" if p == 0 else "pos1_4" if p < 5 else "zzz")
                keys.append("template_prefix" if p < prefix_len else "USER_CONTENT")
                for kk in keys:
                    if kk in b:
                        b[kk][0] += hit; b[kk][1] += 1
        return {k: (round(100 * v[0] / v[1], 2) if v[1] else None) for k, v in b.items()}

    def mask_drop(kdrop):
        m = torch.ones(L, dtype=torch.bool)
        m[:kdrop] = False
        return m

    res = {"model": args.model, "n_prompts": len(obs), "candidates": len(cand),
           "template_prefix_len": prefix_len,
           "random_top1_pct": round(100.0 / len(cand), 4),
           "no_guardrail_all_layers": recover(torch.ones(L, dtype=torch.bool)),
           "relocate_first_k_layers": {
               f"k={k}": recover(mask_drop(k)) for k in (1, 2, 3, 4, 6)},
           "per_layer_pos0": {
               f"layer{ell}": recover(torch.tensor(
                   [j == ell for j in range(L)]))["pos0"] for ell in range(0, L, 4)},
           }
    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
