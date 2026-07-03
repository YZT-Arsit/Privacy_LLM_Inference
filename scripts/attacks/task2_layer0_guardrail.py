#!/usr/bin/env python
"""Task 2: quantify the layer-0 TEE-relocation guardrail on REAL multi-token prompts.

The isolated-token fingerprint attack (A3) recovers 98.5% because every layer is
context-free for a length-1 input. In real deployment only LAYER 0's diagonal
self-logit is a context-free function of token_p; layers 1..27 are contextualized.
The proposed guardrail moves layer-0 attention (RMSNorm0 + QK + softmax + O + residual)
into the TEE at the *existing* single handoff (no extra round-trip), so the GPU never
sees layer-0 attention logits nor the raw masked embedding.

We measure, on real prompts, the per-token recovery an adversary gets from the
GPU-visible attention diagonals, using an isolated-token table (attacker's public
precompute). We compare:
  * FULL      : all 28 layers visible (no guardrail)         -> 784-dim
  * NO_LAYER0 : layers 1..27 visible (guardrail applied)     -> 756-dim
  * per-layer single-layer recovery (28-dim) to show the context gradient
Recovery is bucketed by token position (0, 1-4, 5-15, 16+) because the guardrail's
value grows with available context.
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


def register(model):
    cap = {}
    handles = []
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
    """Per-(layer,head) diagonal self-logit for every position -> [B,S,L,Hq]."""
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_q
    grp = n_q // n_kv
    L = cfg.num_hidden_layers
    out = torch.empty(B, S, L, n_q, dtype=torch.float32)
    for i in range(L):
        q = cap[(i, "q")].reshape(B, S, n_q, hd).float()
        k = cap[(i, "k")].reshape(B, S, n_kv, hd).float()
        kg = k.repeat_interleave(grp, dim=2)
        out[:, :, i, :] = (q * kg).sum(-1) / (hd ** 0.5)   # diagonal (p,p): RoPE cancels
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-cand", type=int, default=10000)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--out", default="/root/task2_layer0_guardrail_result.json")
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    cfg = model.config
    V = cfg.vocab_size
    L, Hq = cfg.num_hidden_layers, cfg.num_attention_heads

    # --- real prompts -------------------------------------------------------
    seqs = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = str(d.get("instruction") or d.get("question") or d.get("prompt") or "")
            if len(t) > 20:
                seqs.append(t)
            if len(seqs) >= args.n_seq:
                break
    print(f"{len(seqs)} prompts", flush=True)

    # --- candidate table: isolated-token diagonal fp per layer --------------
    # candidate set = all tokens that appear in the prompts + random fillers
    appearing = set()
    enc_seqs = []
    for t in seqs:
        ids = tok(t, truncation=True, max_length=args.max_len)["input_ids"]
        enc_seqs.append(ids)
        appearing.update(ids)
    appearing = list(appearing)
    g = torch.Generator().manual_seed(0)
    fillers = torch.randperm(V, generator=g).tolist()
    cand = list(dict.fromkeys(appearing + fillers))[:max(args.n_cand, len(appearing))]
    cand_t = torch.tensor(cand)
    pos_in_cand = {int(v): i for i, v in enumerate(cand)}
    print(f"candidate pool = {len(cand)} tokens ({len(appearing)} appearing)", flush=True)

    cap, handles = register(model)
    # table (isolated)
    print("building isolated table...", flush=True)
    tbl = []
    for s in range(0, len(cand_t), 256):
        ids = cand_t[s:s + 256].to(device).unsqueeze(1)
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        tbl.append(diag_fp(cap, cfg, ids.shape[0], 1)[:, 0])   # [b,L,Hq]
    TABLE = torch.cat(tbl, 0)                                    # [C,L,Hq]

    # observed (contextual) fingerprints for real prompts
    print("running real prompts...", flush=True)
    obs = []          # list of (fp[S,L,Hq], token_ids[S])
    for ids in enc_seqs:
        idt = torch.tensor(ids, device=device).unsqueeze(0)
        cap.clear()
        with torch.no_grad():
            model(input_ids=idt, use_cache=False)
        fp = diag_fp(cap, cfg, 1, idt.shape[1])[0]              # [S,L,Hq]
        obs.append((fp, ids))
    for h in handles:
        h.remove()

    # --- recovery under different visible-layer sets ------------------------
    def recover(layer_mask):
        # layer_mask: bool over L. returns dict of bucket -> (hits, total)
        Tsel = TABLE[:, layer_mask, :].reshape(TABLE.shape[0], -1)   # [C, d]
        Tn = Tsel / (Tsel.norm(dim=1, keepdim=True) + 1e-9)
        buckets = {"pos0": [0, 0], "pos1_4": [0, 0], "pos5_15": [0, 0], "pos16+": [0, 0], "all": [0, 0]}
        for fp, ids in obs:
            F = fp[:, layer_mask, :].reshape(fp.shape[0], -1)
            Fn = F / (F.norm(dim=1, keepdim=True) + 1e-9)
            sims = Fn @ Tn.T                                          # [S, C]
            pred = sims.argmax(1)
            for p, tid in enumerate(ids):
                if tid not in pos_in_cand:
                    continue
                hit = int(pred[p].item() == pos_in_cand[tid])
                b = ("pos0" if p == 0 else "pos1_4" if p < 5 else
                     "pos5_15" if p < 16 else "pos16+")
                for key in (b, "all"):
                    buckets[key][0] += hit
                    buckets[key][1] += 1
        return {k: (round(100 * v[0] / v[1], 2) if v[1] else None) for k, v in buckets.items()}

    full = torch.ones(L, dtype=torch.bool)
    no0 = full.clone(); no0[0] = False
    no01 = no0.clone(); no01[1] = False
    res = {
        "model": args.model, "n_prompts": len(obs), "candidates": len(cand),
        "recovery_FULL_all_layers": recover(full),
        "recovery_GUARDRAIL_drop_layer0": recover(no0),
        "recovery_drop_layer0_and_1": recover(no01),
        "per_layer_top1_all_positions": {
            f"layer{ell}": recover(torch.tensor(
                [j == ell for j in range(L)]))["all"] for ell in range(L)},
    }
    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
