#!/usr/bin/env python
"""Real-weight attention-logit fingerprint attack on Qwen2.5-7B.

Validates the synthetic A3 result on the actual model. By the (proven, exact)
mask + pad invariance of the A_rightmul path, the attention logit the GPU observes
equals the plaintext logit, so the ONLY open question is whether real Qwen
attention-logit fingerprints are distinctive enough to identify tokens. We test
exactly that, with a precision gap (fp32 table vs bf16 query) to emulate the
observed-vs-table numeric mismatch a real attacker faces.

Fingerprint f(token) in R^{L*Hq}: the position-0 diagonal self-attention logit
q_i . k_{group(i)} / sqrt(head_dim) per (layer, query-head). RoPE(0)=I, and for a
same-position pair RoPE cancels anyway, so this is exactly what the GPU sees on the
attention-score diagonal -- no offline re-inference needed.

Attacker procedure emulated:
  1. build table F_table[v] for a candidate vocab set (public weights)   [fp32]
  2. observe F_query[token] (== table by invariance, up to precision)     [bf16]
  3. NN match F_query -> F_table, report top-1/10/100 recovery.

Usage (on H800):
  python real_attention_fingerprint.py --model <path> --n-tokens 20000
"""

from __future__ import annotations

import argparse
import torch


def collect_fingerprints(model, tok, token_ids, device, dtype, batch=512):
    """Return F [N, L*Hq] of position-0 self-attention logits for each token id."""
    layers = model.model.layers
    L = len(layers)
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = cfg.hidden_size // n_q
    grp = n_q // n_kv
    cap = {}

    def mk_hook(idx, which):
        def hook(mod, inp, out):
            cap[(idx, which)] = out.detach()
        return hook

    handles = []
    for i, lyr in enumerate(layers):
        handles.append(lyr.self_attn.q_proj.register_forward_hook(mk_hook(i, "q")))
        handles.append(lyr.self_attn.k_proj.register_forward_hook(mk_hook(i, "k")))

    feats = []
    try:
        for s in range(0, len(token_ids), batch):
            ids = token_ids[s:s + batch].to(device).unsqueeze(1)  # [B,1]
            cap.clear()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            B = ids.shape[0]
            fb = torch.empty(B, L * n_q, dtype=torch.float32)
            for i in range(L):
                q = cap[(i, "q")][:, 0, :].reshape(B, n_q, hd).float()
                k = cap[(i, "k")][:, 0, :].reshape(B, n_kv, hd).float()
                kg = k.repeat_interleave(grp, dim=1)                # [B, n_q, hd]
                logit = (q * kg).sum(-1) / (hd ** 0.5)              # [B, n_q]
                fb[:, i * n_q:(i + 1) * n_q] = logit
            feats.append(fb)
            print(f"  fingerprinted {s + B}/{len(token_ids)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return torch.cat(feats, 0)                                       # [N, L*n_q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tokens", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/attack_fingerprint_result.json")
    args = ap.parse_args()

    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    V = model.config.vocab_size
    g = torch.Generator().manual_seed(args.seed)
    n = min(args.n_tokens, V)
    token_ids = torch.randperm(V, generator=g)[:n]

    # TABLE: attacker's precomputed fingerprints (compute in the model's dtype;
    # then upcast to fp32 as the stored table).
    print(f"building fingerprint table over {n} candidate tokens...", flush=True)
    F = collect_fingerprints(model, tok, token_ids, device, torch.bfloat16)

    # QUERY vs TABLE with a precision gap: table = F (fp32), query = F + bf16-round
    # noise (emulates observed folded-path logit vs public-weight table).
    F_table = F.float()
    F_query = F.bfloat16().float()                # round-trip bf16 == observed side

    # normalize + cosine NN
    Ft = F_table / (F_table.norm(dim=1, keepdim=True) + 1e-9)
    Fq = F_query / (F_query.norm(dim=1, keepdim=True) + 1e-9)
    # chunked argsort to bound memory
    top1 = top10 = top100 = 0
    idx = torch.arange(n)
    CH = 1024
    for s in range(0, n, CH):
        sims = Fq[s:s + CH] @ Ft.T                 # [chunk, N]
        order = sims.argsort(dim=1, descending=True)
        tgt = idx[s:s + CH].unsqueeze(1)
        top1 += int((order[:, :1] == tgt).any(1).sum())
        top10 += int((order[:, :10] == tgt).any(1).sum())
        top100 += int((order[:, :100] == tgt).any(1).sum())
    res = {
        "model": args.model, "candidate_tokens": n,
        "fingerprint_dim": F.shape[1],
        "top1_pct": round(top1 / n * 100, 3),
        "top10_pct": round(top10 / n * 100, 3),
        "top100_pct": round(top100 / n * 100, 3),
        "note": "table=fp32, query=bf16-roundtrip (precision gap); cosine NN; "
                "candidate set = n random vocab tokens (full vocab would add "
                "more collisions).",
    }
    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
