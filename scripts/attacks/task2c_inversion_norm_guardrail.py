#!/usr/bin/env python
"""Closing loop: do the embedding-inversion (A1b, permutation-invariant sorted-|coords|)
and norm (A2) attacks also collapse under the layer-0 TEE guardrail?

Both attacks are mask-invariant (signed-perm N0 preserves sorted|coords| and norm),
so we attack the raw residual stream h_ell directly. The GPU-visible residual is
h_ell@N0; the guardrail relocates layer-0 attention into the TEE, so the GPU's
first-visible residual is layer-0 OUTPUT (hidden_states[1]) onward -- hidden_states[0]
(the raw embedding) is hidden. We measure per-layer, per-position recovery on real
chat prompts, bucketed template-prefix vs USER-CONTENT.

Attacker table = isolated-token residual at each layer (context-free per token):
  A1b: L2-NN on sort(|h_ell[v]|)   A2: 1-D NN on ||h_ell[v]||
hidden_states[0]=embedding (guardrail HIDES); [1]=layer-0 output; ...
"""
from __future__ import annotations
import argparse, json
import torch


def top1_bucket(obs_by_pos, table, mode, prefix_len, chunkT=2048):
    # table: [C, d] (sortedabs) or [C] (norm). obs_by_pos: list of (feat[P,*], ids)
    C = table.shape[0]
    hits = {"pos0": [0, 0], "template_prefix": [0, 0], "USER_CONTENT": [0, 0]}
    for feat, ids in obs_by_pos:
        P = feat.shape[0]
        dev = feat.device
        # nearest table row for each position
        pred = torch.empty(P, dtype=torch.long, device=dev)
        for s in range(0, P, 64):
            fs = feat[s:s + 64]
            if mode == "sortedabs":
                # L2 distance in chunks over table
                best = torch.full((fs.shape[0],), 1e30, device=dev)
                bidx = torch.zeros(fs.shape[0], dtype=torch.long, device=dev)
                for c in range(0, C, chunkT):
                    d = torch.cdist(fs, table[c:c + chunkT])
                    mv, mi = d.min(dim=1)
                    upd = mv < best
                    best[upd] = mv[upd]; bidx[upd] = mi[upd] + c
                pred[s:s + 64] = bidx
            else:  # norm 1-D
                d = (fs.view(-1, 1) - table.view(1, -1)).abs()
                pred[s:s + 64] = d.argmin(dim=1)
        for p in range(P):
            tid = ids[p]
            if tid < 0:
                continue
            hit = int(pred[p].item() == tid)   # table row index == candidate slot
            b = "pos0" if p == 0 else ("template_prefix" if p < prefix_len else "USER_CONTENT")
            hits[b][0] += hit; hits[b][1] += 1
    return {k: (round(100 * v[0] / v[1], 2) if v[1] else None) for k, v in hits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cand", type=int, default=8000)
    ap.add_argument("--n-seq", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--layers", default="0,1,2,4,8,16,27")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device).eval()
    V = model.config.vocab_size
    sel = [int(x) for x in args.layers.split(",")]

    # candidate set = tokens appearing in prompts + fillers; slot index = position in cand
    sysm = "You are a helpful assistant."
    def render(q):
        s = tok.apply_chat_template(
            [{"role": "system", "content": sysm}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False)["input_ids"]
    prefix_len = len(tok(tok.apply_chat_template(
        [{"role": "system", "content": sysm}, {"role": "user", "content": ""}],
        add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"])

    raws = []
    with open("/root/autodl-tmp/datasets/privacy_llm_lora/gsm8k_train/"
              "gsm8k_train_200_lora_dolly_compat.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q = str(d.get("instruction") or "")
            if len(q) > 20:
                raws.append(q)
            if len(raws) >= args.n_seq:
                break
    enc_seqs = [render(q)[:args.max_len] for q in raws]
    appearing = []
    for s in enc_seqs:
        appearing += s
    appearing = list(dict.fromkeys(appearing))
    g = torch.Generator().manual_seed(0)
    fillers = torch.randperm(V, generator=g).tolist()
    cand = list(dict.fromkeys(appearing + fillers))[:max(args.n_cand, len(appearing))]
    slot = {int(v): i for i, v in enumerate(cand)}
    cand_t = torch.tensor(cand)
    print(f"{len(enc_seqs)} prompts, prefix~{prefix_len}, {len(cand)} candidates", flush=True)

    # ---- isolated-token tables per selected layer --------------------------
    tbl_sa = {ell: [] for ell in sel}     # sorted-abs
    tbl_nm = {ell: [] for ell in sel}     # norm
    for s in range(0, len(cand_t), 256):
        ids = cand_t[s:s + 256].to(device).unsqueeze(1)
        with torch.no_grad():
            hs = model(input_ids=ids, use_cache=False,
                       output_hidden_states=True).hidden_states
        for ell in sel:
            h = hs[ell][:, 0, :].float()
            tbl_sa[ell].append(h.abs().sort(dim=1).values.cpu())
            tbl_nm[ell].append(h.norm(dim=1).cpu())
        print(f"  table {s+ids.shape[0]}/{len(cand_t)}", flush=True)
    TSA = {ell: torch.cat(tbl_sa[ell], 0).to(device) for ell in sel}
    TNM = {ell: torch.cat(tbl_nm[ell], 0).to(device) for ell in sel}

    # ---- observed on real prompts ------------------------------------------
    obs_sa = {ell: [] for ell in sel}
    obs_nm = {ell: [] for ell in sel}
    for ids in enc_seqs:
        idt = torch.tensor(ids, device=device).unsqueeze(0)
        with torch.no_grad():
            hs = model(input_ids=idt, use_cache=False,
                       output_hidden_states=True).hidden_states
        slots = [slot.get(t, -1) for t in ids]
        for ell in sel:
            h = hs[ell][0].float()
            obs_sa[ell].append((h.abs().sort(dim=1).values, slots))
            obs_nm[ell].append((h.norm(dim=1), slots))

    res = {"model": args.model, "n_prompts": len(enc_seqs),
           "candidates": len(cand), "template_prefix_len": prefix_len,
           "random_top1_pct": round(100.0 / len(cand), 4),
           "layer_meaning": "hidden_states[ell]: 0=embedding(guardrail HIDES), "
                            "1=layer0 output(GPU sees after guardrail), ...",
           "A1b_sortedabs_inversion": {}, "A2_norm": {}}
    for ell in sel:
        res["A1b_sortedabs_inversion"][f"hs{ell}"] = top1_bucket(
            obs_sa[ell], TSA[ell], "sortedabs", prefix_len)
        res["A2_norm"][f"hs{ell}"] = top1_bucket(
            obs_nm[ell], TNM[ell], "norm", prefix_len)
        print(f"layer {ell}: inv={res['A1b_sortedabs_inversion'][f'hs{ell}']} "
              f"norm={res['A2_norm'][f'hs{ell}']}", flush=True)

    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
