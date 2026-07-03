#!/usr/bin/env python
"""GO/NO-GO security attack on OUR A_rightmul folded scheme: can an honest-but-
curious server recover the secret residual mask N_res (and thus user tokens) from
the offloaded folded weights + the public model?

Threat model (matches our real deployment):
  - Attacker holds the PUBLIC Qwen weights W and RMSNorm gains (we fold EXACT
    public weights -- no fine-tuning), the offloaded folded weights W', and the
    masked residual states X@N_res that cross to the GPU.
  - Secret = the mask family {N_res (signed permutation, GLOBAL/shared in the
    A_rightmul default), per-head pairwise-rotation attn masks, SwiGLU perm}.

The exploitable invariant (derived + verified here). For every left-shared
projection i in {q,k,v,gate,up}, the fold is  W'_i = N_res^{-1} (g_i ⊙ W_i) M_i
with M_i an orthogonal OUTPUT mask. The LEFT (self) Gram cancels M_i:
      W'_i W'_i^T = N_res^{-1} A_i A_i^T N_res^{-T} = P (A_i A_i^T) P^T,
where A_i = g_i⊙W_i is PUBLIC and P = N_res^{-1} = N_res^T is a signed permutation.
=> diag(W'_i W'_i^T) is a PERMUTATION of diag(A_i A_i^T); if those diagonals are
distinct, P's permutation is recovered EXACTLY by matching, and signs from the
off-diagonals. (The cross-Gram (W'_K)^T W'_Q does NOT work for us: our masks are on
the RIGHT, so it leaves the attn masks uncancelled -- the LEFT self-Gram is the
right surface. Two-sided per-family ALS is unidentifiable; see synthetic probe.)

Attack 1  per-projection + aggregated N_res recovery (perm via diagonal match,
          signs via off-diagonal), reported with the identifiability metric
          (diag distinctness) and the WW^T spectrum.
Attack 3  end-to-end token recovery: un-mask X@N_res with the recovered N_res and
          cosine-NN against the public embedding table; top-1/10/100 vs a random-
          mask baseline. CONJFORMER-Table-3-comparable.

Weight-only + one embedding NN => fast, no generation. Prints a GO/NO-GO verdict.
"""
from __future__ import annotations

import argparse
import json

import torch


def sample_signed_perm(d, g, device):
    perm = torch.randperm(d, generator=g)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(torch.float64)
    N = torch.zeros(d, d, dtype=torch.float64)
    N[torch.arange(d), perm] = signs                      # N[i, perm[i]] = signs[i]
    return N.to(device), perm.to(device), signs.to(device)


def sample_pairwise_rotation(d, g, device):
    ang = torch.rand(d // 2, generator=g, dtype=torch.float64) * 6.283185307179586
    c, s = torch.cos(ang), torch.sin(ang)
    M = torch.zeros(d, d, dtype=torch.float64)
    idx = torch.arange(d // 2)
    M[2 * idx, 2 * idx] = c; M[2 * idx, 2 * idx + 1] = -s
    M[2 * idx + 1, 2 * idx] = s; M[2 * idx + 1, 2 * idx + 1] = c
    return M.to(device)


def sample_perm(d, g, device):
    perm = torch.randperm(d, generator=g)
    P = torch.zeros(d, d, dtype=torch.float64)
    P[torch.arange(d), perm] = 1.0
    return P.to(device)


def recover_signed_perm_from_gram(Gt, G):
    """Recover P (signed perm) s.t. Gt = P G P^T, from diagonals + off-diagonals.

    Returns (perm_hat, signs_hat, diag_distinct_frac). perm_hat[i] = j means row i
    of P has its nonzero in column j (i.e. Gt_ii == G_jj)."""
    d = G.shape[0]
    dgt, dg = torch.diagonal(Gt), torch.diagonal(G)
    # match each observed diagonal to the nearest public diagonal (greedy 1-NN;
    # exact when the public diagonal is distinct)
    perm_hat = (dgt.view(-1, 1) - dg.view(1, -1)).abs().argmin(dim=1)
    # identifiability: fraction of distinct public diagonals (rounded)
    distinct = torch.unique(torch.round(dg * 1e4)).numel() / d
    # signs: Gt_ij = s_i s_j G_{p(i),p(j)} => s_i s_anchor = sign(Gt_ia * G_{p(i)p(a)}).
    # global sign is unrecoverable from a Gram, so fix s_anchor=+1.
    Gp = G[perm_hat][:, perm_hat]                          # G_{p(i),p(j)}
    anchor = int(dgt.abs().argmax())
    signs_hat = torch.sign(Gt[:, anchor] * Gp[:, anchor])  # s_i (up to global flip)
    signs_hat[signs_hat == 0] = 1.0
    signs_hat[anchor] = 1.0
    return perm_hat, signs_hat, float(distinct)


def build_N_from_perm_signs(perm_hat, signs_hat, d, device):
    N = torch.zeros(d, d, dtype=torch.float64, device=device)
    N[torch.arange(d, device=device), perm_hat] = signs_hat
    return N


def nn_topk(Q, T, ks=(1, 10, 100), chunk=1024):
    n = Q.shape[0]
    Qn = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
    Tn = T / (T.norm(dim=1, keepdim=True) + 1e-9)
    idx = torch.arange(n, device=Q.device)
    hit = {k: 0 for k in ks}
    for s in range(0, n, chunk):
        order = (Qn[s:s + chunk] @ Tn.T).argsort(dim=1, descending=True)
        tgt = idx[s:s + chunk].unsqueeze(1)
        for k in ks:
            hit[k] += int((order[:, :k] == tgt).any(1).sum())
    return {f"top{k}": round(100 * hit[k] / n, 3) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-layers", type=int, default=-1, help="-1 = all")
    ap.add_argument("--n-tokens", type=int, default=6000)
    ap.add_argument("--mask-seed", type=int, default=2035)
    ap.add_argument("--wtilde-store-dtype", default="float64",
                    choices=["float64", "float32", "bfloat16"],
                    help="simulate the offloaded folded-weight storage precision")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    from transformers import AutoModelForCausalLM
    print(f"[load] {a.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    cfg = model.config
    d = cfg.hidden_size
    nL = cfg.num_hidden_layers if a.n_layers < 0 else min(a.n_layers, cfg.num_hidden_layers)
    E = model.get_input_embeddings().weight.detach().to(torch.float64).to(dev)
    V = E.shape[0]
    g = torch.Generator().manual_seed(a.mask_seed)

    # secret GLOBAL residual mask (signed permutation), shared across all layers
    N_res, perm_true, signs_true = sample_signed_perm(d, g, dev)
    P_true = N_res.T                                       # = N_res^{-1}, the left factor
    perm_P = P_true.abs().argmax(dim=1)                    # ground-truth perm of P
    signs_P = P_true[torch.arange(d, device=dev), perm_P]

    proj_names = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    per_proj = []
    votes = torch.zeros(d, d, dtype=torch.float64, device=dev)   # accumulate perm votes
    sign_accum = torch.zeros(d, d, dtype=torch.float64, device=dev)

    for li in range(nL):
        layer = model.model.layers[li]
        rms1 = layer.input_layernorm.weight.detach().to(torch.float64).to(dev)     # [d]
        rms2 = layer.post_attention_layernorm.weight.detach().to(torch.float64).to(dev)
        mods = {"q_proj": (layer.self_attn.q_proj, rms1), "k_proj": (layer.self_attn.k_proj, rms1),
                "v_proj": (layer.self_attn.v_proj, rms1), "gate_proj": (layer.mlp.gate_proj, rms2),
                "up_proj": (layer.mlp.up_proj, rms2)}
        gl = torch.Generator().manual_seed(a.mask_seed + 101 * (li + 1))
        for name, (mod, rms) in mods.items():
            W = mod.weight.detach().to(torch.float64).to(dev).T          # [in=d, out]
            A = rms.unsqueeze(1) * W                                     # g ⊙ W (public), [d,out]
            dout = A.shape[1]
            # faithful OUTPUT mask (cancels in the left Gram); attn=pairwise rot, mlp=perm
            M_out = (sample_pairwise_rotation(dout, gl, dev) if name in ("q_proj", "k_proj", "v_proj")
                     else sample_perm(dout, gl, dev))
            W_tilde = P_true @ A @ M_out                                 # W' = N_res^{-1} (g⊙W) M_out
            if a.wtilde_store_dtype != "float64":                        # simulate stored precision
                sd = torch.float32 if a.wtilde_store_dtype == "float32" else torch.bfloat16
                W_tilde = W_tilde.to(sd).to(torch.float64)
            G = A @ A.T                                                  # public
            Gt = W_tilde @ W_tilde.T                                     # observed
            ph, sh, distinct = recover_signed_perm_from_gram(Gt, G)
            perm_acc = float((ph == perm_P).float().mean())
            # sign accuracy, global-flip-invariant (Gram can't fix the global sign)
            match = (torch.sign(sh) == torch.sign(signs_P)).float().mean()
            sacc = float(torch.maximum(match, 1.0 - match))
            per_proj.append({"layer": li, "proj": name, "perm_acc": round(perm_acc, 4),
                             "sign_acc_rel": round(sacc, 4), "diag_distinct_frac": round(distinct, 4)})
            votes[torch.arange(d, device=dev), ph] += 1.0
            sign_accum[torch.arange(d, device=dev), ph] += sh

    # aggregate: majority-vote permutation + majority sign
    perm_agg = votes.argmax(dim=1)
    perm_agg_acc = float((perm_agg == perm_P).float().mean())
    signs_agg = torch.sign(sign_accum[torch.arange(d, device=dev), perm_agg])
    signs_agg[signs_agg == 0] = 1.0
    P_hat = build_N_from_perm_signs(perm_agg, signs_agg, d, dev)   # recovered N_res^{-1}

    # ---- Attack 3: token recovery -------------------------------------------
    n = min(a.n_tokens, V)
    gt = torch.Generator().manual_seed(0)
    toks = torch.randperm(V, generator=gt)[:n].to(dev)
    X = E[toks]                                                   # true input embeddings
    X_masked = X @ N_res                                          # what the GPU sees
    # un-mask with recovered P_hat = N_res^{-1}:  X_masked @ N_res^{-1} = X
    X_rec = X_masked @ P_hat
    rec = nn_topk(X_rec, E[toks])
    # global-sign fallback (Gram loses global sign): try -X_rec too, keep best top1
    rec_neg = nn_topk(-X_rec, E[toks])
    if rec_neg["top1"] > rec["top1"]:
        rec = rec_neg
    # random-mask baseline
    Nr, _, _ = sample_signed_perm(d, torch.Generator().manual_seed(999), dev)
    base = nn_topk(X_masked @ Nr.T, E[toks])

    aggregate = {
        "perm_agg_acc": round(perm_agg_acc, 4),
        "single_proj_median_perm_acc": round(
            float(torch.tensor([p["perm_acc"] for p in per_proj]).median()), 4),
        "median_diag_distinct_frac": round(
            float(torch.tensor([p["diag_distinct_frac"] for p in per_proj]).median()), 4),
        "n_projections_attacked": len(per_proj),
    }
    token_rec = {"attack_recovered_mask": rec, "random_mask_baseline": base,
                 "random_top1_chance_pct": round(100.0 / n, 4)}
    verdict = ("NOT_SECURE (exact signed-perm N_res recovery -> token collapse)"
               if perm_agg_acc > 0.99 and rec["top1"] > 50
               else "PARTIAL/UNDETERMINED -- inspect per-layer")
    report = {"model": a.model, "hidden": d, "n_layers": nL, "vocab": V, "n_tokens": n,
              "aggregate": aggregate, "token_recovery": token_rec,
              "verdict": verdict, "per_projection": per_proj}
    print(json.dumps({"aggregate": aggregate, "token_recovery": token_rec,
                      "verdict": verdict}, indent=2), flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
