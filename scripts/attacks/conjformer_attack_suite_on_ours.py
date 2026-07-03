#!/usr/bin/env python
"""Run CONJFORMER's attack suite against OUR A_rightmul folded scheme
(signed-permutation residual/input mask, pairwise-rotation attention mask,
EXACT public-weight folding -- we do NOT fine-tune). Honest self-audit: are our
STRUCTURED masks + exact weights at least as robust as CONJFORMER's dense
rotation + fine-tuning?

Attacks (all with the adversary holding the PUBLIC weights, which for us == the
deployed weights since we fold exact public W):
  A1 embedding inversion on the masked residual embedding X@N0
     a) naive cosine NN vs public E            (expect: defeated, like a rotation)
     b) permutation-invariant NN: sort(|.|)    (expect: BREAKS signed-perm mask)
  A2 norm attack: ||X@N0|| vs ||E||            (orthogonal -> norm leaks)
  A3 Gram mask-recovery: right-Gram of a folded weight = P (W W^T) P^T with P a
     signed permutation -> its DIAGONAL is a permutation of diag(W W^T); if those
     are distinct, recover N0 EXACTLY (only possible because we fold exact public
     W). Then un-mask the residual and re-invert -> full collapse.
  A4 two-sided ALS-Procrustes per family: with independent output masks the
     continuous-orthogonal objective is unidentifiable (residual ~0 for many A) --
     but A3 shows a STRUCTURE-AWARE attacker bypasses that entirely.
"""
from __future__ import annotations
import argparse, json
import torch


def signed_perm(d, g, device):
    perm = torch.randperm(d, generator=g)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float()
    M = torch.zeros(d, d)
    M[torch.arange(d), perm] = signs
    return M.to(device), perm.to(device), signs.to(device)


def pairwise_rotation(d, g, device):
    # block-diagonal SO(2) (d even)
    ang = torch.rand(d // 2, generator=g) * 6.2831853
    c, s = torch.cos(ang), torch.sin(ang)
    M = torch.zeros(d, d)
    idx = torch.arange(d // 2)
    M[2 * idx, 2 * idx] = c
    M[2 * idx, 2 * idx + 1] = -s
    M[2 * idx + 1, 2 * idx] = s
    M[2 * idx + 1, 2 * idx + 1] = c
    return M.to(device)


def nn_top1(Q, T, chunk=512):
    # cosine NN, returns top-1 recovery assuming row i of Q corresponds to row i of T
    n = Q.shape[0]
    Qn = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
    Tn = T / (T.norm(dim=1, keepdim=True) + 1e-9)
    hit1 = hit10 = 0
    idx = torch.arange(n, device=Q.device)
    for s in range(0, n, chunk):
        sims = Qn[s:s + chunk] @ Tn.T
        order = sims.argsort(dim=1, descending=True)
        tgt = idx[s:s + chunk].unsqueeze(1)
        hit1 += int((order[:, :1] == tgt).any(1).sum())
        hit10 += int((order[:, :10] == tgt).any(1).sum())
    return round(100 * hit1 / n, 3), round(100 * hit10 / n, 3)


def l2_top1(Q, T, chunk=256):
    n = Q.shape[0]
    hit1 = hit10 = 0
    idx = torch.arange(n, device=Q.device)
    for s in range(0, n, chunk):
        d = torch.cdist(Q[s:s + chunk], T)      # [chunk, n]
        order = d.argsort(dim=1)
        tgt = idx[s:s + chunk].unsqueeze(1)
        hit1 += int((order[:, :1] == tgt).any(1).sum())
        hit10 += int((order[:, :10] == tgt).any(1).sum())
    return round(100 * hit1 / n, 3), round(100 * hit10 / n, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tokens", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    from transformers import AutoModelForCausalLM
    print("loading", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device).eval()
    cfg = model.config
    d = cfg.hidden_size
    E = model.get_input_embeddings().weight.detach().float().to(device)   # [V,d]
    Wq_full = model.model.layers[0].self_attn.q_proj.weight.detach().float().T.to(device)  # [d,d_q]
    V = E.shape[0]
    del model
    torch.cuda.empty_cache()
    g = torch.Generator().manual_seed(0)
    n = min(args.n_tokens, V)
    toks = torch.randperm(V, generator=g)[:n].to(device)
    Etab = E[toks]                                                # attacker table rows == targets

    N0, perm, signs = signed_perm(d, g, device)                  # residual/input mask
    M = Etab @ N0                                                # what the GPU sees (residual embedding)

    res = {"model": args.model, "hidden": d, "n_tokens": n, "vocab": V,
           "random_top1_pct": round(100.0 / n, 4)}

    # ---- A1 embedding inversion ------------------------------------------------
    res["A1a_naive_cosine_top1_top10"] = nn_top1(M, Etab)
    res["A1b_perm_invariant_sortedabs_top1_top10"] = l2_top1(
        M.abs().sort(dim=1).values, Etab.abs().sort(dim=1).values)
    # ---- A2 norm attack -------------------------------------------------------
    nq = M.norm(dim=1)
    nt = Etab.norm(dim=1)
    idxa = torch.arange(n, device=device)
    h1 = h10 = 0
    for s in range(0, n, 512):
        order = (nq[s:s + 512].view(-1, 1) - nt.view(1, -1)).abs().argsort(dim=1)
        tgt = idxa[s:s + 512].unsqueeze(1)
        h1 += int((order[:, :1] == tgt).any(1).sum())
        h10 += int((order[:, :10] == tgt).any(1).sum())
    res["A2_norm_top1_top10"] = (round(100 * h1 / n, 3), round(100 * h10 / n, 3))

    # ---- A3 Gram mask recovery on a real folded weight ------------------------
    # W_ours [d, d_out] (row-vector conv): transformers Linear weight is [out,in]
    Wq = Wq_full                                                 # [d, d_q]
    dq = Wq.shape[1]
    N_out = pairwise_rotation(dq, g, device)                     # output mask (irrelevant to right-Gram)
    W_tilde = N0.T @ Wq @ N_out                                  # exact public-weight folding
    G_tilde = W_tilde @ W_tilde.T                                # = P (Wq Wq^T) P^T, P = N0^T
    G = Wq @ Wq.T
    dg_t = torch.diagonal(G_tilde)
    dg = torch.diagonal(G)
    # recover permutation: match each observed diag to nearest true diag
    o = (dg_t.view(-1, 1) - dg.view(1, -1)).abs().argsort(dim=1)[:, 0]  # recovered pi(i)?
    # ground truth: G_tilde_ii = G_{pi(i),pi(i)} where pi maps via P=N0^T. Build truth.
    # N0[i, perm[i]] = signs[i]; N0^T[perm[i], i] = signs[i]. right-Gram index map:
    # (P G P^T)_ii = G_{a,a} where P_{i,a}!=0 -> a = perm[i]. so truth pi(i)=perm[i].
    truth = perm
    perm_acc = round(100 * float((o == truth).float().mean()), 3)
    # reconstruct N0_hat as signed permutation from recovered perm + signs from off-diag
    N0_hat = torch.zeros(d, d, device=device)
    N0_hat[torch.arange(d, device=device), o] = 1.0             # sign-agnostic first
    # unmask residual with recovered permutation (ignore sign -> use abs match anyway)
    M_unmasked = M @ N0_hat                                      # approx E if perm correct (up to sign)
    res["A3_gram_perm_recovery_acc_pct"] = perm_acc
    res["A3_diag_distinct_frac"] = round(float(
        (torch.unique(torch.round(dg * 1e3)).numel()) / d), 4)
    # full-collapse check: cosine NN of |unmasked| vs |E| (sign-robust)
    res["A3_after_unmask_inversion_top1_top10"] = nn_top1(
        M_unmasked.abs(), Etab.abs())

    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
