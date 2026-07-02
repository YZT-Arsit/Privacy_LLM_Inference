#!/usr/bin/env python
"""Attention-logit fingerprint attack against the A_rightmul folded scheme.

CONJFORMER Sec. 3.5 / App. D.5: after weights are fixed, each token processed in
isolation yields a characteristic single-position self-attention logit per (layer,
head). Orthogonal obfuscation PRESERVES attention logits (Q_hat K_hat^T = Q K^T),
so the server can read the true logit vector f(token) in R^{B*H} and NN-match it
against a fingerprint table it precomputes from the PUBLIC weights.

Why this matters for us (worse than CONJFORMER):
  * A_rightmul certifies `attention_qk_scores_preserved` (pairwise-rotation QK) --
    the GPU softmax runs over the TRUE logits, so f is exposed and mask-invariant.
  * The linear-boundary pad is compensated (C_pad) BEFORE attention, so it does not
    change the logit -- no protection here either.
  * We do NOT fine-tune, so the attacker's public-weight fingerprint table is EXACT.
    CONJFORMER's only defense (App. D.5: fine-tuning shifts f away from the table)
    is unavailable to us.

This probe (a) verifies f is invariant to the mask + pad, and (b) measures
token-recovery accuracy by NN in fingerprint space. Fast, CPU, synthetic --
validates the attack before running it on the real Qwen-7B folded package.
"""

from __future__ import annotations

import argparse
import torch

torch.manual_seed(0)


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def signed_permutation(d, g):
    perm = torch.randperm(d, generator=g)
    s = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).double()
    M = torch.zeros(d, d, dtype=torch.float64)
    M[torch.arange(d), perm] = s
    return M


def fingerprint(r: torch.Tensor, WQs, WKs, head_dim: int) -> torch.Tensor:
    """f(token) in R^{n_features}: per (layer,head) single-position self-logit
    (q . k / sqrt(head_dim)) for the normed token state r. RoPE at position 0 = I."""
    feats = []
    for WQ, WK in zip(WQs, WKs):
        q = r @ WQ                       # [n, head_dim]
        k = r @ WK
        feats.append((q * k).sum(-1, keepdim=True) / (head_dim ** 0.5))
    return torch.cat(feats, dim=-1)      # [n, n_features]


def run(d=64, vocab=2000, n_tok=256, n_features=96, head_dim=8,
        pad_scale=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Public embedding table + probe tokens.
    E = torch.randn(vocab, d, generator=g, dtype=torch.float64)
    E = E * (0.5 + torch.rand(vocab, 1, generator=g, dtype=torch.float64))
    tokens = torch.randint(0, vocab, (n_tok,), generator=g)
    X = E[tokens]

    # Public per-(layer,head) Q/K projections (attacker knows these exactly).
    WQs = [torch.randn(d, head_dim, generator=g, dtype=torch.float64)
           for _ in range(n_features)]
    WKs = [torch.randn(d, head_dim, generator=g, dtype=torch.float64)
           for _ in range(n_features)]

    # Attacker precomputes the fingerprint table over the whole vocab (public W).
    F_table = fingerprint(rmsnorm(E), WQs, WKs, head_dim)          # [V, n_feat]

    # What the GPU observes for the probe tokens = the TRUE logits (mask-invariant).
    F_obs = fingerprint(rmsnorm(X), WQs, WKs, head_dim)           # [n, n_feat]

    # --- invariance checks: mask + pad do not change the logit -----------------
    N0 = signed_permutation(d, g)                                  # residual mask
    # masked normed state r_tilde = rmsnorm(X) N0 ; folded QK carry N0^T on the left
    r_tilde = rmsnorm(X) @ N0
    WQt = [N0.T @ WQ for WQ in WQs]      # folded Q proj (left input mask)
    WKt = [N0.T @ WK for WK in WKs]
    F_masked = fingerprint_masked(r_tilde, WQt, WKt, head_dim)
    mask_inv_err = float((F_masked - F_obs).abs().max())
    # pad: operand (r - T)N0 with compensation restoring r N0 -> logit unchanged.
    T = pad_scale * torch.randn(d, generator=g, dtype=torch.float64)
    q_op = (rmsnorm(X) - T) @ N0                                   # padded operand
    # compensation C = T N0 added back before the QK bilinear (folded_worker path)
    r_comp = q_op + (T @ N0)
    pad_inv_err = float((r_comp - r_tilde).abs().max())

    # --- attack: NN in fingerprint space --------------------------------------
    Fo = F_obs / (F_obs.norm(dim=1, keepdim=True) + 1e-9)
    Ft = F_table / (F_table.norm(dim=1, keepdim=True) + 1e-9)
    order = (Fo @ Ft.T).argsort(dim=1, descending=True)
    rec = {}
    for k in (1, 10, 100):
        rec[f"top{k}"] = round(float((order[:, :k] == tokens.unsqueeze(1))
                                     .any(1).float().mean()) * 100, 2)
    return {"n_features": n_features, "pad_scale": pad_scale,
            "mask_invariance_maxerr": round(mask_inv_err, 8),
            "pad_invariance_maxerr": round(pad_inv_err, 8),
            "fingerprint_recovery": rec}


def fingerprint_masked(r_tilde, WQt, WKt, head_dim):
    feats = []
    for WQ, WK in zip(WQt, WKt):
        q = r_tilde @ WQ
        k = r_tilde @ WK
        feats.append((q * k).sum(-1, keepdim=True) / (head_dim ** 0.5))
    return torch.cat(feats, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--head-dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import json
    print("=== Attention-logit fingerprint attack (A_rightmul folded) ===")
    print(f"dim={args.dim} vocab={args.vocab} tokens={args.tokens}\n")
    for n_feat in (12, 48, 96, 336):   # 336 ~ 28 layers x 12 heads (Qwen2.5-7B-ish)
        r = run(args.dim, args.vocab, args.tokens, n_feat, args.head_dim,
                0.1, args.seed)
        print(json.dumps(r))
    print("\nmask/pad invariance ~0 => the fingerprint is exposed unchanged.")
    print("recovery high => attack works; ~1/vocab => safe. random top1~0.05%.")


if __name__ == "__main__":
    main()
