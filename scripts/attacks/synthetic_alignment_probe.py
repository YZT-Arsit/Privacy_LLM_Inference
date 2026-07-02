#!/usr/bin/env python
"""Synthetic adversary probe for the A_rightmul folded scheme (fast, CPU, local).

Goal: decide -- in a controlled setting that matches OUR paper-facing design --
whether an honest-but-curious GPU operator can recover input tokens from what the
GPU actually sees. This validates the attack code before pointing it at the real
Qwen-7B folded package, and directly tests the security claim.

Modeled exactly after the deployed A_rightmul path:
  * masks are ORTHOGONAL and STRUCTURED (required by the compatible-mask
    conditions): residual/input = signed permutation (orthogonal monomial),
    attention out = pairwise rotation (block-diagonal 2x2 SO(2)); we also test a
    dense-orthogonal mask as a stress upper bound.
  * NO fine-tuning: the folded weights use the EXACT public weights,
    W_tilde = N_in^{-1} W N_out with W == W_public. (This is the double-edged
    property vs CONJFORMER, which relies on fine-tuning to make W_tilde != W.)
  * Linear-boundary additive pad: the GPU input operand is X_tilde = (X - T) N_in
    with a single broadcast pad vector T (matches masked_input_pad_and_compensation).

Adversary knowledge (all GPU-visible): public embedding table E, public weights
{W_i} for the families that share the input mask (q/k/v/gate/up), the folded
weights {W_tilde_i}, and the masked layer-0 input X_tilde. Secret: N_in, N_out_i, T.

Attacks implemented:
  A0  raw NN inversion (no defense)                       -- sanity, expect ~100%
  A1  norm-based token recovery on X_tilde                -- expect defeated by pad
  A2  LEFT-Gram mask recovery: W_tilde W_tilde^T =
        N_in^{-1} (W W^T) N_in^{-T}  (N_out cancels, W known)
        -> recover N_in (up to sign/degenerate symmetry) by joint
        orthogonal alignment across the shared-input families, then invert
        X_tilde and NN against E.                          -- the decisive attack

Reports top-1 / top-10 token-recovery accuracy per (mask family, pad on/off).
"""

from __future__ import annotations

import argparse
import torch

torch.manual_seed(0)


# --------------------------------------------------------------------------
# Structured orthogonal masks (match the compatible-mask families)
# --------------------------------------------------------------------------

def signed_permutation(d: int, g: torch.Generator) -> torch.Tensor:
    """Orthogonal monomial matrix: a permutation with random +-1 signs."""
    perm = torch.randperm(d, generator=g)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(torch.float64)
    M = torch.zeros(d, d, dtype=torch.float64)
    M[torch.arange(d), perm] = signs
    return M


def pairwise_rotation(d: int, g: torch.Generator) -> torch.Tensor:
    """Block-diagonal 2x2 rotations (orthogonal), matching RoPE-compatible QK."""
    assert d % 2 == 0
    M = torch.zeros(d, d, dtype=torch.float64)
    ang = torch.rand(d // 2, generator=g, dtype=torch.float64) * 6.28318530718
    for i in range(d // 2):
        c, s = torch.cos(ang[i]), torch.sin(ang[i])
        M[2 * i, 2 * i] = c;      M[2 * i, 2 * i + 1] = -s
        M[2 * i + 1, 2 * i] = s;  M[2 * i + 1, 2 * i + 1] = c
    return M


def dense_orthogonal(d: int, g: torch.Generator) -> torch.Tensor:
    """Dense orthogonal (QR). NOT compatible with A_rightmul -- stress upper bound."""
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q * torch.sign(torch.diag(r)).unsqueeze(0)


MASK_FAMILIES = {
    "signed_permutation": signed_permutation,   # our residual/input mask
    "pairwise_rotation": pairwise_rotation,      # our attention mask
    "dense_orthogonal": dense_orthogonal,        # incompatible stress bound
}


# --------------------------------------------------------------------------
# Token recovery helpers
# --------------------------------------------------------------------------

def topk_recovery(X_est: torch.Tensor, E: torch.Tensor, tokens: torch.Tensor,
                  ks=(1, 10)) -> dict:
    """cosine NN of each recovered row against the public embedding table E."""
    Xe = X_est / (X_est.norm(dim=1, keepdim=True) + 1e-9)
    Ee = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    sims = Xe @ Ee.T                       # [n, V]
    order = sims.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (order[:, :k] == tokens.unsqueeze(1)).any(dim=1).float().mean()
        out[f"top{k}"] = round(float(hit) * 100, 2)
    return out


# --------------------------------------------------------------------------
# Attack A2: LEFT-Gram mask recovery (exploits exact public W, no fine-tuning)
# --------------------------------------------------------------------------

def _procrustes(M: torch.Tensor) -> torch.Tensor:
    """Nearest orthogonal matrix to M (argmin_O ||O - M||_F): O = U V^T."""
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


def recover_input_mask_als(Ws, Wt, iters: int = 800, restarts: int = 8):
    """STRONG attack -- LEFT-Gram simultaneous orthogonal conjugation.

    NOTE on why the naive two-sided per-family ALS is useless here: for ANY
    orthogonal A, ``A W_i`` and ``W_tilde_i`` share the same singular values, so
    there is always an orthogonal B_i with ``||W_tilde_i - A W_i B_i|| = 0``. The
    per-family objective is therefore flat in A -- the independent output masks
    N_out_i make A unidentifiable that way. (This is exactly why CONJFORMER's
    attack, which relies on a GLOBAL shared right factor U, does not port over.)

    The only exploitable structure is the INPUT mask A = N_in^{-1} shared across
    families, recovered from the left Gram (B_i cancels, W_i known exactly):

        S_i  = W_i W_i^T      (known)
        St_i = W_tilde_i W_tilde_i^T = A S_i A^T   (observed)

    We recover orthogonal A minimizing  sum_i || A S_i A^T - St_i ||_F^2  by
    Riemannian gradient descent on O(d) with multiple restarts (best residual).
    Returns (A_hat, gram_residual).
    """
    d = Ws[0].shape[0]
    S = [W @ W.T for W in Ws]
    St = [Wt_i @ Wt_i.T for Wt_i in Wt]
    g = torch.Generator().manual_seed(1234)

    def loss(A):
        return sum(float(((A @ Si @ A.T) - Sti).pow(2).sum())
                   for Si, Sti in zip(S, St))

    best_A, best_res = torch.eye(d, dtype=torch.float64), loss(torch.eye(d, dtype=torch.float64))
    for r in range(restarts):
        A = (torch.eye(d, dtype=torch.float64) if r == 0
             else _procrustes(torch.randn(d, d, generator=g, dtype=torch.float64)))
        lr = 0.05
        prev = loss(A)
        for _ in range(iters):
            # Euclidean grad of sum ||A S A^T - St||^2  is  sum 4 (A S A^T - St) A S
            G = sum(4.0 * ((A @ Si @ A.T) - Sti) @ A @ Si
                    for Si, Sti in zip(S, St))
            # Riemannian (tangent) grad on O(d): skew(A^T G) then move
            skew = A.T @ G
            skew = 0.5 * (skew - skew.T)
            A_new = _procrustes(A - lr * (A @ skew))
            cur = loss(A_new)
            if cur < prev:
                A, prev, lr = A_new, cur, lr * 1.1
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        if prev < best_res:
            best_res, best_A = prev, A
    return best_A, best_res


def run(mask_name: str, d: int, vocab: int, n_tok: int, pad_scale: float,
        n_families: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    fam = MASK_FAMILIES[mask_name]

    # Public embedding table + a probe token sequence.
    E = torch.randn(vocab, d, generator=g, dtype=torch.float64)
    # give tokens non-uniform norms (realistic; enables norm attack when unmasked)
    E = E * (0.5 + torch.rand(vocab, 1, generator=g, dtype=torch.float64))
    tokens = torch.randint(0, vocab, (n_tok,), generator=g)
    X = E[tokens]                                            # [n, d] true embeds

    # Shared INPUT mask N_in + per-family public weights and independent out masks.
    N_in = fam(d, g)
    N_in_inv = N_in.T                                         # orthogonal
    Ws, Wt = [], []
    for _ in range(n_families):
        W = torch.randn(d, d, generator=g, dtype=torch.float64)   # public weight
        N_out = fam(d, g)
        Ws.append(W)
        Wt.append(N_in_inv @ W @ N_out)                      # folded (exact W!)

    # Linear-boundary additive pad: single broadcast vector T. IMPORTANT: the pad
    # is boundary-LOCAL -- it only enters the per-Linear matmul operand
    # X_op = (X - T) N_in. It does NOT enter the residual stream / RMSNorm input.
    T = pad_scale * torch.randn(d, generator=g, dtype=torch.float64)
    X_op = (X - T) @ N_in                                     # padded Linear operand

    # What the GPU worker ACTUALLY receives as its layer-0 input (folded_worker.py):
    #   h_tilde = X @ N_0   (N_0 = orthogonal residual mask, PAD-FREE)
    # RMSNorm + residual adds run on this pad-free tensor; the pad is applied only
    # inside _linear. So the norm-attack surface is h_tilde, NOT X_op.
    N0 = fam(d, g)                                            # residual mask (orthogonal)
    h_tilde = X @ N0                                          # GPU-visible, pad-free

    res = {"mask": mask_name, "pad_scale": pad_scale, "families": n_families}

    # A0: no defense (adversary somehow had raw X) -- sanity ceiling.
    res["A0_raw_nn"] = topk_recovery(X, E, tokens)

    # A1a: norm attack on the REAL GPU input h_tilde = X N_0 (pad-free, orthogonal).
    #      ||h_tilde|| = ||X|| exactly -> the pad gives NO protection here.
    e_norm = E.norm(dim=1, keepdim=True)
    nn_h = (h_tilde.norm(dim=1, keepdim=True) - e_norm.T).abs().argmin(dim=1)
    res["A1a_norm_on_h_tilde_top1"] = round(
        float((nn_h == tokens).float().mean()) * 100, 2)
    # A1b: norm attack on the padded Linear operand X_op = (X-T)N_in (secondary
    #      surface). Here ||X_op|| = ||X - T|| so the pad DOES perturb this view --
    #      but it is not the surface that leaks the embedding norm.
    nn_op = (X_op.norm(dim=1, keepdim=True) - e_norm.T).abs().argmin(dim=1)
    res["A1b_norm_on_padded_operand_top1"] = round(
        float((nn_op == tokens).float().mean()) * 100, 2)

    # A2: LEFT-Gram mask recovery -> invert -> NN.
    A_hat, gram_res = recover_input_mask_als(Ws, Wt)   # ~ N_in^{-1} (=N_in^T)
    res["A2_gram_residual"] = round(gram_res, 4)
    #   check how well A_hat recovers the mask action: X_tilde A_hat^T ~ (X - T)
    X_rec = X_op @ A_hat.T                                 # estimate of (X - T) D
    res["A2_gram_recovers_maskerr"] = round(
        float((X_rec - (X - T)).abs().mean()), 4)
    # adversary does NOT know T; NN with the residual pad still present:
    res["A2_gram_nn_no_padknow"] = topk_recovery(X_rec, E, tokens)
    # oracle upper bound: if adversary ALSO knew T (isolates mask-recovery power)
    res["A2_gram_nn_pad_oracle"] = topk_recovery(X_rec + T, E, tokens)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--families", type=int, default=5)   # q,k,v,gate,up share N_in
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json
    print("=== Synthetic adversary probe (A_rightmul-style folded scheme) ===")
    print(f"dim={args.dim} vocab={args.vocab} tokens={args.tokens} "
          f"shared-input families={args.families}\n")
    for mask in ("signed_permutation", "pairwise_rotation", "dense_orthogonal"):
        for pad in (0.0, 0.1, 0.5):
            r = run(mask, args.dim, args.vocab, args.tokens, pad,
                    args.families, args.seed)
            print(json.dumps(r))
    print("\nReading: A0 ~100% = ceiling; A1/A2_*_nn high = BROKEN, ~1/vocab = safe.")
    print("A2 pad_oracle isolates mask-recovery; A2 no_padknow is the real attacker.")


if __name__ == "__main__":
    main()
