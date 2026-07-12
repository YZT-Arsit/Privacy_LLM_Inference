"""Trusted AdamW state protocol (O1-C L12) — runs INSIDE the TDX enclave.

Frozen Gate-0.5 split (proven exact in scripts/full_matrix_equivalence.py, worst 3.8e-13):
  TRUSTED (enclave holds theta_plain + m + v + step; A10 never sees these or gamma or plaintext grad):
    - A factors of {q,k,v,gate,up}  (input transform T_in = diag(gamma) @ Nr is a SCALED signed-perm:
      monomial but NOT unit, so AdamW's elementwise 2nd moment + eps does not cancel gamma -> enclave)
    - B factors of {q,k}            (output transform is a 2D RoPE-commuting rotation: NON-monomial)
  GPU-EXACT (A10 runs AdamW directly on the masked factor; monomial transform commutes with AdamW):
    - A of {o,down}; B of {o,down,v,gate,up}  (transforms are pure signed-perm/perm: unit monomial)

Enclave op flow per trusted factor:
  init : receive initial masked factor theta_tilde -> un-fold to theta_plain, set m=v=0, t=0
  step : receive masked grad g_tilde -> g_plain = un-fold(g_tilde); exact plaintext AdamW on
         (theta_plain, m, v); re-fold theta_plain -> theta_tilde; return theta_tilde only.
Never returns gamma / m / v / plaintext theta. Fails closed on missing/stale factor state.

Fold conventions (build_bundle, full_matrix_equivalence): A_tilde = A_plain @ Tin_iT ;
B_tilde = Tout.T @ B_plain ; gA_plain = gAc @ Tin_inv ; gB_plain = Tout_iT @ gBc.
"""
from __future__ import annotations
import torch

DT = torch.float64
TRUSTED_A = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")   # A-factor -> enclave
TRUSTED_B = ("q_proj", "k_proj")                                      # B-factor -> enclave
ATTN = ("q_proj", "k_proj", "v_proj")


def orthogonal_signed_perm(n, seed, dtype=DT):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype); N[torch.arange(n), perm] = signs
    return N


def rope_rot(hd, seed, dtype=DT):
    half = hd // 2
    g = torch.Generator().manual_seed(int(seed))
    ang = torch.rand(half, generator=g, dtype=dtype) * 6.283185307179586
    B = torch.eye(hd, dtype=dtype)
    c, s = torch.cos(ang), torch.sin(ang)
    for i in range(half):
        j = i + half
        B[i, i] = c[i]; B[j, j] = c[i]; B[i, j] = -s[i]; B[j, i] = s[i]
    return B


def adamw_step(theta, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta = theta - lr * (mhat / (vhat.sqrt() + eps) + wd * theta)
    return theta, m, v


class TrustedAdamW:
    """Holds trusted plaintext factors + Adam moments for A of {q,k,v,gate,up} and B of {q,k}.

    gb: gamma bundle {"ga":{l:tensor}, "gm":{l:tensor}, "nr_seed", "num_layers", "hidden"}.
    """
    def __init__(self, gb, cfg, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.L = gb["num_layers"]; H = gb["hidden"]
        self.nh = cfg["num_attention_heads"]; self.nkv = cfg["num_key_value_heads"]
        self.hd = H // self.nh
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        Nr = orthogonal_signed_perm(H, gb["nr_seed"])
        NrT = Nr.T
        # DEPLOYED package convention (verified against the trusted verifier + gram_inv=Nr^T diag(g^2) Nr):
        #   fold  M = diag(gamma) @ Nr        so  A_tilde = A_plain @ M
        #   grad  M^T = Nr^T diag(gamma)      so  gA_plain = gA_tilde @ M^T   (=> gA_plain@M = gA_tilde@gram_inv)
        #   unfold M^-1 = Nr^T diag(1/gamma)  so  A_plain  = A_tilde @ M^-1
        self.tinA = {}      # per (l,proj) -> (foldM, gradMT, unfoldMinv)
        self.toutB = {}     # per (l,proj) -> Tout (Bq/Bk); B is orthogonal (no gamma), already correct
        for l in range(self.L):
            ga = gb["ga"][l].to(DT); gm = gb["gm"][l].to(DT)
            for p in TRUSTED_A:
                g = ga if p in ATTN else gm
                foldM = torch.diag(g) @ Nr           # A_tilde = A_plain @ foldM
                gradMT = NrT @ torch.diag(g)         # gA_plain = gA_tilde @ gradMT
                unfoldMinv = NrT @ torch.diag(1.0 / g)  # A_plain = A_tilde @ unfoldMinv
                self.tinA[(l, p)] = (foldM, gradMT, unfoldMinv)
            Bl = rope_rot(self.hd, 1000 + l)
            self.toutB[(l, "q_proj")] = torch.block_diag(*([Bl] * self.nh))
            self.toutB[(l, "k_proj")] = torch.block_diag(*([Bl] * self.nkv))
        self.stateA = {}    # (l,proj) -> [theta_plain, m, v]
        self.stateB = {}
        self.t = 0

    # ---- init: un-fold initial masked factors to plaintext, zero moments ----
    def init_factor(self, l, proj, A_tilde=None, B_tilde=None):
        if A_tilde is not None and proj in TRUSTED_A:
            foldM, gradMT, unfoldMinv = self.tinA[(l, proj)]
            Ap = A_tilde.to(DT) @ unfoldMinv          # A_plain = A_tilde @ M^-1
            self.stateA[(l, proj)] = [Ap, torch.zeros_like(Ap), torch.zeros_like(Ap)]
        if B_tilde is not None and proj in TRUSTED_B:
            Tout = self.toutB[(l, proj)]
            # B_tilde = Tout.T @ B_plain  ->  B_plain = Tout @ B_tilde
            Bp = Tout @ B_tilde.to(DT)
            self.stateB[(l, proj)] = [Bp, torch.zeros_like(Bp), torch.zeros_like(Bp)]

    # ---- step: masked grads in -> exact plaintext AdamW -> re-folded masked factors out ----
    def step(self, gA_tilde: dict, gB_tilde: dict):
        self.t += 1
        outA, outB = {}, {}
        expectedA = {(l, p) for l in range(self.L) for p in TRUSTED_A}
        expectedB = {(l, p) for l in range(self.L) for p in TRUSTED_B}
        gotA = set(); gotB = set()
        for k, gt in gA_tilde.items():
            l, proj = int(k.split(".", 1)[0]), k.split(".", 1)[1]
            if proj not in TRUSTED_A or (l, proj) not in self.stateA:
                raise KeyError(f"stale/missing trusted A state {k}")
            foldM, gradMT, unfoldMinv = self.tinA[(l, proj)]
            gA_plain = gt.to(DT) @ gradMT             # gA_plain = gA_tilde @ M^T
            Ap, m, v = self.stateA[(l, proj)]
            Ap, m, v = adamw_step(Ap, gA_plain, m, v, self.t, self.lr, self.b1, self.b2, self.eps, self.wd)
            self.stateA[(l, proj)] = [Ap, m, v]
            outA[k] = (Ap @ foldM)           # re-fold -> masked (A_tilde = A_plain @ M)
            gotA.add((l, proj))
        for k, gt in gB_tilde.items():
            l, proj = int(k.split(".", 1)[0]), k.split(".", 1)[1]
            if proj not in TRUSTED_B or (l, proj) not in self.stateB:
                raise KeyError(f"stale/missing trusted B state {k}")
            Tout = self.toutB[(l, proj)]
            gB_plain = Tout @ gt.to(DT)       # Tout_iT @ gBc, Tout_iT == Tout (orthogonal)
            Bp, m, v = self.stateB[(l, proj)]
            Bp, m, v = adamw_step(Bp, gB_plain, m, v, self.t, self.lr, self.b1, self.b2, self.eps, self.wd)
            self.stateB[(l, proj)] = [Bp, m, v]
            outB[k] = (Tout.T @ Bp)           # re-fold -> masked
            gotB.add((l, proj))
        missing = len(expectedA - gotA) + len(expectedB - gotB)
        return outA, outB, missing

    def state_present(self):
        return len(self.stateA) == self.L * len(TRUSTED_A) and len(self.stateB) == self.L * len(TRUSTED_B)
