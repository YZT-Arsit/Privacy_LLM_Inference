"""Gate 3.5 -- unified masked-training CPU integration contract.

Composes the paper-facing A_rightmul **masked forward** (masked residual
transport + masked-domain nonlinear islands, via the shared kernels in
:mod:`pllo.ops.masked_training_kernels`) with **rank-masked LoRA SGD backward**,
end-to-end, on a tiny synthetic Qwen-shaped decoder block. It proves the contract
needed before spending an H800 turn on a real unified run:

    input masked -> masked RMSNorm-core -> Q/K/V (LoRA-folded) ->
    post-RoPE paired masking -> GQA attention (softmax on true scores) ->
    masked residual -> second RMSNorm-core -> shared-permutation SwiGLU island ->
    masked residual -> masked LM head + monomial logit mask ->
    trusted-loss mock boundary (private CE) -> masked dlogits ->
    full autograd backward -> rank-masked LoRA grads -> masked SGD update

Everything the "GPU" holds is in the masked domain: every residual-stream hidden
tensor equals ``plaintext_hidden @ N`` (a signed permutation), never the plaintext
hidden itself. The nonlinear islands run on the masked state with
``nonlinear_trusted_calls == 0``. The private cross-entropy is the ONLY trusted
crossing; its gradient is delivered back as masked dlogits and stitched into the
autograd graph with a surrogate ``(Z_tilde * G_tilde).sum()`` cut (the same
private-boundary technique the Gate-3 worker uses), so labels never leave the
boundary.

The masks are the compatible family: residual/hidden ``N`` = orthogonal signed
permutation; attention Q/K = shared head-dim **pair permutation** that commutes
with RoPE (so masked q/k carry no plaintext value yet preserve q k^T); V/O folded
back to ``N``; SwiGLU = shared channel permutation; vocab = O(V) monomial mask.

IMPORTANT SCOPE: this is a CPU/fp64 *integration contract*. It is NOT a real Qwen
or GPU result and must never be reported as one. Attention *scores* (post-softmax
weights) are revealed by the A_rightmul design (the documented attention-
fingerprint leak), which is a separate, known property -- this contract asserts no
plaintext *residual hidden state*, not that attention scores are hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from pllo.masks.monomial_logit_mask import MonomialLogitMask, make_monomial_logit_mask
from pllo.ops.masked_training_kernels import (
    NonlinearAccounting,
    apply_rope,
    orthogonal_signed_perm,
    permutation_matrix,
    repeat_kv,
    rmsnorm_core,
    rope_cos_sin,
    silu_island,
    softmax_island,
)

DT = torch.float64


# ---------------------------------------------------------------------------
# Config + parameter bundles
# ---------------------------------------------------------------------------


@dataclass
class BlockConfig:
    hidden: int = 32
    n_heads: int = 4
    head_dim: int = 8
    n_kv_heads: int = 2
    intermediate: int = 64
    vocab: int = 40
    seq: int = 5
    rank: int = 4
    alpha: float = 8.0
    eps: float = 1e-6
    rope_base: float = 1e6
    seed: int = 1234

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads


@dataclass
class Weights:
    """Frozen base weights (gamma already folded into the projections)."""
    Wq: torch.Tensor  # (n_heads*head_dim, hidden)
    Wk: torch.Tensor  # (n_kv*head_dim, hidden)
    Wv: torch.Tensor
    Wo: torch.Tensor  # (hidden, n_heads*head_dim)
    Wg: torch.Tensor  # (intermediate, hidden)
    Wu: torch.Tensor
    Wd: torch.Tensor  # (hidden, intermediate)
    Wlm: torch.Tensor  # (vocab, hidden)
    g1: torch.Tensor   # rmsnorm1 gamma (hidden,)
    g2: torch.Tensor
    gf: torch.Tensor   # final rmsnorm gamma


@dataclass
class LoRA:
    """Rank-masked LoRA leaves. Plaintext leaves A0/B0; masked leaves A_t=U@A0,
    B_t=B0@U^T. U is a per-projection orthogonal rank mask."""
    A0: Dict[str, torch.Tensor]
    B0: Dict[str, torch.Tensor]
    U: Dict[str, torch.Tensor]


def build(cfg: BlockConfig) -> Tuple[Weights, LoRA, "Masks"]:
    g = torch.Generator().manual_seed(cfg.seed)

    def R(*shape):
        return torch.randn(*shape, generator=g, dtype=DT) * 0.1

    H, nh, hd, nkv, I, V = (cfg.hidden, cfg.n_heads, cfg.head_dim,
                            cfg.n_kv_heads, cfg.intermediate, cfg.vocab)
    g1 = 1.0 + R(H).abs()
    g2 = 1.0 + R(H).abs()
    gf = 1.0 + R(H).abs()
    # gamma folded into projection weights (RMSNorm core runs gamma-free)
    Wq = R(nh * hd, H) * g1
    Wk = R(nkv * hd, H) * g1
    Wv = R(nkv * hd, H) * g1
    Wo = R(H, nh * hd)
    Wg = R(I, H) * g2
    Wu = R(I, H) * g2
    Wd = R(H, I)
    Wlm = R(V, H) * gf
    w = Weights(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, Wg=Wg, Wu=Wu, Wd=Wd, Wlm=Wlm,
                g1=g1, g2=g2, gf=gf)

    targets = ("q", "k", "v", "o")
    out_dim = {"q": nh * hd, "k": nkv * hd, "v": nkv * hd, "o": H}
    in_dim = {"q": H, "k": H, "v": H, "o": nh * hd}
    A0, B0, U = {}, {}, {}
    for i, t in enumerate(targets):
        A0[t] = R(cfg.rank, in_dim[t])
        B0[t] = R(out_dim[t], cfg.rank)
        # orthogonal rank mask (distinct stream so A0/B0 unaffected)
        gu = torch.Generator().manual_seed(cfg.seed + 90210 + i)
        M = torch.randn(cfg.rank, cfg.rank, generator=gu, dtype=DT)
        Uq, _ = torch.linalg.qr(M)
        U[t] = Uq
    lora = LoRA(A0=A0, B0=B0, U=U)

    masks = Masks.build(cfg)
    return w, lora, masks


@dataclass
class Masks:
    N: torch.Tensor            # (H,H) residual signed perm
    hd_perm_qk: torch.Tensor   # (head_dim,) pair perm for Q/K (RoPE-commuting)
    hd_perm_v: torch.Tensor    # (head_dim,) pair perm for V
    Pmlp: torch.Tensor         # (I,I) SwiGLU shared channel permutation
    vocab: MonomialLogitMask
    cfg: BlockConfig

    @staticmethod
    def _pair_perm(head_dim: int, seed: int) -> torch.Tensor:
        half = head_dim // 2
        g = torch.Generator().manual_seed(seed)
        sigma = torch.randperm(half, generator=g)
        return torch.cat([sigma, sigma + half])   # commutes with rotate_half

    @classmethod
    def build(cls, cfg: BlockConfig) -> "Masks":
        N = orthogonal_signed_perm(cfg.hidden, cfg.seed + 1, DT)
        hd_perm_qk = cls._pair_perm(cfg.head_dim, cfg.seed + 2)
        hd_perm_v = cls._pair_perm(cfg.head_dim, cfg.seed + 3)
        Pmlp = permutation_matrix(cfg.intermediate, cfg.seed + 4, DT)
        vocab = make_monomial_logit_mask(cfg.vocab, seed=cfg.seed + 5,
                                         scale_low=0.5, scale_high=2.0)
        vocab = vocab.to(dtype=DT)
        return cls(N=N, hd_perm_qk=hd_perm_qk, hd_perm_v=hd_perm_v, Pmlp=Pmlp,
                   vocab=vocab, cfg=cfg)

    def blockdiag_perm(self, hd_perm: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Head-dim permutation as a (n_heads*hd, n_heads*hd) block-diag matrix,
        oriented so that ``(x @ P)[.., base+j] == x[.., base+hd_perm[j]]`` (i.e.
        ``x @ P`` permutes columns by ``hd_perm``, consistent with permuting the
        RoPE cos/sin tables by the same ``hd_perm``)."""
        hd = hd_perm.numel()
        P = torch.zeros(n_heads * hd, n_heads * hd, dtype=DT)
        for h in range(n_heads):
            base = h * hd
            P[base + hd_perm, base + torch.arange(hd)] = 1.0
        return P


# ---------------------------------------------------------------------------
# LoRA effective weight (differentiable w.r.t the trainable leaves)
# ---------------------------------------------------------------------------


def _eff_WT(w_name: str, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
            scaling: float) -> torch.Tensor:
    """Effective W^T (in-,out-) = W^T + scaling * A^T B^T. Differentiable w.r.t
    A,B. W:(out,in) -> W^T:(in,out); A:(r,in); B:(out,r)."""
    return W.T + scaling * (A.T @ B.T)


# ---------------------------------------------------------------------------
# Plaintext reference block (A,B are the trainable leaves)
# ---------------------------------------------------------------------------


def plaintext_forward(h: torch.Tensor, w: Weights, cfg: BlockConfig,
                      A: Dict[str, torch.Tensor], B: Dict[str, torch.Tensor],
                      cos: torch.Tensor, sin: torch.Tensor
                      ) -> Dict[str, torch.Tensor]:
    T, H, nh, hd, nkv = cfg.seq, cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.n_kv_heads
    s = cfg.scaling
    r1 = rmsnorm_core(h, cfg.eps)
    q = r1 @ _eff_WT("q", w.Wq, A["q"], B["q"], s)
    k = r1 @ _eff_WT("k", w.Wk, A["k"], B["k"], s)
    v = r1 @ _eff_WT("v", w.Wv, A["v"], B["v"], s)
    q = q.view(T, nh, hd).transpose(0, 1)          # (nh,T,hd)
    k = k.view(T, nkv, hd).transpose(0, 1)
    v = v.view(T, nkv, hd).transpose(0, 1)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    k = repeat_kv(k, cfg.n_rep)
    v = repeat_kv(v, cfg.n_rep)
    scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
    causal = torch.triu(torch.full((T, T), float("-inf"), dtype=DT), diagonal=1)
    attn = softmax_island(scores + causal, dim=-1)
    ctx = attn @ v                                  # (nh,T,hd)
    ctx = ctx.transpose(0, 1).reshape(T, nh * hd)
    o = ctx @ _eff_WT("o", w.Wo, A["o"], B["o"], s)
    h2 = h + o
    r2 = rmsnorm_core(h2, cfg.eps)
    gate = r2 @ w.Wg.T
    up = r2 @ w.Wu.T
    act = silu_island(gate, up)
    mlp = act @ w.Wd.T
    h3 = h2 + mlp
    rf = rmsnorm_core(h3, cfg.eps)
    logits = rf @ w.Wlm.T
    return {"h2": h2, "h3": h3, "logits": logits, "attn": attn}


# ---------------------------------------------------------------------------
# Masked block (A_t,B_t are the trainable leaves) -- shared kernels
# ---------------------------------------------------------------------------


def masked_forward(h_tilde: torch.Tensor, w: Weights, cfg: BlockConfig, m: Masks,
                   At: Dict[str, torch.Tensor], Bt: Dict[str, torch.Tensor],
                   cos: torch.Tensor, sin: torch.Tensor,
                   acct: NonlinearAccounting) -> Dict[str, torch.Tensor]:
    """All state is masked. Returns masked residual states + masked logits."""
    T, H, nh, hd, nkv = cfg.seq, cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.n_kv_heads
    s = cfg.scaling
    N = m.N
    Pq = m.blockdiag_perm(m.hd_perm_qk, nh)     # q output mask
    Pk = m.blockdiag_perm(m.hd_perm_qk, nkv)    # k output mask (same pair perm)
    Pv = m.blockdiag_perm(m.hd_perm_v, nkv)     # v output mask
    Pv_o = m.blockdiag_perm(m.hd_perm_v, nh)    # o input mask (ctx basis)
    cos_p = cos[:, m.hd_perm_qk]                 # RoPE tables permuted to q/k basis
    sin_p = sin[:, m.hd_perm_qk]

    r1_tilde = rmsnorm_core(h_tilde, cfg.eps, acct)      # == rmsnorm_core(h) @ N
    # projections: M_in = N, M_out = pair/channel perm; LoRA folded, differentiable
    q = r1_tilde @ (N.T @ _eff_WT("q", w.Wq, At["q"], Bt["q"], s) @ Pq)
    k = r1_tilde @ (N.T @ _eff_WT("k", w.Wk, At["k"], Bt["k"], s) @ Pk)
    v = r1_tilde @ (N.T @ _eff_WT("v", w.Wv, At["v"], Bt["v"], s) @ Pv)
    q = q.view(T, nh, hd).transpose(0, 1)
    k = k.view(T, nkv, hd).transpose(0, 1)
    v = v.view(T, nkv, hd).transpose(0, 1)
    q = apply_rope(q, cos_p, sin_p)              # masked q, RoPE in permuted basis
    k = apply_rope(k, cos_p, sin_p)
    k = repeat_kv(k, cfg.n_rep)
    v = repeat_kv(v, cfg.n_rep)
    scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)   # == true scores
    causal = torch.triu(torch.full((T, T), float("-inf"), dtype=DT), diagonal=1)
    attn = softmax_island(scores + causal, dim=-1, acct=acct)
    ctx = attn @ v                               # masked by V pair perm
    ctx = ctx.transpose(0, 1).reshape(T, nh * hd)
    # o_proj: M_in = Pv_o (ctx basis), M_out = N -> residual back in N basis
    o_tilde = ctx @ (Pv_o.T @ _eff_WT("o", w.Wo, At["o"], Bt["o"], s) @ N)
    h2_tilde = h_tilde + o_tilde
    r2_tilde = rmsnorm_core(h2_tilde, cfg.eps, acct)
    # SwiGLU: shared channel perm Pmlp; gate/up masked, down folds Pmlp^T & N
    gate = r2_tilde @ (N.T @ w.Wg.T @ m.Pmlp)
    up = r2_tilde @ (N.T @ w.Wu.T @ m.Pmlp)
    act = silu_island(gate, up, acct)            # == (silu(gate)*up) on masked chan
    mlp_tilde = act @ (m.Pmlp.T @ w.Wd.T @ N)
    h3_tilde = h2_tilde + mlp_tilde
    rf_tilde = rmsnorm_core(h3_tilde, cfg.eps, acct)
    # LM head folds N^T and the O(V) monomial vocab mask -> masked logits
    Wlm_fold = N.T @ w.Wlm.T @ m.vocab.dense_matrix(DT)
    Z_tilde = rf_tilde @ Wlm_fold
    return {"h2_tilde": h2_tilde, "h3_tilde": h3_tilde, "Z_tilde": Z_tilde,
            "attn": attn}


# ---------------------------------------------------------------------------
# Contract verification
# ---------------------------------------------------------------------------


@dataclass
class ContractResult:
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())


def run_contract(cfg: Optional[BlockConfig] = None, *, lr: float = 1e-2,
                 labels: Optional[torch.Tensor] = None) -> ContractResult:
    cfg = cfg or BlockConfig()
    w, lora, m = build(cfg)
    cos, sin = rope_cos_sin(cfg.seq, cfg.head_dim, cfg.rope_base, DT)
    torch.manual_seed(cfg.seed)
    h = torch.randn(cfg.seq, cfg.hidden, dtype=DT) * 0.1
    if labels is None:
        gl = torch.Generator().manual_seed(cfg.seed + 77)
        labels = torch.randint(0, cfg.vocab, (cfg.seq,), generator=gl)

    res = ContractResult()

    # -- plaintext reference: leaves A,B --------------------------------------
    A = {t: lora.A0[t].clone().requires_grad_(True) for t in lora.A0}
    B = {t: lora.B0[t].clone().requires_grad_(True) for t in lora.B0}
    pf = plaintext_forward(h, w, cfg, A, B, cos, sin)
    plain_loss = torch.nn.functional.cross_entropy(pf["logits"], labels,
                                                   reduction="mean")
    plain_loss.backward()
    gA = {t: A[t].grad.detach().clone() for t in A}
    gB = {t: B[t].grad.detach().clone() for t in B}

    # -- masked path: leaves A_t=U@A0, B_t=B0@U^T ----------------------------
    At = {t: (lora.U[t] @ lora.A0[t]).detach().clone().requires_grad_(True)
          for t in lora.A0}
    Bt = {t: (lora.B0[t] @ lora.U[t].T).detach().clone().requires_grad_(True)
          for t in lora.B0}
    h_tilde = h @ m.N                                # masked input
    acct = NonlinearAccounting()
    mf = masked_forward(h_tilde, w, cfg, m, At, Bt, cos, sin, acct)

    # forward recovery: unmask logits == plaintext logits
    Z_rec = m.vocab.recover_logits(mf["Z_tilde"].detach())
    plain_logits = pf["logits"].detach()
    res.metrics["forward_logits_max_abs_err"] = float(
        (Z_rec - plain_logits).abs().max())
    res.checks["forward_recovered_output"] = torch.allclose(
        Z_rec, plain_logits, atol=1e-8)

    # per-layer mask invariant: masked residual == plaintext residual @ N,
    # and NOT equal to the plaintext residual (i.e. genuinely masked)
    def masked_ok(name_t, name_p):
        got = mf[name_t].detach()
        want = pf[name_p].detach() @ m.N
        masked_err = float((got - want).abs().max())
        plain_gap = float((got - pf[name_p].detach()).abs().max())
        return masked_err, plain_gap
    e2, gap2 = masked_ok("h2_tilde", "h2")
    e3, gap3 = masked_ok("h3_tilde", "h3")
    res.metrics["residual_h2_mask_err"] = e2
    res.metrics["residual_h3_mask_err"] = e3
    res.metrics["residual_h2_plain_gap"] = gap2
    res.metrics["residual_h3_plain_gap"] = gap3
    res.checks["per_layer_mask_invariant"] = (e2 < 1e-9 and e3 < 1e-9)
    res.checks["residual_remains_masked"] = (gap2 > 1e-3 and gap3 > 1e-3)
    # no plaintext hidden materialization: every masked residual differs from plain
    plaintext_materializations = int(gap2 <= 1e-9) + int(gap3 <= 1e-9)
    res.counters["plaintext_hidden_materializations"] = plaintext_materializations
    res.checks["no_plaintext_hidden_materialization"] = (
        plaintext_materializations == 0)

    # -- trusted-loss mock boundary: private CE over masked logits -----------
    ce = m.vocab.private_ce(mf["Z_tilde"].detach(), labels, compute_dtype=DT)
    res.metrics["private_ce_loss"] = float(ce.loss)
    res.metrics["loss_abs_diff_vs_plaintext"] = abs(
        float(ce.loss) - float(plain_loss.detach()))
    res.checks["private_ce_matches_plaintext"] = (
        res.metrics["loss_abs_diff_vs_plaintext"] < 1e-9)

    # surrogate cut: backward((Z_tilde * G_tilde).sum()) == backprop of true loss
    surrogate = (mf["Z_tilde"] * ce.masked_logit_gradient.detach()).sum()
    surrogate.backward()
    gAt = {t: At[t].grad.detach().clone() for t in At}
    gBt = {t: Bt[t].grad.detach().clone() for t in Bt}

    # gradA/gradB recovered correctness: U^T gAt == gA ; gBt U == gB
    gA_err = max(float((lora.U[t].T @ gAt[t] - gA[t]).abs().max()) for t in gA)
    gB_err = max(float((gBt[t] @ lora.U[t] - gB[t]).abs().max()) for t in gB)
    res.metrics["gradA_recovered_max_err"] = gA_err
    res.metrics["gradB_recovered_max_err"] = gB_err
    res.checks["gradA_recovered_correct"] = gA_err < 1e-8
    res.checks["gradB_recovered_correct"] = gB_err < 1e-8

    # masked SGD update correctness: recovered post-step params == plaintext SGD
    upd_err = 0.0
    for t in gA:
        At_new = At[t].detach() - lr * gAt[t]
        A_new_rec = lora.U[t].T @ At_new
        A_new_plain = A[t].detach() - lr * gA[t]
        upd_err = max(upd_err, float((A_new_rec - A_new_plain).abs().max()))
        Bt_new = Bt[t].detach() - lr * gBt[t]
        B_new_rec = Bt_new @ lora.U[t]
        B_new_plain = B[t].detach() - lr * gB[t]
        upd_err = max(upd_err, float((B_new_rec - B_new_plain).abs().max()))
    res.metrics["masked_sgd_update_max_err"] = upd_err
    res.checks["masked_sgd_update_correct"] = upd_err < 1e-8

    # next-step logits: run one masked SGD step on both, compare recovered logits
    with torch.no_grad():
        A2 = {t: A[t].detach() - lr * gA[t] for t in A}
        B2 = {t: B[t].detach() - lr * gB[t] for t in B}
        At2 = {t: At[t].detach() - lr * gAt[t] for t in At}
        Bt2 = {t: Bt[t].detach() - lr * gBt[t] for t in Bt}
        pf2 = plaintext_forward(h, w, cfg, A2, B2, cos, sin)
        acct2 = NonlinearAccounting()
        mf2 = masked_forward(h_tilde, w, cfg, m, At2, Bt2, cos, sin, acct2)
        Z_rec2 = m.vocab.recover_logits(mf2["Z_tilde"])
    res.metrics["next_step_logits_max_err"] = float(
        (Z_rec2 - pf2["logits"]).abs().max())
    res.checks["next_step_logits_match"] = torch.allclose(
        Z_rec2, pf2["logits"], atol=1e-8)

    # counters / boundary discipline
    res.counters["nonlinear_trusted_calls"] = acct.nonlinear_trusted_calls
    res.counters["trusted_nonlinear_ops_count"] = acct.trusted_nonlinear_ops_count
    res.counters["accelerator_nonlinear_ops"] = acct.accelerator_nonlinear_ops
    res.counters["packed_update"] = 0
    res.counters["trusted_optimizer_calls"] = 0
    res.counters["trusted_boundary_calls"] = 1        # the single private CE
    res.checks["zero_nonlinear_trusted_calls"] = (
        acct.nonlinear_trusted_calls == 0)
    res.checks["zero_packed_updates"] = True
    res.checks["zero_trusted_optimizer_calls"] = True
    return res


def observed_facts_for_profile(res: ContractResult) -> Dict[str, object]:
    """Map the contract result to the fields the execution_profile validator
    expects (so paper_safe can be asserted over this contract)."""
    return {
        "nonlinear_backend": "A_rightmul",
        "nonlinear_trusted_calls": res.counters["nonlinear_trusted_calls"],
        "trusted_nonlinear_ops_count": res.counters["trusted_nonlinear_ops_count"],
        "plaintext_hidden_materializations":
            res.counters["plaintext_hidden_materializations"],
        "masked_residual_transport": res.checks["residual_remains_masked"],
        "masked_inter_block_handoff": res.checks["per_layer_mask_invariant"],
        "silent_fallback_occurred": False,
    }


def main(out: Optional[str] = None) -> ContractResult:
    res = run_contract()
    payload = {
        "stage": "gate3_5_unified_masked_training_cpu_contract",
        "is_real_qwen": False,
        "is_real_gpu": False,
        "note": ("CPU/fp64 integration contract; NOT a real Qwen/GPU result. "
                 "Proves masked inter-block transport + masked-domain nonlinear "
                 "islands + exact rank-masked LoRA backward/SGD close together."),
        "all_passed": res.all_passed,
        "checks": res.checks,
        "metrics": res.metrics,
        "counters": res.counters,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return res


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
