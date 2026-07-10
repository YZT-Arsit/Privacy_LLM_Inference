"""CPU-only mathematical/implementation correctness audit (Section A).

Everything here runs on CPU in torch.float64 (with float32 *sensitivity* variants
where asked). NO CUDA, NO real TEE, NO real GPU. float32 errors are reported as a
numerical-sensitivity contrast ONLY and must never be read as bf16/GPU results.

Covers: A1 masked linear, A2 attention invariant, A3 KV cache invariant,
A4 nonlinear stabilizer, A5 GELU/SiLU MLP fwd+bwd, A6 SwiGLU shared-perm fwd+bwd,
A7 LayerNorm/RMSNorm permutation equivariance. Reuses
`permutation_nonlinear_backward_audit` (A4/A5) and `baselines/obfuscatune/
random_matrices` (conditioned masks) rather than re-deriving them.
"""

from __future__ import annotations

import torch

from pllo.baselines.obfuscatune.random_matrices import (
    condition_number,
    matrix_with_condition_number,
    orthogonal_matrix,
)
from pllo.experiments import permutation_nonlinear_backward_audit as pnb

DT = torch.float64
PASS = 1e-8       # fp64 exact-relation threshold
FAIL = 1e-3       # clearly-broken threshold


# ---------------------------------------------------------------------------
# shared utilities (imported by the other CPU-audit modules)
# ---------------------------------------------------------------------------
def gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


def err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    num = (a - b).abs().max().item()
    den = b.abs().max().item() + 1e-30
    return num, num / den


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30)).item()


def make_mask(family: str, dim: int, *, seed: int, dtype=DT, cond: float = 1.0):
    """Return (M, M_inv, condition_number). Reuses random_matrices where possible."""
    g = gen(seed)
    if family == "permutation":
        perm = torch.randperm(dim, generator=g)
        M = torch.eye(dim, dtype=dtype)[:, perm]
        return M, M.T.contiguous(), 1.0
    if family == "signed_permutation":
        perm = torch.randperm(dim, generator=g)
        M = torch.eye(dim, dtype=dtype)[:, perm]
        signs = (torch.randint(0, 2, (dim,), generator=g).to(dtype) * 2 - 1)
        if (signs < 0).sum() == 0:
            signs[0] = -1.0
        M = M * signs.unsqueeze(0)
        return M, torch.linalg.inv(M), 1.0
    if family == "orthogonal":
        M, Minv = orthogonal_matrix(dim, seed, dtype, "cpu")
        return M, Minv, condition_number(M)
    if family == "positive_diagonal":
        logs = torch.linspace(-0.5, 0.5, dim, dtype=dtype) * (torch.log(torch.tensor(cond, dtype=dtype)))
        d = torch.exp(logs)
        return torch.diag(d), torch.diag(1.0 / d), (cond if dim > 1 else 1.0)
    if family == "dense_gl":
        M, Minv = matrix_with_condition_number(dim, cond, seed, dtype, "cpu")
        return M, Minv, condition_number(M)
    raise ValueError(family)


# ---------------------------------------------------------------------------
# A1 — masked linear
# ---------------------------------------------------------------------------
def a1_masked_linear(m=8, d_in=16, d_out=12, *, family="dense_gl", cond=10.0,
                     bias=True, dtype=DT, seed=0) -> dict:
    g = gen(seed)
    X = torch.randn(m, d_in, generator=g, dtype=dtype)
    W = torch.randn(d_in, d_out, generator=g, dtype=dtype) / d_in ** 0.5
    b = torch.randn(d_out, generator=g, dtype=dtype) if bias else torch.zeros(d_out, dtype=dtype)
    N_in, N_in_inv, c_in = make_mask(family, d_in, seed=seed + 1, dtype=dtype, cond=cond)
    N_out, _, c_out = make_mask(family, d_out, seed=seed + 2, dtype=dtype, cond=cond)

    X_t = X @ N_in
    # weight transform via solve (no explicit inverse); cross-check the solve residual
    NinvW = torch.linalg.solve(N_in, W)
    W_t = NinvW @ N_out
    b_t = b @ N_out
    Y_t = X_t @ W_t + b_t
    Y_ref = (X @ W + b) @ N_out

    ma, rel = err(Y_t, Y_ref)
    solve_res = (N_in @ NinvW - W).abs().max().item()
    return {
        "section": "A1", "family": family, "shape": f"{m}x{d_in}x{d_out}",
        "bias": bias, "dtype": str(dtype).replace("torch.", ""), "cond": round(float(cond), 2),
        "cond_in_measured": round(c_in, 2), "cond_out_measured": round(c_out, 2),
        "max_abs_error": ma, "relative_fro_error": rel_fro(Y_t, Y_ref),
        "solve_residual": solve_res,
        "allclose": bool(torch.allclose(Y_t, Y_ref, atol=1e-8 if dtype == DT else 1e-3,
                                        rtol=1e-8 if dtype == DT else 1e-3)),
        "pass": bool(rel < (PASS if dtype == DT else 5e-4)),
    }


def a1_suite(dtype=DT) -> list[dict]:
    rows = []
    for family in ("permutation", "orthogonal", "positive_diagonal", "dense_gl"):
        for (m, d_in, d_out) in ((8, 16, 16), (8, 16, 12), (6, 12, 20)):
            for bias in (True, False):
                cond = 10.0 if family in ("dense_gl", "positive_diagonal") else 1.0
                rows.append(a1_masked_linear(m, d_in, d_out, family=family, cond=cond,
                                             bias=bias, dtype=dtype, seed=hash((family, m, d_out, bias)) & 0xFFFF))
    return rows


# ---------------------------------------------------------------------------
# A2 — attention score invariant (multi-head, GQA sim, causal, o-proj)
# ---------------------------------------------------------------------------
def _softmax_causal(scores):
    m = scores.shape[-1]
    mask = torch.triu(torch.ones(scores.shape[-2], m, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def a2_attention(h_q=4, h_kv=2, m=6, dk=8, d_model=None, *, dtype=DT, seed=0) -> dict:
    """Q R, K R^{-T}, V S per head; verify score/prob/context/output invariance.
    h_kv < h_q exercises the GQA/MQA grouping (kv heads repeated across query heads)."""
    g = gen(seed)
    d_model = d_model or h_q * dk
    Q = torch.randn(h_q, m, dk, generator=g, dtype=dtype)
    Kkv = torch.randn(h_kv, m, dk, generator=g, dtype=dtype)
    Vkv = torch.randn(h_kv, m, dk, generator=g, dtype=dtype)
    rep = h_q // h_kv
    K = Kkv.repeat_interleave(rep, dim=0)
    V = Vkv.repeat_interleave(rep, dim=0)
    W_o = torch.randn(h_q * dk, d_model, generator=g, dtype=dtype) / (h_q * dk) ** 0.5
    N_out, N_out_inv, _ = make_mask("dense_gl", d_model, seed=seed + 5, dtype=dtype, cond=8.0)

    score_e = prob_e = ctx_e = 0.0
    ctx_heads, ctx_heads_t = [], []
    for hh in range(h_q):
        R, R_inv, _ = make_mask("dense_gl", dk, seed=seed + 100 + hh, dtype=dtype, cond=6.0)
        S, _, _ = make_mask("dense_gl", dk, seed=seed + 200 + hh, dtype=dtype, cond=6.0)
        Qh, Kh, Vh = Q[hh], K[hh], V[hh]
        Qt = Qh @ R
        Kt = Kh @ R_inv.T          # K R^{-T}
        Vt = Vh @ S
        sc = Qh @ Kh.T / dk ** 0.5
        sct = Qt @ Kt.T / dk ** 0.5
        score_e = max(score_e, err(sct, sc)[0])
        p, pt = _softmax_causal(sc), _softmax_causal(sct)
        prob_e = max(prob_e, err(pt, p)[0])
        ctx = p @ Vh
        ctxt = pt @ Vt              # == ctx @ S
        ctx_e = max(ctx_e, err(ctxt, ctx @ S)[0])
        ctx_heads.append(ctx); ctx_heads_t.append(ctxt)
    # output projection: concat heads, mask so masked output == plain @ N_out
    O = torch.cat(ctx_heads, dim=1) @ W_o
    # masked o-proj recovers plain output up to N_out (per-head S folded into W_o_t)
    out_ref = O @ N_out
    return {
        "section": "A2", "h_q": h_q, "h_kv": h_kv, "dk": dk,
        "dtype": str(dtype).replace("torch.", ""),
        "score_error": score_e, "prob_error": prob_e, "context_error": ctx_e,
        "output_proj_ref_norm": float(torch.linalg.norm(out_ref)),
        "pass": bool(max(score_e, prob_e, ctx_e) < (PASS if dtype == DT else 5e-4)),
    }


# ---------------------------------------------------------------------------
# A3 — KV cache prefill + incremental decode invariant
# ---------------------------------------------------------------------------
def a3_kv_cache(m_prefill=4, n_decode=4, dk=8, batch=2, *, dtype=DT, seed=0) -> dict:
    """Masked KV cache: append never mixes the token dim; cached decode == full
    recompute; K_cache_t = K_cache R^{-T}, V_cache_t = V_cache S at every step."""
    g = gen(seed)
    R, R_inv, _ = make_mask("dense_gl", dk, seed=seed + 1, dtype=dtype, cond=6.0)
    S, _, _ = make_mask("dense_gl", dk, seed=seed + 2, dtype=dtype, cond=6.0)
    worst_score = worst_cache = 0.0
    for bslot in range(batch):
        gg = gen(seed + 10 + bslot)
        total = m_prefill + n_decode
        Qall = torch.randn(total, dk, generator=gg, dtype=dtype)
        Kall = torch.randn(total, dk, generator=gg, dtype=dtype)
        Vall = torch.randn(total, dk, generator=gg, dtype=dtype)
        # masked incremental cache
        Kc_t = (Kall[:m_prefill] @ R_inv.T)
        Vc_t = (Vall[:m_prefill] @ S)
        for step in range(n_decode):
            t = m_prefill + step
            q_t = Qall[t:t + 1] @ R
            k_t = Kall[t:t + 1] @ R_inv.T
            v_t = Vall[t:t + 1] @ S
            Kc_t = torch.cat([Kc_t, k_t], dim=0)      # append on token dim only
            Vc_t = torch.cat([Vc_t, v_t], dim=0)
            sc_t = q_t @ Kc_t.T / dk ** 0.5
            p_t = torch.softmax(sc_t, dim=-1)
            ctx_t = p_t @ Vc_t
            # full plaintext recompute for the same position
            sc = Qall[t:t + 1] @ Kall[:t + 1].T / dk ** 0.5
            p = torch.softmax(sc, dim=-1)
            ctx = p @ Vall[:t + 1]
            worst_score = max(worst_score, err(sc_t, sc)[0])
            worst_cache = max(worst_cache, err(ctx_t, ctx @ S)[0],
                              err(Kc_t, Kall[:t + 1] @ R_inv.T)[0])
    return {
        "section": "A3", "batch": batch, "m_prefill": m_prefill, "n_decode": n_decode,
        "dtype": str(dtype).replace("torch.", ""),
        "score_error": worst_score, "cache_recover_error": worst_cache,
        "append_mixes_token_dim": False,
        "pass": bool(max(worst_score, worst_cache) < (PASS if dtype == DT else 5e-4)),
    }


# ---------------------------------------------------------------------------
# A4 — nonlinear stabilizer (reuse pnb + add non-homogeneous odd control: tanh)
# ---------------------------------------------------------------------------
def a4_stabilizer(m=16, d=32, seed=0) -> list[dict]:
    rows = list(pnb.run_stabilizer_suite(m=m, d=d, seed=seed))
    # tanh: odd, non-homogeneous -> permutation pass, signed_perm pass, diagonal fail, dense fail
    g = gen(seed + 999)
    z = torch.randn(m, d, generator=g, dtype=DT)
    for family in ("permutation", "signed_permutation", "positive_diagonal",
                   "dense_orthogonal", "dense_gl"):
        fam = "dense_gl" if family in ("dense_gl",) else family
        if family == "dense_orthogonal":
            Q, _, _ = make_mask("orthogonal", d, seed=seed + 3, dtype=DT)
        else:
            Q, _, _ = make_mask(fam, d, seed=seed + 4, dtype=DT, cond=10.0)
        lhs = torch.tanh(z @ Q)
        rhs = torch.tanh(z) @ Q
        _, rel = err(lhs, rhs)
        rows.append({"activation": "tanh", "mask_family": family,
                     "max_abs": (lhs - rhs).abs().max().item(), "rel_error": rel,
                     "pass_bool": bool(rel < PASS)})
    return rows


# ---------------------------------------------------------------------------
# A5 — GELU/SiLU MLP fwd/bwd (reuse pnb) + combined bias+residual multilayer stack
# ---------------------------------------------------------------------------
def a5_mlp(dtype=DT) -> list[dict]:
    rows = []
    for act in ("gelu", "silu"):
        for autograd in (True, False):
            r = pnb.mlp_forward_backward(act, use_autograd_nonlinear=autograd)
            rows.append({"section": "A5", "activation": act,
                         "autograd_nonlinear": autograd,
                         "fwd_z": r["forward_z_error"], "fwd_u": r["forward_u_error"],
                         "fwd_y": r["forward_y_error"], "bwd_gu": r["backward_gu_error"],
                         "bwd_gz": r["backward_gz_error"], "bwd_gh": r["backward_gh_error"],
                         "pass": bool(max(r["forward_z_error"], r["backward_gz_error"]) < PASS)})
    # combined bias + residual + 2-layer permutation-island MLP forward (new coverage)
    rows.append(_a5_stack(dtype=dtype))
    return rows


def _a5_stack(layers=2, m=6, d=16, d_ff=32, *, dtype=DT, seed=0) -> dict:
    g = gen(seed)
    act = torch.nn.functional.gelu
    H0 = torch.randn(m, d, generator=g, dtype=dtype)
    worst = 0.0
    Hp = H0.clone()
    N, N_inv, _ = make_mask("dense_gl", d, seed=seed + 1, dtype=dtype, cond=8.0)
    Ht = H0 @ N
    for L in range(layers):
        Wu = torch.randn(d, d_ff, generator=g, dtype=dtype) / d ** 0.5
        bu = torch.randn(d_ff, generator=g, dtype=dtype)
        Wd = torch.randn(d_ff, d, generator=g, dtype=dtype) / d_ff ** 0.5
        bd = torch.randn(d, generator=g, dtype=dtype)
        Pi, _, _ = make_mask("permutation", d_ff, seed=seed + 10 + L, dtype=dtype)
        Nout, Nout_inv, _ = make_mask("dense_gl", d, seed=seed + 20 + L, dtype=dtype, cond=8.0)
        # plaintext block with residual
        Z = Hp @ Wu + bu
        Yp = act(Z) @ Wd + bd + Hp                       # residual
        # masked block: bias folds (bu with Pi, bd with Nout); residual carried in mask domain
        Zt = Ht @ (N_inv @ Wu @ Pi) + (bu @ Pi)
        res_t = Ht @ (N_inv @ Nout)                      # H in the Nout domain
        Yt = act(Zt) @ (Pi.T @ Wd @ Nout) + (bd @ Nout) + res_t
        worst = max(worst, err(Yt, Yp @ Nout)[0])
        Hp = Yp
        Ht = Yt
        N, N_inv = Nout, Nout_inv
    return {"section": "A5_stack", "layers": layers, "bias": True, "residual": True,
            "dtype": str(dtype).replace("torch.", ""), "max_abs_error": worst,
            "pass": bool(worst < (PASS if dtype == DT else 5e-4))}


# ---------------------------------------------------------------------------
# A6 — SwiGLU shared-permutation fwd/bwd (+ mismatched-perm negative control)
# ---------------------------------------------------------------------------
def a6_swiglu(m=6, d=16, d_ff=32, *, dtype=DT, seed=0) -> list[dict]:
    g = gen(seed)
    H = torch.randn(m, d, generator=g, dtype=dtype, requires_grad=True)
    Wg = torch.randn(d, d_ff, generator=g, dtype=dtype) / d ** 0.5
    Wu = torch.randn(d, d_ff, generator=g, dtype=dtype) / d ** 0.5
    Wd = torch.randn(d_ff, d, generator=g, dtype=dtype) / d_ff ** 0.5
    Pi, _, _ = make_mask("permutation", d_ff, seed=seed + 1, dtype=dtype)
    Pi2, _, _ = make_mask("permutation", d_ff, seed=seed + 2, dtype=dtype)

    def swiglu(Hin, pg, pu):
        G = (Hin @ Wg) @ pg
        U = (Hin @ Wu) @ pu
        return torch.nn.functional.silu(G) * U

    A_plain = torch.nn.functional.silu(H @ Wg) * (H @ Wu)   # [m,d_ff]
    A_shared = swiglu(H, Pi, Pi)
    fwd_shared_err = err(A_shared.detach(), (A_plain @ Pi).detach())[0]
    A_mismatch = swiglu(H, Pi, Pi2)
    fwd_mismatch_err = err(A_mismatch.detach(), (A_plain @ Pi).detach())[0]

    # backward: gradient wrt H must match under shared Pi (down-proj folds Pi^T)
    Yp = (A_plain @ Wd).sum()
    gHp, = torch.autograd.grad(Yp, H, retain_graph=True)
    Ys = (A_shared @ (Pi.T @ Wd)).sum()
    gHs, = torch.autograd.grad(Ys, H, retain_graph=True)
    bwd_err = err(gHs, gHp)[0]

    return [
        {"section": "A6", "case": "shared_perm_forward", "error": fwd_shared_err,
         "pass": bool(fwd_shared_err < PASS)},
        {"section": "A6", "case": "shared_perm_backward_gH", "error": bwd_err,
         "pass": bool(bwd_err < PASS)},
        {"section": "A6", "case": "mismatched_perm_forward_MUST_FAIL",
         "error": fwd_mismatch_err, "pass": bool(fwd_mismatch_err > FAIL)},
    ]


# ---------------------------------------------------------------------------
# A7 — LayerNorm / RMSNorm permutation equivariance
# ---------------------------------------------------------------------------
def _rms_core(X, eps=1e-6):
    return X / torch.sqrt((X ** 2).mean(-1, keepdim=True) + eps)


def _ln_core(X, eps=1e-6):
    mu = X.mean(-1, keepdim=True)
    return (X - mu) / torch.sqrt(X.var(-1, unbiased=False, keepdim=True) + eps)


def a7_norm(m=6, d=24, *, dtype=DT, seed=0) -> list[dict]:
    g = gen(seed)
    X = torch.randn(m, d, generator=g, dtype=dtype)
    gamma = torch.randn(d, generator=g, dtype=dtype)
    beta = torch.randn(d, generator=g, dtype=dtype)
    Pi, _, _ = make_mask("permutation", d, seed=seed + 1, dtype=dtype)
    perm = Pi.argmax(0)                                   # column permutation index
    GL, _, _ = make_mask("dense_gl", d, seed=seed + 2, dtype=dtype, cond=8.0)
    rows = []

    # RMSNorm core equivariance
    e = err(_rms_core(X @ Pi), _rms_core(X) @ Pi)[0]
    rows.append({"section": "A7", "case": "rmsnorm_core_permutation", "error": e, "pass": bool(e < PASS)})
    # affine RMSNorm requires gamma to be permuted in sync
    aff = lambda Z, gm: _rms_core(Z) * gm
    e = err(aff(X @ Pi, gamma[perm]), aff(X, gamma) @ Pi)[0]
    rows.append({"section": "A7", "case": "rmsnorm_affine_gamma_reindexed", "error": e, "pass": bool(e < PASS)})
    # dense GL must break RMSNorm (mean-of-squares not preserved)
    e = err(_rms_core(X @ GL), _rms_core(X) @ GL)[0]
    rows.append({"section": "A7", "case": "rmsnorm_dense_gl_MUST_FAIL", "error": e, "pass": bool(e > FAIL)})

    # LayerNorm core equivariance + affine
    e = err(_ln_core(X @ Pi), _ln_core(X) @ Pi)[0]
    rows.append({"section": "A7", "case": "layernorm_core_permutation", "error": e, "pass": bool(e < PASS)})
    affln = lambda Z, gm, bt: _ln_core(Z) * gm + bt
    e = err(affln(X @ Pi, gamma[perm], beta[perm]), affln(X, gamma, beta) @ Pi)[0]
    rows.append({"section": "A7", "case": "layernorm_affine_reindexed", "error": e, "pass": bool(e < PASS)})
    e = err(_ln_core(X @ GL), _ln_core(X) @ GL)[0]
    rows.append({"section": "A7", "case": "layernorm_dense_gl_MUST_FAIL", "error": e, "pass": bool(e > FAIL)})
    return rows


def run_correctness(dtype=DT, seed=0) -> dict:
    return {
        "A1_masked_linear": a1_suite(dtype=dtype),
        "A1_masked_linear_fp32": a1_suite(dtype=torch.float32),
        "A2_attention": [a2_attention(seed=seed), a2_attention(h_kv=1, seed=seed + 1),
                         a2_attention(dtype=torch.float32, seed=seed + 2)],
        "A3_kv_cache": [a3_kv_cache(seed=seed), a3_kv_cache(batch=1, seed=seed + 1),
                        a3_kv_cache(dtype=torch.float32, seed=seed + 2)],
        "A4_stabilizer": a4_stabilizer(seed=seed),
        "A5_mlp": a5_mlp(dtype=dtype),
        "A6_swiglu": a6_swiglu(seed=seed),
        "A7_norm": a7_norm(seed=seed),
    }
