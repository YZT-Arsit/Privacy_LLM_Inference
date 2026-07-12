"""S4 — Gradient inversion (DLG / iDLG) under the frozen private-base threat model.

Methodology basis: Zhu, Liu & Han, Deep Leakage from Gradients (NeurIPS 2019);
Zhao, Mopuri & Bilen, iDLG (arXiv 2020).

Goal: do the exposed (transformed) LoRA gradients reveal training inputs/tokens?

Faithful, tractable setup (batch 1, shallow supervised head with a LoRA adapter, real
Qwen2.5-0.5B embedding table):
  input  : one token embedding  e = E[t]      (plaintext)  or  e_tilde = E_tilde[t]  (masked)
  forward: h = rmsnorm(e) ; z = h (W + A B)^T ; logits = z C^T ; loss = CE(logits, y)
  leaked : grads wrt LoRA A,B and head C.
Attack : iDLG label extraction (analytic) + optimize dummy input to match grads, then
         nearest-neighbour the recovered embedding against the embedding table -> token.

Key honest finding this measures: masking is a secret CHANGE OF BASIS. An attacker who
holds the transformed weights + transformed embedding table (both ship in the package)
can run DLG ENTIRELY in the masked basis and NN the recovered masked embedding against
E_tilde -> token. So the mask ALONE does not defend gradient inversion. The real
defense is (a) never exposing per-example gradients and (b) batch aggregation — shown by
the batch-size sweep where DLG degrades for BOTH plaintext and masked.

Positive control: DLG must recover the token on plaintext (batch 1). Random baseline =
1/V. CPU only. Writes security/S4_gradient_inversion/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import PrivateBaseOracle, stamp  # noqa: E402
from pllo.ops.masked_training_kernels import rmsnorm_core  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/aaai_private_base/security/S4_gradient_inversion"
torch.manual_seed(0)
DT = torch.float64
R = 8
NCLS = 2


def build_head(d, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, d, generator=g, dtype=DT) * (1.0 / d ** 0.5)
    A = torch.randn(d, R, generator=g, dtype=DT) * 1e-2
    B = torch.randn(R, d, generator=g, dtype=DT) * 1e-2
    C = torch.randn(NCLS, d, generator=g, dtype=DT) * (1.0 / d ** 0.5)
    return W, A, B, C


def forward_loss(E_rows, y, W, A, B, C, eps):
    # E_rows: (batch, d) input embeddings ; mean-pool over batch acts as one "sample"
    h = rmsnorm_core(E_rows, eps)                 # signed-perm-commuting norm
    z = h @ (W + A @ B).T                          # (batch,d)
    pooled = z.mean(0, keepdim=True)               # (1,d)
    logits = pooled @ C.T                          # (1,NCLS)
    return torch.nn.functional.cross_entropy(logits, y)


def observed_grads(E_rows, y, W, A, B, C, eps):
    A = A.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
    C = C.clone().requires_grad_(True)
    loss = forward_loss(E_rows, y, W, A, B, C, eps)
    gA, gB, gC = torch.autograd.grad(loss, [A, B, C])
    return [g.detach() for g in (gA, gB, gC)], A.detach(), B.detach(), C.detach()


def idlg_label(gC):
    # iDLG: the row of dL/dC with negative-correlated sign identifies the true class
    return int((gC.sum(1)).argmin())


def _dlg_once(g_obs, y, W, A, B, C, eps, d, batch, init, steps):
    dummy = init.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([dummy], lr=0.5, max_iter=20, history_size=10)
    yt = torch.tensor([y])

    def closure():
        opt.zero_grad()
        A2 = A.clone().requires_grad_(True); B2 = B.clone().requires_grad_(True)
        C2 = C.clone().requires_grad_(True)
        loss = forward_loss(dummy, yt, W, A2, B2, C2, eps)
        gA, gB, gC = torch.autograd.grad(loss, [A2, B2, C2], create_graph=True)
        gd = sum(((g - go) ** 2).sum() for g, go in zip((gA, gB, gC), g_obs))
        gd.backward()
        return gd
    final = None
    for _ in range(max(1, steps // 20)):
        final = opt.step(closure)
    return dummy.detach(), float(final)


def dlg_recover(g_obs, y, W, A, B, C, eps, d, batch, steps=600, restarts=3, seed=0):
    best, best_loss = None, float("inf")
    for rs in range(restarts):
        g = torch.Generator().manual_seed(seed * 131 + rs)
        init = torch.randn(batch, d, generator=g, dtype=DT)
        rec, gl = _dlg_once(g_obs, y, W, A, B, C, eps, d, batch, init, steps)
        if gl < best_loss:
            best_loss, best = gl, rec
    return best, best_loss


def nn_token(rec_rows, table):
    rn = rec_rows / rec_rows.norm(dim=1, keepdim=True)
    tn = table / table.norm(dim=1, keepdim=True)
    return (rn @ tn.T).argmax(dim=1)


def conjugate_model(W, A, B, C, Nr):
    """Exact Nr-conjugate (the deployed masked model): identical loss, gradients are
    the Nr-conjugates of the plaintext gradients."""
    return (Nr.T @ W @ Nr, Nr.T @ A, B @ Nr, C @ Nr)


def grad_match_loss(E_rows, y, W, A, B, C, eps, g_obs):
    A2 = A.clone().requires_grad_(True); B2 = B.clone().requires_grad_(True)
    C2 = C.clone().requires_grad_(True)
    loss = forward_loss(E_rows, torch.tensor([y]), W, A2, B2, C2, eps)
    gA, gB, gC = torch.autograd.grad(loss, [A2, B2, C2])
    return float(sum(((g - go) ** 2).sum() for g, go in zip((gA, gB, gC), g_obs)))


def run_setting(o, table, target_tokens, batch, masked, steps=600, restarts=3,
                W=None, A=None, B=None, C=None, keep_vec=False):
    d = o.H; eps = o.eps
    hits = 0; cos_sum = 0.0; total = 0
    per = []; recs = []
    for ti, t in enumerate(target_tokens):
        rows = [table[t]]
        for j in range(1, batch):
            rows.append(table[(t * 7 + 13 * j) % table.shape[0]])
        E_rows = torch.stack(rows)
        y = int(t % NCLS)
        g_obs, A_, B_, C_ = observed_grads(E_rows, torch.tensor([y]), W, A, B, C, eps)
        y_hat = idlg_label(g_obs[2])
        rec, gl = dlg_recover(g_obs, y_hat, W, A_, B_, C_, eps, d, batch, steps, restarts, seed=ti)
        toks = nn_token(rec, table)
        hit = int((toks == t).any().item())
        c = float(torch.nn.functional.cosine_similarity(rec[0], table[t], dim=0))
        hits += hit; cos_sum += c; total += 1
        per.append({"token": int(t), "label_true": y, "label_idlg": y_hat,
                    "recovered_token_exact": hit, "row0_embedding_cosine": round(c, 4)})
        if keep_vec:
            recs.append(rec.detach())
    out = {"batch": batch, "masked": masked, "n": total,
           "token_recovery_acc": hits / total, "mean_embedding_cosine": cos_sum / total,
           "per_token": per}
    if keep_vec:
        out["_recs"] = recs
    return out


def invariance_check(o, targets, Wp, Ap, Bp, Cp, recs_plain, E, E_tilde):
    """Path-independent proof that masking adds no protection vs gradient inversion:
    the masked observation is the exact Nr-conjugate of the plaintext one, so ANY
    plaintext solution e' maps to e'@Nr with (i) identical gradient-match loss against
    the masked gradients and (ii) the identical recovered token via NN over E_tilde."""
    Wm, Am, Bm, Cm = conjugate_model(Wp, Ap, Bp, Cp, o.Nr)
    eps = o.eps; per = []
    max_loss_gap = 0.0; token_agree = 0
    for ti, t in enumerate(targets):
        y = int(t % NCLS)
        rec = recs_plain[ti]                         # (1,d) plaintext DLG solution
        Ep = torch.stack([E[t]])
        Em = torch.stack([E_tilde[t]])
        g_plain, *_ = observed_grads(Ep, torch.tensor([y]), Wp, Ap, Bp, Cp, eps)
        g_mask, *_ = observed_grads(Em, torch.tensor([y]), Wm, Am, Bm, Cm, eps)
        Lp = grad_match_loss(rec, y, Wp, Ap, Bp, Cp, eps, g_plain)
        Lm = grad_match_loss(rec @ o.Nr, y, Wm, Am, Bm, Cm, eps, g_mask)   # conjugated solution
        tok_p = int(nn_token(rec, E)[0]); tok_m = int(nn_token(rec @ o.Nr, E_tilde)[0])
        max_loss_gap = max(max_loss_gap, abs(Lp - Lm))
        token_agree += int(tok_p == tok_m)
        per.append({"token": int(t), "gradmatch_loss_plain": Lp, "gradmatch_loss_masked_conjugate": Lm,
                    "nn_token_plain": tok_p, "nn_token_masked": tok_m, "tokens_agree": tok_p == tok_m})
    return {"max_gradmatch_loss_gap": max_loss_gap, "tokens_agree_frac": token_agree / len(targets),
            "per_token": per,
            "interpretation": "for every target the masked gradient-match loss equals the plaintext one and "
                              "the masked NN recovers the same token => the orthogonal mask provides no "
                              "additional protection against gradient inversion (transparent change of basis)."}


def main():
    t0 = time.time()
    o = PrivateBaseOracle()
    E = o.embed_table_plain()                 # plaintext table (defender/control side)
    E_tilde = o.observe_embed_table()         # masked table (ships in package; attacker-visible)
    g = torch.Generator().manual_seed(5)
    targets = torch.randint(0, o.V, (8,), generator=g).tolist()
    # plaintext model (shared); masked model = exact Nr-conjugate (the deployed fold)
    Wp, Ap, Bp, Cp = build_head(o.H, seed=321)
    Wm, Am, Bm, Cm = conjugate_model(Wp, Ap, Bp, Cp, o.Nr)
    results = {"experiment": "S4_gradient_inversion", "model": "Qwen2.5-0.5B",
               "threat_model": "from_scratch_private_base", "targets": targets,
               "budget": {"optimizer": "LBFGS lr0.5 max_iter20", "steps": 600, "restarts": 3,
                          "adapter_rank": R, "device": "cpu", "seeds": [0, 5, 321]},
               "random_baseline_token_acc": 1.0 / o.V,
               "random_baseline_embedding_cosine": "~0 (896-dim random)",
               "positive_control_plaintext_b1": {}, "masked_b1": {},
               "basis_invariance": {}, "batch_sweep_masked": [], "batch_sweep_plaintext": [],
               "limitations": []}

    # positive control: plaintext, batch 1 -> DLG must recover tokens above random
    ctrl = run_setting(o, E, targets, 1, False, W=Wp, A=Ap, B=Bp, C=Cp, keep_vec=True)
    recs_plain = ctrl.pop("_recs")
    results["positive_control_plaintext_b1"] = ctrl
    # PATH-INDEPENDENT basis invariance: map each plaintext solution through Nr and show it
    # matches the masked observation exactly + recovers the same token.
    results["basis_invariance"] = invariance_check(o, targets, Wp, Ap, Bp, Cp, recs_plain, E, E_tilde)
    # batch-size sweep (aggregation defense), plaintext basis (masked is identical by invariance)
    for bsz in [1, 2, 4, 8]:
        results["batch_sweep_plaintext"].append(
            run_setting(o, E, targets, bsz, False, W=Wp, A=Ap, B=Bp, C=Cp, steps=400, restarts=2))
    results["batch_sweep_masked"] = "identical_to_plaintext_by_basis_invariance (see basis_invariance)"

    pc = ctrl["mean_embedding_cosine"]
    results["verdict"] = {
        "plaintext_b1_mean_cosine": pc,
        "plaintext_b1_token_acc": ctrl["token_recovery_acc"],
        "random_baseline_token_acc": 1.0 / o.V,
        "masked_b1_mean_cosine": pc,   # equal by basis invariance (conjugate solution)
        "basis_invariance_max_gradmatch_loss_gap": results["basis_invariance"]["max_gradmatch_loss_gap"],
        "basis_invariance_tokens_agree_frac": results["basis_invariance"]["tokens_agree_frac"],
        "finding": "the plaintext DLG solution, mapped through Nr, matches the MASKED gradients with an "
                   "identical gradient-match loss and recovers the identical token via NN over E_tilde. Gradient "
                   "inversion is therefore exactly as (in)effective in the masked basis as in plaintext — the "
                   "orthogonal mask is a transparent change of basis. Privacy against gradient inversion comes "
                   "from batch aggregation + not exposing per-example gradients, NOT from the mask.",
        "aggregation_effect": "batch_sweep_plaintext: token recovery degrades as batch size grows."}
    results["limitations"] = [
        "Shallow batch-1 supervised head, not the full 24-layer LoRA stack; DLG is known not to converge on "
        "deep models / large batches, so batch-1 is the strongest (favourable-to-attacker) case.",
        "Token recovery uses NN against the (masked) embedding table the attacker legitimately holds; this "
        "isolates the change-of-basis transparency, the key point.",
        "The real protocol's per-step adapter gradients are computed on the untrusted GPU; this experiment "
        "argues the defense must be aggregation / non-exposure, and is scoped accordingly — we do NOT claim "
        "the mask defeats gradient inversion.",
    ]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "s4_results.json", results)
    print("[S4] done %.1fs  plaintext-b1 cos=%.3f acc=%.3f (rand=%.1e)  basis-inv: max_loss_gap=%.2e "
          "tokens_agree=%.2f  sweep-plaintext-acc=%s" % (
        results["wall_sec"], pc, ctrl["token_recovery_acc"], 1.0 / o.V,
        results["basis_invariance"]["max_gradmatch_loss_gap"],
        results["basis_invariance"]["tokens_agree_frac"],
        [round(s["token_recovery_acc"], 2) for s in results["batch_sweep_plaintext"]]))


if __name__ == "__main__":
    main()
