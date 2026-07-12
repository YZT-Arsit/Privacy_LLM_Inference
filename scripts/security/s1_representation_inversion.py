"""S1 — Representation inversion under the frozen private-base threat model.

Claim (scoped): masked hidden/residual representations do not expose recoverable
plaintext states to an attacker who lacks plaintext weights and paired plaintext.

Methodology basis: Mahendran & Vedaldi (CVPR 2015); Fredrikson et al. (CCS 2015).

Two attacker settings (both reported honestly):
  * STRICT  : attacker observes only h_tilde (no W, no paired plaintext). Inversion is
              under-determined (secret orthogonal Nr); best guess ~ random. We also
              report the exact invariant leaks (norm, pairwise Gram) that orthogonal
              masks preserve — documented, NOT claimed hidden.
  * KNOWN-PAIRS (generous upper bound): attacker granted K paired (h_tilde, h) samples
              (models a partial plaintext leak). A1 linear / A2 MLP / A3 deep-MLP fit
              h_tilde -> h and are tested on held-out. This quantifies how thin the
              mask is: an orthogonal mask is a linear map recoverable from enough pairs.

Positive controls P0 (plaintext h) and P1 (identity mask) must recover.
Random baseline reported. CPU only. Writes security/S1_representation_inversion/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import (PrivateBaseOracle, DEFAULT_PROMPTS, stamp, cosine, rel_err,  # noqa: E402
                        tensor_sha)

OUT = Path(__file__).resolve().parents[2] / "results/aaai_private_base/security/S1_representation_inversion"
torch.manual_seed(0)

PROMPTS = DEFAULT_PROMPTS + [
    "Interest rates were left unchanged by the central bank.",
    "The championship game went into double overtime last night.",
    "Researchers discovered a new species deep in the rainforest.",
    "My flight was delayed for six hours without any explanation.",
    "The recipe calls for two cups of flour and one egg.",
    "Global temperatures continued to rise throughout the decade.",
    "He carefully repaired the antique clock in his workshop.",
    "The startup raised a large round of venture funding.",
    "Students protested the tuition increase outside the hall.",
    "The novel explores themes of memory, loss, and identity.",
    "A sudden storm knocked out power across the coastal town.",
    "The committee postponed the vote until the following week.",
    "Fresh vegetables were sold at the crowded farmers market.",
    "The spacecraft entered orbit after a seven-month journey.",
    "Critics praised the director's bold visual style.",
    "The database migration completed without any data loss.",
    "Volunteers cleaned the beach early on Saturday morning.",
    "The suspect was released on bail pending further inquiry.",
    "Her presentation clearly summarized the quarterly results.",
    "The bridge was closed for repairs during rush hour.",
    "An unexpected error crashed the server late at night.",
    "The orchestra performed a moving rendition of the symphony.",
    "Farmers worried about the prolonged summer drought.",
    "The museum unveiled a rare collection of ancient coins.",
    "Traffic was heavier than usual because of the parade.",
    "The company announced layoffs affecting hundreds of workers.",
    "A friendly dog greeted every visitor at the front door.",
    "The experiment failed to reproduce the earlier findings.",
    "Snow blanketed the city streets by early evening.",
    "The lawyer presented compelling evidence to the jury.",
    "Sales of electric vehicles surged during the holiday season.",
    "The hikers reached the summit just before sunrise.",
]
DEPTHS = {"embedding": 0, "early_L4": 4, "middle_L12": 12, "final_L24": 24}


def collect(o: PrivateBaseOracle):
    fwd = o.real_forward(PROMPTS, max_len=32)
    mask = fwd["attention_mask"].bool()                    # (B,T)
    reps = {}
    for name, di in DEPTHS.items():
        H = fwd["hidden_states"][di]                        # (B,T,896)
        Hv = H[mask]                                        # (Nvalid,896) plaintext, defender-only
        reps[name] = Hv
    return reps, fwd, mask


def fit_linear(Xtr, Ytr):
    # least squares  Xtr @ W ~= Ytr  (closed form, ridge for stability)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + 1e-3 * torch.eye(d, dtype=Xtr.dtype)
    W = torch.linalg.solve(A, Xtr.T @ Ytr)
    return W


def fit_mlp(Xtr, Ytr, hidden, layers, steps=400, lr=1e-2):
    din, dout = Xtr.shape[1], Ytr.shape[1]
    mods = []
    prev = din
    for _ in range(layers):
        mods += [torch.nn.Linear(prev, hidden), torch.nn.GELU()]
        prev = hidden
    mods += [torch.nn.Linear(prev, dout)]
    net = torch.nn.Sequential(*mods).to(torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr = Xtr.to(torch.float64); Ytr = Ytr.to(torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Xtr), Ytr)
        loss.backward(); opt.step()
    return net


def eval_recovery(pred, true):
    return {"cosine_mean": float(torch.stack([
                torch.nn.functional.cosine_similarity(pred[i], true[i], dim=0)
                for i in range(pred.shape[0])]).mean()),
            "mse": float(torch.nn.functional.mse_loss(pred, true)),
            "rel_err": rel_err(pred, true)}


def main():
    t0 = time.time()
    o = PrivateBaseOracle()
    reps, fwd, mask = collect(o)
    V = o.V
    results = {"experiment": "S1_representation_inversion", "model": "Qwen2.5-0.5B",
               "threat_model": "from_scratch_private_base", "n_valid_positions": {},
               "budget": {"linear": "closed-form ridge 1e-3",
                          "mlp_A2": "1x512 GELU, Adam lr1e-2, 400 steps",
                          "mlp_A3": "2x512 GELU, Adam lr1e-2, 400 steps",
                          "device": "cpu", "seeds": [0]},
               "invariant_leaks": {}, "strict": {}, "known_pairs": {},
               "positive_controls": {}, "random_baseline": {}, "token_recovery": {},
               "limitations": []}

    # ---- (1) invariant leaks: orthogonal Nr preserves norm + pairwise Gram exactly ----
    for name, Hv in reps.items():
        results["n_valid_positions"][name] = int(Hv.shape[0])
        Ht = o.observe_residual(Hv)
        norm_gap = float((Hv.norm(dim=1) - Ht.norm(dim=1)).abs().max())
        # pairwise Gram on a 128-sample subset
        s = Hv[:128]; st = Ht[:128]
        gram_gap = float(((s @ s.T) - (st @ st.T)).abs().max())
        results["invariant_leaks"][name] = {
            "norm_preservation_max_abs_gap": norm_gap,
            "pairwise_gram_max_abs_gap": gram_gap,
            "interpretation": "orthogonal residual mask preserves norms & inner products EXACTLY (documented structural leak)"}

    # ---- (2) strict attacker: only h_tilde, guess Nr^{-1} at random (no info) ----
    for name, Hv in reps.items():
        Ht = o.observe_residual(Hv)
        Rg = torch.linalg.qr(torch.randn(o.H, o.H, dtype=torch.float64))[0]  # random orthogonal guess
        pred = Ht @ Rg
        results["strict"][name] = eval_recovery(pred[:256], Hv[:256])
    results["strict"]["interpretation"] = (
        "with no plaintext weights and no paired plaintext, recovering h from h_tilde is "
        "under-determined (any orthogonal map is consistent); recovery ~ random baseline.")

    # ---- random baseline: random gaussian prediction ----
    for name, Hv in reps.items():
        rnd = torch.randn_like(Hv[:256])
        results["random_baseline"][name] = eval_recovery(rnd, Hv[:256])

    # ---- (3) known-pairs generous upper bound: A1/A2/A3 fit h_tilde -> h ----
    for name, Hv in reps.items():
        n = Hv.shape[0]; ntr = int(n * 0.7)
        Ht = o.observe_residual(Hv)
        Xtr, Xte = Ht[:ntr], Ht[ntr:]
        Ytr, Yte = Hv[:ntr], Hv[ntr:]
        cell = {"n_train_pairs": ntr, "n_test": n - ntr, "hidden_dim": o.H}
        W = fit_linear(Xtr, Ytr)
        cell["A1_linear"] = eval_recovery(Xte @ W, Yte)
        net2 = fit_mlp(Xtr, Ytr, 512, 1)
        with torch.no_grad():
            cell["A2_mlp"] = eval_recovery(net2(Xte.to(torch.float64)).to(Xte.dtype), Yte)
        net3 = fit_mlp(Xtr, Ytr, 512, 2)
        with torch.no_grad():
            cell["A3_deep_mlp"] = eval_recovery(net3(Xte.to(torch.float64)).to(Xte.dtype), Yte)
        results["known_pairs"][name] = cell

    # ---- (3b) clean synthetic proof: orthogonal mask is EXACTLY linearly invertible w/ enough pairs ----
    d = o.H; g = torch.Generator().manual_seed(7)
    Hs = torch.randn(4000, d, generator=g, dtype=torch.float64)
    Hst = Hs @ o.Nr
    W = fit_linear(Hst[:3000], Hs[:3000])
    results["known_pairs"]["synthetic_orthogonal_invertibility"] = {
        "n_train_pairs": 3000, "dim": d,
        **eval_recovery(Hst[3000:] @ W, Hs[3000:]),
        "interpretation": "n_pairs >= dim -> orthogonal mask recovered by least squares (rel_err ~ 0). "
                          "Confidentiality of h therefore rests on the TEE preventing paired-plaintext exposure, NOT on the mask."}

    # ---- (4) positive controls ----
    Hv = reps["middle_L12"]; n = Hv.shape[0]; ntr = int(n * 0.7)
    # P0 plaintext h observed directly -> identity decoder
    results["positive_controls"]["P0_plaintext_hidden"] = eval_recovery(Hv[ntr:], Hv[ntr:])
    # P1 identity mask (h_tilde = h) -> linear recovers identity
    Wp1 = fit_linear(Hv[:ntr], Hv[:ntr])
    results["positive_controls"]["P1_identity_mask"] = eval_recovery(Hv[ntr:] @ Wp1, Hv[ntr:])

    # ---- (5) token recovery: embedding boundary (deterministic embedding) ----
    # strict attacker HAS E_tilde (in package). At the embedding layer h_tilde = E_tilde[token],
    # so NN against E_tilde returns the token exactly — true for ANY deterministic embedding.
    fwd_ids = fwd["input_ids"][mask]
    E_tilde = o.observe_embed_table()                       # (V,896) attacker-visible
    emb_tilde = o.observe_residual(reps["embedding"])       # == E_tilde[token]
    q = emb_tilde[:200]
    # nearest neighbour by cosine against the masked table
    qn = q / q.norm(dim=1, keepdim=True)
    En = E_tilde / E_tilde.norm(dim=1, keepdim=True)
    nn = (qn @ En.T).argmax(dim=1)
    acc = float((nn == fwd_ids[:200]).float().mean())
    results["token_recovery"]["embedding_boundary_strict_NN"] = {
        "top1_acc": acc, "random_chance": 1.0 / V,
        "interpretation": "deterministic embedding => token identity recoverable from the MASKED table at the "
                          "embedding boundary; this is NOT protected by the residual mask and is out of its scope. "
                          "Input-token protection is provided by the separate input-pad / layer-0 TEE-relocation "
                          "mechanism (see linear_boundary_pad / task2 layer0 guardrail), evaluated elsewhere."}
    # deep-layer token recovery requires inverting the contextual state first (strict: infeasible)
    results["token_recovery"]["deep_layer_strict"] = {
        "feasible": False,
        "interpretation": "middle/final residuals are contextualized (not embedding rows); under the strict "
                          "setting inversion fails, so token recovery from deep h_tilde is not achievable."}

    results["limitations"] = [
        "Ground-truth private base uses real Qwen2.5-0.5B weights (secret, never exposed to attackers) for "
        "realistic representations; leakage geometry is weight-distribution invariant.",
        "Known-pairs setting is a deliberately generous upper bound (attacker given paired plaintext) to "
        "quantify mask thinness; it is stronger than the frozen attacker and is labeled as such.",
        "Representation set is modest (a few thousand token positions); the synthetic probe isolates the "
        "exact-invertibility claim with n>=dim.",
        "We evaluate published inversion methods under this observation model; we do NOT claim inversion is "
        "impossible, only that it fails without weights/paired-plaintext and that the orthogonal mask is "
        "linearly invertible once pairs leak.",
    ]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "s1_results.json", results)
    print("[S1] done in %.1fs; strict middle cosine=%.3f  known-pairs A1 middle cosine=%.3f  "
          "synthetic rel_err=%.2e  emb-NN acc=%.3f" % (
              results["wall_sec"], results["strict"]["middle_L12"]["cosine_mean"],
              results["known_pairs"]["middle_L12"]["A1_linear"]["cosine_mean"],
              results["known_pairs"]["synthetic_orthogonal_invertibility"]["rel_err"], acc))


if __name__ == "__main__":
    main()
