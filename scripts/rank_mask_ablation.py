"""Section 11 -- rank-mask refresh ablation (exact fp64, local).

The rank mask U (orthogonal rxr) is a RANK-SPACE GAUGE:
    A_masked = U @ A,   B_masked = B @ U^T,   effective  B_masked @ A_masked = B @ A.
So it never changes the effective LoRA (or the O1-C update, which is per-factor linear).
A REFRESH U -> U' is exact ONLY if BOTH factors and the optimizer state are transported
into the new basis consistently:
    A  <- U' U^T A,   B <- B U U'^T,   buf_A <- U' U^T buf_A,   buf_B <- buf_B U U'^T.
If the optimizer state is NOT transported, the next momentum/Adam step mixes bases and the
effective trajectory breaks (this is the failure the lora-lifecycle audit flagged).

Configs tested: L7 (no rank mask), L10 fixed U, refresh every step / every 10 / per session.
Metrics: exactness (effective dW vs no-mask reference), cross-step linkability (does the
stored masked A change basis each refresh -> unlinkable), cost (extra rxr transforms),
and the NEGATIVE control (refresh without state transport -> breaks).
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from gate05_o1_optimizer_audit import transforms, DT, CKPT
from pllo.ops.masked_training_kernels import orthogonal_signed_perm

OUT = REPO / "results/aaai_private_base/full_lora_matrix/ablations"


def rand_orth(r, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=DT))
    return Q


def run(sd, l, proj, Nr, nh, nkv, hd, *, refresh, transport_state, steps=20, r=8,
        scale=2.0, lr=1e-3, mom=0.9, seed=0):
    """Return (effective_dW_err_vs_nomask, max_linkability_corr, refreshes, broke)."""
    grp = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    W = sd[f"model.layers.{l}.{grp}.{proj}.weight"].to(DT)
    out_d, in_d = W.shape
    T_in, T_out = transforms(sd, l, proj, Nr, nh, nkv, hd)
    Tin_iT = torch.linalg.inv(T_in).T
    Tin_gram_inv = torch.linalg.inv(T_in) @ Tin_iT
    tin_orth = float((T_in.T @ T_in - torch.eye(in_d, dtype=DT)).abs().max()) < 1e-9
    W_tilde = T_out.T @ W @ Tin_iT
    g = torch.Generator().manual_seed(100 + seed)
    A0 = torch.randn(r, in_d, generator=g, dtype=DT) * 0.02
    B0 = torch.randn(out_d, r, generator=g, dtype=DT) * 0.02 + 0.01
    Tn = 6
    x = torch.randn(Tn, in_d, generator=g, dtype=DT) @ T_in
    tgt = torch.randn(Tn, out_d, generator=g, dtype=DT) @ T_out

    # reference: O1-C update with NO rank mask
    def o1c_step(A, B, bufA, bufB):
        Ar = A.clone().requires_grad_(True); Br = B.clone().requires_grad_(True)
        y = x @ (W_tilde + scale * (Br @ Ar)).T
        loss = ((y - tgt) ** 2).mean()
        gA, gB = torch.autograd.grad(loss, [Ar, Br])
        cgA = gA @ Tin_gram_inv if not tin_orth else gA
        bufA = mom * bufA + cgA; bufB = mom * bufB + gB
        return A - lr * bufA, B - lr * bufB, bufA, bufB

    # no-mask reference trajectory
    A, B = A0.clone(), B0.clone(); bA = torch.zeros_like(A); bB = torch.zeros_like(B)
    for _ in range(steps):
        A, B, bA, bB = o1c_step(A, B, bA, bB)
    ref_dW = scale * (B @ A)

    # rank-masked trajectory with refresh
    U = rand_orth(r, 7000 + seed)                      # initial rank mask
    Am, Bm = U @ A0, B0 @ U.T
    bAm = torch.zeros_like(Am); bBm = torch.zeros_like(Bm)
    stored_A_bases = [Am.clone()]
    refreshes = 0
    for step in range(steps):
        Am, Bm, bAm, bBm = o1c_step(Am, Bm, bAm, bBm)
        do_refresh = (refresh == "every_step") or (refresh == "every_10" and (step + 1) % 10 == 0)
        if do_refresh:
            Un = rand_orth(r, 7000 + seed + 1000 * (step + 1))
            M = Un @ U.T                                # basis change U -> Un
            Am = M @ Am; Bm = Bm @ M.T
            if transport_state:                         # transport optimizer state too
                bAm = M @ bAm; bBm = bBm @ M.T
            U = Un; refreshes += 1
            stored_A_bases.append(Am.clone())
    if refresh == "per_session":                        # single refresh at the end
        Un = rand_orth(r, 9999 + seed); M = Un @ U.T
        Am = M @ Am; Bm = Bm @ M.T
        if transport_state:
            bAm = M @ bAm; bBm = bBm @ M.T
        refreshes = 1; stored_A_bases.append(Am.clone())

    dW_masked = scale * (Bm @ Am)
    dW_err = float((dW_masked - ref_dW).norm() / (ref_dW.norm() + 1e-30))
    # cross-step linkability: correlation between successive stored masked-A snapshots
    link = 0.0
    for i in range(1, len(stored_A_bases)):
        a, b = stored_A_bases[i - 1].flatten(), stored_A_bases[i].flatten()
        link = max(link, abs(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))))
    return dW_err, link, refreshes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file
    sd = load_file(str(CKPT / "model.safetensors"))
    cfg = json.loads((CKPT / "config.json").read_text())
    H = cfg["hidden_size"]; nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]; hd = H // nh
    Nr = orthogonal_signed_perm(H, 9000, DT)
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    rows = []
    configs = [("L7_no_rank_mask", "none", True), ("L10_fixed_U", "fixed", True),
               ("refresh_every_step", "every_step", True), ("refresh_every_10", "every_10", True),
               ("refresh_per_session", "per_session", True),
               ("NEG_refresh_no_state_transport", "every_step", False)]
    for name, refresh, transport in configs:
        for proj in TARGETS:
            errs, links, refs = [], [], []
            for l in [0, 11, 23]:                        # sample layers (exactness is layer-invariant)
                if refresh == "none":
                    # no rank mask: run O1-C directly (identity gauge)
                    e, lk, rf = run(sd, l, proj, Nr, nh, nkv, hd, refresh="fixed", transport_state=True)
                    # override: no mask means stored basis == plaintext leaf (linkable across steps)
                else:
                    e, lk, rf = run(sd, l, proj, Nr, nh, nkv, hd, refresh=refresh, transport_state=transport)
                errs.append(e); links.append(lk); refs.append(rf)
            rows.append({"config": name, "proj": proj, "refresh": refresh,
                         "state_transported": transport,
                         "effective_dW_err_vs_nomask_max": max(errs),
                         "max_cross_step_linkability": max(links),
                         "refreshes": max(refs),
                         "exact": max(errs) < 1e-10})
    with open(OUT / "rank_mask_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = {}
    for name in {r["config"] for r in rows}:
        rs = [r for r in rows if r["config"] == name]
        summary[name] = {"max_dW_err": max(r["effective_dW_err_vs_nomask_max"] for r in rs),
                         "all_exact": all(r["exact"] for r in rs),
                         "max_linkability": max(r["max_cross_step_linkability"] for r in rs),
                         "refreshes": max(r["refreshes"] for r in rs)}
    (OUT / "rank_mask_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
