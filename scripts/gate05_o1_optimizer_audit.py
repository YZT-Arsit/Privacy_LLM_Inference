"""Gate 0.5 -- O1 gamma-fold & optimizer-equivalence audit (exact arithmetic).

Answers: does the package-native LoRA update (O1-A, naive masked SGD on the H800) equal
ordinary plaintext SGD folded into the masked domain, or does the RMSNorm-gamma fold make
the LoRA INPUT transform non-orthogonal, requiring a corrected update?

Row-vector convention (code: y = x @ W.T, x is (T,in)). For each target we define the
effective input/output transforms  x_tilde = x_plain @ T_in,  y_tilde = y_plain @ T_out,
and the fold  A_tilde = A_plain @ T_in^{-T},  B_tilde = T_out^T @ B_plain.

Derived (see symbolic_derivation.md):
  naive masked SGD on A_tilde == folded plaintext SGD  <=>  T_in^T T_in = I
  naive masked SGD on B_tilde == folded plaintext SGD  <=>  T_out^T T_out = I
Correction (O1-B/C):  A_tilde' = A_tilde - lr * gA_tilde @ (T_in^T T_in)^{-1}.
For our masks T_out is always orthogonal (Bq/Bk/Sv/Nr/P) so B is always exact; T_in is
orthogonal only for o_proj (So) and down_proj (P); it is diag(1/gamma) @ Nr (NON-orthogonal)
for q/k/v/gate/up.  Correction tensor = (T_in^T T_in)^{-1} = Nr^T diag(gamma^2) Nr.

fp64 exact-arithmetic diagnostics on the REAL checkpoint. Emits results/.../gate05_o1/*.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src"))
from pllo.ops.masked_training_kernels import orthogonal_signed_perm, permutation_matrix

CKPT = Path("/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B")
OUT = REPO / "results/aaai_private_base/gate05_o1"
DT = torch.float64
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def rope_rot(hd, seed):
    half = hd // 2
    g = torch.Generator().manual_seed(int(seed))
    ang = torch.rand(half, generator=g, dtype=DT) * 6.283185307179586
    B = torch.eye(hd, dtype=DT); c, s = torch.cos(ang), torch.sin(ang)
    for i in range(half):
        j = i + half; B[i, i] = c[i]; B[j, j] = c[i]; B[i, j] = -s[i]; B[j, i] = s[i]
    return B


def transforms(sd, l, proj, Nr, nh, nkv, hd):
    """Return (T_in, T_out) with x_tilde=x_plain@T_in, y_tilde=y_plain@T_out."""
    ga = sd[f"model.layers.{l}.input_layernorm.weight"].to(DT)
    gm = sd[f"model.layers.{l}.post_attention_layernorm.weight"].to(DT)
    B = rope_rot(hd, 1000 + l); S = orthogonal_signed_perm(hd, 2000 + l, DT)
    P = permutation_matrix(sd[f"model.layers.{l}.mlp.gate_proj.weight"].shape[0], 3000 + l, DT)
    bd = lambda M, r: torch.block_diag(*([M] * r))
    inv_ga = torch.diag(1.0 / ga); inv_gm = torch.diag(1.0 / gm)
    if proj == "q_proj":    return inv_ga @ Nr, bd(B, nh)
    if proj == "k_proj":    return inv_ga @ Nr, bd(B, nkv)
    if proj == "v_proj":    return inv_ga @ Nr, bd(S, nkv)
    if proj == "o_proj":    return bd(S, nh), Nr
    if proj == "gate_proj": return inv_gm @ Nr, P
    if proj == "up_proj":   return inv_gm @ Nr, P
    if proj == "down_proj": return P, Nr
    raise ValueError(proj)


def orthog_dev(M):
    G = M.T @ M
    I = torch.eye(G.shape[0], dtype=DT)
    return float((G - I).abs().max())


def cond(M):
    s = torch.linalg.svdvals(M)
    return float(s.max() / s.min())


def one_step_target(sd, l, proj, Nr, nh, nkv, hd, r=8, scale=2.0, lr=1e-3,
                    steps=1, momentum=0.0, seed=0):
    """Exact per-target diagnostic. Returns metrics comparing plaintext SGD to O1-A
    (naive masked) and O1-B/C (corrected) over `steps`, with optional momentum."""
    W = sd[f"model.layers.{l}.self_attn.{proj}.weight" if "proj" in proj and proj in
           ("q_proj", "k_proj", "v_proj", "o_proj") else
           f"model.layers.{l}.mlp.{proj}.weight"].to(DT)
    out_d, in_d = W.shape
    T_in, T_out = transforms(sd, l, proj, Nr, nh, nkv, hd)
    Tin_inv = torch.linalg.inv(T_in)                  # precomputed once (perf)
    Tin_iT = Tin_inv.T                                # T_in^{-T}
    Tin_T = T_in.T
    Tout_inv_T = torch.linalg.inv(T_out).T            # T_out^{-T}
    Tin_gram_inv = Tin_inv @ Tin_iT                   # (T_in^T T_in)^{-1} = correction
    g = torch.Generator().manual_seed(100 + seed)
    A0 = torch.randn(r, in_d, generator=g, dtype=DT) * 0.02
    B0 = torch.randn(out_d, r, generator=g, dtype=DT) * 0.02 + 0.01
    T = 6
    x_plain = torch.randn(T, in_d, generator=g, dtype=DT)
    tgt_plain = torch.randn(T, out_d, generator=g, dtype=DT)
    # masked-domain versions
    x_tilde = x_plain @ T_in
    tgt_tilde = tgt_plain @ T_out
    W_tilde = T_out.T @ W @ Tin_iT

    # state
    Ap, Bp = A0.clone(), B0.clone()
    AtA, BtA = (A0 @ Tin_iT).clone(), (T_out.T @ B0).clone()   # O1-A masked leaves
    AtB, BtB = (A0 @ Tin_iT).clone(), (T_out.T @ B0).clone()   # O1-B/C corrected
    mAp = mBp = mAtA = mBtA = mAtB = mBtB = 0.0
    hist = []

    def grads(A, B, W_, x, tgt):
        A = A.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
        y = x @ (W_ + scale * (B @ A)).T
        loss = ((y - tgt) ** 2).mean()
        gA, gB = torch.autograd.grad(loss, [A, B])
        return float(loss), gA, gB

    for step in range(steps):
        Lp, gAp, gBp = grads(Ap, Bp, W, x_plain, tgt_plain)
        LtA, gAtA, gBtA = grads(AtA, BtA, W_tilde, x_tilde, tgt_tilde)
        _, gAtB, gBtB = grads(AtB, BtB, W_tilde, x_tilde, tgt_tilde)
        # momentum buffers (v = mom*v + grad; theta -= lr*v)
        mAp = momentum * mAp + gAp; mBp = momentum * mBp + gBp
        Ap = Ap - lr * mAp; Bp = Bp - lr * mBp
        # O1-A naive masked
        mAtA = momentum * mAtA + gAtA; mBtA = momentum * mBtA + gBtA
        AtA = AtA - lr * mAtA; BtA = BtA - lr * mBtA
        # O1-B/C corrected: A grad right-multiplied by (T_in^T T_in)^{-1}; B exact
        cgAtB = gAtB @ Tin_gram_inv
        mAtB = momentum * mAtB + cgAtB; mBtB = momentum * mBtB + gBtB
        AtB = AtB - lr * mAtB; BtB = BtB - lr * mBtB

        # references: fold of plaintext-updated params
        foldAp = Ap @ Tin_iT; foldBp = T_out.T @ Bp
        dWp = scale * (Bp @ Ap)                        # plaintext effective dW (out,in)
        # fold plaintext dW into the masked domain (dW_t = T_out^T dW_p T_in^{-T});
        # compare against each profile's masked effective dW -- no inversions.
        dW_ref_masked = T_out.T @ dWp @ Tin_iT
        dW_A_masked = scale * (BtA @ AtA)
        dW_B_masked = scale * (BtB @ AtB)
        relerr = lambda a, b: float((a - b).norm() / (b.norm() + 1e-30))
        cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        gA_relation = relerr(gAp, gAtA @ Tin_inv)      # gAp should == gAt @ T_in^{-1}
        hist.append({
            "step": step, "proj": proj,
            "gradA_relation_err(gAp==gAt@Tin^-1)": gA_relation,
            "O1A_A_update_relerr_vs_plaintextfold": relerr(AtA, foldAp),
            "O1A_B_update_relerr_vs_plaintextfold": relerr(BtA, foldBp),
            "O1B_A_update_relerr_vs_plaintextfold": relerr(AtB, foldAp),
            "O1A_recoveredPlaintextA_err": relerr(AtA @ Tin_T, Ap),
            "O1A_dW_relerr_vs_plaintext": relerr(dW_A_masked, dW_ref_masked),
            "O1A_dW_cosine_vs_plaintext": cos(dW_A_masked, dW_ref_masked),
            "O1B_dW_relerr_vs_plaintext": relerr(dW_B_masked, dW_ref_masked),
            "O1B_dW_cosine_vs_plaintext": cos(dW_B_masked, dW_ref_masked),
            "loss_plaintext": Lp, "loss_O1A_masked": LtA,
            "loss_match": abs(Lp - LtA)})
    return hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file
    sd = load_file(str(CKPT / "model.safetensors"))
    cfg = json.loads((CKPT / "config.json").read_text())
    H = cfg["hidden_size"]; nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = H // nh
    Nr = orthogonal_signed_perm(H, 9000, DT)

    # ---------- 1. orthogonality audit + classification (all 24 layers) ----------
    registry = []; audit = {}
    for proj in TARGETS:
        rows = []
        for l in range(cfg["num_hidden_layers"]):
            T_in, T_out = transforms(sd, l, proj, Nr, nh, nkv, hd)
            rows.append((orthog_dev(T_in), cond(T_in), orthog_dev(T_out), cond(T_out)))
        tin_dev = max(r[0] for r in rows); tin_cond = max(r[1] for r in rows)
        tout_dev = max(r[2] for r in rows); tout_cond = max(r[3] for r in rows)
        gamma = proj in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
        tin_orth = tin_dev < 1e-9; tout_orth = tout_dev < 1e-9
        cls = "O" if (tin_orth and tout_orth) else ("NO" if (not tin_orth and not tout_orth) else "MIXED")
        registry.append({"target": proj, "gamma_folded_input": gamma,
                         "T_in_orthogonal": tin_orth, "T_in_orthogonality_dev": tin_dev,
                         "T_in_condition_number": tin_cond,
                         "T_out_orthogonal": tout_orth, "T_out_orthogonality_dev": tout_dev,
                         "T_out_condition_number": tout_cond,
                         "A_update_exact": tin_orth, "B_update_exact": tout_orth,
                         "class": cls,
                         "correction_needed": "none" if tin_orth else "A_grad @ (Nr^T diag(gamma^2) Nr)"})
        audit[proj] = {"class": cls, "A_update_exact": tin_orth, "B_update_exact": tout_orth,
                       "T_in_cond_max": tin_cond, "T_out_cond_max": tout_cond}
    with open(OUT / "target_transform_registry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(registry[0].keys())); w.writeheader(); w.writerows(registry)
    (OUT / "orthogonality_audit.json").write_text(json.dumps(audit, indent=2))

    # ---------- 5+6. one-step & ten-step numerical diagnostics (fp64) ----------
    one_rows, ten_rows = [], []
    for proj in TARGETS:
        h1 = one_step_target(sd, 0, proj, Nr, nh, nkv, hd, steps=1)
        one_rows += h1
        h10 = one_step_target(sd, 0, proj, Nr, nh, nkv, hd, steps=10)
        ten_rows += h10
    for name, rows in [("one_step_results.csv", one_rows), ("ten_step_trajectory.csv", ten_rows)]:
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(one_rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- 7. momentum diagnostic ----------
    mom_rows = []
    for proj in TARGETS:
        mom_rows += one_step_target(sd, 0, proj, Nr, nh, nkv, hd, steps=5, momentum=0.9)
    with open(OUT / "momentum_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mom_rows[0].keys())); w.writeheader(); w.writerows(mom_rows)

    # ---------- per-target summary ----------
    per = []
    for proj in TARGETS:
        r1 = [r for r in one_rows if r["proj"] == proj][0]
        r10 = [r for r in ten_rows if r["proj"] == proj][-1]
        rm = [r for r in mom_rows if r["proj"] == proj][-1]
        per.append({"target": proj, "class": audit[proj]["class"],
                    "A_update_exact": audit[proj]["A_update_exact"],
                    "B_update_exact": audit[proj]["B_update_exact"],
                    "T_in_cond_max": audit[proj]["T_in_cond_max"],
                    "O1A_A_update_err_step1": r1["O1A_A_update_relerr_vs_plaintextfold"],
                    "O1A_A_update_err_step10": r10["O1A_A_update_relerr_vs_plaintextfold"],
                    "O1B_A_update_err_step1": r1["O1B_A_update_relerr_vs_plaintextfold"],
                    "O1A_dW_cos_step10": r10["O1A_dW_cosine_vs_plaintext"],
                    "O1B_dW_cos_step10": r10["O1B_dW_cosine_vs_plaintext"],
                    "O1A_momentum_A_err_step5": rm["O1A_A_update_relerr_vs_plaintextfold"],
                    "O1B_momentum_A_err_step5": rm["O1B_A_update_relerr_vs_plaintextfold"],
                    "loss_match_O1A": r1["loss_match"]})
    with open(OUT / "per_target_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)

    print(json.dumps({"classification": {p["target"]: p["class"] for p in registry},
                      "O1A_A_err_step10": {p["target"]: round(p["O1A_A_update_err_step10"], 4) for p in per},
                      "O1B_A_err_step10": {p["target"]: f"{[r for r in ten_rows if r['proj']==p['target']][-1]['O1B_A_update_relerr_vs_plaintextfold']:.1e}" for p in per},
                      "O1A_dW_cos_step10": {p["target"]: round(p["O1A_dW_cos_step10"], 4) for p in per}}, indent=2))
    return per


if __name__ == "__main__":
    main()
