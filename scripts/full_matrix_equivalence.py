"""Full LoRA validation matrix -- exact per-target optimizer-equivalence engine (fp64).

Parameter-trajectory equivalence is an EXACT-ARITHMETIC property; it is decided here at
fp64 on the REAL Qwen2.5-0.5B checkpoint, per the plan ("decided at fp64"). The real
BF16 + real TDX run (scripts/l10_real_bf16_tdx_gate via the H800/TDX services) validates
numerics + in-enclave correction; it does not decide exactness.

Profiles compared, per target x layer x seed x optimizer family x step-count:
  L1  plaintext_lora_<opt>            (reference; plaintext basis)
  L2  O1-A uniform GPU masked <opt>   (naive masked update; efficiency approximation)
  L10/L11/L12  O1-C hybrid            (o/down GPU-exact; gamma-fed corrected/trusted)

Row-vector convention (y = x @ W.T). Fold:  A_tilde = A_plain @ T_in^{-T},
B_tilde = T_out^T @ B_plain,  W_tilde = T_out^T W T_in^{-T},  x_tilde = x_plain @ T_in,
y_tilde = y_plain @ T_out.  Established relation (Gate 0.5): gA_plain = gA_tilde @ T_in^{-1}.

Equivalence conditions:
  SGD/momentum  A-update exact  <=>  T_in orthogonal.  Correction (linear, O1-C):
      corrected gA = gA_tilde @ (T_in^T T_in)^{-1}  ==  gA_plain @ T_in^{-T}  (== fold of gA_plain)
      -> applied INSIDE TDX for q/k/v/gate/up; identity for o/down.
      momentum buffer accumulates the corrected grad -> buffer lives in the folded metric,
      so it equals the fold of the plaintext buffer exactly (same linear recursion).
  AdamW  needs the STRONGER condition: the factor transform must be an ORTHOGONAL MONOMIAL
      with UNIT magnitudes (signed permutation / permutation), because the 2nd moment is an
      ELEMENTWISE square and only monomial maps commute with it. Where it fails (all gamma-fed
      A-factors -- scaled monomial; q/k B-factor -- 2D rotation, non-monomial), no linear
      correction suffices; AdamW MUST run in the plaintext basis inside TDX, then re-fold.
      GPU-exact-AdamW-safe factors: A of {o,down}; B of {v,o,gate,up,down}.
      TDX-AdamW factors:           A of {q,k,v,gate,up}; B of {q,k}.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from gate05_o1_optimizer_audit import transforms, DT, CKPT, TARGETS  # reuse verified transforms
from pllo.ops.masked_training_kernels import orthogonal_signed_perm

OUT = REPO / "results/aaai_private_base/full_lora_matrix"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def wkey(proj, l):
    grp = "self_attn" if proj in ATTN else "mlp"
    return f"model.layers.{l}.{grp}.{proj}.weight"


def is_orthogonal(M, tol=1e-9):
    I = torch.eye(M.shape[0], dtype=M.dtype)
    return float((M.T @ M - I).abs().max()) < tol


def is_orthogonal_monomial_unit(M, tol=1e-9):
    """Exactly one nonzero per row & col, with |entry| == 1 (signed perm / perm)."""
    A = M.abs()
    row_nz = (A > tol).sum(dim=1)
    col_nz = (A > tol).sum(dim=0)
    if not (torch.all(row_nz == 1) and torch.all(col_nz == 1)):
        return False
    vals = A[A > tol]
    return bool(torch.all((vals - 1.0).abs() < tol))


def relerr(a, b):
    return float((a - b).norm() / (b.norm() + 1e-300))


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


# ---------------- optimizer step primitives (all fp64) ----------------
def sgd_step(theta, g, buf, lr, mom):
    buf = mom * buf + g
    return theta - lr * buf, buf


def adamw_step(theta, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    theta = theta - lr * (mhat / (vhat.sqrt() + eps) + wd * theta)
    return theta, m, v


def build_bundle(sd, l, proj, Nr, nh, nkv, hd):
    """Precompute per-(proj,layer) transforms + inversions (cached across seeds)."""
    W = sd[wkey(proj, l)].to(DT)
    T_in, T_out = transforms(sd, l, proj, Nr, nh, nkv, hd)
    Tin_inv = torch.linalg.inv(T_in); Tin_iT = Tin_inv.T
    Tin_gram_inv = Tin_inv @ Tin_iT
    Tout_inv = torch.linalg.inv(T_out); Tout_iT = Tout_inv.T
    return {"W": W, "T_in": T_in, "T_out": T_out, "Tin_inv": Tin_inv, "Tin_iT": Tin_iT,
            "Tin_gram_inv": Tin_gram_inv, "Tout_iT": Tout_iT, "W_tilde": T_out.T @ W @ Tin_iT,
            "tin_orth": is_orthogonal(T_in), "tout_orth": is_orthogonal(T_out),
            "A_gpu_adamw_safe": is_orthogonal_monomial_unit(T_in),
            "B_gpu_adamw_safe": is_orthogonal_monomial_unit(T_out)}


def run_target(bundle, l, proj, *, optimizer, steps, seed, record_steps=None,
               r=8, scale=2.0, lr=1e-3, mom=0.9, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """Run plaintext / O1-A / O1-C trajectories for one target and return per-step metrics."""
    record_steps = record_steps or set(range(1, steps + 1))
    W = bundle["W"]; out_d, in_d = W.shape
    T_in, T_out = bundle["T_in"], bundle["T_out"]
    Tin_inv, Tin_iT, Tin_gram_inv = bundle["Tin_inv"], bundle["Tin_iT"], bundle["Tin_gram_inv"]
    Tout_iT, W_tilde = bundle["Tout_iT"], bundle["W_tilde"]
    tin_orth, tout_orth = bundle["tin_orth"], bundle["tout_orth"]
    A_gpu_adamw_safe, B_gpu_adamw_safe = bundle["A_gpu_adamw_safe"], bundle["B_gpu_adamw_safe"]

    g = torch.Generator().manual_seed(1000 + seed)
    A0 = torch.randn(r, in_d, generator=g, dtype=DT) * 0.02
    B0 = torch.randn(out_d, r, generator=g, dtype=DT) * 0.02 + 0.01
    Tn = 6
    x_plain = torch.randn(Tn, in_d, generator=g, dtype=DT)
    tgt_plain = torch.randn(Tn, out_d, generator=g, dtype=DT)
    x_tilde = x_plain @ T_in; tgt_tilde = tgt_plain @ T_out

    def fold_A(Ap): return Ap @ Tin_iT
    def fold_B(Bp): return T_out.T @ Bp

    # analytic grads of  loss = mean((x @ (W_ + s B A).T - tgt)^2), factored to O(Tn r dim)
    # (no autograd graph, no out x in materialization):
    #   y = xW + s (x A^T) B^T ;  G = 2 (y - tgt)/(Tn*out) ;
    #   gA = s (G B)^T x ;  gB = s G^T (x A^T)
    def grads_an(A, B, xW, x, tgt, s, Tn_out):
        y = xW + s * ((x @ A.T) @ B.T)
        R = y - tgt
        loss = float((R * R).mean())
        G = (2.0 / Tn_out) * R
        gA = s * (G @ B).T @ x
        gB = s * G.T @ (x @ A.T)
        return loss, gA, gB

    # ---- state ----
    Ap, Bp = A0.clone(), B0.clone()                        # plaintext (L1/L3/L5 ref)
    Aa, Ba = fold_A(A0).clone(), fold_B(B0).clone()        # O1-A uniform GPU masked
    Ac, Bc = fold_A(A0).clone(), fold_B(B0).clone()        # O1-C hybrid
    # optimizer buffers
    zp = lambda t: torch.zeros_like(t)
    bufAp, bufBp = zp(Ap), zp(Bp)
    bufAa, bufBa = zp(Aa), zp(Ba)
    bufAc, bufBc = zp(Ac), zp(Bc)
    mAp = vAp = zp(Ap); mBp = vBp = zp(Bp)
    mAa = vAa = zp(Aa); mBa = vBa = zp(Ba)
    mAc = vAc = zp(Ac); mBc = vBc = zp(Bc)
    # O1-C AdamW keeps plaintext-basis state for TDX-side factors:
    tdxA_m = zp(Ap); tdxA_v = zp(Ap); tdxA_p = A0.clone()   # plaintext A state in TDX
    tdxB_m = zp(Bp); tdxB_v = zp(Bp); tdxB_p = B0.clone()   # plaintext B state in TDX

    xWp = x_plain @ W.T                       # precompute once (W fixed)
    xWt = x_tilde @ W_tilde.T
    To_p = Tn * out_d                          # mean normalizer

    hist = []
    for step in range(1, steps + 1):
        Lp, gAp, gBp = grads_an(Ap, Bp, xWp, x_plain, tgt_plain, scale, To_p)
        La, gAa, gBa = grads_an(Aa, Ba, xWt, x_tilde, tgt_tilde, scale, To_p)
        Lc, gAc, gBc = grads_an(Ac, Bc, xWt, x_tilde, tgt_tilde, scale, To_p)

        if optimizer in ("sgd", "momentum"):
            m = mom if optimizer == "momentum" else 0.0
            Ap, bufAp = sgd_step(Ap, gAp, bufAp, lr, m)
            Bp, bufBp = sgd_step(Bp, gBp, bufBp, lr, m)
            Aa, bufAa = sgd_step(Aa, gAa, bufAa, lr, m)
            Ba, bufBa = sgd_step(Ba, gBa, bufBa, lr, m)
            # O1-C: correct A grad in TDX metric (identity if T_in orthogonal); B always exact
            cgAc = gAc @ Tin_gram_inv if not tin_orth else gAc
            Ac, bufAc = sgd_step(Ac, cgAc, bufAc, lr, m)
            Bc, bufBc = sgd_step(Bc, gBc, bufBc, lr, m)
        else:  # adamw
            Ap, mAp, vAp = adamw_step(Ap, gAp, mAp, vAp, step, lr, b1, b2, eps, wd)
            Bp, mBp, vBp = adamw_step(Bp, gBp, mBp, vBp, step, lr, b1, b2, eps, wd)
            Aa, mAa, vAa = adamw_step(Aa, gAa, mAa, vAa, step, lr, b1, b2, eps, wd)
            Ba, mBa, vBa = adamw_step(Ba, gBa, mBa, vBa, step, lr, b1, b2, eps, wd)
            # O1-C AdamW: GPU-exact for monomial factors; plaintext-basis (TDX) otherwise
            if A_gpu_adamw_safe:
                Ac, mAc, vAc = adamw_step(Ac, gAc, mAc, vAc, step, lr, b1, b2, eps, wd)
            else:
                gA_plain = gAc @ Tin_inv                         # recover plaintext grad in TDX
                tdxA_p, tdxA_m, tdxA_v = adamw_step(tdxA_p, gA_plain, tdxA_m, tdxA_v,
                                                    step, lr, b1, b2, eps, wd)
                Ac = fold_A(tdxA_p)                              # re-fold to masked domain
            if B_gpu_adamw_safe:
                Bc, mBc, vBc = adamw_step(Bc, gBc, mBc, vBc, step, lr, b1, b2, eps, wd)
            else:
                gB_plain = Tout_iT @ gBc
                tdxB_p, tdxB_m, tdxB_v = adamw_step(tdxB_p, gB_plain, tdxB_m, tdxB_v,
                                                    step, lr, b1, b2, eps, wd)
                Bc = fold_B(tdxB_p)

        # ---- metrics only at checkpoint steps (dense transform multiplies are costly) ----
        if step not in record_steps:
            continue
        foldAp, foldBp = fold_A(Ap), fold_B(Bp)
        dW_ref = scale * (foldBp @ foldAp)     # = T_out^T (s Bp Ap) Tin_iT, factored
        dW_a = scale * (Ba @ Aa); dW_c = scale * (Bc @ Ac)
        o1a_dw_relerr, o1a_dw_cos = relerr(dW_a, dW_ref), cos(dW_a, dW_ref)
        o1c_dw_relerr, o1c_dw_cos = relerr(dW_c, dW_ref), cos(dW_c, dW_ref)
        finite = lambda *ts: all(bool(torch.isfinite(t).all()) for t in ts)
        hist.append({
            "proj": proj, "layer": l, "seed": seed, "optimizer": optimizer, "step": step,
            "loss_plaintext": Lp, "loss_O1A": La, "loss_O1C": Lc,
            "loss_absdiff_O1A": abs(Lp - La), "loss_absdiff_O1C": abs(Lp - Lc),
            "gradA_relation_err": relerr(gAp, gAa @ Tin_inv),   # gAp == gAt @ Tin^-1
            "O1A_A_update_relerr": relerr(Aa, foldAp), "O1A_B_update_relerr": relerr(Ba, foldBp),
            "O1C_A_update_relerr": relerr(Ac, foldAp), "O1C_B_update_relerr": relerr(Bc, foldBp),
            "O1A_dW_relerr": o1a_dw_relerr, "O1A_dW_cos": o1a_dw_cos,
            "O1C_dW_relerr": o1c_dw_relerr, "O1C_dW_cos": o1c_dw_cos,
            "O1A_optstate_relerr": (relerr(bufAa, fold_A(bufAp)) if optimizer != "adamw"
                                    else relerr(mAa, fold_A(mAp))),
            "O1C_optstate_relerr": (relerr(bufAc, fold_A(bufAp)) if optimizer != "adamw"
                                    else (relerr(mAc, fold_A(mAp)) if A_gpu_adamw_safe
                                          else relerr(tdxA_p, Ap))),  # TDX A-state == plaintext exactly
            "T_in_orthogonal": tin_orth, "T_out_orthogonal": tout_orth,
            "A_gpu_adamw_safe": A_gpu_adamw_safe, "B_gpu_adamw_safe": B_gpu_adamw_safe,
            "O1C_A_in_tdx": (optimizer == "adamw" and not A_gpu_adamw_safe) or
                            (optimizer != "adamw" and not tin_orth),
            "O1C_B_in_tdx": optimizer == "adamw" and not B_gpu_adamw_safe,
            "nan_inf": not finite(Ac, Bc, Aa, Ba),
        })
    return hist


def aggregate(rows, key_err):
    """Group-wise + worst-layer/median/p95 over the final-step rows for one metric."""
    import statistics
    groups = {"o_down": ("o_proj", "down_proj"), "qkv": ("q_proj", "k_proj", "v_proj"),
              "gate_up": ("gate_proj", "up_proj")}
    out = {}
    for gname, projs in groups.items():
        vals = [r[key_err] for r in rows if r["proj"] in projs]
        if not vals:
            continue
        svals = sorted(vals)
        out[gname] = {
            "max": max(vals), "median": statistics.median(vals),
            "p95": svals[min(len(svals) - 1, int(0.95 * len(svals)))],
            "worst_layer": max((r for r in rows if r["proj"] in projs),
                               key=lambda r: r[key_err])["layer"],
            "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 2025, 7])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 10, 50])
    ap.add_argument("--optimizers", nargs="+", default=["sgd", "momentum", "adamw"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file
    sd = load_file(str(CKPT / "model.safetensors"))
    cfg = json.loads((CKPT / "config.json").read_text())
    H = cfg["hidden_size"]; nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = H // nh
    Nr = orthogonal_signed_perm(H, 9000, DT)

    # cache transform bundles once per (proj,layer) -- reused across seeds & optimizers
    print("building transform bundles ...", flush=True)
    bundles = {}
    for proj in TARGETS:
        for l in range(args.layers):
            bundles[(proj, l)] = build_bundle(sd, l, proj, Nr, nh, nkv, hd)
    print(f"  {len(bundles)} bundles ready", flush=True)

    subdir = {"sgd": "equivalence", "momentum": "momentum", "adamw": "adamw"}
    all_final = {}
    for optimizer in args.optimizers:
        max_steps = max(args.steps)
        allrows = []
        for seed in args.seeds:
            for proj in TARGETS:
                for l in range(args.layers):
                    h = run_target(bundles[(proj, l)], l, proj, optimizer=optimizer,
                                   steps=max_steps, seed=seed, record_steps=set(args.steps))
                    allrows.extend(h)
        d = OUT / subdir[optimizer]; d.mkdir(parents=True, exist_ok=True)
        # write full per-step trajectory (all targets/layers/seeds)
        with open(d / f"{optimizer}_full_trajectory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        # checkpoints at requested step counts
        summary = {}
        for S in args.steps:
            step_rows = [r for r in allrows if r["step"] == S]
            for prof, key in [("O1A_A_update", "O1A_A_update_relerr"),
                              ("O1C_A_update", "O1C_A_update_relerr"),
                              ("O1A_dW_cos", "O1A_dW_cos"), ("O1C_dW_cos", "O1C_dW_cos"),
                              ("O1C_optstate", "O1C_optstate_relerr"),
                              ("O1A_optstate", "O1A_optstate_relerr")]:
                summary[f"step{S}_{prof}"] = aggregate(step_rows, key)
        (d / f"{optimizer}_aggregate.json").write_text(json.dumps(summary, indent=2))
        all_final[optimizer] = summary
        # console: worst O1-C A-update error and O1-A A-update error at max steps
        fin = [r for r in allrows if r["step"] == max_steps]
        wc = max(r["O1C_A_update_relerr"] for r in fin)
        wa = max(r["O1A_A_update_relerr"] for r in fin)
        print(f"[{optimizer}] steps={max_steps} seeds={args.seeds}  "
              f"worst O1C A-update relerr={wc:.2e}  worst O1A A-update relerr={wa:.3f}")
    (OUT / "equivalence" / "matrix_index.json").write_text(json.dumps(
        {"optimizers": args.optimizers, "seeds": args.seeds, "steps": args.steps,
         "layers": args.layers, "targets": TARGETS,
         "head_hash": Path(REPO / "results/aaai_private_base/code_baseline/head_hash.txt").read_text().strip(),
         "precision": "fp64", "checkpoint_sha_ref": "see experiment_registry fixed_model"}, indent=2))
    print("done:", json.dumps({o: list(all_final[o].keys())[:2] for o in all_final}))


if __name__ == "__main__":
    main()
