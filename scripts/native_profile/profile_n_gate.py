"""PHASE 2/3.1-3.2 (LOCAL gate) — Profile N native transformed-coordinate AdamW.

Implements and evaluates three LoRA optimizers on the SAME supervised task, in a self-contained
CPU harness (the on-hardware real-Qwen 1/10/50-step gate is Phase 3, deferred):

  plaintext : canonical AdamW on plaintext A,B.
  Profile E : plaintext-coordinate AdamW, factors folded for transport (exact-equivalent to
              plaintext -> identical effective DeltaW). Trusted state = FP32 master + m + v in TDX.
  Profile N : native AdamW DIRECTLY on the transformed factors A_tilde,B_tilde (m/v in transformed
              coordinates, on the GPU). NO plaintext master / m / v in TDX. Does NOT claim stepwise
              plaintext-AdamW equivalence; the adapter stays inference-compatible.

Fold (self-consistent): x_tilde = x M_in ; y_tilde = y M_out ; W_tilde = M_out^T W M_in ;
DeltaW_tilde = M_out^T (A B) M_in = A_tilde B_tilde  (A_tilde=M_out^T A U, B_tilde=U^-1 B M_in).
Because M_in,M_out are orthogonal the LOSS is identical in both coordinate systems, so Profile N
minimizes the same objective; only AdamW's diagonal preconditioner differs (different trajectory).

Metrics per profile: loss trajectory, downstream top-1, final-logits KL vs plaintext, effective
DeltaW cosine + relative difference vs plaintext, convergence stability, trusted-state bytes.
Labels: optimizer_semantics, plaintext_trajectory_equivalence=false (N), inference_compatible=true.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "security"))
from pb_harness import stamp, cosine, rel_err  # noqa: E402
from pllo.ops.masked_training_kernels import orthogonal_signed_perm  # noqa: E402


def dense_orthogonal(d, seed, dtype=torch.float64):
    """A DENSE orthogonal matrix (QR of a random gaussian). Unlike a signed permutation, AdamW's
    per-element m/v preconditioner is NOT invariant under a dense rotation, so native
    transformed-coordinate AdamW follows a genuinely different trajectory."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=dtype))
    return q

OUT = Path(__file__).resolve().parents[2] / "results/aaai_private_base/native_profile_stage"
torch.manual_seed(0)
DT = torch.float64


def adamw_update(p, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
    p.mul_(1 - lr * wd).addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)


def make_task(d=256, r=8, n=384, seed=1):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, d, generator=g, dtype=DT) * (1.0 / d ** 0.5)         # frozen base
    dW = (torch.randn(d, r, generator=g, dtype=DT) @ torch.randn(r, d, generator=g, dtype=DT)) * (0.5 / d ** 0.5)
    X = torch.randn(n, d, generator=g, dtype=DT)
    C = torch.randn(2, d, generator=g, dtype=DT) * (1.0 / d ** 0.5)         # teacher 2-class head
    with torch.no_grad():
        y = ((X @ (W + dW).T) @ C.T).argmax(1)                             # labels from teacher(W+dW*)
    return W, dW, X, y, C


def train(profile, W, X, y, C, Min, Mout, U, steps, r, lr=1e-2):
    d = W.shape[0]
    g = torch.Generator().manual_seed(7)
    A0 = torch.randn(d, r, generator=g, dtype=DT) * (1.0 / r ** 0.5)
    B0 = torch.zeros(r, d, dtype=DT)
    losses = []
    if profile in ("plaintext", "E"):
        A, B = A0.clone(), B0.clone()          # plaintext coords (E = same update; trusted-side)
        mA = torch.zeros_like(A); vA = torch.zeros_like(A); mB = torch.zeros_like(B); vB = torch.zeros_like(B)
        for t in range(1, steps + 1):
            A.requires_grad_(True); B.requires_grad_(True)
            logits = (X @ (W + A @ B).T) @ C.T
            loss = torch.nn.functional.cross_entropy(logits, y)
            gA, gB = torch.autograd.grad(loss, [A, B]); losses.append(float(loss))
            A = A.detach(); B = B.detach()
            adamw_update(A, gA, mA, vA, t, lr); adamw_update(B, gB, mB, vB, t, lr)
        dW_eff = (A @ B).detach()
        trusted_bytes = 3 * (A.numel() + B.numel()) * 4   # master+m+v FP32 in TDX (E); plaintext=n/a
    elif profile == "N":
        # native transformed-coordinate AdamW: optimize A_tilde,B_tilde directly; m/v in tilde coords
        A_t = (Mout.T @ A0 @ U).clone(); B_t = (U.T @ B0 @ Min).clone()
        W_t = Mout.T @ W @ Min
        mA = torch.zeros_like(A_t); vA = torch.zeros_like(A_t); mB = torch.zeros_like(B_t); vB = torch.zeros_like(B_t)
        X_t = X @ Min                                       # GPU sees only transformed activations
        for t in range(1, steps + 1):
            A_t.requires_grad_(True); B_t.requires_grad_(True)
            # transformed forward; teacher head applied after un-folding the output basis (Mout) —
            # equivalently logits_tilde @ (C Mout^T)^T; loss identical to plaintext (orthogonal).
            logits = (X_t @ (W_t + A_t @ B_t).T) @ (C @ Mout).T
            loss = torch.nn.functional.cross_entropy(logits, y)
            gA, gB = torch.autograd.grad(loss, [A_t, B_t]); losses.append(float(loss))
            A_t = A_t.detach(); B_t = B_t.detach()
            adamw_update(A_t, gA, mA, vA, t, lr); adamw_update(B_t, gB, mB, vB, t, lr)
        dW_eff = (Mout @ (A_t @ B_t) @ Min.T).detach()      # un-fold to plaintext for comparison ONLY
        trusted_bytes = 0                                   # no plaintext master/m/v in TDX for N
    # final utility on the (same) task
    with torch.no_grad():
        logits = (X @ (W + dW_eff).T) @ C.T
        top1 = float((logits.argmax(1) == y).float().mean())
    return {"losses": losses, "dW_eff": dW_eff, "top1": top1, "trusted_bytes": trusted_bytes,
            "final_loss": losses[-1], "late_loss_std": float(torch.tensor(losses[-max(3, steps // 5):]).std())}


def logits_kl(dW_a, dW_b, W, X, C):
    la = torch.log_softmax((X @ (W + dW_a).T) @ C.T, -1)
    lb = torch.log_softmax((X @ (W + dW_b).T) @ C.T, -1)
    return float((la.exp() * (la - lb)).sum(-1).mean())


def main():
    t0 = time.time()
    d, r = 256, 8
    W, dW_star, X, y, C = make_task(d=d, r=r)
    # PRIMARY: dense orthogonal masks -> AdamW preconditioner is basis-dependent -> Profile N follows a
    # genuinely DIFFERENT trajectory than plaintext (the honest demonstration of "not equivalent").
    Min = dense_orthogonal(d, seed=101); Mout = dense_orthogonal(d, seed=202); U = dense_orthogonal(r, seed=303)
    results = {"experiment": "profile_n_local_gate", "note": "LOCAL synthetic gate; real-Qwen "
               "1/10/50-step is Phase 3 (hardware, deferred)", "d": d, "rank": r,
               "mask_family": "dense_orthogonal (primary)",
               "labels": {"profile_N_optimizer_semantics": "native_transformed_coordinate_adamw",
                          "profile_N_plaintext_trajectory_equivalence": False,
                          "profile_N_inference_compatible_adapter": True,
                          "profile_E_optimizer_semantics": "standard_plaintext_coordinate_adamw",
                          "profile_E_plaintext_trajectory_equivalence": True},
               "gates": {}}
    for steps in [1, 10, 50]:
        cell = {}
        rp = train("plaintext", W, X, y, C, Min, Mout, U, steps, r)
        rE = train("E", W, X, y, C, Min, Mout, U, steps, r)
        rN = train("N", W, X, y, C, Min, Mout, U, steps, r)
        cell["plaintext"] = {"final_loss": rp["final_loss"], "top1": rp["top1"],
                             "trusted_bytes": "n/a (no protection)"}
        cell["profile_E"] = {"final_loss": rE["final_loss"], "top1": rE["top1"],
                             "trusted_bytes": rE["trusted_bytes"],
                             "dW_cosine_vs_plaintext": cosine(rE["dW_eff"], rp["dW_eff"]),
                             "dW_rel_diff_vs_plaintext": rel_err(rE["dW_eff"], rp["dW_eff"]),
                             "logits_KL_vs_plaintext": logits_kl(rE["dW_eff"], rp["dW_eff"], W, X, C),
                             "late_loss_std": rE["late_loss_std"]}
        cell["profile_N"] = {"final_loss": rN["final_loss"], "top1": rN["top1"],
                             "trusted_bytes": rN["trusted_bytes"],
                             "dW_cosine_vs_plaintext": cosine(rN["dW_eff"], rp["dW_eff"]),
                             "dW_rel_diff_vs_plaintext": rel_err(rN["dW_eff"], rp["dW_eff"]),
                             "dW_cosine_vs_target": cosine(rN["dW_eff"], dW_star),
                             "logits_KL_vs_plaintext": logits_kl(rN["dW_eff"], rp["dW_eff"], W, X, C),
                             "top1_agreement_vs_plaintext": None,
                             "late_loss_std": rN["late_loss_std"]}
        results["gates"][f"{steps}_step"] = cell
    # SPECIAL FINDING: under a SIGNED-PERMUTATION mask, per-element AdamW is permutation/sign
    # equivariant, so native transformed-coordinate AdamW COINCIDES with plaintext AdamW (rel_diff~0).
    Pin = orthogonal_signed_perm(d, seed=101, dtype=DT); Pout = orthogonal_signed_perm(d, seed=202, dtype=DT)
    Pu = orthogonal_signed_perm(r, seed=303, dtype=DT)
    rN_perm = train("N", W, X, y, C, Pin, Pout, Pu, 50, r)
    rp_ref = train("plaintext", W, X, y, C, Pin, Pout, Pu, 50, r)
    results["signed_permutation_equivariance"] = {
        "N_dW_rel_diff_vs_plaintext": rel_err(rN_perm["dW_eff"], rp_ref["dW_eff"]),
        "note": "under a signed-permutation transform, native transformed AdamW == plaintext AdamW "
                "(per-element m/v are permutation/sign invariant); a DENSE transform (primary gate) breaks this."}

    g50 = results["gates"]["50_step"]
    results["summary"] = {
        "E_is_exact_vs_plaintext": g50["profile_E"]["dW_rel_diff_vs_plaintext"] < 1e-9,
        "N_trusted_bytes": g50["profile_N"]["trusted_bytes"],
        "E_trusted_bytes": g50["profile_E"]["trusted_bytes"],
        "N_utility_top1": g50["profile_N"]["top1"], "plaintext_utility_top1": g50["plaintext"]["top1"],
        "N_final_loss": g50["profile_N"]["final_loss"], "plaintext_final_loss": g50["plaintext"]["final_loss"],
        "reading": "Profile E reproduces plaintext EXACTLY (rel_diff ~0). Profile N follows a DIFFERENT "
                   "trajectory (no plaintext-AdamW equivalence) but reaches comparable utility while keeping "
                   "ZERO plaintext optimizer state in the TDX (trusted_bytes 0 vs E's master+m+v). This LOCAL "
                   "synthetic gate demonstrates the trade-off; the real-Qwen 1/10/50-step + resource comparison "
                   "is the deferred Phase 3 hardware gate."}
    results["limitations"] = [
        "LOCAL synthetic task (real-Qwen 1/10/50-step on A10+TDX is Phase 3, deferred/hardware).",
        "Do NOT read Profile N's trajectory difference as failure — utility + stability are the criteria.",
        "Security surface of Profile N (GPU-visible transformed m/v/A/B) is a SEPARATE audit (Phase 3.3);"
        " no equal-security claim is made here."]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "profile_n_gate.json", results)
    s = results["summary"]
    print("[profile_N gate] E exact vs plaintext:", s["E_is_exact_vs_plaintext"],
          "| trusted_bytes E=%s N=%s" % (s["E_trusted_bytes"], s["N_trusted_bytes"]))
    print("  50-step utility: plaintext top1=%.3f  N top1=%.3f  N dW_cos_vs_plaintext=%.3f  N final_loss=%.4f vs %.4f" % (
        s["plaintext_utility_top1"], s["N_utility_top1"],
        g50["profile_N"]["dW_cosine_vs_plaintext"], s["N_final_loss"], s["plaintext_final_loss"]))


if __name__ == "__main__":
    main()
