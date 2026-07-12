"""L5-plaintext-AdamW reference oracle vs L12 trusted-enclave AdamW — runs on the control plane (Mac).

Legitimately holds the gamma secret (validation harness, NOT the deployed GPU threat model). For each
trusted factor it reconstructs the PLAINTEXT-domain AdamW trajectory from the REAL gradients the enclave
consumed (masked grad SENT, logged by the runner, mapped to plaintext via the public metric:
g_plain = g_tilde @ M^T for A; g_plain = Tout @ g_tilde for B), running an INDEPENDENT FP32 AdamW
identical to an L5 plaintext run. It compares, in the masked domain, the reference re-fold (plaintext
master @ M) against the enclave's REAL returned re-folded factor.

Verification oracle only: the L12 numbers come from real A10 GPU fwd/bwd + real TDX enclave AdamW; this
script checks them. It is NOT an offline substitution for the L12 arm.

Reports per trusted target: master-param rel err (A and B). Reports per q/k layer: effective DeltaW
(scale * B @ A, both trusted) rel err + cosine. m/v are the reference's (enclave m/v equal these by
construction — identical plaintext grads fed to identical AdamW); master + DeltaW agreement validates
the enclave's internal m/v trajectory transitively.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from tdx_adamw_protocol import (orthogonal_signed_perm, rope_rot, adamw_step,
                                TRUSTED_A, TRUSTED_B, ATTN)
from h800_d4_worker import rank_masked_init

CFG = {"num_attention_heads": 14, "num_key_value_heads": 2}
BUNDLE = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
DT = torch.float32   # L5 reference dtype == L12 enclave master dtype (fp32)


def build_transforms(gb):
    H = gb["hidden"]; hd = H // CFG["num_attention_heads"]
    Nr = orthogonal_signed_perm(H, gb["nr_seed"]).to(DT); NrT = Nr.T
    foldA, gradMTA, invA = {}, {}, {}
    for l in range(gb["num_layers"]):
        ga = gb["ga"][l].to(DT); gm = gb["gm"][l].to(DT)
        for p in TRUSTED_A:
            g = ga if p in ATTN else gm
            M = torch.diag(g) @ Nr
            foldA[(l, p)] = M; gradMTA[(l, p)] = NrT @ torch.diag(g); invA[(l, p)] = torch.linalg.inv(M)
    toutB = {}
    for l in range(gb["num_layers"]):
        Bl = rope_rot(hd, 1000 + l).to(DT)
        toutB[(l, "q_proj")] = torch.block_diag(*([Bl] * CFG["num_attention_heads"]))
        toutB[(l, "k_proj")] = torch.block_diag(*([Bl] * CFG["num_key_value_heads"]))
    return foldA, gradMTA, invA, toutB


def rel(a, b):
    n = b.norm().item()
    return (a - b).norm().item() / n if n > 1e-30 else (a - b).norm().item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tlog", required=True); ap.add_argument("--run-json", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--master-rel-max", type=float, default=5e-3)
    ap.add_argument("--deltaw-cos-min", type=float, default=0.999)
    a = ap.parse_args()
    b1, b2, eps, wd, scale = 0.9, 0.999, 1e-8, 0.01, 16 / 8
    gb = torch.load(BUNDLE, map_location="cpu")
    foldA, gradMTA, invA, toutB = build_transforms(gb)
    tlog = torch.load(a.tlog, map_location="cpu", weights_only=False)
    seed_base = 7000 + a.seed
    H = gb["hidden"]; hd = H // CFG["num_attention_heads"]

    # true projection output dims (rank_masked_init draws A0,B0(size=out_d),U in sequence from ONE
    # generator, so A_t=U@A0 DEPENDS on out_d — must pass the real out_d or U desyncs).
    inter = 4864
    OUTD = {"q_proj": CFG["num_attention_heads"] * hd, "k_proj": CFG["num_key_value_heads"] * hd,
            "v_proj": CFG["num_key_value_heads"] * hd, "gate_proj": inter, "up_proj": inter}
    refA, refB = {}, {}
    for l in range(gb["num_layers"]):
        for p in TRUSTED_A:
            A_t, _ = rank_masked_init(l, p, H, OUTD[p], seed_base=seed_base)
            Ap = A_t.to(DT) @ invA[(l, p)]
            refA[(l, p)] = [Ap, torch.zeros_like(Ap), torch.zeros_like(Ap)]
        for p in TRUSTED_B:
            _, B_t = rank_masked_init(l, p, H, OUTD[p], seed_base=seed_base)
            Bp = toutB[(l, p)] @ B_t.to(DT)
            refB[(l, p)] = [Bp, torch.zeros_like(Bp), torch.zeros_like(Bp)]

    per_step = []
    W = {"masterA": 0.0, "masterB": 0.0, "dw_rel": 0.0, "dw_cos": 1.0, "tgt": None, "step": None}
    for slog in tlog:
        step = slog["step"]; t = step + 1
        mA, mB = [], []
        for k, g_tilde in slog.get("gA_tilde", {}).items():
            l, p = int(k.split(".")[0]), k.split(".", 1)[1]
            g_plain = g_tilde.to(DT) @ gradMTA[(l, p)]
            Ap, m, v = refA[(l, p)]
            Ap, m, v = adamw_step(Ap, g_plain, m, v, t, a.lr, b1, b2, eps, wd)
            refA[(l, p)] = [Ap, m, v]
            enc = slog.get("refoldA", {}).get(k)
            if enc is not None:
                r = rel(Ap @ foldA[(l, p)], enc.to(DT)); mA.append(r)
                if r > W["masterA"]: W.update(masterA=r, tgt=k, step=step)
        for k, g_tilde in slog.get("gB_tilde", {}).items():
            l, p = int(k.split(".")[0]), k.split(".", 1)[1]
            g_plain = toutB[(l, p)] @ g_tilde.to(DT)
            Bp, m, v = refB[(l, p)]
            Bp, m, v = adamw_step(Bp, g_plain, m, v, t, a.lr, b1, b2, eps, wd)
            refB[(l, p)] = [Bp, m, v]
            enc = slog.get("refoldB", {}).get(k)
            if enc is not None:
                r = rel(toutB[(l, p)].T @ Bp, enc.to(DT)); mB.append(r)
                if r > W["masterB"]: W.update(masterB=r)
        # effective DeltaW for q/k (both A and B trusted -> fully reconstructable in plaintext domain)
        dwr, dwc = [], []
        for p in ("q_proj", "k_proj"):
            for l in range(gb["num_layers"]):
                if (l, p) in refA and (l, p) in refB:
                    dw_ref = scale * (refB[(l, p)][0] @ refA[(l, p)][0])
                    encA = slog.get("refoldA", {}).get(f"{l}.{p}"); encB = slog.get("refoldB", {}).get(f"{l}.{p}")
                    if encA is not None and encB is not None:
                        A_pl = encA.to(DT) @ invA[(l, p)]; B_pl = toutB[(l, p)] @ encB.to(DT)
                        dw_enc = scale * (B_pl @ A_pl)
                        rr = rel(dw_enc, dw_ref); cc = cos(dw_enc, dw_ref)
                        dwr.append(rr); dwc.append(cc)
                        if rr > W["dw_rel"]: W["dw_rel"] = rr
                        if cc < W["dw_cos"]: W["dw_cos"] = cc
        per_step.append({"step": step, "adam_t": t,
                         "max_master_rel_A": max(mA, default=0.0), "max_master_rel_B": max(mB, default=0.0),
                         "median_master_rel_A": float(torch.tensor(mA).median()) if mA else None,
                         "max_deltaw_rel_qk": max(dwr, default=0.0), "min_deltaw_cos_qk": min(dwc, default=1.0)})

    run = json.loads(Path(a.run_json).read_text()); traj = run.get("trajectory", [])
    passed = (W["masterA"] <= a.master_rel_max and W["masterB"] <= a.master_rel_max
              and W["dw_cos"] >= a.deltaw_cos_min)
    out = {
        "profile_compared": "L12_mixed_bf16compute_fp32master  vs  L5_plaintext_adamw_reference (fp32)",
        "seed": a.seed, "steps": len(tlog),
        "reference_construction": "INDEPENDENT plaintext-domain FP32 AdamW fed the REAL enclave gradients (A: g_tilde@M^T, B: Tout@g_tilde); verification oracle, not a substitute for the hardware L12 arm",
        "trusted_A_per_step": len(tlog[0].get("gA_tilde", {})) if tlog else 0,
        "trusted_B_per_step": len(tlog[0].get("gB_tilde", {})) if tlog else 0,
        "worst_master_rel_err_A": W["masterA"], "worst_master_rel_err_B": W["masterB"],
        "worst_target": W["tgt"], "worst_step": W["step"],
        "worst_deltaw_rel_qk": W["dw_rel"], "worst_deltaw_cos_qk": W["dw_cos"],
        "per_step": per_step,
        "hardware_state_version_final": traj[-1].get("state_version") if traj else None,
        "runtime_copy_matches_master_transform_all": all(bool(s.get("runtime_copy_matches_master_transform")) for s in traj) if traj else None,
        "finite_all": all(s.get("finite") for s in traj) if traj else None,
        "PREREGISTERED": {"master_rel_max": a.master_rel_max, "deltaw_cos_min": a.deltaw_cos_min},
        "PASS": bool(passed),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"seed": a.seed, "worst_master_rel_A": W["masterA"], "worst_master_rel_B": W["masterB"],
                      "worst_dw_cos_qk": W["dw_cos"], "PASS": bool(passed)}, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
