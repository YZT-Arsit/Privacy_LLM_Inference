"""Build the TRUSTED O1-C gradient-correction bundle (runs on the trusted packager / Mac).

The correction that makes naive masked A-updates exact for the gamma-fed targets is a
right-multiplication of the masked A-gradient by
    gram_inv = (T_in^T T_in)^{-1} = Nr^T diag(gamma^2) Nr
where Nr is the global orthogonal signed-perm residual mask (seed 9000) and gamma is a
PRIVATE RMSNorm gain:
    q/k/v  use  ga = input_layernorm.weight           -> gram_inv_attn[l]
    gate/up use gm = post_attention_layernorm.weight   -> gram_inv_mlp[l]
o_proj/down_proj need NO correction (orthogonal T_in).

This bundle encodes the private gamma spectrum, so it is a TRUSTED secret: it is
provisioned ONLY into the TDX enclave and MUST NEVER be materialized on the untrusted GPU
worker (that is exactly why O1-B -- applying it GPU-side -- is rejected as paper_unsafe).

Emits a single .pt (per-layer attn/mlp gram_inv, fp64) + a manifest with a self-check that
gram_inv @ (Nr^T diag(1/gamma^2) Nr) == I and a cross-check vs the fp64 audit engine.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from pllo.ops.masked_training_kernels import orthogonal_signed_perm
from gate05_o1_optimizer_audit import transforms, CKPT

DT = torch.float64
OUT = REPO / "results/aaai_private_base/full_lora_matrix/profile_validation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"))
    args = ap.parse_args()
    from safetensors.torch import load_file
    sd = load_file(str(CKPT / "model.safetensors"))
    cfg = json.loads((CKPT / "config.json").read_text())
    H = cfg["hidden_size"]; L = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]; hd = H // nh
    Nr = orthogonal_signed_perm(H, 9000, DT)
    I = torch.eye(H, dtype=DT)

    # Ship only the tiny PRIVATE gamma vectors + Nr seed (~350 KB); TDX reconstructs the
    # dense gram_inv matrices IN-ENCLAVE (they are never transferred over the wire).
    bundle = {"ga": {}, "gm": {}, "nr_seed": 9000, "hidden": H, "num_layers": L}
    checks = []
    for l in range(L):
        ga = sd[f"model.layers.{l}.input_layernorm.weight"].to(DT)
        gm = sd[f"model.layers.{l}.post_attention_layernorm.weight"].to(DT)
        bundle["ga"][l] = ga.clone()
        bundle["gm"][l] = gm.clone()
        gram_inv_attn = Nr.T @ torch.diag(ga ** 2) @ Nr
        gram_inv_mlp = Nr.T @ torch.diag(gm ** 2) @ Nr
        # self-check: gram_inv @ Tin^T Tin == I  (Tin^T Tin = Nr^T diag(1/gamma^2) Nr)
        gram_attn = Nr.T @ torch.diag(1.0 / ga ** 2) @ Nr
        gram_mlp = Nr.T @ torch.diag(1.0 / gm ** 2) @ Nr
        err_attn = float((gram_inv_attn @ gram_attn - I).abs().max())
        err_mlp = float((gram_inv_mlp @ gram_mlp - I).abs().max())
        # cross-check vs the audit engine's Tin_gram_inv for q_proj / gate_proj
        Tin_q, _ = transforms(sd, l, "q_proj", Nr, nh, nkv, hd)
        Tin_g, _ = transforms(sd, l, "gate_proj", Nr, nh, nkv, hd)
        ref_q = torch.linalg.inv(Tin_q) @ torch.linalg.inv(Tin_q).T
        ref_g = torch.linalg.inv(Tin_g) @ torch.linalg.inv(Tin_g).T
        checks.append({"layer": l,
                       "selfcheck_attn_max_err": err_attn, "selfcheck_mlp_max_err": err_mlp,
                       "vs_audit_qproj_max_err": float((gram_inv_attn - ref_q).abs().max()),
                       "vs_audit_gate_max_err": float((gram_inv_mlp - ref_g).abs().max())})

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, outp)
    bhash = hashlib.sha256(outp.read_bytes()).hexdigest()
    worst = {k: max(c[k] for c in checks) for k in
             ["selfcheck_attn_max_err", "selfcheck_mlp_max_err", "vs_audit_qproj_max_err", "vs_audit_gate_max_err"]}
    manifest = {
        "artifact": "o1c_gamma_bundle.pt", "sha256": bhash,
        "contents": "private RMSNorm gains ga/gm per layer + Nr seed (gram_inv rebuilt IN-ENCLAVE)",
        "gram_inv_matrices_transferred": False,
        "trust_domain": "TRUSTED_ONLY", "provision_to": "TDX_enclave",
        "forbidden_on": "untrusted_gpu_worker",
        "encodes_private_gamma_spectrum": True,
        "reason_o1b_rejected": "materializing gram_inv GPU-side reveals eigenvalues=gamma^2 (private RMSNorm gains)",
        "num_layers": L, "hidden": H, "per_layer_matrices_rebuilt_in_enclave": 2,
        "matrices_total": 2 * L, "dtype": "float64",
        "correction_targets": {"attn": ["q_proj", "k_proj", "v_proj"], "mlp": ["gate_proj", "up_proj"]},
        "no_correction_targets": ["o_proj", "down_proj"],
        "verification_worst": worst,
        "verified_identity": all(v < 1e-9 for v in worst.values())}
    (outp.parent / "correction_bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"bundle_sha256": bhash[:16], "matrices": 2 * L,
                      "verification_worst": {k: f"{v:.2e}" for k, v in worst.items()},
                      "verified": manifest["verified_identity"]}, indent=2))


if __name__ == "__main__":
    main()
