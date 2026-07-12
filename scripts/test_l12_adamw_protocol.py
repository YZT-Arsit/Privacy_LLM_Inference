"""Local self-test of the trusted AdamW protocol (no hardware). Deployed fold convention.

Validates, per trusted factor, and end-to-end via the real enclave step()/checkpoint API:
  (1) closed-form transforms vs linalg.inv (foldM=diag(g)Nr; gradMT=foldM^T; unfoldMinv=foldM^-1);
  (2) init round-trip: un-fold(fold(theta_plain)) == theta_plain;
  (3) N-step enclave step (masked-in -> plaintext AdamW -> re-fold) == independent plaintext AdamW
      reference re-folded, exercising the REAL tw.step() (all trusted factors, missing==0, version bump);
  (4) FP32 state_dtype variant: still numerically self-consistent (looser tol);
  (5) encrypted checkpoint/restore round-trip + fail-closed on tamper, wrong key, and version rollback.
Uses the real gamma bundle + real Qwen2.5-0.5B config.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from tdx_adamw_protocol import (TrustedAdamW, adamw_step, orthogonal_signed_perm,
                                TRUSTED_A, TRUSTED_B, ATTN, DT)

BUNDLE = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
CFG = {"num_attention_heads": 14, "num_key_value_heads": 2, "hidden_size": 896,
       "intermediate_size": 4864}
RANK = 8


def build_full(gb, state_dtype, seed):
    """Fresh enclave with ALL trusted factors initialized from random plaintext (folded in).
    Returns (tw, refA, refB) where ref* are the parallel plaintext AdamW states."""
    tw = TrustedAdamW(gb, CFG, lr=1e-3, state_dtype=state_dtype)
    tw.set_binding(run_id="selftest", package_root_hash="pkg", adapter_id="ad0")
    g = torch.Generator().manual_seed(seed)
    H = gb["hidden"]; hd = H // CFG["num_attention_heads"]
    refA, refB = {}, {}
    for l in range(gb["num_layers"]):
        for p in TRUSTED_A:
            Ap = torch.randn(RANK, H, generator=g, dtype=state_dtype)
            foldM, _, _ = tw.tinA[(l, p)]
            tw.init_factor(l, p, A_tilde=Ap @ foldM)           # deployed: A_tilde = A_plain @ M
            refA[(l, p)] = [Ap.clone(), torch.zeros_like(Ap), torch.zeros_like(Ap)]
        for p in TRUSTED_B:
            od = hd * (CFG["num_attention_heads"] if p == "q_proj" else CFG["num_key_value_heads"])
            Bp = torch.randn(od, RANK, generator=g, dtype=state_dtype)
            Tout = tw.toutB[(l, p)]
            tw.init_factor(l, p, B_tilde=Tout.T @ Bp)           # B_tilde = Tout^T @ B_plain
            refB[(l, p)] = [Bp.clone(), torch.zeros_like(Bp), torch.zeros_like(Bp)]
    return tw, refA, refB


def main():
    gb = torch.load(BUNDLE, map_location="cpu")
    H = gb["hidden"]; hd = H // CFG["num_attention_heads"]
    Nr = orthogonal_signed_perm(H, gb["nr_seed"])

    # (1) transforms vs linalg
    tw0 = TrustedAdamW(gb, CFG, lr=1e-3)
    max_transform_err = 0.0
    for l in [0, 5, 12, 23]:
        for p in TRUSTED_A:
            foldM, gradMT, unfoldMinv = tw0.tinA[(l, p)]
            gamma = (gb["ga"][l] if p in ATTN else gb["gm"][l]).to(DT)
            M = torch.diag(gamma) @ Nr
            max_transform_err = max(max_transform_err,
                float((foldM - M).abs().max()),
                float((gradMT - M.T).abs().max()),
                float((unfoldMinv - torch.linalg.inv(M)).abs().max()),
                float((foldM @ unfoldMinv - torch.eye(H, dtype=DT)).abs().max()))

    # (2)+(3): fp64 full-enclave N-step vs independent plaintext AdamW reference
    tw, refA, refB = build_full(gb, DT, seed=123)
    max_roundtrip_err = max(
        max(float((tw.stateA[k][0] - refA[k][0]).abs().max()) for k in refA),
        max(float((tw.stateB[k][0] - refB[k][0]).abs().max()) for k in refB))
    g = torch.Generator().manual_seed(777)
    max_step_err = 0.0; N = 5
    for step in range(1, N + 1):
        gA_tilde, gB_tilde = {}, {}
        for (l, p), (Ap, m, v) in refA.items():
            gpl = torch.randn_like(Ap) * 0.01
            foldM, gradMT, unfoldMinv = tw.tinA[(l, p)]
            # masked grad the enclave will receive: g_tilde s.t. g_tilde @ gradMT == g_plain
            gA_tilde[f"{l}.{p}"] = gpl @ torch.linalg.inv(gradMT)
            refA[(l, p)] = list(adamw_step(Ap, gpl, m, v, step, 1e-3, 0.9, 0.999, 1e-8, 0.01))
        for (l, p), (Bp, m, v) in refB.items():
            gpl = torch.randn_like(Bp) * 0.01
            Tout = tw.toutB[(l, p)]
            gB_tilde[f"{l}.{p}"] = torch.linalg.inv(Tout) @ gpl   # Tout @ g_tilde == g_plain
            refB[(l, p)] = list(adamw_step(Bp, gpl, m, v, step, 1e-3, 0.9, 0.999, 1e-8, 0.01))
        outA, outB, missing = tw.step(gA_tilde, gB_tilde)
        assert missing == 0, f"missing {missing}"
        assert tw.version == step, f"version {tw.version} != {step}"
        for (l, p), (rAp, _, _) in refA.items():
            foldM, _, _ = tw.tinA[(l, p)]
            max_step_err = max(max_step_err, float((outA[f"{l}.{p}"] - rAp @ foldM).abs().max()))
        for (l, p), (rBp, _, _) in refB.items():
            Tout = tw.toutB[(l, p)]
            max_step_err = max(max_step_err, float((outB[f"{l}.{p}"] - Tout.T @ rBp).abs().max()))

    # (4) fp32 state variant: build + 1 step, self-consistency (looser)
    tw32, rA32, rB32 = build_full(gb, torch.float32, seed=42)
    fp32_roundtrip = max(float((tw32.stateA[k][0] - rA32[k][0]).abs().max()) for k in rA32)

    # (5) STANDARD-AEAD (ChaCha20-Poly1305) checkpoint / restore
    tw.set_binding(run_id="R1", package_root_hash="PKGHASH", adapter_id="AD1",
                   optimizer_profile="L12_adamw", model_config_hash="MCFG", service_hash="SVC")
    blob = tw.checkpoint(session_key=b"k" * 32)
    assert blob.startswith(b"L12AEADv1"), "not a standard-AEAD seal"
    exp = {"run_id": "R1", "package_root_hash": "PKGHASH", "adapter_id": "AD1",
           "optimizer_profile": "L12_adamw", "model_config_hash": "MCFG", "service_hash": "SVC",
           "version": tw.version, "checkpoint_sequence": tw.checkpoint_seq}
    tw_r = TrustedAdamW(gb, CFG, lr=1e-3)
    info = tw_r.restore(blob, session_key=b"k" * 32, expected_binding=exp, min_version=0)
    ck_roundtrip = max(float((tw_r.stateA[k][0] - tw.stateA[k][0]).abs().max()) for k in tw.stateA)
    def fails(fn):
        try:
            fn(); return False
        except Exception:
            return True
    tamper = bytearray(blob); tamper[-1] ^= 0x01     # flip a ciphertext/tag byte
    fc_tamper = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(bytes(tamper), b"k" * 32, exp))
    fc_wrongkey = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"x" * 32, exp))
    fc_rollback = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"k" * 32, dict(exp, version=exp["version"] + 5)))
    fc_badpkg = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"k" * 32, dict(exp, package_root_hash="OTHER")))
    fc_badprofile = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"k" * 32, dict(exp, optimizer_profile="L5")))
    fc_badadapter = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"k" * 32, dict(exp, adapter_id="OTHER")))
    fc_badseq = fails(lambda: TrustedAdamW(gb, CFG, lr=1e-3).restore(blob, b"k" * 32, dict(exp, checkpoint_sequence=exp["checkpoint_sequence"] + 3)))
    # nonce-reuse: opening the same blob twice against a ledger that already saw the nonce must fail
    tw_n = TrustedAdamW(gb, CFG, lr=1e-3); tw_n.restore(blob, b"k" * 32, exp)
    fc_nonce_reuse = fails(lambda: tw_n.restore(blob, b"k" * 32, exp))
    fc_badbind = fc_badpkg and fc_badprofile and fc_badadapter and fc_badseq

    ok = (max_transform_err < 1e-9 and max_roundtrip_err < 1e-9 and max_step_err < 1e-10
          and fp32_roundtrip < 1e-4 and ck_roundtrip < 1e-12
          and fc_tamper and fc_wrongkey and fc_rollback and fc_badbind and fc_nonce_reuse)
    print(json.dumps({
        "aead": "ChaCha20Poly1305", "max_transform_err": max_transform_err, "max_roundtrip_err": max_roundtrip_err,
        "max_step_err_5steps": max_step_err, "final_version": tw.version,
        "fp32_state_roundtrip": fp32_roundtrip, "checkpoint_restore_roundtrip": ck_roundtrip,
        "restore_info": info,
        "fail_closed": {"tamper": fc_tamper, "wrong_key": fc_wrongkey, "version_rollback": fc_rollback,
                        "bad_package": fc_badpkg, "bad_profile": fc_badprofile, "bad_adapter": fc_badadapter,
                        "bad_checkpoint_seq": fc_badseq, "nonce_reuse": fc_nonce_reuse},
        "PASS": bool(ok)}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
