"""L10 (O1-C hybrid) orchestrator -- real H800 worker + real Intel TDX, Mac-driven.

Extends the D4 protocol with the THIRD trusted invocation: an in-enclave gradient
correction for the gamma-fed targets (q/k/v/gate/up). Per step:

  forward(H800) -> ferry masked logits -> TDX loss (CE+dlogits, in-enclave)
    -> ferry masked dlogits -> backward(H800, profile=o1c: applies o/down A + all B on GPU,
       DEFERS q/k/v/gate/up masked A-grads) -> ferry corr_grads -> TDX gradient correction
       (in-enclave gA @ Nr^T diag(gamma^2) Nr) -> ferry corrected grads
    -> apply_correction(H800: applies the 5x24 corrected A-updates).

The trusted correction bundle is provisioned into TDX ONLY (never to H800). Fresh
attestation is bound to the O1-C manifest (optimizer_profile=o1c_hybrid, correction
service + bundle hashes). Keys on Mac; secrets/bundle in TDX; worker package-only.
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, secrets, time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
sys.path.insert(0, str(REPO / "scripts"))
from gate0_d4_orchestrator import (  # reuse verified ferry helpers
    sh, h800, tdx, json_q, pull_h800, push_tdx, pull_tdx, push_h800, sha_local,
    CM_H800, H800, H800_PORT, TDX_KEY, TDX, RH800, RTDX, PY_H800, PY_TDX, ENV_H800)

OUT = REPO / "results/aaai_private_base/full_lora_matrix/profile_validation"
BUNDLE_LOCAL = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
CORR_A = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]


def hmac_req(path, seq, run_id, key):
    payload = hashlib.sha256(Path(path).read_bytes()).digest() + str(seq).encode() + run_id.encode()
    return {"seq": seq, "run_id": run_id,
            "hmac": hmac.new(key, payload, hashlib.sha256).hexdigest(),
            "worker_send_time": time.time()}


def resp_hmac_ok(path, seq, run_id, key, tag):
    p = hashlib.sha256(Path(path).read_bytes()).digest() + str(seq).encode() + run_id.encode()
    return hmac.compare_digest(hmac.new(key, p, hashlib.sha256).hexdigest(), tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--run-tag", default="L10")
    ap.add_argument("--input-ids", default=str(
        REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MSG = OUT / f"msg_{args.run_tag}_s{args.seed}"; MSG.mkdir(exist_ok=True)
    ENV = ENV_H800.replace("PB_DTYPE=fp32", f"PB_DTYPE={args.dtype}")

    run_id = f"{args.run_tag}-{args.seed}-{int(time.time())}-{secrets.token_hex(4)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    ephem_pub = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    ids_path = Path(args.input_ids)
    labels = torch.tensor(json.loads(ids_path.read_text())["input_ids"][:args.seq_len])
    head_hash = (REPO / "results/aaai_private_base/code_baseline/head_hash.txt").read_text().strip()
    rc, wtdiff, _ = sh(f"cd {REPO} && git diff HEAD | shasum -a 256 | cut -d' ' -f1")

    # ---- fresh attestation binding to the O1-C manifest ----
    worker_rt = hashlib.sha256(
        (REPO / "scripts/h800_d4_worker.py").read_bytes()
        + (REPO / "scripts/h800_unified_worker.py").read_bytes()).hexdigest()
    binding = {
        # ---- required schema fields (bound into report_data) ----
        "d4_run_id": run_id,
        "package_root_hash": PKG_ROOT, "worker_runtime_hash": worker_rt,
        "tdx_service_hash": sha_local(REPO / "scripts/tdx_trusted_loss_service.py"),
        "model_config_hash": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
        "tokenizer_hash": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "execution_profile": "paper_safe",
        "feature_mask_family": "orthogonal_signed_perm",
        "nonlinear_permutation_profile": "swiglu_shared_perm_P",
        "qk_bias_handling_profile": "rope_commuting_planar_rotation_OI1",
        "rank_mask_profile": "orthogonal_U",
        "optimizer_profile": "o1c_hybrid_sgd",            # re-quoted per design (binds O1-C)
        "vocabulary_mask_profile": "monomial_perm_only",
        "gradient_convention": "masked_domain_dlogits",
        "dataset_batch_manifest_hash": sha_local(ids_path),
        "code_revision_or_worktree_hash": head_hash,
        "debug_false_required": True, "nonce": nonce, "ephemeral_key_pub": ephem_pub,
        # ---- O1-C-specific design binding (also covered by report_data hash) ----
        "profile": "L10_O1C_hybrid",
        "tdx_correction_service_hash": sha_local(REPO / "scripts/tdx_grad_correction_service.py"),
        "correction_bundle_hash": sha_local(BUNDLE_LOCAL),
        "correction_bundle_trust_domain": "TDX_only_never_gpu",
        "gpu_side_targets": ["o_proj", "down_proj"], "tdx_corrected_targets": CORR_A,
        "dtype": args.dtype, "seed": args.seed,
        "worktree_diff_sha256": wtdiff.strip(),
        "invocations_per_step_expected": 3}
    bpath = OUT / f"binding_{args.run_tag}_s{args.seed}.json"; bpath.write_text(json.dumps(binding, indent=2))
    push_tdx(str(bpath), "/tmp/l10_binding.json")
    print("[attest] fresh quote bound to", run_id)
    tdx(f"cd {RTDX} && rm -rf /tmp/l10_attest && nohup env PYTHONPATH={RTDX}/src {PY_TDX} "
        f"scripts/gate0_d4_attestation.py --binding-manifest /tmp/l10_binding.json "
        f"--out-dir /tmp/l10_attest > /tmp/l10_attest.log 2>&1 & echo started", timeout=60)
    att = {}
    for _ in range(60):
        rc, o, _ = tdx("test -f /tmp/l10_attest/attestation_evidence.json && echo DONE || echo W", timeout=40)
        if "DONE" in o:
            pull_tdx("/tmp/l10_attest/attestation_evidence.json", str(OUT / f"attestation_{args.run_tag}_s{args.seed}.json"))
            att = json.loads((OUT / f"attestation_{args.run_tag}_s{args.seed}.json").read_text())
            break
        time.sleep(8)
    print("[attest] verified:", att.get("attestation_verified"), att.get("verifier_overall_appraisal_result"))

    # ---- provision TRUSTED correction bundle into TDX only ----
    print("[provision] pushing correction bundle to TDX (trusted->trusted) ...")
    push_tdx(str(BUNDLE_LOCAL), "/tmp/o1c_bundle.pt")
    push_tdx(str(ids_path), "/tmp/l10_labels.json")
    rc, o, _ = tdx(f"test -f /tmp/o1c_bundle.pt && shasum -a 256 /tmp/o1c_bundle.pt", timeout=120)
    print("[provision] tdx bundle:", o.strip()[:80])

    # ---- init masked LoRA (seed-specific) on H800 ----
    h800(f"cd {RH800} && {ENV} {PY_H800} scripts/h800_d4_worker.py --mode init --seq-len {args.seq_len}", timeout=600)

    steps = []; last_seq = -1
    for step in range(args.steps):
        t_step0 = time.time()
        # (1) forward
        rc, o, e = h800(f"cd {RH800} && {ENV} {PY_H800} scripts/h800_d4_worker.py --mode forward "
                        f"--profile o1c --step {step} --seq-len {args.seq_len} --lr {args.lr}", timeout=600)
        if rc != 0:
            print("[forward FAIL]", e[-400:]); break
        llog = MSG / f"logits_step{step}.pt"
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/msg/logits_step{step}.pt", str(llog))
        push_tdx(str(llog), f"/tmp/l10_logits_{step}.pt")
        # (2) TDX loss
        seq = last_seq + 1
        lreq = MSG / f"req_loss_{step}.json"; lreq.write_text(json.dumps(hmac_req(llog, seq, run_id, session_key)))
        push_tdx(str(lreq), f"/tmp/l10_req_loss_{step}.json")
        t_loss0 = time.time()
        rc, o, e = tdx(f"cd {RTDX} && {PY_TDX} scripts/tdx_trusted_loss_service.py "
                       f"--request /tmp/l10_req_loss_{step}.json --masked-logits /tmp/l10_logits_{step}.pt "
                       f"--labels /tmp/l10_labels.json --session-key-hex {session_key.hex()} "
                       f"--last-seq {last_seq} --out-dlogits /tmp/l10_dlog_{step}.pt "
                       f"--out-response /tmp/l10_resp_loss_{step}.json --vocab-seed 8000", timeout=600)
        if rc != 0:
            print("[tdx loss FAIL]", e[-400:], o[-200:]); break
        pull_tdx(f"/tmp/l10_resp_loss_{step}.json", str(MSG / f"resp_loss_{step}.json"))
        pull_tdx(f"/tmp/l10_dlog_{step}.pt", str(MSG / f"dlog_{step}.pt"))
        resp_loss = json.loads((MSG / f"resp_loss_{step}.json").read_text())
        tdx_ce = resp_loss["aggregate"]["ce_loss"]; t_loss = time.time() - t_loss0
        loss_seq = resp_loss["seq"]
        # independent Mac boundary check
        masked = torch.load(llog, map_location="cpu").float()
        gseed = torch.Generator().manual_seed(8000)
        perm = torch.randperm(masked.shape[1], generator=gseed)
        perm_inv = torch.empty_like(perm); perm_inv[perm] = torch.arange(masked.shape[1])
        plain = masked[:, perm]; exp_ce = float(F.cross_entropy(plain[:-1], labels[1:]))
        # (2') ferry dlogits -> H800, backward o1c
        push_h800(str(MSG / f"dlog_{step}.pt"), f"{RH800}/results/aaai_private_base/gate0_d4/msg/dlog_{step}.pt")
        rc, o, e = h800(f"cd {RH800} && {ENV} {PY_H800} scripts/h800_d4_worker.py --mode backward "
                        f"--profile o1c --step {step} --seq-len {args.seq_len} --lr {args.lr} "
                        f"--dlogits {RH800}/results/aaai_private_base/gate0_d4/msg/dlog_{step}.pt", timeout=600)
        if rc != 0:
            print("[backward FAIL]", e[-400:]); break
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/backward_result_step{step}.json",
                  str(MSG / f"backward_{step}.json"))
        bwd = json.loads((MSG / f"backward_{step}.json").read_text())
        # (3) ferry corr_grads -> TDX correction service
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/msg/corr_grads_step{step}.pt",
                  str(MSG / f"corr_grads_{step}.pt"))
        push_tdx(str(MSG / f"corr_grads_{step}.pt"), f"/tmp/l10_corr_in_{step}.pt")
        cseq = loss_seq + 1
        creq = MSG / f"req_corr_{step}.json"
        creq.write_text(json.dumps(hmac_req(MSG / f"corr_grads_{step}.pt", cseq, run_id, session_key)))
        push_tdx(str(creq), f"/tmp/l10_req_corr_{step}.json")
        t_corr0 = time.time()
        rc, o, e = tdx(f"cd {RTDX} && {PY_TDX} scripts/tdx_grad_correction_service.py "
                       f"--request /tmp/l10_req_corr_{step}.json --grads /tmp/l10_corr_in_{step}.pt "
                       f"--bundle /tmp/o1c_bundle.pt --session-key-hex {session_key.hex()} "
                       f"--last-seq {loss_seq} --num-layers 24 --out-grads /tmp/l10_corr_out_{step}.pt "
                       f"--out-response /tmp/l10_resp_corr_{step}.json", timeout=600)
        if rc != 0:
            print("[tdx correction FAIL]", e[-400:], o[-200:]); break
        pull_tdx(f"/tmp/l10_resp_corr_{step}.json", str(MSG / f"resp_corr_{step}.json"))
        pull_tdx(f"/tmp/l10_corr_out_{step}.pt", str(MSG / f"corr_out_{step}.pt"))
        resp_corr = json.loads((MSG / f"resp_corr_{step}.json").read_text())
        t_corr = time.time() - t_corr0
        corr_hmac_ok = resp_hmac_ok(MSG / f"corr_out_{step}.pt", resp_corr["seq"], run_id, session_key, resp_corr["hmac"])
        # (3') ferry corrected -> H800, apply
        push_h800(str(MSG / f"corr_out_{step}.pt"),
                  f"{RH800}/results/aaai_private_base/gate0_d4/msg/corr_out_{step}.pt")
        rc, o, e = h800(f"cd {RH800} && {ENV} {PY_H800} scripts/h800_d4_worker.py --mode apply_correction "
                        f"--profile o1c --step {step} --seq-len {args.seq_len} --lr {args.lr} "
                        f"--corrected-grads {RH800}/results/aaai_private_base/gate0_d4/msg/corr_out_{step}.pt", timeout=600)
        if rc != 0:
            print("[apply FAIL]", e[-400:]); break
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/apply_correction_step{step}.json",
                  str(MSG / f"apply_{step}.json"))
        appl = json.loads((MSG / f"apply_{step}.json").read_text())
        last_seq = resp_corr["seq"]
        steps.append({
            "step": step, "tdx_ce": tdx_ce, "expected_ce_mac": exp_ce,
            "tdx_vs_expected_ce_abs": abs(tdx_ce - exp_ce),
            "loss_hmac_ok": resp_hmac_ok(MSG / f"dlog_{step}.pt", loss_seq, run_id, session_key, resp_loss["hmac"]),
            "correction_hmac_ok": corr_hmac_ok,
            "all_targets_connected": bwd["all_targets_connected"],
            "gpu_exact_A_applied": bwd["gpu_exact_A_applied"], "deferred_A_to_tdx": bwd["deferred_A_to_tdx"],
            "correction_targets_applied": appl["correction_targets_applied"],
            "all_corrections_applied": appl["all_corrections_applied"],
            "correction_missing_targets": resp_corr["counters"]["correction_missing_targets"],
            "corrected_grad_zeroed_fraction": appl["corrected_grad_zeroed_fraction"],
            "base_tensors_require_grad": bwd["base_tensors_require_grad"],
            "step_finite": bwd["step_finite"] and appl["step_finite"],
            "backward_counters": bwd["counters"], "apply_counters": appl["counters"],
            "tdx_correction_numerics": resp_corr["corrected_numerics"],
            "bf16_raw_A_numerics_layer0": bwd["bf16_raw_A_numerics_layer0"],
            "logical_invocations": 3,
            "comm_bytes": {"logits_to_tdx": resp_loss["timing"]["logits_bytes_received"],
                           "dlogits_from_tdx": resp_loss["timing"]["gradient_bytes_returned"],
                           "corr_grads_to_tdx": resp_corr["timing"]["grads_bytes_received"],
                           "corrected_from_tdx": resp_corr["timing"]["grads_bytes_returned"]},
            "timing": {"tdx_loss_sec": t_loss, "tdx_correction_sec": t_corr,
                       "tdx_correction_compute": resp_corr["timing"]["tdx_correction_sec"],
                       "step_wall_sec": time.time() - t_step0}})
        print(f"[step {step}] ce={tdx_ce:.5f} dCE={abs(tdx_ce-exp_ce):.1e} "
              f"gpuA={bwd['gpu_exact_A_applied']} tdxA={appl['correction_targets_applied']} "
              f"miss={resp_corr['counters']['correction_missing_targets']} "
              f"zeroed={appl['corrected_grad_zeroed_fraction']:.2e} "
              f"hmac={corr_hmac_ok} finite={steps[-1]['step_finite']}")

    # ---- effective equivalence (trusted-eval on H800) at the FINAL trained state ----
    # Run one forward at the post-training state so the logits and the LoRA snapshot are
    # temporally aligned (comparing post-step state vs pre-step logits gives a false 0.93).
    equiv = {}
    if steps:
        fs = args.steps           # fresh forward index at the final trained state
        rc, o, e = h800(f"cd {RH800} && {ENV} {PY_H800} scripts/h800_d4_worker.py --mode forward "
                        f"--profile o1c --step {fs} --seq-len {args.seq_len} --lr {args.lr}", timeout=600)
        if rc == 0:
            rc, o, e = h800(f"cd {RH800} && {ENV} {PY_H800} scripts/d4_trusted_verifier.py "
                            f"--masked-logits {RH800}/results/aaai_private_base/gate0_d4/msg/logits_step{fs}.pt "
                            f"--lora-state {RH800}/results/aaai_private_base/gate0_d4/msg/lora_state_at_step{fs}.pt "
                            f"--input-ids {RH800}/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json "
                            f"--seq-len {args.seq_len} --out {RH800}/results/aaai_private_base/gate0_d4/l10_equiv.json", timeout=900)
        if rc == 0:
            pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/l10_equiv.json", str(MSG / "effective_equivalence.json"))
            equiv = json.loads((MSG / "effective_equivalence.json").read_text())
        else:
            equiv = {"error": e[-300:]}

    report = {"run_id": run_id, "profile": "L10_O1C_hybrid", "dtype": args.dtype, "seed": args.seed,
              "steps_run": len(steps),
              "attestation": {k: att.get(k) for k in
                              ["attestation_verified", "verifier_overall_appraisal_result",
                               "reportdata_binds_d4_manifest", "debug_false", "mr_td"]},
              "binding_manifest": binding, "trajectory": steps, "effective_equivalence": equiv,
              "gate_pass": bool(steps) and all(
                  s["all_corrections_applied"] and s["correction_missing_targets"] == 0
                  and s["step_finite"] and s["correction_hmac_ok"] and s["loss_hmac_ok"]
                  for s in steps)}
    rpath = OUT / f"L10_{args.dtype}_s{args.seed}_{args.steps}step.json"
    rpath.write_text(json.dumps(report, indent=2))
    print(json.dumps({"run_id": run_id, "steps": len(steps), "gate_pass": report["gate_pass"],
                      "attestation_verified": att.get("attestation_verified"),
                      "equiv_top1": equiv.get("effective_equivalence_top1_agreement"),
                      "equiv_kl": equiv.get("effective_equivalence_next_logit_kl")}, indent=2))


if __name__ == "__main__":
    main()
