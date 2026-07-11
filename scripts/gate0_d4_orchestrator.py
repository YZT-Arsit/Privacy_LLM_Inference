"""D4 orchestrator (runs on the Mac). Drives the private-base protected training step
across the untrusted H800 worker and the trusted Intel TDX loss boundary, with a fresh
attestation. Keys stay on the Mac; secrets stay in TDX; the worker stays package-only.

Per step:
  forward on H800  -> ferry masked logits -> real TDX loss (CE+dlogits in-enclave)
  -> ferry masked dlogits -> backward+masked-SGD on H800.
The Mac also (a) independently recomputes the expected CE/dlogits to CHECK the TDX
boundary, and (b) runs the trusted-eval verifier on H800 for the plaintext equivalence.
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, os, secrets, subprocess, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/gate0_d4"
CM_H800 = "/Users/Hoshino/.ssh/cm-westb-19948"
H800 = "root@connect.westb.seetacloud.com"; H800_PORT = "19948"
TDX_KEY = os.path.expanduser("~/Downloads/passkey.pem")
TDX = "root@39.96.4.252"
RH800 = "/root/pllo_pb"
RTDX = "/root/privacy_llm_obfuscation"
PY_H800 = "/root/miniconda3/bin/python"
PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
ENV_H800 = (f"PYTHONPATH={RH800}/src PB_PKG_DIR={RH800}/results/aaai_private_base/"
            f"private_package/gpu_package PB_CKPT_DIR=/root/autodl-tmp/modelscope_cache/"
            f"models/Qwen/Qwen2___5-0___5B PB_DTYPE=fp32 HF_HUB_OFFLINE=1")


def sh(cmd, timeout=1200):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def h800(cmd, timeout=1200):
    return sh(f"ssh -S {CM_H800} -p {H800_PORT} {H800} {json_q(cmd)}", timeout)


def tdx(cmd, timeout=1200):
    return sh(f"ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} {json_q(cmd)}", timeout)


def json_q(s):
    return "'" + s.replace("'", "'\\''") + "'"


def pull_h800(remote, local):
    sh(f"ssh -S {CM_H800} -p {H800_PORT} {H800} 'cat {remote}' > {local}")


def push_tdx(local, remote):
    sh(f"cat {local} | ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} 'cat > {remote}'")


def pull_tdx(remote, local):
    sh(f"ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} 'cat {remote}' > {local}")


def push_h800(local, remote):
    sh(f"cat {local} | ssh -S {CM_H800} -p {H800_PORT} {H800} 'cat > {remote}'")


def sha_local(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--run-tag", default="D4")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL_MSG = OUT / "local_msg"; LOCAL_MSG.mkdir(exist_ok=True)

    run_id = f"{args.run_tag}-{int(time.time())}-{secrets.token_hex(4)}"
    nonce = secrets.token_hex(16)
    session_key = secrets.token_bytes(32)
    ephem_pub = hashlib.sha256(secrets.token_bytes(32)).hexdigest()  # ephemeral session pub proxy
    ids_path = REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"
    labels = torch.tensor(json.loads(ids_path.read_text())["input_ids"][:args.seq_len])

    # ---------- fresh attestation binding ----------
    worker_rt = hashlib.sha256(
        (REPO / "scripts/h800_d4_worker.py").read_bytes()
        + (REPO / "scripts/h800_unified_worker.py").read_bytes()).hexdigest()
    svc_hash = sha_local(REPO / "scripts/tdx_trusted_loss_service.py")
    rc, gitrev, _ = sh(f"cd {REPO} && git rev-parse HEAD 2>/dev/null || echo worktree")
    binding = {
        "d4_run_id": run_id,
        "package_root_hash": "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
        "worker_runtime_hash": worker_rt, "tdx_service_hash": svc_hash,
        "model_config_hash": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
        "tokenizer_hash": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "execution_profile": "paper_safe", "feature_mask_family": "orthogonal_signed_perm",
        "nonlinear_permutation_profile": "swiglu_shared_perm_P",
        "qk_bias_handling_profile": "rope_commuting_planar_rotation_OI1",
        "rank_mask_profile": "orthogonal_U", "optimizer_profile": "gpu_masked_sgd_O1A",
        "vocabulary_mask_profile": "monomial_perm_only",
        "gradient_convention": "masked_domain_dlogits",
        "dataset_batch_manifest_hash": sha_local(ids_path),
        "code_revision_or_worktree_hash": gitrev.strip(),
        "debug_false_required": True, "nonce": nonce, "ephemeral_key_pub": ephem_pub}
    bpath = OUT / "attestation" / "d4_binding_manifest.json"
    bpath.parent.mkdir(parents=True, exist_ok=True)
    bpath.write_text(json.dumps(binding, indent=2))
    push_tdx(str(bpath), "/tmp/d4_binding.json")
    print("[attest] generating fresh quote bound to", run_id)
    tdx(f"cd {RTDX} && rm -rf /tmp/d4_attest && nohup env PYTHONPATH={RTDX}/src {PY_TDX} "
        f"scripts/gate0_d4_attestation.py --binding-manifest /tmp/d4_binding.json "
        f"--out-dir /tmp/d4_attest > /tmp/d4_attest.log 2>&1 & echo started", timeout=60)
    for _ in range(50):
        rc, o, _ = tdx("test -f /tmp/d4_attest/attestation_evidence.json && echo DONE || echo W", timeout=40)
        if "DONE" in o:
            break
        time.sleep(8)
    pull_tdx("/tmp/d4_attest/attestation_evidence.json", str(OUT / "attestation" / "attestation_evidence.json"))
    att = json.loads((OUT / "attestation" / "attestation_evidence.json").read_text())
    print("[attest] verified:", att.get("attestation_verified"), att.get("verifier_overall_appraisal_result"))

    # push private labels to TDX (trusted side holds labels)
    push_tdx(str(ids_path), "/tmp/d4_labels.json")

    # ---------- init masked LoRA on H800 ----------
    rc, o, e = h800(f"cd {RH800} && {ENV_H800} {PY_H800} scripts/h800_d4_worker.py --mode init "
                    f"--seq-len {args.seq_len}", timeout=600)
    print("[init]", o.strip().splitlines()[-1] if o.strip() else e[-300:])

    # ---------- training steps ----------
    steps = []
    last_seq = -1
    for step in range(args.steps):
        # forward on H800
        rc, o, e = h800(f"cd {RH800} && {ENV_H800} {PY_H800} scripts/h800_d4_worker.py "
                        f"--mode forward --step {step} --seq-len {args.seq_len} --lr {args.lr}",
                        timeout=600)
        if rc != 0:
            print("[forward FAIL]", e[-500:]); break
        # ferry logits H800 -> Mac -> TDX
        llog = LOCAL_MSG / f"logits_step{step}.pt"
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/msg/logits_step{step}.pt", str(llog))
        push_tdx(str(llog), f"/tmp/logits_step{step}.pt")
        # authenticated request envelope
        seq = last_seq + 1
        ml_hash = hashlib.sha256(llog.read_bytes()).digest()
        payload = ml_hash + str(seq).encode() + run_id.encode()
        tag = hmac.new(session_key, payload, hashlib.sha256).hexdigest()
        req = {"seq": seq, "run_id": run_id, "hmac": tag, "worker_send_time": time.time()}
        lreq = LOCAL_MSG / f"req_step{step}.json"; lreq.write_text(json.dumps(req))
        push_tdx(str(lreq), f"/tmp/req_step{step}.json")
        # real TDX loss
        t_net0 = time.time()
        rc, o, e = tdx(
            f"cd {RTDX} && {PY_TDX} scripts/tdx_trusted_loss_service.py "
            f"--request /tmp/req_step{step}.json --masked-logits /tmp/logits_step{step}.pt "
            f"--labels /tmp/d4_labels.json --session-key-hex {session_key.hex()} "
            f"--last-seq {last_seq} --out-dlogits /tmp/dlogits_step{step}.pt "
            f"--out-response /tmp/resp_step{step}.json --vocab-seed 8000", timeout=600)
        if rc != 0:
            print("[tdx loss FAIL]", e[-500:], o[-300:]); break
        pull_tdx(f"/tmp/resp_step{step}.json", str(LOCAL_MSG / f"resp_step{step}.json"))
        pull_tdx(f"/tmp/dlogits_step{step}.pt", str(LOCAL_MSG / f"dlogits_step{step}.pt"))
        resp = json.loads((LOCAL_MSG / f"resp_step{step}.json").read_text())
        tdx_ce = resp["aggregate"]["ce_loss"]
        # verify response HMAC
        dl = LOCAL_MSG / f"dlogits_step{step}.pt"
        rp = hashlib.sha256(dl.read_bytes()).digest() + str(resp["seq"]).encode() + run_id.encode()
        resp_ok = hmac.compare_digest(hmac.new(session_key, rp, hashlib.sha256).hexdigest(), resp["hmac"])
        net_time = time.time() - t_net0

        # INDEPENDENT boundary check on Mac: expected CE + dlogits vs TDX
        masked = torch.load(llog, map_location="cpu").float()
        g = torch.Generator().manual_seed(8000)
        perm = torch.randperm(masked.shape[1], generator=g)
        perm_inv = torch.empty_like(perm); perm_inv[perm] = torch.arange(masked.shape[1])
        plain = masked[:, perm]
        exp_ce = float(F.cross_entropy(plain[:-1], labels[1:]))
        probs = torch.softmax(plain[:-1], -1); dp = probs.clone()
        dp[torch.arange(labels[1:].shape[0]), labels[1:]] -= 1.0; dp /= labels[1:].shape[0]
        exp_d = torch.zeros_like(plain); exp_d[:-1] = dp
        exp_dm = exp_d[:, perm_inv]
        tdx_dm = torch.load(dl, map_location="cpu").float()
        dlogits_match = float((exp_dm - tdx_dm).abs().max())

        # ferry dlogits -> H800, backward+step
        push_h800(str(dl), f"{RH800}/results/aaai_private_base/gate0_d4/msg/dlogits_step{step}.pt")
        rc, o, e = h800(f"cd {RH800} && {ENV_H800} {PY_H800} scripts/h800_d4_worker.py "
                        f"--mode backward --step {step} --seq-len {args.seq_len} --lr {args.lr} "
                        f"--dlogits {RH800}/results/aaai_private_base/gate0_d4/msg/dlogits_step{step}.pt",
                        timeout=600)
        if rc != 0:
            print("[backward FAIL]", e[-500:]); break
        pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/backward_result_step{step}.json",
                  str(LOCAL_MSG / f"backward_step{step}.json"))
        bwd = json.loads((LOCAL_MSG / f"backward_step{step}.json").read_text())

        steps.append({
            "step": step, "tdx_ce": tdx_ce, "expected_ce_mac": exp_ce,
            "tdx_vs_expected_ce_abs": abs(tdx_ce - exp_ce),
            "tdx_dlogits_vs_expected_max_abs": dlogits_match,
            "response_hmac_ok": resp_ok,
            "all_targets_connected": bwd["all_targets_connected"],
            "base_tensors_require_grad": bwd["base_tensors_require_grad"],
            "step_finite": bwd["step_finite"],
            "counters": bwd["counters"], "tdx_timing": resp["timing"],
            "logits_bytes": resp["timing"]["logits_bytes_received"],
            "gradient_bytes": resp["timing"]["gradient_bytes_returned"],
            "ferry_net_sec": net_time})
        last_seq = resp["seq"]
        print(f"[step {step}] tdx_ce={tdx_ce:.5f} exp_ce={exp_ce:.5f} "
              f"dCE={abs(tdx_ce-exp_ce):.2e} dlogits_match={dlogits_match:.2e} "
              f"connected={bwd['all_targets_connected']} hmac={resp_ok}")

    # ---------- effective equivalence on last step (trusted-eval on H800) ----------
    equiv = {}
    if steps:
        s = args.steps - 1
        # NOTE: the effective-dW file (multi-GB) stays on H800; the trusted-eval
        # verifier runs ON H800 and reads it locally -- do NOT ferry it to the Mac.
        rc, o, e = h800(
            f"cd {RH800} && {ENV_H800} {PY_H800} scripts/d4_trusted_verifier.py "
            f"--masked-logits {RH800}/results/aaai_private_base/gate0_d4/msg/logits_step{s}.pt "
            f"--lora-state {RH800}/results/aaai_private_base/gate0_d4/msg/lora_state_at_step{s}.pt "
            f"--input-ids {RH800}/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json "
            f"--seq-len {args.seq_len} --out {RH800}/results/aaai_private_base/gate0_d4/equiv.json",
            timeout=900)
        if rc == 0:
            pull_h800(f"{RH800}/results/aaai_private_base/gate0_d4/equiv.json",
                      str(OUT / "effective_equivalence.json"))
            equiv = json.loads((OUT / "effective_equivalence.json").read_text())
        else:
            equiv = {"error": e[-400:]}

    report = {"run_id": run_id, "steps_run": len(steps),
              "attestation": {k: att.get(k) for k in
                              ["attestation_verified", "verifier_overall_appraisal_result",
                               "reportdata_binds_d4_manifest", "debug_false", "mr_td",
                               "nonce_fresh", "ephemeral_key_fresh"]},
              "binding_manifest": binding, "trajectory": steps,
              "effective_equivalence": equiv}
    (OUT / ("one_step_metrics.json" if args.steps == 1 else "trajectory_metrics.json")
     ).write_text(json.dumps(report, indent=2))
    print(json.dumps({"run_id": run_id, "steps": len(steps),
                      "attestation_verified": att.get("attestation_verified"),
                      "equiv": {k: equiv.get(k) for k in
                                ["effective_equivalence_top1_agreement",
                                 "effective_equivalence_next_logit_kl"]}}, indent=2))


if __name__ == "__main__":
    main()
