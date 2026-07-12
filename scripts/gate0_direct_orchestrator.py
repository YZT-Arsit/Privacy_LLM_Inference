"""Direct-transport control-plane orchestrator (runs on the Mac).

CONTROL-PLANE ONLY. The Mac: (1) generates the session key + binding manifest, (2) provisions
the session config + gamma bundle + service/runner code to TDX and H800, (3) launches the
persistent H800 runner, (4) collects the result + attestation. It NEVER touches per-step
payloads and does NO step-by-step orchestration. All per-step logits/dlogits/gradients flow
H800 <-> TDX directly over one persistent SSH channel (transport_profile=direct_h800_tdx).
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/direct_transport"
CM_H800 = "/Users/Hoshino/.ssh/cm-westb-19948"
H800 = "root@connect.westb.seetacloud.com"; H800_PORT = "19948"
TDX_KEY = "/Users/Hoshino/Downloads/passkey.pem"; TDX = "root@39.96.4.252"; TDX_IP = "39.96.4.252"
RH800 = "/root/pllo_pb"; RTDX = "/root/privacy_llm_obfuscation"
PY_H800 = "/root/miniconda3/bin/python"; PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
ENV_H800 = (f"PYTHONPATH={RH800}/src PB_PKG_DIR={RH800}/results/aaai_private_base/private_package/"
            f"gpu_package PB_CKPT_DIR=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen2___5-0___5B "
            f"HF_HUB_OFFLINE=1")
IDS = "/root/pllo_pb/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"


def sh(cmd, timeout=1800):
    p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def h800(cmd, timeout=1800):
    return sh(f"ssh -S {CM_H800} -p {H800_PORT} {H800} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def tdx(cmd, timeout=600):
    return sh(f"ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def push_h800(local, remote):
    sh(f"cat {local} | ssh -S {CM_H800} -p {H800_PORT} {H800} 'cat > {remote}'")


def push_tdx(local, remote):
    sh(f"cat {local} | ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} 'cat > {remote}'")


def pull_h800(remote, local):
    sh(f"ssh -S {CM_H800} -p {H800_PORT} {H800} 'cat {remote}' > {local}")


def pull_tdx(remote, local):
    sh(f"ssh -i {TDX_KEY} -o StrictHostKeyChecking=no {TDX} 'cat {remote}' > {local}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--run-tag", default="DIRECT")
    ap.add_argument("--attest", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.run_tag}-{args.steps}s-{int(time.time())}-{secrets.token_hex(3)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head_hash.txt").read_text().strip()
    ids = json.loads((REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json").read_text())
    labels = ids["input_ids"][:args.seq_len]
    binding = {  # for fresh attestation at session setup
        "d4_run_id": run_id, "package_root_hash": "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1",
        "worker_runtime_hash": hashlib.sha256((REPO/"scripts/h800_direct_runner.py").read_bytes()).hexdigest(),
        "tdx_service_hash": hashlib.sha256((REPO/"scripts/tdx_persistent_service.py").read_bytes()).hexdigest(),
        "model_config_hash": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
        "tokenizer_hash": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "execution_profile": "paper_safe", "feature_mask_family": "orthogonal_signed_perm",
        "nonlinear_permutation_profile": "swiglu_shared_perm_P",
        "qk_bias_handling_profile": "rope_commuting_planar_rotation_OI1", "rank_mask_profile": "orthogonal_U",
        "optimizer_profile": ("o1c_hybrid_momentum" if args.momentum > 0 else "o1c_hybrid_sgd"),
        "vocabulary_mask_profile": "monomial_perm_only", "gradient_convention": "masked_domain_dlogits",
        "dataset_batch_manifest_hash": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
        "code_revision_or_worktree_hash": head, "transport_profile": "direct_h800_tdx",
        "debug_false_required": True, "nonce": nonce, "ephemeral_key_pub": hashlib.sha256(nonce.encode()).hexdigest()}

    # ---- provision code + gamma bundle + session config ----
    push_tdx(str(REPO / "scripts/tdx_persistent_service.py"), f"{RTDX}/scripts/tdx_persistent_service.py")
    push_h800(str(REPO / "scripts/h800_direct_runner.py"), f"{RH800}/scripts/h800_direct_runner.py")
    tdx("test -f /tmp/o1c_bundle.pt || echo MISSING_BUNDLE")     # provisioned earlier; re-push if needed
    push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": labels,
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/direct_attest",
                "binding_manifest": binding, "counters_out": "/tmp/direct_session_counters.json"}
    Path(OUT / "tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(OUT / "tdx_session.json"), "/tmp/direct_session.json")
    h800_sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    Path(OUT / "h800_session.json").write_text(json.dumps(h800_sess))
    push_h800(str(OUT / "h800_session.json"), "/tmp/direct_session_h800.json")

    # ---- launch the persistent H800 runner (it connects to TDX directly) ----
    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RH800}/results/aaai_private_base/direct_transport/{args.run_tag}_{args.dtype}_{args.steps}step.json"
    h800(f"mkdir -p {RH800}/results/aaai_private_base/direct_transport")
    cmd = (f"cd {RH800} && {ENV_H800} {PY_H800} scripts/h800_direct_runner.py "
           f"--session /tmp/direct_session_h800.json --steps {args.steps} --seq-len {args.seq_len} "
           f"--lr {args.lr} --momentum {args.momentum} --dtype {args.dtype} --input-ids {IDS} "
           f"--tdx root@{TDX_IP} --key /root/.ssh/h800_to_tdx "
           f"--service-cmd \"{service_cmd}\" --out {out_json}")
    print("[launch] direct runner on H800 (data plane = H800<->TDX direct) ...")
    t0 = time.time()
    rc, o, e = h800(cmd, timeout=3600)
    wall = time.time() - t0
    print(o[-1500:] if o else "", "\n[stderr]", e[-600:] if rc != 0 else "")

    # ---- collect (control-plane): result + attestation + tdx counters ----
    lout = OUT / f"{args.run_tag}_{args.dtype}_{args.steps}step.json"
    pull_h800(out_json, str(lout))
    pull_h800(out_json.replace(".json", ".adapter.pt"), str(OUT / f"{args.run_tag}_{args.dtype}_{args.steps}step.adapter.pt"))
    if args.attest:
        pull_tdx("/tmp/direct_attest/session_attestation.json", str(OUT / f"{args.run_tag}_session_attestation.json"))
    pull_tdx("/tmp/direct_session_counters.json", str(OUT / f"{args.run_tag}_tdx_counters.json"))
    try:
        r = json.loads(lout.read_text())
        print(json.dumps({"run_id": run_id, "transport": "direct_h800_tdx", "steps": r["steps_run"],
                          "control_plane_wall_sec": round(wall, 1),
                          "ce_first_last": [r["trajectory"][0]["ce"], r["trajectory"][-1]["ce"]] if r["trajectory"] else None,
                          "mean_step_wall": round(sum(s["timing"]["step_wall_sec"] for s in r["trajectory"])/max(1,len(r["trajectory"])), 2)}, indent=2))
    except Exception as ex:
        print("collect parse error:", ex)


if __name__ == "__main__":
    main()
