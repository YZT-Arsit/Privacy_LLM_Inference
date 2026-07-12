"""A10+TDX direct-transport control-plane orchestrator (runs on the Mac).

CONTROL-PLANE ONLY. Mac reaches the A10 GPU host via a ProxyJump control-master through the TDX
public host; provisions the session key + binding manifest + gamma bundle + code to TDX (public
passkey) and to the A10; launches the persistent A10 runner; collects results. The per-step DATA
PLANE is A10 <-> TDX over the PRIVATE VPC IP (172.30.25.153), never the Mac, never the public IP.
"""
from __future__ import annotations
import argparse, hashlib, json, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/alicloud_a10_runs"
CM_A10 = "/tmp/cm-a10"                     # ProxyJump control master to A10 (via TDX public)
A10 = "root@172.30.25.154"
PK = "/Users/Hoshino/Downloads/passkey.pem"; TDX = "root@39.96.4.252"      # TDX public (control)
TDX_PRIV = "172.30.25.153"                 # TDX private (data plane)
RA10 = "/root/pllo_pb"; RTDX = "/root/privacy_llm_obfuscation"
PY_A10 = "/usr/bin/python3"; PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
CKPT_A10 = "/root/qwen25_05b"
ENV_A10 = (f"PYTHONPATH={RA10}/src PB_PKG_DIR={RA10}/results/aaai_private_base/private_package/"
           f"gpu_package PB_CKPT_DIR={CKPT_A10} HF_HUB_OFFLINE=1")
IDS = f"{RA10}/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"


def sh(cmd, timeout=3600):
    p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def a10(cmd, timeout=3600):
    return sh(f"ssh -S {CM_A10} {A10} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def tdx(cmd, timeout=600):
    return sh(f"ssh -i {PK} -o StrictHostKeyChecking=no {TDX} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def push_a10(local, remote):
    sh(f"cat {local} | ssh -S {CM_A10} {A10} 'cat > {remote}'")


def push_tdx(local, remote):
    sh(f"cat {local} | ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat > {remote}'")


def pull_a10(remote, local):
    sh(f"ssh -S {CM_A10} {A10} 'cat {remote}' > {local}")


def pull_tdx(remote, local):
    sh(f"ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat {remote}' > {local}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--run-tag", default="A10")
    ap.add_argument("--attest", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.run_tag}-{args.steps}s-{int(time.time())}-{secrets.token_hex(3)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    hmac_key_commitment = hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest()
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head.txt").read_text().strip()
    ids = json.loads((REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json").read_text())
    labels = ids["input_ids"][:args.seq_len]
    binding = {
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
        "code_revision_or_worktree_hash": head, "transport_profile": "direct_a10_tdx_private",
        "compute_host": "alicloud_a10", "data_plane_ip": TDX_PRIV,
        "hmac_key_commitment": hmac_key_commitment, "session_key_derivation": "sha256(session_key||nonce)",
        "debug_false_required": True, "nonce": nonce, "ephemeral_key_pub": hashlib.sha256(nonce.encode()).hexdigest()}

    # ---- provision code + gamma bundle + session config ----
    push_tdx(str(REPO / "scripts/tdx_persistent_service.py"), f"{RTDX}/scripts/tdx_persistent_service.py")
    push_a10(str(REPO / "scripts/h800_direct_runner.py"), f"{RA10}/scripts/h800_direct_runner.py")
    push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": labels,
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/a10_attest",
                "binding_manifest": binding, "counters_out": "/tmp/a10_session_counters.json"}
    Path(OUT / "tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(OUT / "tdx_session.json"), "/tmp/direct_session.json")
    h_sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    Path(OUT / "a10_session.json").write_text(json.dumps(h_sess))
    push_a10(str(OUT / "a10_session.json"), "/tmp/direct_session_a10.json")

    # ---- launch the persistent A10 runner (data plane = A10 <-> TDX PRIVATE) ----
    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/{args.run_tag}_{args.dtype}_{args.steps}step.json"
    a10(f"mkdir -p {RA10}/results/aaai_private_base/alicloud_a10_runs")
    cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/h800_direct_runner.py "
           f"--session /tmp/direct_session_a10.json --steps {args.steps} --seq-len {args.seq_len} "
           f"--lr {args.lr} --momentum {args.momentum} --dtype {args.dtype} --input-ids {IDS} "
           f"--tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx "
           f"--service-cmd \"{service_cmd}\" --out {out_json}")
    print(f"[launch] A10 runner; data plane A10<->TDX PRIVATE {TDX_PRIV} ...")
    t0 = time.time(); rc, o, e = a10(cmd, timeout=3600); wall = time.time() - t0
    print(o[-1500:] if o else "", "\n[stderr]", e[-800:] if rc != 0 else "")

    # ---- effective equivalence on A10 (trusted-eval at final trained state) ----
    equiv_remote = out_json.replace(".json", ".equiv.json")
    eq_cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/d4_trusted_verifier.py "
              f"--masked-logits {out_json.replace('.json','.final_logits.pt')} "
              f"--lora-state {out_json.replace('.json','.adapter.pt')} "
              f"--input-ids {IDS} --seq-len {args.seq_len} --out {equiv_remote}")
    erc, eo, ee = a10(eq_cmd, timeout=1800)
    if erc == 0:
        pull_a10(equiv_remote, str(OUT / f"{args.run_tag}_{args.dtype}_{args.steps}step.equiv.json"))
    else:
        print("[equiv] verifier failed:", ee[-400:])

    # ---- collect ----
    lout = OUT / f"{args.run_tag}_{args.dtype}_{args.steps}step.json"
    pull_a10(out_json, str(lout))
    pull_a10(out_json.replace(".json", ".adapter.pt"), str(OUT / f"{args.run_tag}_{args.dtype}_{args.steps}step.adapter.pt"))
    if args.attest:
        pull_tdx("/tmp/a10_attest/session_attestation.json", str(OUT / f"{args.run_tag}_session_attestation.json"))
    pull_tdx("/tmp/a10_session_counters.json", str(OUT / f"{args.run_tag}_tdx_counters.json"))
    try:
        r = json.loads(lout.read_text())
        print(json.dumps({"run_id": run_id, "transport": "direct_a10_tdx_private", "steps": r["steps_run"],
                          "control_wall_sec": round(wall, 1),
                          "ce_first_last": [r["trajectory"][0]["ce"], r["trajectory"][-1]["ce"]] if r["trajectory"] else None,
                          "mean_step_wall": round(sum(s["timing"]["step_wall_sec"] for s in r["trajectory"])/max(1,len(r["trajectory"])), 3)}, indent=2))
    except Exception as ex:
        print("collect parse error:", ex)


if __name__ == "__main__":
    main()
