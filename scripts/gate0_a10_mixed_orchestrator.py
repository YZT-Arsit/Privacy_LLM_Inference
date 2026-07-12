"""A10+TDX control-plane orchestrator for L12 MIXED-PRECISION (BF16 compute / FP32 master).

Drives scripts/a10_mixed_runner.py for a given --seed/--steps; data plane A10<->TDX PRIVATE VPC.
Pulls the per-step transport log and runs scripts/compare_l5_l12.py (plaintext-AdamW reference oracle)
on the Mac. Supports the restart-continuity test (--checkpoint-at / --restore-run).
"""
from __future__ import annotations
import argparse, hashlib, json, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/alicloud_a10_runs/l12_mixed"
CM_A10 = "/tmp/cm-a10"; A10 = "root@172.30.25.154"
PK = "/Users/Hoshino/Downloads/passkey.pem"; TDX = "root@39.96.4.252"
TDX_PRIV = "172.30.25.153"
RA10 = "/root/pllo_pb"; RTDX = "/root/privacy_llm_obfuscation"
PY_A10 = "/usr/bin/python3"; PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
CKPT_A10 = "/root/qwen25_05b"
PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
ENV_A10 = (f"PYTHONPATH={RA10}/src PB_PKG_DIR={RA10}/results/aaai_private_base/private_package/"
           f"gpu_package PB_CKPT_DIR={CKPT_A10} HF_HUB_OFFLINE=1")
IDS = f"{RA10}/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"


def sh(cmd, timeout=3600):
    p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def a10(cmd, timeout=3600):
    return sh(f"ssh -S {CM_A10} {A10} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def push_a10(local, remote): sh(f"cat {local} | ssh -S {CM_A10} {A10} 'cat > {remote}'")
def push_tdx(local, remote): sh(f"cat {local} | ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat > {remote}'")
def pull_a10(remote, local): sh(f"ssh -S {CM_A10} {A10} 'cat {remote}' > {local}")
def pull_tdx(remote, local): sh(f"ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat {remote}' > {local}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-tag", default="L12MIX")
    ap.add_argument("--attest", action="store_true")
    ap.add_argument("--checkpoint-at", type=int, default=-1)
    ap.add_argument("--restore-first", action="store_true")
    ap.add_argument("--ckpt-path", default=""); ap.add_argument("--expected-binding", default="")
    ap.add_argument("--reuse-run-id", default="")     # for restart test: keep run_id + session key
    ap.add_argument("--reuse-key", default="")
    ap.add_argument("--skip-compare", action="store_true")  # long runs: skip giant per-step tlog pull+compare
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tag = f"{args.run_tag}_s{args.seed}_{args.steps}step"
    if args.reuse_run_id:
        run_id = args.reuse_run_id; session_key = bytes.fromhex(args.reuse_key)
    else:
        run_id = f"{tag}-{int(time.time())}-{secrets.token_hex(3)}"
        session_key = secrets.token_bytes(32)
    nonce = secrets.token_hex(16)
    hmac_key_commitment = hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest()
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head.txt").read_text().strip()
    ids = json.loads((REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json").read_text())
    labels = ids["input_ids"][:args.seq_len]
    binding = {
        "d4_run_id": run_id, "package_root_hash": PKG_ROOT,
        "tdx_service_hash": hashlib.sha256((REPO/"scripts/tdx_persistent_service.py").read_bytes()).hexdigest(),
        "tokenizer_hash": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "execution_profile": "paper_safe", "optimizer_profile": "L12_mixed_bf16compute_fp32master",
        "code_revision_or_worktree_hash": head, "transport_profile": "direct_a10_tdx_private",
        "compute_host": "alicloud_a10", "data_plane_ip": TDX_PRIV, "seed": args.seed,
        "hmac_key_commitment": hmac_key_commitment, "debug_false_required": True, "nonce": nonce,
        "ephemeral_key_pub": hashlib.sha256(nonce.encode()).hexdigest()}

    push_tdx(str(REPO / "scripts/tdx_persistent_service.py"), f"{RTDX}/scripts/tdx_persistent_service.py")
    push_tdx(str(REPO / "scripts/tdx_adamw_protocol.py"), f"{RTDX}/scripts/tdx_adamw_protocol.py")
    push_a10(str(REPO / "scripts/h800_direct_runner.py"), f"{RA10}/scripts/h800_direct_runner.py")
    push_a10(str(REPO / "scripts/a10_mixed_runner.py"), f"{RA10}/scripts/a10_mixed_runner.py")
    push_a10(str(REPO / "scripts/tdx_adamw_protocol.py"), f"{RA10}/scripts/tdx_adamw_protocol.py")
    push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": labels,
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/a10_attest",
                "binding_manifest": binding, "counters_out": "/tmp/a10_session_counters.json",
                "package_root_hash": PKG_ROOT, "ckpt_dir": "/tmp/l12_ckpt",
                "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2,
                              "hidden_size": 896, "intermediate_size": 4864}}
    Path(OUT / f"{tag}.tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(OUT / f"{tag}.tdx_session.json"), "/tmp/direct_session.json")
    h_sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    Path(OUT / f"{tag}.a10_session.json").write_text(json.dumps(h_sess))
    push_a10(str(OUT / f"{tag}.a10_session.json"), "/tmp/direct_session_a10.json")

    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/l12_mixed/{tag}.json"
    tlog_remote = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/l12_mixed/{tag}.tlog.pt"
    a10(f"mkdir -p {RA10}/results/aaai_private_base/alicloud_a10_runs/l12_mixed")
    extra = ""
    if args.checkpoint_at >= 0: extra += f" --checkpoint-at {args.checkpoint_at}"
    if args.restore_first:
        extra += f" --restore-first --ckpt-path {args.ckpt_path} --expected-binding '{args.expected_binding}'"
    gpu_state = f"/tmp/gpustate_{run_id}.pt"
    logt = "" if args.skip_compare else f" --log-transport {tlog_remote}"
    cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/a10_mixed_runner.py "
           f"--session /tmp/direct_session_a10.json --steps {args.steps} --seq-len {args.seq_len} "
           f"--lr {args.lr} --seed {args.seed} --input-ids {IDS} --tdx root@{TDX_PRIV} "
           f"--key /root/.ssh/a10_to_tdx --service-cmd \"{service_cmd}\" --out {out_json} "
           f"--adapter-id {run_id}{logt} --gpu-state-path {gpu_state}{extra}")
    print(f"[launch] {tag}; data plane A10<->TDX PRIVATE {TDX_PRIV} ...")
    t0 = time.time(); rc, o, e = a10(cmd, timeout=3600); wall = time.time() - t0
    print(o[-1800:] if o else "", "\n[stderr]", e[-1000:] if rc != 0 else "")

    # effective equivalence on A10
    equiv_remote = out_json.replace(".json", ".equiv.json")
    eq_cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/d4_trusted_verifier.py "
              f"--masked-logits {out_json.replace('.json','.final_logits.pt')} "
              f"--lora-state {out_json.replace('.json','.adapter.pt')} "
              f"--input-ids {IDS} --seq-len {args.seq_len} --out {equiv_remote}")
    erc, eo, ee = a10(eq_cmd, timeout=1800)
    if erc == 0: pull_a10(equiv_remote, str(OUT / f"{tag}.equiv.json"))
    else: print("[equiv] failed:", ee[-400:])

    pull_a10(out_json, str(OUT / f"{tag}.json"))
    pull_a10(out_json.replace(".json", ".adapter.pt"), str(OUT / f"{tag}.adapter.pt"))  # small; for restart cmp
    if args.attest: pull_tdx("/tmp/a10_attest/session_attestation.json", str(OUT / f"{tag}.attestation.json"))
    pull_tdx("/tmp/a10_session_counters.json", str(OUT / f"{tag}.tdx_counters.json"))

    if not args.skip_compare:
        pull_a10(tlog_remote, str(OUT / f"{tag}.tlog.pt"))
        cmp_out = OUT / f"{tag}.compare.json"
        crc, co, ce = sh(f"cd {REPO} && python3 scripts/compare_l5_l12.py "
                         f"--tlog {OUT / (tag + '.tlog.pt')} --run-json {OUT / (tag + '.json')} "
                         f"--seed {args.seed} --out {cmp_out}")
        print("[compare]", co.strip() if crc == 0 else ce[-600:])
        sh(f"rm -f {OUT / (tag + '.tlog.pt')}")   # drop the large tlog after compare
    print(json.dumps({"run_id": run_id, "session_key_hex": session_key.hex(), "tag": tag,
                      "control_wall_sec": round(wall, 1), "checkpoint_at": args.checkpoint_at}, indent=2))


if __name__ == "__main__":
    main()
