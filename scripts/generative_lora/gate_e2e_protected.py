"""G2 — exact protected LoRA on E2E NLG: real A10 GPU + real Intel TDX. Control-plane (Mac).

Reuses the frozen batched protected protocol (a10_batch_runner clm training over the immutable
transformed package + TDX trusted AdamW/CE/dlogits), then runs PROTECTED GENERATION (protected_generate
+ TDX decode_argmax boundary) with the trained masked adapter. Data plane = A10<->TDX PRIVATE VPC.

Endpoints (post VM-rebuild): A10 pub 39.107.123.173 (priv 172.30.25.154), TDX pub 39.96.43.122
(priv 172.30.25.153). Same passkey. Same private data-plane IPs as the frozen runs.

Usage:
  python3 scripts/generative_lora/gate_e2e_protected.py --profile L12 --seed 1234 --lr 1e-4 \
      --max-steps 250 --gen-max 200 --attest
"""
from __future__ import annotations
import argparse, hashlib, json, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
GL = REPO / "results/aaai_private_base/generative_lora"
OUT = GL / "protected_runs"
CM_A10 = "/tmp/cm-a10"; A10 = "root@172.30.25.154"; A10_PUB = "root@39.107.123.173"
PK = "/Users/Hoshino/Downloads/passkey.pem"; KH = "/Users/Hoshino/.ssh/known_hosts"
TDX = "root@39.96.43.122"; TDX_PRIV = "172.30.25.153"
RA10 = "/root/pllo_pb"; RTDX = "/root/privacy_llm_obfuscation"
PY_A10 = "/usr/bin/python3"; PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
CKPT_A10 = "/root/qwen25_05b"
PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
TOK_HASH = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
MODEL_CFG_HASH = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
ENV_A10 = (f"PYTHONPATH={RA10}/src PB_PKG_DIR={RA10}/results/aaai_private_base/private_package/"
           f"gpu_package PB_CKPT_DIR={CKPT_A10} HF_HUB_OFFLINE=1")
DATA = f"{RA10}/results/aaai_private_base/generative_lora/data"
E2E_TEMPLATE = "User:\nGenerate a natural-language description for the following restaurant attributes:\n{mr}\n\nAssistant:\n"
LABEL_SCHEMA = "causal_lm_shifted_ignore_index_-100"


def sh(cmd, timeout=7200):
    p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def a10(cmd, timeout=7200):
    return sh(f"ssh -S {CM_A10} {A10_PUB} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def tdx(cmd, timeout=600):
    return sh(f"ssh -i {PK} -o UserKnownHostsFile={KH} -o StrictHostKeyChecking=accept-new {TDX} "
              + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def push_a10(l, r): sh(f"cat {l} | ssh -S {CM_A10} {A10_PUB} 'cat > {r}'")
def push_tdx(l, r): sh(f"cat {l} | ssh -i {PK} -o UserKnownHostsFile={KH} -o StrictHostKeyChecking=accept-new {TDX} 'cat > {r}'")
def pull_a10(r, l): sh(f"ssh -S {CM_A10} {A10_PUB} 'cat {r}' > {l}")
def pull_tdx(r, l): sh(f"ssh -i {PK} -o UserKnownHostsFile={KH} -o StrictHostKeyChecking=accept-new {TDX} 'cat {r}' > {l}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="L12", choices=["L0", "L5", "L12"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--gen-in", default="e2e_test_gen.json")
    ap.add_argument("--gen-max", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--attest", action="store_true")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--skip-train", action="store_true")   # reuse an existing adapter
    ap.add_argument("--adapter-path", default="")
    ap.add_argument("--sync-code", action="store_true")
    ap.add_argument("--timeout", type=int, default=10800)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or f"e2e_{args.profile}_s{args.seed}_st{args.max_steps}"
    run_id = f"{tag}-{int(time.time())}-{secrets.token_hex(3)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    hmac_key_commitment = hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest()
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head.txt").read_text().strip()
    tmpl_hash = hashlib.sha256(E2E_TEMPLATE.encode()).hexdigest()
    lsch_hash = hashlib.sha256(LABEL_SCHEMA.encode()).hexdigest()
    svc_hash = hashlib.sha256((REPO / "scripts/tdx_persistent_service.py").read_bytes()).hexdigest()

    if args.sync_code:
        for f in ["tdx_persistent_service.py", "tdx_adamw_protocol.py", "batch_dataplane.py"]:
            push_tdx(str(REPO / "scripts" / f), f"{RTDX}/scripts/{f}")
        for f in ["h800_direct_runner.py", "a10_batch_runner.py", "tdx_adamw_protocol.py", "batch_dataplane.py",
                  "h800_unified_worker.py", "h800_d4_worker.py"]:
            push_a10(str(REPO / "scripts" / f), f"{RA10}/scripts/{f}")
        push_a10(str(REPO / "scripts/generative_lora/protected_generate.py"),
                 f"{RA10}/scripts/generative_lora/protected_generate.py")
        print("[sync] pushed current code to A10 + TDX")

    push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    # train label table -> TDX (trusted side only)
    lt = GL / "data/e2e_train_labels.pt"; rlt = "/tmp/e2e_train_labels.pt"; push_tdx(str(lt), rlt)
    local_sched = GL / f"data/e2e_schedule_s{args.seed}.json"
    rsched = f"/tmp/e2e_schedule_s{args.seed}.json"; push_tdx(str(local_sched), rsched)

    binding = {"d4_run_id": run_id, "package_root_hash": PKG_ROOT, "seed": args.seed,
               "optimizer_profile": f"{args.profile}_adamw", "execution_profile": "paper_safe",
               "transport_profile": "direct_a10_tdx_private", "compute_host": "alicloud_a10",
               "data_plane_ip": TDX_PRIV, "code_revision_or_worktree_hash": head,
               "hmac_key_commitment": hmac_key_commitment, "nonce": nonce, "debug_false_required": True,
               "tokenizer_hash": TOK_HASH, "dataset_id": "e2e_nlg", "task": "clm"}
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": [0],
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/e2e_attest",
                "binding_manifest": binding, "counters_out": "/tmp/e2e_batch_counters.json",
                "package_root_hash": PKG_ROOT, "ckpt_dir": "/tmp/l12_ckpt_e2e",
                "model_config_hash": MODEL_CFG_HASH, "service_hash": svc_hash,
                "dataset_id": "e2e_nlg", "task": "clm", "template_hash": tmpl_hash,
                "label_schema_hash": lsch_hash, "tokenizer_hash": TOK_HASH,
                "batch_label_tables": {"train": rlt}, "batch_schedules": {"train": rsched},
                "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2,
                              "hidden_size": 896, "intermediate_size": 4864}}
    (OUT / f"{tag}.tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(OUT / f"{tag}.tdx_session.json"), "/tmp/direct_session.json")
    a10sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    (OUT / f"{tag}.a10_session.json").write_text(json.dumps(a10sess))
    push_a10(str(OUT / f"{tag}.a10_session.json"), "/tmp/e2e_session_a10.json")

    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RA10}/results/aaai_private_base/generative_lora/protected_runs/{tag}.json"
    adapter_remote = out_json.replace(".json", ".adapter.pt")
    a10(f"mkdir -p {RA10}/results/aaai_private_base/generative_lora/protected_runs")
    result = {"tag": tag, "run_id": run_id, "profile": args.profile, "seed": args.seed,
              "lr": args.lr, "max_steps": args.max_steps, "attest": bool(args.attest),
              "endpoints": {"a10_pub": A10_PUB, "tdx_pub": TDX, "tdx_priv": TDX_PRIV}}

    # ---------------- TRAIN (protected) ----------------
    if not args.skip_train:
        train_data = f"{DATA}/e2e_train_a10.pt"; sched_a10 = f"{DATA}/e2e_schedule_s{args.seed}.json"
        cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/a10_batch_runner.py "
               f"--session /tmp/e2e_session_a10.json --profile {args.profile} --task clm "
               f"--dataset-id e2e_nlg --train-split train --eval-split validation --eval-max 0 "
               f"--train-data {train_data} --schedule {sched_a10} --max-steps {args.max_steps} "
               f"--lr {args.lr} --seed {args.seed} --max-seq 256 "
               f"--template-hash {tmpl_hash} --tokenizer-hash {TOK_HASH} --label-schema-hash {lsch_hash} "
               f"--tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx --service-cmd \"{service_cmd}\" "
               f"--out {out_json} --adapter-id {run_id}"
               + (" --require-attestation" if args.attest else ""))
        print(f"[train] {tag} profile={args.profile} steps={args.max_steps} lr={args.lr}; A10<->TDX PRIVATE {TDX_PRIV}")
        t0 = time.time(); rc, o, e = a10(cmd, timeout=args.timeout); wall = time.time() - t0
        print(o[-1500:] if o else "", "\n[train stderr]", e[-1500:] if rc != 0 else "")
        result["train"] = {"rc": rc, "wall_sec": round(wall, 1)}
        pull_a10(out_json, str(OUT / f"{tag}.json"))
        pull_a10(adapter_remote, str(OUT / f"{tag}.adapter.pt"))
        pull_tdx("/tmp/e2e_batch_counters.json", str(OUT / f"{tag}.tdx_counters.json"))
        if args.attest:
            pull_tdx("/tmp/e2e_attest/session_attestation.json", str(OUT / f"{tag}.attestation.json"))
        if rc != 0:
            (OUT / f"{tag}.result.json").write_text(json.dumps(result, indent=2)); print("TRAIN FAILED"); return
    adapter_use = args.adapter_path or adapter_remote

    # ---------------- PROTECTED GENERATION ----------------
    gen_out = f"{RA10}/results/aaai_private_base/generative_lora/generations/g2_protected_{tag}.jsonl"
    a10(f"mkdir -p {RA10}/results/aaai_private_base/generative_lora/generations")
    gcmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/generative_lora/protected_generate.py "
            f"--session /tmp/e2e_session_a10.json --tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx "
            f"--service-cmd \"{service_cmd}\" --adapter {adapter_use} --tok /root/genlora_tok "
            f"--gen-in {DATA}/{args.gen_in} --gen-out {gen_out} --gen-max {args.gen_max} "
            f"--max-new {args.max_new} --cell G2_protected_{args.profile} --dtype fp32"
            + (" --require-attestation" if args.attest else ""))
    print(f"[protected-gen] {tag} gen_max={args.gen_max}")
    t0 = time.time(); rc, o, e = a10(gcmd, timeout=args.timeout); wall = time.time() - t0
    print(o[-1500:] if o else "", "\n[gen stderr]", e[-1500:] if rc != 0 else "")
    result["protected_gen"] = {"rc": rc, "wall_sec": round(wall, 1)}
    pull_a10(gen_out, str(GL / f"generations/g2_protected_{tag}.jsonl"))
    pull_a10(gen_out + ".profile.json", str(GL / f"generations/g2_protected_{tag}.jsonl.profile.json"))
    (OUT / f"{tag}.result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
