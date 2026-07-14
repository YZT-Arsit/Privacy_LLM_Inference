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
import argparse, hashlib, json, os, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
GL = REPO / "results/aaai_private_base/generative_lora"
OUT = GL / "protected_runs"
CM_A10 = os.environ.get("G2_CM_A10", "/tmp/cm-a10")
A10 = os.environ.get("G2_A10", "root@172.30.25.154")
A10_PUB = os.environ.get("G2_A10_PUB", "root@39.107.123.173")
PK = os.environ.get("G2_SSH_KEY", "/Users/Hoshino/Downloads/passkey.pem")
KH = os.environ.get("G2_KNOWN_HOSTS", "/Users/Hoshino/.ssh/known_hosts")
TDX = os.environ.get("G2_TDX_PUB", "root@39.96.43.122")
TDX_PRIV = os.environ.get("G2_TDX_PRIV", "172.30.25.153")
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
ALL_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


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
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--targets", default=ALL_TARGETS)
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
    ap.add_argument("--output-root", default="",
                    help="append-only path relative to generative_lora; empty preserves the historical layout")
    args = ap.parse_args()
    if not (args.weight_decay >= 0.0):
        raise ValueError("--weight-decay must be finite and non-negative")
    targets = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    allowed_targets = set(ALL_TARGETS.split(","))
    if not targets or len(set(targets)) != len(targets) or set(targets) - allowed_targets:
        raise ValueError("--targets must be a non-empty duplicate-free subset of " + ALL_TARGETS)
    local_out = (GL / args.output_root / "protected_runs") if args.output_root else OUT
    local_gen = (GL / args.output_root / "generations") if args.output_root else (GL / "generations")
    local_out.mkdir(parents=True, exist_ok=True); local_gen.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or f"e2e_{args.profile}_s{args.seed}_st{args.max_steps}"
    run_id = f"{tag}-{int(time.time())}-{secrets.token_hex(3)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    hmac_key_commitment = hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest()
    head = sh(f"git -C {REPO} rev-parse HEAD", timeout=60)[1].strip()
    bound_sources = [
        REPO / "scripts/a10_batch_runner.py",
        REPO / "scripts/tdx_persistent_service.py",
        REPO / "scripts/batch_dataplane.py",
        REPO / "scripts/h800_unified_worker.py",
        REPO / "scripts/h800_d4_worker.py",
        REPO / "scripts/generative_lora/protected_generate.py",
        REPO / "scripts/generative_lora/gate_e2e_protected.py",
    ]
    source_hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in bound_sources}
    source_bundle_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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

    wd_label = format(float(args.weight_decay), ".12g").replace(".", "p")
    optimizer_profile = f"{args.profile}_adamw_wd{wd_label}"
    binding = {"d4_run_id": run_id, "package_root_hash": PKG_ROOT, "seed": args.seed,
               "optimizer_profile": optimizer_profile, "effective_weight_decay": float(args.weight_decay),
               "execution_profile": "paper_safe",
               "transport_profile": "direct_a10_tdx_private", "compute_host": "alicloud_a10",
               "data_plane_ip": TDX_PRIV, "source_revision": head,
               "code_revision_or_worktree_hash": source_bundle_hash,
               "source_hashes": source_hashes,
               "hmac_key_commitment": hmac_key_commitment, "nonce": nonce, "debug_false_required": True,
               "tokenizer_hash": TOK_HASH, "dataset_id": "e2e_nlg", "task": "clm"}
    binding["lora_targets"] = list(targets)
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": [0],
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/e2e_attest",
                "binding_manifest": binding, "counters_out": "/tmp/e2e_batch_counters.json",
                "package_root_hash": PKG_ROOT, "ckpt_dir": "/tmp/l12_ckpt_e2e",
                "model_config_hash": MODEL_CFG_HASH, "service_hash": svc_hash,
                "dataset_id": "e2e_nlg", "task": "clm", "template_hash": tmpl_hash,
                "lora_targets": list(targets),
                "label_schema_hash": lsch_hash, "tokenizer_hash": TOK_HASH,
                "batch_label_tables": {"train": rlt}, "batch_schedules": {"train": rsched},
                "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2,
                              "hidden_size": 896, "intermediate_size": 4864}}
    (local_out / f"{tag}.tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(local_out / f"{tag}.tdx_session.json"), "/tmp/direct_session.json")
    a10sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    (local_out / f"{tag}.a10_session.json").write_text(json.dumps(a10sess))
    push_a10(str(local_out / f"{tag}.a10_session.json"), "/tmp/e2e_session_a10.json")

    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    remote_root = f"{RA10}/results/aaai_private_base/generative_lora/{args.output_root}" if args.output_root else f"{RA10}/results/aaai_private_base/generative_lora"
    remote_runs = f"{remote_root}/protected_runs"; remote_gens = f"{remote_root}/generations"
    out_json = f"{remote_runs}/{tag}.json"
    adapter_remote = out_json.replace(".json", ".adapter.pt")
    a10(f"mkdir -p {remote_runs} {remote_gens}")
    result = {"tag": tag, "run_id": run_id, "profile": args.profile, "seed": args.seed,
              "lr": args.lr, "weight_decay": float(args.weight_decay),
              "optimizer_profile": optimizer_profile,
              "max_steps": args.max_steps, "attest": bool(args.attest),
              "lora_targets": list(targets),
              "endpoints": {"a10_pub": A10_PUB, "tdx_pub": TDX, "tdx_priv": TDX_PRIV}}

    # ---------------- build robust A10-side nohup driver (survives control-master drops) ----------------
    train_data = f"{DATA}/e2e_train_a10.pt"; sched_a10 = f"{DATA}/e2e_schedule_s{args.seed}.json"
    gen_out = f"{remote_gens}/g2_protected_{tag}.jsonl"
    adapter_use = args.adapter_path or adapter_remote
    req = " --require-attestation" if args.attest else ""
    train_line = "" if args.skip_train else (
        f"{ENV_A10} {PY_A10} scripts/a10_batch_runner.py --session /tmp/e2e_session_a10.json "
        f"--profile {args.profile} --task clm --dataset-id e2e_nlg --train-split train "
        f"--eval-split validation --eval-max 0 --train-data {train_data} --schedule {sched_a10} "
        f"--max-steps {args.max_steps} --lr {args.lr} --weight-decay {args.weight_decay} --seed {args.seed} --max-seq 256 "
        f"--targets {','.join(targets)} "
        f"--template-hash {tmpl_hash} --tokenizer-hash {TOK_HASH} --label-schema-hash {lsch_hash} "
        f"--tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx --service-cmd \"{service_cmd}\" "
        f"--out {out_json} --adapter-id {run_id}{req} || {{ echo G2_TRAIN_FAILED; exit 1; }}")
    gen_line = (
        f"{ENV_A10} {PY_A10} scripts/generative_lora/protected_generate.py --session /tmp/e2e_session_a10.json "
        f"--tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx --service-cmd \"{service_cmd}\" "
        f"--adapter {adapter_use} --tok /root/genlora_tok --gen-in {DATA}/{args.gen_in} --gen-out {gen_out} "
        f"--gen-max {args.gen_max} --max-new {args.max_new} --cell G2_protected_{args.profile} --dtype fp32{req} "
        f"|| {{ echo G2_GEN_FAILED; exit 1; }}")
    freeze_train_counters = (
        f"ssh -i /root/.ssh/a10_to_tdx -o BatchMode=yes root@{TDX_PRIV} "
        "'cp /tmp/e2e_batch_counters.json /tmp/e2e_training_counters.frozen.json' "
        "|| { echo G2_COUNTER_FREEZE_FAILED; exit 1; }")
    driver = ("#!/bin/bash\nset -e\ncd " + RA10 + "\n"
              f"mkdir -p {remote_gens}\n"
              + (train_line + "\n" + freeze_train_counters + "\n" if train_line else "")
              + gen_line + "\necho G2_ALL_DONE\n")
    dpath = local_out / f"{tag}.driver.sh"; dpath.write_text(driver)
    push_a10(str(dpath), f"{RA10}/g2_driver_{tag}.sh")
    logf = f"{RA10}/g2_{tag}.log"
    rc, launch_out, launch_err = a10(
        f"nohup bash {RA10}/g2_driver_{tag}.sh > {logf} 2>&1 < /dev/null & echo $!", timeout=60)
    if rc != 0 or not launch_out.strip().splitlines():
        raise RuntimeError(f"failed to launch durable G2 driver: rc={rc} stderr={launch_err}")
    remote_pid = int(launch_out.strip().splitlines()[-1])
    launched = {
        **result, "status": "running", "host": A10_PUB, "pid": remote_pid,
        "log_path": logf, "driver_path": f"{RA10}/g2_driver_{tag}.sh",
        "source_revision": head, "source_bundle_hash": source_bundle_hash,
        "start_unix": time.time(),
    }
    (local_out / f"{tag}.launch.json").write_text(json.dumps(launched, indent=2))
    print(f"[g2] launched nohup driver for {tag}, pid {remote_pid}; polling {logf}")

    # poll until G2_ALL_DONE / *_FAILED (robust to master drops: reconnect + re-poll)
    import subprocess as _sp
    t0 = time.time(); done = False
    while time.time() - t0 < args.timeout:
        rc, o, _ = sh(f"ssh -S {CM_A10} {A10_PUB} 'tail -3 {logf} 2>/dev/null; "
                      f"grep -qE \"G2_ALL_DONE|G2_TRAIN_FAILED|G2_GEN_FAILED|G2_COUNTER_FREEZE_FAILED\" "
                      f"{logf} && echo POLL_DONE'", timeout=60)
        if "POLL_DONE" in o:
            done = True; print(o.strip()[-500:]); break
        time.sleep(20)
    result["g2_wall_sec"] = round(time.time() - t0, 1); result["completed"] = done
    # pull everything produced
    pull_a10(out_json, str(local_out / f"{tag}.json"))
    pull_a10(adapter_remote, str(local_out / f"{tag}.adapter.pt"))
    pull_a10(gen_out, str(local_gen / f"g2_protected_{tag}.jsonl"))
    pull_a10(gen_out + ".profile.json", str(local_gen / f"g2_protected_{tag}.jsonl.profile.json"))
    if not args.skip_train:
        pull_tdx("/tmp/e2e_training_counters.frozen.json",
                 str(local_out / f"{tag}.training_tdx_counters.json"))
    pull_tdx("/tmp/e2e_batch_counters.json", str(local_out / f"{tag}.generation_tdx_counters.json"))
    if args.attest:
        pull_tdx("/tmp/e2e_attest/session_attestation.json", str(local_out / f"{tag}.attestation.json"))
    (local_out / f"{tag}.result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
