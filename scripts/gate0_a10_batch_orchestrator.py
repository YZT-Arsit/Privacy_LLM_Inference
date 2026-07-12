"""Control-plane orchestrator for the batched utility data plane (A10 compute + TDX trusted).

Pushes the private label table + deterministic batch schedule to TDX, the A10 input artifact +
runner + shared modules to A10, launches scripts/a10_batch_runner.py for a chosen
profile/task/dataset, and collects results. Data plane runs A10<->TDX over the PRIVATE VPC.

Usage (one-batch SST-2 L12 gate):
  python3 scripts/gate0_a10_batch_orchestrator.py --dataset sst2 --task cls --profile L12 \
      --seed 1234 --max-steps 1 --eval-max 64
"""
from __future__ import annotations
import argparse, hashlib, json, secrets, subprocess, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane"
CM_A10 = "/tmp/cm-a10"; A10 = "root@172.30.25.154"
PK = "/Users/Hoshino/Downloads/passkey.pem"; TDX = "root@39.96.4.252"; TDX_PRIV = "172.30.25.153"
RA10 = "/root/pllo_pb"; RTDX = "/root/privacy_llm_obfuscation"
PY_A10 = "/usr/bin/python3"; PY_TDX = "/root/miniconda3/envs/tdx310/bin/python"
CKPT_A10 = "/root/qwen25_05b"
PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
TOK_HASH = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
MODEL_CFG_HASH = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
ENV_A10 = (f"PYTHONPATH={RA10}/src PB_PKG_DIR={RA10}/results/aaai_private_base/private_package/"
           f"gpu_package PB_CKPT_DIR={CKPT_A10} HF_HUB_OFFLINE=1")
TOKD = REPO / "results/aaai_private_base/datasets/tokenized"

TEMPLATES = {"sst2": "Review: {sentence}\nSentiment:", "gsm8k": "Question: {q}\nAnswer:"}
LABEL_SCHEMA = {"sst2": json.dumps({"0": [8225, "negative"], "1": [6785, "positive"]}, sort_keys=True),
                "gsm8k": "causal_lm_shifted_ignore_index_-100"}


def sh(cmd, timeout=7200):
    p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def a10(cmd, timeout=7200):
    return sh(f"ssh -S {CM_A10} {A10} " + "'" + cmd.replace("'", "'\\''") + "'", timeout)


def push_a10(l, r): sh(f"cat {l} | ssh -S {CM_A10} {A10} 'cat > {r}'")
def push_tdx(l, r): sh(f"cat {l} | ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat > {r}'")
def pull_a10(r, l): sh(f"ssh -S {CM_A10} {A10} 'cat {r}' > {l}")
def pull_tdx(r, l): sh(f"ssh -i {PK} -o StrictHostKeyChecking=no {TDX} 'cat {r}' > {l}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sst2", "gsm8k"])
    ap.add_argument("--task", required=True, choices=["cls", "clm"])
    ap.add_argument("--profile", required=True, choices=["L0", "L5", "L12"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--eval-max", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="")
    ap.add_argument("--attest", action="store_true")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--checkpoint-at", type=int, default=-1)
    ap.add_argument("--ledger-checkpoint-at", type=int, default=-1)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--restore-first", action="store_true")
    ap.add_argument("--reuse-run-id", default=""); ap.add_argument("--reuse-key", default="")
    ap.add_argument("--ckpt-path", default=""); ap.add_argument("--expected-binding", default="")
    ap.add_argument("--ledger-path", default="")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = args.dataset
    eval_split = args.eval_split or ("dev" if ds == "sst2" else "test")
    max_seq = 128 if ds == "sst2" else 512
    tag = args.run_tag or f"{ds}_{args.profile}_s{args.seed}_{args.max_steps}b"
    if args.reuse_run_id:
        run_id = args.reuse_run_id; session_key = bytes.fromhex(args.reuse_key)
    else:
        run_id = f"{tag}-{int(time.time())}-{secrets.token_hex(3)}"; session_key = secrets.token_bytes(32)
    nonce = secrets.token_hex(16)
    hmac_key_commitment = hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest()
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head.txt").read_text().strip()
    tmpl_hash = hashlib.sha256(TEMPLATES[ds].encode()).hexdigest()
    lsch_hash = hashlib.sha256(LABEL_SCHEMA[ds].encode()).hexdigest()
    svc_hash = hashlib.sha256((REPO / "scripts/tdx_persistent_service.py").read_bytes()).hexdigest()

    # push code + shared modules
    for f in ["tdx_persistent_service.py", "tdx_adamw_protocol.py", "batch_dataplane.py"]:
        push_tdx(str(REPO / "scripts" / f), f"{RTDX}/scripts/{f}")
    for f in ["h800_direct_runner.py", "a10_batch_runner.py", "tdx_adamw_protocol.py", "batch_dataplane.py"]:
        push_a10(str(REPO / "scripts" / f), f"{RA10}/scripts/{f}")
    # gamma bundle for the enclave AdamW
    push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    # push label tables (TRUSTED side only) + schedule to TDX
    label_tables = {}
    for split in ({"train", eval_split}):
        lt = TOKD / f"{ds}_{split}_labels.pt"
        rlt = f"/tmp/{ds}_{split}_labels.pt"; push_tdx(str(lt), rlt); label_tables[split] = rlt
    rsched = f"/tmp/{ds}_schedule_s{args.seed}.json"
    push_tdx(str(TOKD / f"{ds}_schedule_s{args.seed}.json"), rsched)
    schedules = {"train": rsched}

    binding = {"d4_run_id": run_id, "package_root_hash": PKG_ROOT, "seed": args.seed,
               "optimizer_profile": f"{args.profile}_adamw", "execution_profile": "paper_safe",
               "transport_profile": "direct_a10_tdx_private", "compute_host": "alicloud_a10",
               "data_plane_ip": TDX_PRIV, "code_revision_or_worktree_hash": head,
               "hmac_key_commitment": hmac_key_commitment, "nonce": nonce, "debug_false_required": True,
               "tokenizer_hash": TOK_HASH, "dataset_id": ds, "task": args.task}
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": [0],
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
                "attest": bool(args.attest), "attest_out": "/tmp/a10_attest",
                "binding_manifest": binding, "counters_out": "/tmp/a10_batch_counters.json",
                "package_root_hash": PKG_ROOT, "ckpt_dir": "/tmp/l12_ckpt",
                "model_config_hash": MODEL_CFG_HASH, "service_hash": svc_hash,
                "dataset_id": ds, "task": args.task, "template_hash": tmpl_hash,
                "label_schema_hash": lsch_hash, "tokenizer_hash": TOK_HASH,
                "batch_label_tables": label_tables, "batch_schedules": schedules,
                "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2,
                              "hidden_size": 896, "intermediate_size": 4864}}
    (OUT / f"{tag}.tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    push_tdx(str(OUT / f"{tag}.tdx_session.json"), "/tmp/direct_session.json")
    (OUT / f"{tag}.a10_session.json").write_text(json.dumps({"session_key_hex": session_key.hex(), "run_id": run_id}))
    push_a10(str(OUT / f"{tag}.a10_session.json"), "/tmp/batch_session_a10.json")

    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/utility_dataplane/{tag}.json"
    a10(f"mkdir -p {RA10}/results/aaai_private_base/alicloud_a10_runs/utility_dataplane")
    train_data = f"{RA10}/results/aaai_private_base/datasets/tokenized/{ds}_{args.train_split}_a10.pt"
    eval_data = f"{RA10}/results/aaai_private_base/datasets/tokenized/{ds}_{eval_split}_a10.pt"
    sched_a10 = f"{RA10}/results/aaai_private_base/datasets/tokenized/{ds}_schedule_s{args.seed}.json"
    extra = ""
    if args.profile != "L0":
        extra += f" --train-data {train_data} --schedule {sched_a10} --max-steps {args.max_steps}"
    if args.checkpoint_at >= 0: extra += f" --checkpoint-at {args.checkpoint_at} --gpu-state-path /tmp/gpustate_{run_id}.pt"
    if args.ledger_checkpoint_at >= 0: extra += f" --ledger-checkpoint-at {args.ledger_checkpoint_at}"
    if args.start_step > 0: extra += f" --start-step {args.start_step}"
    if args.restore_first:
        extra += (f" --restore-first --ckpt-path {args.ckpt_path} --expected-binding '{args.expected_binding}'"
                  f" --gpu-state-path /tmp/gpustate_{run_id}.pt")
        if args.ledger_path: extra += f" --ledger-path {args.ledger_path}"
    cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/a10_batch_runner.py "
           f"--session /tmp/batch_session_a10.json --profile {args.profile} --task {args.task} "
           f"--dataset-id {ds} --train-split {args.train_split} --eval-data {eval_data} "
           f"--eval-split {eval_split} --eval-max {args.eval_max} --lr "
           f"{5e-4 if ds == 'sst2' else 2e-4} --seed {args.seed} --max-seq {max_seq} "
           f"--template-hash {tmpl_hash} --tokenizer-hash {TOK_HASH} --label-schema-hash {lsch_hash} "
           f"--tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx --service-cmd \"{service_cmd}\" "
           f"--out {out_json} --adapter-id {run_id}{extra}")
    print(f"[launch] {tag} profile={args.profile} steps={args.max_steps} ds={ds}; A10<->TDX PRIVATE {TDX_PRIV}")
    t0 = time.time(); rc, o, e = a10(cmd, timeout=args.timeout); wall = time.time() - t0
    print(o[-2500:] if o else "", "\n[stderr]", e[-1200:] if rc != 0 else "")
    pull_a10(out_json, str(OUT / f"{tag}.json"))
    for suffix in [".adapter.pt", ".preds.pt"]:
        pull_a10(out_json.replace(".json", suffix), str(OUT / f"{tag}{suffix}"))
    pull_tdx("/tmp/a10_batch_counters.json", str(OUT / f"{tag}.tdx_counters.json"))
    if args.attest: pull_tdx("/tmp/a10_attest/session_attestation.json", str(OUT / f"{tag}.attestation.json"))
    # control sidecar (run_id/key/checkpoint/ledger/binding) for the restart driver
    ctrl = {"run_id": run_id, "session_key_hex": session_key.hex(), "tag": tag,
            "control_wall_sec": round(wall, 1), "rc": rc, "dataset": ds, "task": args.task,
            "profile": args.profile, "seed": args.seed, "eval_split": eval_split}
    try:
        res = json.loads((OUT / f"{tag}.json").read_text())
        traj = res.get("trajectory", [])
        if traj:
            last = traj[-1]
            ctrl["checkpoint"] = last.get("checkpoint"); ctrl["ledger_checkpoint"] = last.get("ledger_checkpoint")
        ctrl["eval"] = res.get("eval")
    except Exception:
        pass
    (OUT / f"{tag}.control.json").write_text(json.dumps(ctrl, indent=2))
    print(json.dumps(ctrl, indent=2))


if __name__ == "__main__":
    main()
