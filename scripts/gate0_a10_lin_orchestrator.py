"""A10+TDX control-plane orchestrator for SGD/momentum LoRA (L1/L2/L10, L3/L4/L11).

Reuses gate0_a10_mixed_orchestrator's transport helpers/constants. Launches a10_lin_runner with
--optimizer/--profile; data plane A10<->TDX PRIVATE VPC. Runs the plaintext (L1/L3) reference oracle
(compare_l5_l12.py) for O1-C runs. O1-A runs are judged by effective-equivalence divergence (top1/KL).
"""
from __future__ import annotations
import argparse, hashlib, json, secrets, time
from pathlib import Path
import gate0_a10_mixed_orchestrator as M   # push_a10/tdx, pull_a10/tdx, a10, sh, constants

REPO = M.REPO; OUT = REPO / "results/aaai_private_base/alicloud_a10_runs/l1_l11_matrix"
TDX_PRIV = M.TDX_PRIV; RA10 = M.RA10; RTDX = M.RTDX; PY_A10 = M.PY_A10; PY_TDX = M.PY_TDX
ENV_A10 = M.ENV_A10; IDS = M.IDS; PKG_ROOT = M.PKG_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum"])
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--profile", default="o1c", choices=["o1c", "o1a"])
    ap.add_argument("--run-tag", required=True); ap.add_argument("--attest", action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.run_tag}_s{args.seed}_{args.steps}step"
    run_id = f"{tag}-{int(time.time())}-{secrets.token_hex(3)}"
    session_key = secrets.token_bytes(32); nonce = secrets.token_hex(16)
    head = (REPO / "results/aaai_private_base/code_baseline_closure/head.txt").read_text().strip()
    ids = json.loads((REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json").read_text())
    labels = ids["input_ids"][:args.seq_len]
    binding = {"d4_run_id": run_id, "package_root_hash": PKG_ROOT, "execution_profile": "paper_safe",
               "optimizer_profile": f"{args.profile}_{args.optimizer}", "transport_profile": "direct_a10_tdx_private",
               "compute_host": "alicloud_a10", "data_plane_ip": TDX_PRIV, "seed": args.seed,
               "hmac_key_commitment": hashlib.sha256(session_key + bytes.fromhex(nonce)).hexdigest(),
               "debug_false_required": True, "nonce": nonce,
               "code_revision_or_worktree_hash": head,
               "ephemeral_key_pub": hashlib.sha256(nonce.encode()).hexdigest()}
    M.push_tdx(str(REPO / "scripts/tdx_persistent_service.py"), f"{RTDX}/scripts/tdx_persistent_service.py")
    M.push_tdx(str(REPO / "scripts/tdx_adamw_protocol.py"), f"{RTDX}/scripts/tdx_adamw_protocol.py")
    M.push_a10(str(REPO / "scripts/h800_direct_runner.py"), f"{RA10}/scripts/h800_direct_runner.py")
    M.push_a10(str(REPO / "scripts/a10_lin_runner.py"), f"{RA10}/scripts/a10_lin_runner.py")
    M.push_a10(str(REPO / "scripts/tdx_adamw_protocol.py"), f"{RA10}/scripts/tdx_adamw_protocol.py")
    M.push_tdx(str(REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"), "/tmp/o1c_bundle.pt")
    tdx_sess = {"session_key_hex": session_key.hex(), "run_id": run_id, "labels": labels,
                "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000, "attest": bool(args.attest),
                "attest_out": "/tmp/a10_attest", "binding_manifest": binding,
                "counters_out": "/tmp/a10_session_counters.json", "package_root_hash": PKG_ROOT,
                "ckpt_dir": "/tmp/l12_ckpt",
                "model_config_hash": "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
                "service_hash": hashlib.sha256((REPO/"scripts/tdx_persistent_service.py").read_bytes()).hexdigest(),
                "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2, "hidden_size": 896,
                              "intermediate_size": 4864}}
    Path(OUT / f"{tag}.tdx_session.json").write_text(json.dumps(tdx_sess, indent=2))
    M.push_tdx(str(OUT / f"{tag}.tdx_session.json"), "/tmp/direct_session.json")
    h_sess = {"session_key_hex": session_key.hex(), "run_id": run_id}
    Path(OUT / f"{tag}.a10_session.json").write_text(json.dumps(h_sess))
    M.push_a10(str(OUT / f"{tag}.a10_session.json"), "/tmp/direct_session_a10.json")

    service_cmd = f"{PY_TDX} {RTDX}/scripts/tdx_persistent_service.py /tmp/direct_session.json"
    out_json = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/l1_l11_matrix/{tag}.json"
    tlog_remote = f"{RA10}/results/aaai_private_base/alicloud_a10_runs/l1_l11_matrix/{tag}.tlog.pt"
    M.a10(f"mkdir -p {RA10}/results/aaai_private_base/alicloud_a10_runs/l1_l11_matrix")
    logt = "" if args.skip_compare else f" --log-transport {tlog_remote}"
    cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/a10_lin_runner.py "
           f"--session /tmp/direct_session_a10.json --steps {args.steps} --seq-len {args.seq_len} "
           f"--lr {args.lr} --seed {args.seed} --optimizer {args.optimizer} --mom {args.mom} "
           f"--profile {args.profile} --input-ids {IDS} --tdx root@{TDX_PRIV} --key /root/.ssh/a10_to_tdx "
           f"--service-cmd \"{service_cmd}\" --out {out_json} --adapter-id {run_id}{logt}")
    print(f"[launch] {tag} ({args.profile}/{args.optimizer}); A10<->TDX PRIVATE ...")
    t0 = time.time(); rc, o, e = M.a10(cmd, timeout=3600); wall = time.time() - t0
    print(o[-1200:] if o else "", "\n[stderr]", e[-800:] if rc != 0 else "")

    equiv_remote = out_json.replace(".json", ".equiv.json")
    eq_cmd = (f"cd {RA10} && {ENV_A10} {PY_A10} scripts/d4_trusted_verifier.py "
              f"--masked-logits {out_json.replace('.json','.final_logits.pt')} "
              f"--lora-state {out_json.replace('.json','.adapter.pt')} "
              f"--input-ids {IDS} --seq-len {args.seq_len} --out {equiv_remote}")
    erc, eo, ee = M.a10(eq_cmd, timeout=1800)
    if erc == 0: M.pull_a10(equiv_remote, str(OUT / f"{tag}.equiv.json"))
    else: print("[equiv] failed:", ee[-300:])
    M.pull_a10(out_json, str(OUT / f"{tag}.json"))
    M.pull_a10(out_json.replace(".json", ".adapter.pt"), str(OUT / f"{tag}.adapter.pt"))
    if args.attest: M.pull_tdx("/tmp/a10_attest/session_attestation.json", str(OUT / f"{tag}.attestation.json"))
    M.pull_tdx("/tmp/a10_session_counters.json", str(OUT / f"{tag}.tdx_counters.json"))

    if not args.skip_compare and args.profile == "o1c":
        M.pull_a10(tlog_remote, str(OUT / f"{tag}.tlog.pt"))
        cmp_out = OUT / f"{tag}.compare.json"
        crc, co, cerr = M.sh(f"cd {REPO} && python3 scripts/compare_l5_l12.py --tlog {OUT/(tag+'.tlog.pt')} "
                             f"--run-json {OUT/(tag+'.json')} --seed {args.seed} --optimizer {args.optimizer} "
                             f"--mom {args.mom} --out {cmp_out}")
        print("[compare L1/L3-plaintext oracle]", co.strip() if crc == 0 else cerr[-500:])
        M.sh(f"rm -f {OUT/(tag+'.tlog.pt')}")
    print(json.dumps({"tag": tag, "profile": args.profile, "optimizer": args.optimizer,
                      "control_wall_sec": round(wall, 1)}))


if __name__ == "__main__":
    main()
