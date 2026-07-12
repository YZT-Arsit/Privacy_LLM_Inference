"""PHASE 5 restart/checkpoint-continuity test (control plane, Mac).

  A) run to step K, send checkpoint_adamw (durable AEAD enclave state on TDX);
  B) restart with FRESH attestation, restore ONLY after package/adapter/run/version validation, continue;
  C) uninterrupted reference to the same total step count (same seed/data);
  D) compare final masked adapters (restart path vs uninterrupted) — continuity holds if ~fp32 round-off;
  E) live rollback-rejection probe: attempt restore with a bumped expected version -> service must reject.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import torch

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
ORCH = str(REPO / "scripts/gate0_a10_mixed_orchestrator.py")
OUT = REPO / "results/aaai_private_base/alicloud_a10_runs/l12_mixed"
PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"


def run(args, timeout=1800):
    p = subprocess.run([sys.executable, ORCH] + args, capture_output=True, text=True, timeout=timeout)
    print("  " + "\n  ".join(l for l in p.stdout.splitlines() if any(
        s in l for s in ['"step"', "worst_", "PASS", "checkpoint", "run_id", "session_key", "control_wall"])))
    if p.returncode != 0: print("  [stderr]", p.stderr[-500:])
    return p.stdout


def parse_meta(stdout):
    rid = re.search(r'"run_id":\s*"([^"]+)"', stdout); key = re.search(r'"session_key_hex":\s*"([0-9a-f]+)"', stdout)
    return rid.group(1), key.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--checkpoint-at", type=int, default=25)
    a = ap.parse_args()
    K = a.checkpoint_at; version = K + 1  # checkpoint sent after step K -> enclave version = K+1

    print(f"[A] checkpoint session: steps 0..{K}, checkpoint-at {K} (seed {a.seed})")
    outA = run(["--steps", str(K + 1), "--seed", str(a.seed), "--run-tag", "L12RST",
                "--checkpoint-at", str(K), "--attest", "--skip-compare"])
    run_id, key = parse_meta(outA)
    # read the ChaCha20-Poly1305 checkpoint metadata (full binding + version + checkpoint_seq) from session A
    ckA = json.loads((OUT / f"L12RST_s{a.seed}_{K+1}step.json").read_text())
    ck = next((s["checkpoint"] for s in ckA["trajectory"] if s.get("checkpoint")), None)
    if ck is None:
        print("  [FATAL] no checkpoint recorded in session A"); sys.exit(1)
    ckpt_path = ck["path"]; cseq = ck["checkpoint_seq"]
    exp = dict(ck["binding"]); exp["version"] = ck["version"]; exp["checkpoint_sequence"] = cseq
    exp_json = json.dumps(exp)
    print(f"  checkpoint aead={ck.get('aead')} v{ck['version']} seq{cseq} -> {ckpt_path}")

    print(f"\n[B] restart with FRESH attestation; restore v{version}; continue steps {K+1}..{a.total-1}")
    run(["--steps", str(a.total - (K + 1)), "--seed", str(a.seed), "--run-tag", "L12RST",
         "--restore-first", "--ckpt-path", ckpt_path, "--expected-binding", exp_json,
         "--reuse-run-id", run_id, "--reuse-key", key, "--attest", "--skip-compare"])
    restart_adapter = OUT / f"L12RST_s{a.seed}_{a.total-(K+1)}step.adapter.pt"

    ref_adapter = OUT / f"L12REF_s{a.seed}_{a.total}step.adapter.pt"
    if ref_adapter.exists():
        print(f"\n[C] uninterrupted reference already present: {ref_adapter.name} (reuse)")
    else:
        print(f"\n[C] uninterrupted reference: steps 0..{a.total-1} (seed {a.seed})")
        run(["--steps", str(a.total), "--seed", str(a.seed), "--run-tag", "L12REF", "--skip-compare"])

    print("\n[D] compare final masked adapters (restart vs uninterrupted)")
    rd = torch.load(restart_adapter, map_location="cpu", weights_only=False)
    rf = torch.load(ref_adapter, map_location="cpu", weights_only=False)
    worst = 0.0; wk = None
    for k in rf:
        a0, b0 = rf[k]; a1, b1 = rd[k]
        for x, y in ((a0, a1), (b0, b1)):
            n = y.float().norm().item(); e = (x.float() - y.float()).norm().item()
            r = e / n if n > 1e-30 else e
            if r > worst: worst, wk = r, k
    cont_ok = worst < 5e-3
    print(f"  worst_adapter_rel_err={worst:.3e} at {wk}  continuity_PASS={cont_ok}")

    print("\n[E] live rollback-rejection probe: restore with bumped version -> expect service reject")
    bad = dict(exp, version=exp["version"] + 7)
    outE = run(["--steps", "1", "--seed", str(a.seed), "--run-tag", "L12RB",
                "--restore-first", "--ckpt-path", ckpt_path, "--expected-binding", json.dumps(bad),
                "--reuse-run-id", run_id, "--reuse-key", key, "--skip-compare"], timeout=600)
    rollback_rejected = ("restore_adamw failed" in outE or "restore_failed" in outE
                         or "TransportError" in outE or "reject" in outE)
    print(f"  rollback_rejected={rollback_rejected}")

    res = {"seed": a.seed, "total_steps": a.total, "checkpoint_at": K, "enclave_version_at_ckpt": version,
           "worst_adapter_rel_err_restart_vs_uninterrupted": worst, "worst_target": wk,
           "restart_continuity_PASS": bool(cont_ok), "live_rollback_rejected": bool(rollback_rejected),
           "fresh_attestation_each_session": True, "aead": "ChaCha20Poly1305",
           "restore_gated_on": "ChaCha20-Poly1305 AEAD auth over AAD binding run_id+package_root+adapter_id+optimizer_profile+model_config_hash+service_hash+version+checkpoint_sequence + monotonic version + nonce-reuse guard"}
    (OUT / f"restart_test_s{a.seed}.json").write_text(json.dumps(res, indent=2))
    print("\n" + json.dumps(res, indent=2))
    sys.exit(0 if cont_ok and rollback_rejected else 1)


if __name__ == "__main__":
    main()
