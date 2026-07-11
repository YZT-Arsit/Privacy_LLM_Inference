"""Gate 3 — Mac-side control-plane orchestrator (issues challenge, verifies, authorizes).

Control plane (Mac): issue fresh nonce -> H800 performs /train/challenge directly with
TDX -> Mac verifies the signed DCAP appraisal + measurement policy + report_data +
runtime hash -> Mac authorizes. Data plane (H800<->TDX direct) carries logits/dlogits;
the Mac NEVER relays model tensors. Then H800 runs the real masked-SGD step(s).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.real_tdx_attestation import (  # noqa: E402
    ExternalAttestationVerifier, MeasurementPolicy)

SOCK = "/Users/Hoshino/.ssh/cm-westb-19948"
H = "root@connect.westb.seetacloud.com"
PORT = "19948"
RWORK = "/root/privacy_llm_obfuscation"
PYENV = "export PATH=/root/miniconda3/bin:$PATH; cd " + RWORK


def gpu(cmd, timeout=1200):
    full = f"{PYENV}; {cmd}"
    r = subprocess.run(["ssh", "-S", SOCK, "-p", PORT, H, full],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def scp_from_gpu(remote, local):
    subprocess.run(["scp", "-o", f"ControlPath={SOCK}", "-P", PORT,
                    f"{H}:{remote}", local], check=True, capture_output=True)


def chain_verifier(ev):
    return bool(ev.get("appraisal_ok") and ev.get("signature_verified") is True
                and ev.get("reportdata_binds") and ev.get("jwt_parts", 0) >= 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-id", default="Qwen2.5-0.5B")
    ap.add_argument("--config-digest", required=True)
    ap.add_argument("--runtime-hash", required=True)
    ap.add_argument("--gradient-convention", default="nout_dual")
    ap.add_argument("--optimizer-mode", default="gpu_masked_sgd")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--targets", default="attn")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-local", required=True)
    args = ap.parse_args()
    out = Path(args.out_local); out.mkdir(parents=True, exist_ok=True)

    # 1. Mac issues fresh challenge (nonce single-use)
    policy = MeasurementPolicy(allow_any_mr_td=True, allow_debug=False)
    verifier = ExternalAttestationVerifier(policy=policy)
    ch = verifier.issue_challenge(run_id=args.run_id, model_id=args.model_id,
                                  config_digest=args.config_digest,
                                  expected_runtime_hash=args.runtime_hash,
                                  gradient_convention=args.gradient_convention,
                                  optimizer_mode=args.optimizer_mode)
    print(f"[mac] issued fresh nonce {ch.verifier_nonce[:16]}...")

    # 2. H800 performs /train/challenge DIRECTLY with TDX (control call; no tensors)
    rc, so, se = gpu(
        f"python scripts/gate3_tdx_step.py --phase challenge "
        f"--tdx-url http://127.0.0.1:18091 --run-id {args.run_id} "
        f"--config-digest {args.config_digest} --model-id {args.model_id} "
        f"--gradient-convention {args.gradient_convention} "
        f"--optimizer-mode {args.optimizer_mode} --nonce {ch.verifier_nonce} "
        f"--bundle-out /root/gate3/bundle.json")
    print("[h800 challenge]", so.strip()[-300:] or se.strip()[-300:])
    if rc != 0:
        print("challenge failed"); return 1
    scp_from_gpu("/root/gate3/bundle.json", str(out / "challenge_bundle.json"))
    bundle = json.loads((out / "challenge_bundle.json").read_text())

    # 3. Mac verifies the signed DCAP appraisal + policy + report_data + runtime hash
    evidence = {
        "report_data_hex": bundle.get("report_data_hex"),
        "runtime_hash_hex": bundle.get("runtime_hash_hex"),
        "mr_td": bundle.get("mr_td"), "debug": bundle.get("debug"),
        "quote_hash": bundle.get("quote_hash"), "nonce": ch.verifier_nonce,
        "run_id": args.run_id, "model_id": args.model_id,
        "config_digest": args.config_digest,
        "gradient_convention": args.gradient_convention,
        "optimizer_mode": args.optimizer_mode,
        "guest_ephemeral_public_hex": bundle.get("guest_ephemeral_public_hex"),
        "appraisal_ok": bundle.get("appraisal_ok"),
        "reportdata_binds": bundle.get("reportdata_binds"),
        "signature_verified": bundle.get("signature_verified"),
        "jwt_parts": bundle.get("jwt_parts")}
    result = verifier.verify(evidence=evidence, challenge=ch, chain_verifier=chain_verifier)
    result["logits_mask_family"] = bundle.get("attestation_bundle_public", {}).get(
        "logits_mask_family") or "vocab_permutation"
    (out / "tdx_attestation.json").write_text(json.dumps(
        {"verification": result, "bundle_public": {k: v for k, v in bundle.items()
         if k not in ("quote_b64", "appraisal_jwt")},
         "fresh_nonce": ch.verifier_nonce[:16] + "...(single-use)",
         "reused_gate2": False}, indent=2, default=str))
    print(f"[mac] attestation_verified={result['attestation_verified']} "
          f"policy={result['measurement_policy_passed']} debug={result['debug_mode']} "
          f"report_data_match={result['report_data_match']}")
    if not result["attestation_verified"]:
        print("[mac] NOT AUTHORIZED -> abort (fail-closed, no fallback)"); return 2

    # 4. Mac AUTHORIZES -> H800 runs the real masked-SGD step(s) directly with TDX
    print("[mac] AUTHORIZED. H800 running real step(s)...")
    rc, so, se = gpu(
        f"python scripts/gate3_tdx_step.py --phase run "
        f"--tdx-url http://127.0.0.1:18091 --run-id {args.run_id} "
        f"--config-digest {args.config_digest} --model-id {args.model_id} "
        f"--gradient-convention {args.gradient_convention} "
        f"--optimizer-mode {args.optimizer_mode} --bundle-in /root/gate3/bundle.json "
        f"--model-dir {args.model_dir} --data {args.data} --batch {args.batch} "
        f"--seqlen {args.seqlen} --targets {args.targets} --steps {args.steps} "
        f"--seed {args.seed} --lr {args.lr} --out /root/gate3/out")
    print("[h800 run]", (so.strip()[-600:] or se.strip()[-600:]))
    if rc != 0:
        print("run failed:", se[-400:]); return 1
    scp_from_gpu("/root/gate3/out/result_protected_tdx.json", str(out / "result_protected_tdx.json"))
    scp_from_gpu("/root/gate3/out/tensors_protected_tdx.pt", str(out / "tensors_protected_tdx.pt"))
    print("[mac] pulled protected_tdx results; attestation_verified + step complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
