"""Gate 3 — fail-closed / negative matrix against the LIVE real-TDX loss service.

Runs from the Mac control plane over the H800 tunnel path (challenge control calls) and
directly exercises the service's fail-closed behaviour. No CPU/plaintext/AdamW fallback
is possible by construction (require_real_tdx service). Writes negative_tests.csv.
"""
from __future__ import annotations

import argparse
import base64
import csv
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


def gpu_py(snippet, timeout=180):
    full = ("export PATH=/root/miniconda3/bin:$PATH; cd /root/privacy_llm_obfuscation; "
            "python - <<'PYEOF'\n" + snippet + "\nPYEOF")
    r = subprocess.run(["ssh", "-S", SOCK, "-p", PORT, H, full],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config-digest", required=True)
    ap.add_argument("--runtime-hash", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []

    def rec(name, passed, detail=""):
        rows.append({"test": name, "passed": bool(passed), "detail": str(detail)[:160]})
        print(("PASS" if passed else "FAIL"), name)

    # test negatives that require only the H800 client + the live service. Each runs a
    # tiny python snippet on the H800 that pokes the service and reports the rejection.
    base = ('import sys; sys.path.insert(0,"scripts"); sys.path.insert(0,"src")\n'
            'from gate3_tdx_client import TdxLossSession\n'
            'from pllo.experiments.real_tdx_training_client import TrainingProtocolError\n'
            's=TdxLossSession("http://127.0.0.1:18091")\n')

    # 1. wrong run_id at challenge -> fail closed
    rc, so, se = gpu_py(base + f'''
try:
    s.challenge(run_id="WRONG_RUNID", config_digest="{args.config_digest}", model_id="Qwen2.5-0.5B",
                gradient_convention="nout_dual", optimizer_mode="gpu_masked_sgd", verifier_nonce="{"a"*64}")
    print("ACCEPTED")
except TrainingProtocolError as e:
    print("REJECTED", str(e)[:80])
''')
    rec("wrong_run_id_rejected", "REJECTED" in so and "run_id" in so, so.strip()[-120:] or se[-120:])

    # 2. wrong optimizer_mode at challenge -> fail closed
    rc, so, se = gpu_py(base + f'''
try:
    s.challenge(run_id="{args.run_id}", config_digest="{args.config_digest}", model_id="Qwen2.5-0.5B",
                gradient_convention="nout_dual", optimizer_mode="trusted_adamw", verifier_nonce="{"b"*64}")
    print("ACCEPTED")
except TrainingProtocolError as e:
    print("REJECTED", str(e)[:80])
''')
    rec("wrong_optimizer_mode_rejected", "REJECTED" in so, so.strip()[-120:] or se[-120:])

    # 3. wrong config_digest at challenge -> fail closed
    rc, so, se = gpu_py(base + f'''
try:
    s.challenge(run_id="{args.run_id}", config_digest="{"0"*64}", model_id="Qwen2.5-0.5B",
                gradient_convention="nout_dual", optimizer_mode="gpu_masked_sgd", verifier_nonce="{"c"*64}")
    print("ACCEPTED")
except TrainingProtocolError as e:
    print("REJECTED", str(e)[:80])
''')
    rec("wrong_config_digest_rejected", "REJECTED" in so, so.strip()[-120:] or se[-120:])

    # 4. compute endpoint without an authenticated session -> fail closed
    rc, so, se = gpu_py(base + f'''
s.client.run_id="{args.run_id}"; s.client.config_digest="{args.config_digest}"
try:
    s.client.init_session(run_id="{args.run_id}", config_digest="{args.config_digest}",
        lora_manifest=[], vocab_size=151936, labels_by_step={{}})
    print("ACCEPTED")
except TrainingProtocolError as e:
    print("REJECTED", str(e)[:80])
''')
    rec("compute_without_session_rejected", "REJECTED" in so, so.strip()[-120:] or se[-120:])

    # 5. wrong runtime hash -> Mac external verifier refuses (report_data mismatch)
    ver = ExternalAttestationVerifier(policy=MeasurementPolicy(allow_any_mr_td=True, allow_debug=False))
    ch = ver.issue_challenge(run_id=args.run_id, model_id="Qwen2.5-0.5B",
                             config_digest=args.config_digest, expected_runtime_hash="00" * 32,
                             gradient_convention="nout_dual", optimizer_mode="gpu_masked_sgd")
    rc, so, se = gpu_py(base + f'''
b=s.challenge(run_id="{args.run_id}", config_digest="{args.config_digest}", model_id="Qwen2.5-0.5B",
              gradient_convention="nout_dual", optimizer_mode="gpu_masked_sgd", verifier_nonce="{ch.verifier_nonce}")
import json; print("BUNDLE", json.dumps({{k:b.get(k) for k in ("report_data_hex","runtime_hash_hex","guest_ephemeral_public_hex","mr_td","debug","quote_hash","appraisal_ok","signature_verified","jwt_parts")}}))
''')
    refused = False
    try:
        line = [l for l in so.splitlines() if l.startswith("BUNDLE")][0][len("BUNDLE"):]
        b = json.loads(line)
        ev = {"report_data_hex": b["report_data_hex"], "runtime_hash_hex": b["runtime_hash_hex"],
              "mr_td": b["mr_td"], "debug": b["debug"], "quote_hash": b["quote_hash"],
              "nonce": ch.verifier_nonce, "run_id": args.run_id, "model_id": "Qwen2.5-0.5B",
              "config_digest": args.config_digest, "gradient_convention": "nout_dual",
              "optimizer_mode": "gpu_masked_sgd",
              "guest_ephemeral_public_hex": b["guest_ephemeral_public_hex"],
              "appraisal_ok": b["appraisal_ok"], "signature_verified": b["signature_verified"],
              "jwt_parts": b["jwt_parts"]}
        try:
            ver.verify(evidence=ev, challenge=ch, chain_verifier=lambda e: True)
        except Exception as e:  # AttestationFailClosed on runtime-hash mismatch
            refused = "runtime hash" in str(e) or "report_data" in str(e)
    except Exception as e:  # noqa: BLE001
        refused = False
    rec("wrong_runtime_hash_refused_by_verifier", refused, "external verifier report_data/runtime mismatch")

    Path(args.out).write_text("")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"]); w.writeheader(); w.writerows(rows)
    print("NEGATIVES_ALL_PASS", all(r["passed"] for r in rows))


if __name__ == "__main__":
    main()
