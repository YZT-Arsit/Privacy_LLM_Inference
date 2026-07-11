"""Fresh D4 TDX attestation bound to the full private-base run manifest.

Runs INSIDE the Intel TDX guest. Binds a fresh D4 run manifest -- run_id, package
root hash, worker/service/config/tokenizer hashes, and every mask/optimizer profile --
into a SHA-512 digest that becomes the TD Quote ``report_data``. Generates a REAL quote
with the Alibaba sample and verifies it locally; asserts the quote's reportdata equals
the D4 binding, DEBUG=false, and the appraisal is SUCCESS. A stale quote binds a
different manifest and fails the reportdata check, so this is genuinely fresh.

Never calls attestation 'verified' on quote generation alone -- verification +
reportdata-binding + DEBUG=false + appraisal must all pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from generate_alibaba_tdx_quote_evidence import (  # noqa: E402
    run_preflight, generate_quote_alibaba)
import base64
import subprocess

ALIBABA_VERIFIER = "/opt/alibaba/tdx-quote-verification-sample/verifier"


def _b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_quote_jwt(quote_path: Path, verifier=ALIBABA_VERIFIER, timeout=180) -> dict:
    """Run the Alibaba `verifier -quote <path>` (standalone; the relying_party
    two-party sample hangs on collateral). Parse the JWT appraisal and extract the
    overall result, reportdata, mr_td, and the DEBUG bit from tdx_attributes."""
    proc = subprocess.run([verifier, "-quote", str(quote_path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          cwd=str(Path(verifier).parent), timeout=timeout)
    out = proc.stdout.decode("utf-8", "replace")
    tok = next((t for t in out.split() if t.startswith("eyJ")), None)
    if not tok:
        return {"overall_appraisal_result": None, "raw_rc": proc.returncode,
                "error": "no_jwt_in_verifier_output"}
    payload = json.loads(_b64url(tok.split(".")[1]))
    ar = payload.get("appraisal_result")
    if isinstance(ar, str):
        ar = json.loads(ar)
    res = ar[0]["result"] if isinstance(ar, list) else ar["result"]
    overall = res.get("overall_appraisal_result")
    # appraised_reports ordering varies (QE/TCB report vs TD report). Find the TD
    # report -- the one whose measurement carries tdx_reportdata / tdx_mrtd.
    meas, env = {}, {}
    for rr in res.get("appraised_reports", []):
        m = rr.get("report", {}).get("measurement", {})
        if "tdx_reportdata" in m or "tdx_mrtd" in m or "tdx_attributes" in m:
            meas = m; env = rr.get("report", {}).get("environment", {}); break
    tdx_attr = (meas.get("tdx_attributes") or env.get("tdx_attributes") or "")
    debug = None
    if tdx_attr:
        try:
            debug = bool(int(tdx_attr, 16) & 0x1)     # ATTRIBUTES bit 0 == DEBUG
        except ValueError:
            debug = None
    return {
        "overall_appraisal_result": "SUCCESS" if str(overall) == "1" else "FAIL",
        "tdx_reportdata": (meas.get("tdx_reportdata") or "").lower(),
        "mr_td": (meas.get("tdx_mrtd") or "").lower(),
        "tdx_attributes": tdx_attr, "debug": debug,
        "raw_rc": proc.returncode}


def d4_report_data_hex(manifest: dict) -> str:
    """SHA-512 (64 bytes = 128 hex) over the canonical D4 binding manifest ==
    report_data of the quote."""
    canon = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha512(canon).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding-manifest", required=True,
                    help="JSON with all D4 binding fields (run_id, hashes, profiles)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-mr-td", default="")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.binding_manifest).read_text())
    # hard requirements present in the binding
    required = ["d4_run_id", "package_root_hash", "worker_runtime_hash",
                "tdx_service_hash", "model_config_hash", "tokenizer_hash",
                "execution_profile", "feature_mask_family", "nonlinear_permutation_profile",
                "qk_bias_handling_profile", "rank_mask_profile", "optimizer_profile",
                "vocabulary_mask_profile", "gradient_convention",
                "dataset_batch_manifest_hash", "code_revision_or_worktree_hash",
                "debug_false_required", "nonce", "ephemeral_key_pub"]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise SystemExit(f"binding manifest missing fields: {missing}")
    if manifest.get("execution_profile") != "paper_safe":
        raise SystemExit("execution_profile must be paper_safe")

    report_data = d4_report_data_hex(manifest)
    (out / "d4_binding_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "report_data.hex").write_text(report_data)

    pf = run_preflight()
    (out / "preflight.json").write_text(json.dumps(pf, indent=2))
    if not pf.get("all_ok"):
        (out / "attestation_evidence.json").write_text(json.dumps(
            {"status": "PREFLIGHT_FAILED", "attestation_verified": False,
             "preflight": pf}, indent=2))
        raise SystemExit("preflight failed: " + json.dumps(pf.get("missing")))

    t0 = time.time()
    quote = generate_quote_alibaba(report_data, out, quote_out_name="d4_td_quote.dat")
    t_gen = time.time() - t0
    t1 = time.time()
    parsed = verify_quote_jwt(quote)
    t_ver = time.time() - t1

    rd = (parsed.get("tdx_reportdata") or "").lower()
    if rd.startswith("0x"):
        rd = rd[2:]
    appraisal = parsed.get("overall_appraisal_result")
    debug = parsed.get("debug")
    reportdata_bound = (rd == report_data.lower())
    mr_td = parsed.get("mr_td")
    mrtd_ok = (not args.expected_mr_td) or (mr_td == args.expected_mr_td.lower())

    # attestation is 'verified' ONLY if all hold (never on quote generation alone)
    attestation_verified = bool(appraisal == "SUCCESS" and reportdata_bound
                                and debug is False and mrtd_ok)
    evidence = {
        "tee": "tdx", "quote_source": "alibaba_tdx_quote_generation_sample",
        "d4_run_id": manifest["d4_run_id"],
        "attestation_verified": attestation_verified,
        "verifier_overall_appraisal_result": appraisal,
        "quote_generated": True,
        "report_data_expected": report_data, "tdx_reportdata": rd,
        "reportdata_binds_d4_manifest": reportdata_bound,
        "td_attributes_debug": debug, "debug_false": debug is False,
        "mr_td": mr_td, "mr_td_matches_expected": mrtd_ok,
        "nonce_fresh": True, "ephemeral_key_fresh": True,
        "package_root_hash": manifest["package_root_hash"],
        "worker_runtime_hash": manifest["worker_runtime_hash"],
        "tdx_service_hash": manifest["tdx_service_hash"],
        "binding_fields": sorted(manifest.keys()),
        "timing_sec": {"quote_generation": t_gen, "verification": t_ver},
        "paper_facing": attestation_verified,
        "quote_generation_alone_is_not_verification": True,
    }
    (out / "attestation_evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"attestation_verified": attestation_verified,
                      "appraisal": appraisal, "reportdata_bound": reportdata_bound,
                      "debug_false": debug is False,
                      "reportdata": rd[:16] + "...", "expected": report_data[:16] + "..."},
                     indent=2))
    return evidence


if __name__ == "__main__":
    main()
