"""Gate 2 §8 — consolidate the driver's results.json into the full evidence bundle.

Reads results.json (+ quote_*.bin) produced by gate2_driver.py and the guest-side
identity JSON (gate2_service --print-identity), then writes the §8 file set:
verification.json, measurement_policy.json, runtime_manifest.json,
deployed_artifact_hashes.json, session_handshake.json, endpoint_results.json,
challenge.json, summary.md. Never writes secrets (keys/shared-secrets/plaintext).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sgd-identity", default="", help="guest gate2_service --print-identity JSON (sgd)")
    ap.add_argument("--adamw-identity", default="", help="guest identity JSON (adamw)")
    args = ap.parse_args()
    d = Path(args.dir)
    res = json.loads((d / "results.json").read_text())
    profiles = res["profiles"]

    identities = {}
    for om, path in (("gpu_masked_sgd", args.sgd_identity), ("trusted_adamw", args.adamw_identity)):
        if path and Path(path).exists():
            identities[om] = json.loads(Path(path).read_text())

    # verification.json — the external-verifier decision per profile
    verification = {om: {**p["attestation"],
                         "expected_runtime_hash": p["expected_runtime_hash"],
                         "quote_hash": p["attestation_bundle_public"].get("quote_hash"),
                         "mr_td": p["attestation_bundle_public"].get("mr_td"),
                         "rtmr0": p["attestation_bundle_public"].get("rtmr0"),
                         "rtmr1": p["attestation_bundle_public"].get("rtmr1"),
                         "rtmr2": p["attestation_bundle_public"].get("rtmr2"),
                         "rtmr3": p["attestation_bundle_public"].get("rtmr3"),
                         "debug_mode": p["attestation_bundle_public"].get("debug"),
                         "quote_version": p["attestation_bundle_public"].get("quote_version"),
                         "appraisal_source": p["attestation_bundle_public"].get("appraisal_source"),
                         "overall_appraisal_result": p["attestation_bundle_public"].get("overall_appraisal_result"),
                         "signature_verified": p["attestation_bundle_public"].get("signature_verified"),
                         "reportdata_binds": p["attestation_bundle_public"].get("reportdata_binds")}
                    for om, p in profiles.items()}
    (d / "verification.json").write_text(json.dumps(verification, indent=2, default=str))

    # measurement_policy.json
    (d / "measurement_policy.json").write_text(json.dumps({
        "policy": "DEBUG must be false; overall_appraisal_result must be SUCCESS; "
                  "relying_party -v ES384 policy signature must verify; report_data "
                  "must bind all session-identity fields.",
        "mr_td_pinned": False,
        "mr_td_pinning_note": "mr_td RECORDED per profile (below) but not pinned to a "
                              "golden allowlist in this run (no pre-provisioned golden "
                              "measurement). debug=false IS enforced.",
        "allow_debug": False,
        "recorded_mr_td": {om: verification[om]["mr_td"] for om in verification},
    }, indent=2, default=str))

    # runtime_manifest.json + deployed_artifact_hashes.json
    (d / "runtime_manifest.json").write_text(json.dumps({
        "protocol_version": "pllo.train.v1",
        "profiles": {om: {"runtime_hash": verification[om]["expected_runtime_hash"],
                          "config_digest": profiles[om]["endpoint"].get("config_digest"),
                          "optimizer_mode": om,
                          "gradient_convention": profiles[om]["endpoint"]["gradient_convention"]}
                     for om in profiles},
        "guest_identity": identities,
    }, indent=2, default=str))
    if identities:
        (d / "deployed_artifact_hashes.json").write_text(json.dumps(
            {om: idv.get("artifact_hashes") for om, idv in identities.items()}, indent=2))

    # session_handshake.json (public only — NO keys)
    (d / "session_handshake.json").write_text(json.dumps({om: {
        "guest_ephemeral_public_hex": p["attestation_bundle_public"].get("guest_ephemeral_public_hex"),
        "authenticated_session": bool(p["attestation"].get("attestation_verified")),
        "kex": "X25519 ECDH -> HKDF-SHA256 (salt=quote_hash) -> ChaCha20-Poly1305 AEAD",
        "note": "verifier ephemeral public key + all derived keys are NOT exported"}
        for om, p in profiles.items()}, indent=2, default=str))

    # endpoint_results.json
    (d / "endpoint_results.json").write_text(json.dumps(
        {om: p["endpoint"] for om, p in profiles.items()}, indent=2, default=str))

    # challenge.json (structure, nonces regenerated each run; not stored)
    (d / "challenge.json").write_text(json.dumps({
        "report_data_binding": "SHA512(runtime_hash || protocol || config_digest || "
            "run_id || model_id || gradient_convention || optimizer_mode || "
            "guest_ephemeral_public_key || verifier_nonce)",
        "nonce": "fresh per run, single-use (not persisted)",
    }, indent=2))

    # summary.md
    def row(om):
        a = verification[om]; e = profiles[om]["endpoint"]
        return (f"### {om}\n"
                f"- attestation_verified: **{a['attestation_verified']}** | "
                f"quote_chain(QVL+ES384 sig): {a['signature_verified']} | "
                f"overall_appraisal_result: {a['overall_appraisal_result']} | "
                f"report_data_match: {a['report_data_match']}\n"
                f"- measurement_policy_passed: {a['measurement_policy_passed']} | "
                f"DEBUG: {a['debug_mode']} | quote_version: {a['quote_version']} | "
                f"appraisal_source: {a['appraisal_source']}\n"
                f"- mr_td: `{(a['mr_td'] or '')[:48]}...`\n"
                f"- runtime_hash: `{a['expected_runtime_hash'][:48]}...`\n"
                f"- trusted invocations/step: **{e['trusted_invocations_per_step']}** "
                f"(expected {e['expected_invocations']}, ok={e['invocations_ok']}) | "
                f"trusted_optimizer_calls: {e['trusted_optimizer_calls']}\n"
                f"- logits_loss: loss_abs_err={e['logits_loss']['loss_abs_err']:.2e}, "
                f"dlogits_abs_err={e['logits_loss']['dlogits_abs_err']:.2e}\n"
                + (f"- packed_update: adapter_abs_err={e['packed_update']['adapter_abs_err']:.2e} "
                   f"(real AdamW verified)\n" if 'packed_update' in e else
                   f"- packed_update rejected in SGD mode: {e.get('packed_update_rejected')}\n"))

    negs = "\n".join(f"- [{'PASS' if r['passed'] else 'FAIL'}] {r['test']}: {r['detail'][:80]}"
                     for r in res.get("negative_tests", []))
    md = f"""# Gate 2 — real TDX service validation (evidence)

**gate2_pass = {res['gate2_pass']}**  (sgd={res['sgd_pass']}, adamw={res['adamw_pass']}, negatives={res['negatives_pass']})

Flags: uses_real_tee={res['uses_real_tee']}, uses_real_gpu={res['uses_real_gpu']},
uses_real_qwen={res['uses_real_qwen']}, service_validation_only={res['service_validation_only']}.
guest_hostname={res.get('guest_hostname')}.

The exact trusted training-service code runs INSIDE the real Intel TDX guest. Each
profile: fresh nonce -> /train/challenge (guest binds it + the guest ECDH public key
into report_data and produces a REAL TD Quote) -> DCAP QVL appraisal (verifier +
relying_party -v: PCK cert chain + ES384 policy signature) -> external verifier
re-derives report_data over every bound field and enforces DEBUG=false -> attested
X25519 ECDH -> AEAD-wrapped real endpoints over the SSH tunnel.

This is SERVICE VALIDATION ONLY with SYNTHETIC protocol tensors — NOT Qwen, NOT GPU.

{row('gpu_masked_sgd')}
{row('trusted_adamw')}

## Negative / fail-closed matrix (real wire)
{negs}
"""
    (d / "summary.md").write_text(md)
    print("wrote evidence bundle to", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
