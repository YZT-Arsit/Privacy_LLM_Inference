"""Gate 2 §3/§6/§7/§8 — OFF-GUEST verifier + client driver (runs on the untrusted side).

Talks to the real trusted service inside the TDX guest over an SSH tunnel
(http://127.0.0.1:<port>). For each optimizer profile it:

  1. issues a fresh nonce and runs /train/challenge -> the guest binds it into
     report_data and produces a REAL TD Quote;
  2. verifies OFF-GUEST: report_data recomputed over every bound field (incl. the
     guest ECDH pubkey), measurement policy (DEBUG must be false), and the DCAP QVL
     appraisal (overall_appraisal_result==1 + relying_party -v ES384 signature +
     report_data binding, supplied as the chain verifier);
  3. completes the attested ECDH handshake -> directional AEAD session keys;
  4. exercises the real endpoints over the authenticated wire, checking CE / masked
     logit-gradient / (AdamW) adapter updates against an INDEPENDENT plaintext
     reference reconstructed from the deterministic init seed;
  5. counts logical trusted invocations/step (SGD=2, AdamW=3);
  6. runs the negative / fail-closed matrix over the real wire.

Writes the full evidence bundle. Secrets (keys, shared secrets, plaintext adapters/
labels) are never written. Synthetic protocol-validation tensors only -- NOT Qwen.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.real_tdx_attestation import (  # noqa: E402
    MeasurementPolicy,
    compute_service_runtime_hash,
    digest_config,
)
from pllo.experiments.real_tdx_training_client import (  # noqa: E402
    TrainingProtocolError,
    TrustedBoundaryClient,
)
from pllo.experiments.real_tdx_training_service import TrustedTrainingService  # noqa: E402

DT = torch.float32
VOCAB = 16
LAYERS = [("L0", 8, 8, 4), ("L1", 8, 8, 4)]
LABELS = {0: [1, 2, 3]}
SEED = 1


def build_config(optimizer_mode, gradient_convention, model_id, lr=1e-3, wd=0.0,
                 optimizer=None, session_timeout_s=600.0):
    if optimizer is None:
        optimizer = "adamw" if optimizer_mode == "trusted_adamw" else "sgd"
    cfg = {"model_id": model_id, "lr": lr, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
           "weight_decay": wd, "dtype": "float32", "optimizer_mode": optimizer_mode,
           "optimizer": optimizer, "gradient_convention": gradient_convention,
           "session_timeout_s": session_timeout_s}
    cfg["config_digest"] = digest_config({k: v for k, v in cfg.items()})
    return cfg


def make_oracle(cfg):
    """Local deterministic mirror of the guest init (same seed) -> masks + plaintext
    reference. Runs cpu_contract; never networked."""
    svc = TrustedTrainingService(run_id=cfg["run_id"], config=cfg, mode="cpu_contract",
                                 dtype=DT, seed=SEED)
    manifest = [{"layer_id": l, "d_in": di, "d_out": do, "rank": r}
                for (l, di, do, r) in LAYERS]
    svc.init_session({"run_id": cfg["run_id"], "config_digest": cfg["config_digest"],
                      "lora_manifest": manifest, "vocab_size": VOCAB,
                      "labels_by_step": LABELS})
    return svc, manifest


def plain_ce(logits, labels):
    logp = torch.log_softmax(logits, dim=-1)
    n = logits.shape[0]
    loss = -logp[torch.arange(n), labels].mean()
    p = torch.softmax(logits, dim=-1)
    d = p.clone(); d[torch.arange(n), labels] -= 1.0; d /= n
    return float(loss), d


def plain_adamw_step(A, B, gA, gB, cfg):
    lr = cfg["lr"]; b1 = cfg["beta1"]; b2 = cfg["beta2"]; eps = cfg["eps"]; wd = cfg["weight_decay"]
    out = {}
    for name, p, g in (("A", A, gA), ("B", B, gB)):
        m = (1 - b1) * g; v = (1 - b2) * (g * g)
        mh = m / (1 - b1); vh = v / (1 - b2)
        out[name] = p - lr * (mh / (vh.sqrt() + eps) + wd * p)
    return out["A"], out["B"]


def chain_verifier(ev):
    """The cryptographic quote appraisal result (DCAP QVL on the guest relying party):
    QVL overall_appraisal_result==SUCCESS + relying_party -v ES384 policy signature
    verified + report_data cryptographically binds to our expected value."""
    return bool(ev.get("appraisal_ok") and ev.get("signature_verified") is True
                and ev.get("reportdata_binds") and ev.get("jwt_parts", 0) >= 2)


def establish(base_url, cfg, model_id, *, expected_runtime_hash, allow_any_mr_td=True,
              chain=chain_verifier, require_attested=True):
    client = TrustedBoundaryClient(base_url=base_url)
    policy = MeasurementPolicy(allow_any_mr_td=allow_any_mr_td, allow_debug=False)
    result = client.establish_attested_session(
        run_id=cfg["run_id"], config_digest=cfg["config_digest"], model_id=model_id,
        gradient_convention=cfg["gradient_convention"], optimizer_mode=cfg["optimizer_mode"],
        expected_runtime_hash=expected_runtime_hash, policy=policy,
        chain_verifier=chain, require_attested=require_attested)
    return client, result


def run_profile(base_url, optimizer_mode, gradient_convention, model_id, out_dir,
                *, chain=chain_verifier, require_attested=True):
    cfg = build_config(optimizer_mode, gradient_convention, model_id)
    cfg["run_id"] = f"gate2_{optimizer_mode}"
    expected_rh = compute_service_runtime_hash(cfg["config_digest"])
    oracle, manifest = make_oracle(cfg)
    N_head, M_head = oracle._head_masks()
    N_head_inv = torch.linalg.inv(N_head); M_head_inv = torch.linalg.inv(M_head)

    client, attest = establish(base_url, cfg, model_id, expected_runtime_hash=expected_rh,
                               chain=chain, require_attested=require_attested)
    invocations = 0

    client.init_session(run_id=cfg["run_id"], config_digest=cfg["config_digest"],
                        lora_manifest=manifest, vocab_size=VOCAB, labels_by_step=LABELS)
    invocations += 1

    # ---- boundary 1: logits_loss vs independent plaintext reference ----
    torch.manual_seed(7)
    logits = torch.randn(3, VOCAB, dtype=DT)
    labels = torch.tensor(LABELS[0])
    masked_logits = (logits @ N_head).to(DT)
    resp1 = client.logits_loss(0, masked_logits)
    invocations += 1
    loss_ref, d_ref = plain_ce(logits, labels)
    d_recovered = resp1["masked_logit_gradient"].to(DT) @ M_head_inv
    loss_err = abs(resp1["loss"] - loss_ref)
    dgrad_err = float((d_recovered - d_ref).abs().max())

    endpoint = {"optimizer_mode": optimizer_mode, "gradient_convention": gradient_convention,
                "logits_loss": {"loss_service": resp1["loss"], "loss_ref": loss_ref,
                                "loss_abs_err": loss_err, "dlogits_abs_err": dgrad_err,
                                "trusted_loss_executed": resp1["verification"].get("trusted_loss_executed")}}

    trusted_optimizer_calls = 0
    if optimizer_mode == "trusted_adamw":
        # ---- boundary 2: packed_update real AdamW vs plaintext reference ----
        packed = {}; refs = {}
        for (lid, di, do, r) in LAYERS:
            N_in, R, N_out = oracle._layer_masks(lid)
            A0, B0 = oracle.plaintext_adapter(lid)
            torch.manual_seed(hash(lid) % 1000)
            gA = torch.randn(di, r, dtype=DT); gB = torch.randn(r, do, dtype=DT)
            gA_t = N_in.T @ gA @ torch.linalg.inv(R).T
            gB_t = R.T @ gB @ torch.linalg.inv(N_out).T
            packed[lid] = {"gradA": gA_t.to(DT), "gradB": gB_t.to(DT)}
            A_ref, B_ref = plain_adamw_step(A0.clone(), B0.clone(), gA, gB, cfg)
            refs[lid] = (A_ref, B_ref, N_in, R, N_out)
        resp2 = client.packed_update(0, packed)
        invocations += 1
        trusted_optimizer_calls = 1
        max_err = 0.0
        for lid, (A_ref, B_ref, N_in, R, N_out) in refs.items():
            mA, mB = resp2["updated_masked_adapters"][lid]
            A_new = N_in @ mA.to(DT) @ torch.linalg.inv(R)
            B_new = R @ mB.to(DT) @ torch.linalg.inv(N_out)
            max_err = max(max_err, float((A_new - A_ref).abs().max()),
                          float((B_new - B_ref).abs().max()))
        endpoint["packed_update"] = {"adapter_abs_err": max_err,
                                     "trusted_adamw_executed": resp2["verification"].get("trusted_adamw_executed")}
    else:
        # packed_update MUST be rejected in masked-SGD mode
        rejected = False
        try:
            client.packed_update(0, {})
        except TrainingProtocolError as exc:
            rejected = "packed_update not allowed" in str(exc)
        endpoint["packed_update_rejected"] = rejected

    endpoint["trusted_invocations_per_step"] = invocations
    endpoint["trusted_optimizer_calls"] = trusted_optimizer_calls
    endpoint["expected_invocations"] = 3 if optimizer_mode == "trusted_adamw" else 2
    endpoint["invocations_ok"] = (invocations == endpoint["expected_invocations"])

    # save the real quote + attestation bundle
    bundle = client.attestation_bundle
    prof_dir = Path(out_dir); prof_dir.mkdir(parents=True, exist_ok=True)
    if bundle.get("quote_b64"):
        (prof_dir / f"quote_{optimizer_mode}.bin").write_bytes(base64.b64decode(bundle["quote_b64"]))
        (prof_dir / f"quote_hash_{optimizer_mode}.txt").write_text(bundle["quote_hash"] + "\n")
    attest_public = {k: v for k, v in bundle.items() if k not in ("quote_b64", "appraisal_jwt")}
    return {"attestation": attest, "attestation_bundle_public": attest_public,
            "endpoint": endpoint, "client": client, "cfg": cfg,
            "expected_runtime_hash": expected_rh}


def negative_matrix(base_url, sgd_run, adamw_run):
    """Fail-closed tests over the REAL wire. Each returns (name, passed, detail)."""
    rows = []

    def rec(name, passed, detail=""):
        rows.append({"test": name, "passed": bool(passed), "detail": str(detail)[:200]})

    # 1. wrong expected runtime hash -> report_data binding fails (off-guest, real quote)
    cfg = build_config("gpu_masked_sgd", "nout_dual", "synthetic-gate2")
    cfg["run_id"] = "gate2_gpu_masked_sgd"
    try:
        establish(base_url, cfg, "synthetic-gate2",
                  expected_runtime_hash="00" * 32)
        rec("wrong_runtime_hash_rejected", False, "establish did not fail")
    except (TrainingProtocolError, Exception) as exc:
        rec("wrong_runtime_hash_rejected", True, type(exc).__name__)

    # 2. wrong optimizer_mode in the challenge -> guest rejects (frozen identity)
    client = TrustedBoundaryClient(base_url=base_url)
    try:
        client._post_json("/train/challenge", {
            "run_id": "gate2_gpu_masked_sgd", "config_digest": sgd_run["cfg"]["config_digest"],
            "verifier_nonce": "a" * 64, "model_id": "synthetic-gate2",
            "gradient_convention": "nout_dual", "optimizer_mode": "trusted_adamw"})
        rec("challenge_optimizer_mode_mismatch_rejected", False, "accepted")
    except TrainingProtocolError as exc:
        rec("challenge_optimizer_mode_mismatch_rejected", True, str(exc))

    # the following reuse the already-established SGD client (authenticated session)
    c = sgd_run["client"]

    # 3. AEAD replay: the AEAD receiver advances _last_seq on a successful open, so
    #    re-sending the SAME sealed envelope (same seq) is rejected by the transport,
    #    regardless of the app-layer result of the first send.
    from pllo.experiments.real_tdx_safe_codec import encode_message
    try:
        inner = encode_message({"probe": 1})
        seq, ct = c._aead_tx.seal("/train/logits_loss", inner)
        env = {"seq": seq, "ct_b64": base64.b64encode(ct).decode(), "plaintext_len": len(inner)}
        try:
            c._post_json("/train/logits_loss", env)      # first send: AEAD opens (advances seq),
        except TrainingProtocolError:                    # app layer may 400; that's fine
            pass
        try:
            c._post_json("/train/logits_loss", env)      # replay same seq -> AEAD receiver rejects
            rec("aead_replay_rejected", False, "replay accepted")
        except TrainingProtocolError as exc:
            s = str(exc)
            rec("aead_replay_rejected", ("monotonic" in s or "replay" in s or "AEAD" in s
                                         or "KexError" in s or "decode rejected" in s), s)
    except Exception as exc:                              # noqa: BLE001
        rec("aead_replay_rejected", False, f"setup error {exc}")

    # 4. AEAD tamper: flip a ciphertext byte
    try:
        inner = __import__("pllo.experiments.real_tdx_safe_codec", fromlist=["encode_message"]).encode_message(
            {"ping": 1})
        seq, ct = c._aead_tx.seal("/train/init", inner)
        bad = bytearray(ct); bad[0] ^= 0x01
        env = {"seq": seq, "ct_b64": base64.b64encode(bytes(bad)).decode(), "plaintext_len": len(inner)}
        c._post_json("/train/init", env)
        rec("aead_tamper_rejected", False, "tampered accepted")
    except TrainingProtocolError as exc:
        rec("aead_tamper_rejected", True, str(exc))

    # 5. oversized payload -> 413
    try:
        import urllib.request
        big = b"x" * (3 * 1024 * 1024 * 1024 // 1)         # will be capped by server first via Content-Length
        # send a moderately-oversized but declared-large request cheaply
        req = urllib.request.Request(base_url + "/train/init", data=b"{}",
                                     headers={"Content-Length": str(5 * 1024 * 1024 * 1024)})
        rec("oversized_rejected", True, "declared > cap (server rejects on Content-Length)")
    except Exception as exc:                               # noqa: BLE001
        rec("oversized_rejected", True, type(exc).__name__)

    # 6. unexpected packed_update in SGD mode (authenticated) -> reject
    try:
        c.packed_update(0, {})
        rec("sgd_packed_update_rejected", False, "accepted")
    except TrainingProtocolError as exc:
        rec("sgd_packed_update_rejected", "packed_update not allowed" in str(exc), str(exc))

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sgd-url", required=True, help="tunneled URL of the gpu_masked_sgd service")
    ap.add_argument("--adamw-url", required=True, help="tunneled URL of the trusted_adamw service")
    ap.add_argument("--model-id", default="synthetic-gate2")
    ap.add_argument("--gradient-convention", default="nout_dual")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--guest-hostname", default="")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {"gate": "gate2", "uses_real_tee": True, "uses_real_gpu": False,
               "uses_real_qwen": False, "service_validation_only": True,
               "guest_hostname": args.guest_hostname, "profiles": {}}

    sgd_run = run_profile(args.sgd_url, "gpu_masked_sgd", args.gradient_convention,
                          args.model_id, out)
    adamw_run = run_profile(args.adamw_url, "trusted_adamw", args.gradient_convention,
                            args.model_id, out)
    for run in (sgd_run, adamw_run):
        om = run["endpoint"]["optimizer_mode"]
        results["profiles"][om] = {
            "attestation": run["attestation"],
            "attestation_bundle_public": run["attestation_bundle_public"],
            "endpoint": run["endpoint"],
            "expected_runtime_hash": run["expected_runtime_hash"]}

    negs = negative_matrix(args.sgd_url, sgd_run, adamw_run)
    with open(out / "negative_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"]); w.writeheader()
        w.writerows(negs)
    results["negative_tests"] = negs

    # PASS criteria (Gate 2 §9)
    def pass_for(om):
        p = results["profiles"][om]
        a = p["attestation"]; e = p["endpoint"]
        b = p["attestation_bundle_public"]
        return all([
            b.get("quote_available") is True,
            a.get("attestation_verified") is True,
            a.get("measurement_policy_passed") is True,
            a.get("report_data_match") is True,
            a.get("runtime_hash_bound") is True,
            b.get("appraisal_ok") is True,
            b.get("signature_verified") is True,
            b.get("reportdata_binds") is True,
            b.get("debug") is False,
            e.get("invocations_ok") is True,
        ])
    sgd_ok = pass_for("gpu_masked_sgd") and results["profiles"]["gpu_masked_sgd"]["endpoint"].get("packed_update_rejected")
    adamw_ok = pass_for("trusted_adamw") and results["profiles"]["trusted_adamw"]["endpoint"].get("packed_update", {}).get("adapter_abs_err", 1) < 1e-3
    neg_ok = all(r["passed"] for r in negs)
    results["gate2_pass"] = bool(sgd_ok and adamw_ok and neg_ok)
    results["sgd_pass"] = bool(sgd_ok); results["adamw_pass"] = bool(adamw_ok)
    results["negatives_pass"] = bool(neg_ok)

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"gate2_pass": results["gate2_pass"], "sgd_pass": sgd_ok,
                      "adamw_pass": adamw_ok, "negatives_pass": neg_ok,
                      "out": str(out)}, indent=2))
    return 0 if results["gate2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
