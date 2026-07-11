"""GPU-side training client for the two trusted boundaries.

This is the protocol/transport state machine the GPU worker uses to talk to the
trusted TDX training service. It holds NO private secrets (no plaintext base
weights, no plaintext A/B, no labels, no optimizer state) -- only masked adapters
handed to it by the trusted service, and the masked tensors it computes on the GPU.

Two boundaries:
  logits_loss(step_id, masked_logits)  -> masked_logit_gradient
  packed_update(step_id, packed_grads) -> updated masked adapters

Supports two transports:
  - in-process: pass a TrustedTrainingService instance (Gate 1 contract tests)
  - http:       pass a base URL (Gate 2 deployment over the GPU<->TDX tunnel)
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import torch

from pllo.experiments.real_tdx_safe_codec import Limits, decode_message, encode_message


class TrainingProtocolError(Exception):
    """Raised when the trusted service rejects a request (fail-closed on GPU side too)."""


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


class TrustedBoundaryClient:
    def __init__(self, *, service=None, base_url: str | None = None,
                 codec: str = "safe", limits: Limits | None = None):
        if (service is None) == (base_url is None):
            raise ValueError("provide exactly one of service= or base_url=")
        self._service = service
        self._base_url = base_url.rstrip("/") if base_url else None
        if codec not in ("safe", "torch_save"):
            raise ValueError("codec must be 'safe' or 'torch_save'")
        self._codec = codec              # 'torch_save' is CPU-test only (unsafe pickle)
        self._limits = limits or Limits()
        self.run_id: str | None = None
        self.config_digest: str | None = None
        self._nonce: str | None = None
        self.masked_adapters: dict | None = None
        self.verification: dict | None = None
        self.attestation: dict | None = None
        self.optimizer_mode: str | None = None
        self.gradient_convention: str | None = None
        # attested session (Gate 2 §4): set by establish_attested_session()
        self._aead_tx = None            # AeadSender  (client->service)
        self._aead_rx = None            # AeadReceiver(service->client)
        self.attested = False
        self.attestation_result: dict | None = None
        self.attestation_bundle: dict | None = None

    # ---- attested session establishment (Gate 2 §3/§4) ----
    def establish_attested_session(self, *, run_id, config_digest, model_id,
                                   gradient_convention, optimizer_mode,
                                   expected_runtime_hash, policy=None,
                                   chain_verifier=None, require_attested=True) -> dict:
        """External-verifier + attested-ECDH handshake over the real wire path.

        Issues a fresh nonce, POSTs /train/challenge (guest binds it into report_data
        and produces a REAL quote), verifies the quote+bindings OFF-guest via
        ExternalAttestationVerifier + MeasurementPolicy, then POSTs /train/handshake
        and derives directional AEAD keys. ``chain_verifier(evidence)->bool`` supplies
        the cryptographic quote appraisal result (DCAP QVL). Fail-closed."""
        from pllo.experiments.real_tdx_attestation import ExternalAttestationVerifier
        from pllo.experiments.real_tdx_kex import (
            AeadReceiver,
            AeadSender,
            VerifierKeyExchange,
        )
        self.run_id = run_id
        self.config_digest = config_digest
        verifier = ExternalAttestationVerifier(policy=policy)
        challenge = verifier.issue_challenge(
            run_id=run_id, model_id=model_id, config_digest=config_digest,
            expected_runtime_hash=expected_runtime_hash,
            gradient_convention=gradient_convention, optimizer_mode=optimizer_mode)
        bundle = self._post_json("/train/challenge", {
            "run_id": run_id, "config_digest": config_digest,
            "verifier_nonce": challenge.verifier_nonce, "model_id": model_id,
            "gradient_convention": gradient_convention, "optimizer_mode": optimizer_mode})
        self.attestation_bundle = bundle
        evidence = {
            "report_data_hex": bundle.get("report_data_hex"),
            "runtime_hash_hex": bundle.get("runtime_hash_hex"),
            "mr_td": bundle.get("mr_td"), "debug": bundle.get("debug"),
            "quote_hash": bundle.get("quote_hash"), "nonce": challenge.verifier_nonce,
            "run_id": run_id, "model_id": model_id, "config_digest": config_digest,
            "gradient_convention": gradient_convention, "optimizer_mode": optimizer_mode,
            "guest_ephemeral_public_hex": bundle.get("guest_ephemeral_public_hex"),
            "quote_b64": bundle.get("quote_b64"),
            "appraisal_ok": bundle.get("appraisal_ok"),
            "reportdata_binds": bundle.get("reportdata_binds"),
            "rp_reportdata_binds": bundle.get("rp_reportdata_binds"),
            "signature_verified": bundle.get("signature_verified"),
            "overall_appraisal_result": bundle.get("overall_appraisal_result"),
            "jwt_parts": bundle.get("jwt_parts")}
        result = verifier.verify(evidence=evidence, challenge=challenge,
                                 chain_verifier=chain_verifier)
        self.attestation_result = result
        if require_attested and not result.get("attestation_verified"):
            raise TrainingProtocolError(
                "attestation not verified (fail-closed): "
                f"chain={result.get('quote_chain_verified')} "
                f"policy={result.get('measurement_policy_passed')} "
                f"report_data_match={result.get('report_data_match')}")
        # complete the attested ECDH (binds keys to the verified quote hash)
        vkex = VerifierKeyExchange()
        kx = vkex.complete(
            guest_public_hex=bundle["guest_ephemeral_public_hex"], run_id=run_id,
            config_digest=config_digest, optimizer_mode=optimizer_mode,
            gradient_convention=gradient_convention, quote_hash_hex=bundle["quote_hash"])
        self._post_json("/train/handshake", {"verifier_public_hex": vkex.public_hex})
        self._aead_tx = AeadSender(key=kx.c2s_key, run_id=run_id, direction=1)
        self._aead_rx = AeadReceiver(key=kx.s2c_key, run_id=run_id, direction=2)
        self.attested = True
        return result

    def _post_json(self, path, obj):
        r = urllib.request.Request(self._base_url + path,
                                   data=json.dumps(obj).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=180) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            raise TrainingProtocolError(f"{path}: HTTP {exc.code}: {detail}")
        if isinstance(body, dict) and body.get("error"):
            raise TrainingProtocolError(f"{path}: {body['error']}")
        return body

    # ---- transport ----
    def _call(self, op: str, req: dict) -> dict:
        if self._service is not None:
            fn = {"init": self._service.init_session,
                  "logits_loss": self._service.logits_loss,
                  "packed_update": self._service.packed_update}[op]
            try:
                return fn(req)
            except Exception as exc:                       # fail-closed surfaces as error
                raise TrainingProtocolError(f"{op}: {type(exc).__name__}: {exc}") from exc
        path = {"init": "/train/init", "logits_loss": "/train/logits_loss",
                "packed_update": "/train/packed_update"}[op]
        # attested session: AEAD-seal the safe-codec payload (Gate 2 §6)
        if self.attested:
            import base64
            inner = encode_message(req)
            seq, ct = self._aead_tx.seal(path, inner)
            outer = self._post_json(path, {"seq": seq, "ct_b64": base64.b64encode(ct).decode(),
                                           "plaintext_len": len(inner)})
            import time as _t
            pt = self._aead_rx.open(endpoint=path, seq=int(outer["seq"]),
                                    ciphertext=base64.b64decode(outer["ct_b64"]),
                                    plaintext_len=int(outer["plaintext_len"]), now=_t.monotonic())
            return decode_message(pt, limits=self._limits)
        # unauthenticated http transport (cpu contract tests): safe binary codec
        if self._codec == "safe":
            payload = encode_message(req)
            ctype = "application/x-pllo-safe"
        else:                                    # cpu-test only
            payload = json.dumps(_encode_req(req)).encode()
            ctype = "application/json"
        r = urllib.request.Request(self._base_url + path, data=payload,
                                   headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(r, timeout=180) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            raise TrainingProtocolError(f"{op}: HTTP {exc.code}: {detail}")
        if self._codec == "safe":
            return decode_message(raw, limits=self._limits)
        body = json.loads(raw.decode())
        if body.get("error"):
            raise TrainingProtocolError(f"{op}: {body['error']}")
        return _decode_resp(body)

    # ---- session ----
    def init_session(self, *, run_id, config_digest, lora_manifest, vocab_size,
                     labels_by_step) -> dict:
        self.run_id = run_id
        self.config_digest = config_digest
        resp = self._call("init", {
            "run_id": run_id, "config_digest": config_digest,
            "lora_manifest": lora_manifest, "vocab_size": vocab_size,
            "labels_by_step": labels_by_step})
        self._nonce = resp["next_nonce"]
        self.masked_adapters = resp["masked_adapters"]
        self.verification = resp["verification"]
        self.attestation = resp["attestation"]
        # frozen session identity echoed by the service (no implicit switching)
        self.optimizer_mode = self.verification.get("optimizer_mode")
        self.gradient_convention = self.verification.get("gradient_convention")
        return resp

    def _identity(self) -> dict:
        d = {}
        if self.optimizer_mode is not None:
            d["optimizer_mode"] = self.optimizer_mode
        if self.gradient_convention is not None:
            d["gradient_convention"] = self.gradient_convention
        return d

    # ---- boundary 1 ----
    def logits_loss(self, step_id: int, masked_logits: torch.Tensor) -> dict:
        req = {"run_id": self.run_id, "step_id": int(step_id),
               "config_digest": self.config_digest, "nonce": self._nonce,
               "manifest": {"masked_logits": {"shape": list(masked_logits.shape),
                                              "dtype": _dtype_str(masked_logits)}},
               "masked_logits": masked_logits, **self._identity()}
        resp = self._call("logits_loss", req)
        self._nonce = resp["next_nonce"]
        return resp

    # ---- boundary 2 ----
    def packed_update(self, step_id: int, packed_grads: dict) -> dict:
        manifest = {"layers": {lid: {"gradA": {"shape": list(g["gradA"].shape),
                                               "dtype": _dtype_str(g["gradA"])},
                                     "gradB": {"shape": list(g["gradB"].shape),
                                               "dtype": _dtype_str(g["gradB"])}}
                               for lid, g in packed_grads.items()}}
        req = {"run_id": self.run_id, "step_id": int(step_id),
               "config_digest": self.config_digest, "nonce": self._nonce,
               "manifest": manifest, "packed_grads": packed_grads, **self._identity()}
        resp = self._call("packed_update", req)
        self._nonce = resp["next_nonce"]
        self.masked_adapters = resp["updated_masked_adapters"]
        return resp

    # deliberately expose a way to corrupt the nonce/manifest for fail-closed tests
    def _force_nonce(self, value):
        self._nonce = value


# ---- HTTP (de)serialization: dtype-agnostic tensor codec (torch.save+base64) ----
# torch.save handles every dtype incl. bfloat16 (numpy cannot), so grads/logits in
# bf16/fp32/fp64 all round-trip losslessly over the GPU<->TDX tunnel.
def _t2j(t: torch.Tensor):
    buf = io.BytesIO()
    torch.save(t.contiguous().cpu(), buf)
    return {"__tensor__": base64.b64encode(buf.getvalue()).decode()}


def _j2t(d):
    buf = io.BytesIO(base64.b64decode(d["__tensor__"]))
    return torch.load(buf, weights_only=True)


def _encode_req(obj):
    return _walk(obj, torch.Tensor, _t2j)


def _decode_resp(obj):
    return _walk(obj, dict, _j2t, key="__tensor__")


def _walk(obj, hit_type, fn, key=None):
    if key is None and isinstance(obj, hit_type):
        return fn(obj)
    if isinstance(obj, dict):
        if key is not None and key in obj:
            return fn(obj)
        return {k: _walk(v, hit_type, fn, key) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, hit_type, fn, key) for v in obj]
    return obj
