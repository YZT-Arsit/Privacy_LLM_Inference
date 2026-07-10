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
        # http transport (Gate 2): safe binary codec (no pickle), POST octet-stream
        path = {"init": "/train/init", "logits_loss": "/train/logits_loss",
                "packed_update": "/train/packed_update"}[op]
        if self._codec == "safe":
            payload = encode_message(req)
            ctype = "application/x-pllo-safe"
        else:                                    # cpu-test only
            payload = json.dumps(_encode_req(req)).encode()
            ctype = "application/json"
        r = urllib.request.Request(self._base_url + path, data=payload,
                                   headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
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
        return resp

    # ---- boundary 1 ----
    def logits_loss(self, step_id: int, masked_logits: torch.Tensor) -> dict:
        req = {"run_id": self.run_id, "step_id": int(step_id),
               "config_digest": self.config_digest, "nonce": self._nonce,
               "manifest": {"masked_logits": {"shape": list(masked_logits.shape),
                                              "dtype": _dtype_str(masked_logits)}},
               "masked_logits": masked_logits}
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
               "manifest": manifest, "packed_grads": packed_grads}
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
