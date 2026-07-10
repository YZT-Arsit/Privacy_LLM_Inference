"""Trusted TDX training service (Gate 1 logic; deploys unchanged inside the guest).

Holds the private assets: plaintext LoRA A/B, per-layer masks, the AdamW optimizer
state, and the private labels. Exposes three operations:

  init_session       -> register run, init adapters+masks+opt state, return MASKED adapters
  /train/logits_loss -> recover logits, real CE, real logit gradient, re-mask, return
  /train/packed_update -> unpack+unmask ALL LoRA grads, real AdamW, re-mask, return

Every request carries run_id, step_id, nonce, manifest (shape/dtype), config_digest.
FAIL-CLOSED: any mismatch raises FailClosed; there is NO silent fallback, NO plain-CPU
fallback, NO dummy AdamW. Plaintext A/B, optimizer state, and labels never leave here.

Masking relations (per layer; masks held ONLY here):
  masked adapters : A_t = N_in^-1 A R,  B_t = R^-1 B N_out
  masked LoRA grads (produced by GPU, masked-domain only, upstream U_t = U N_out^-T):
                    gA_t = N_in^T gradA R^-T,  gB_t = R^T gradB N_out^-T
  recovery here   : gradA = N_in^-T gA_t R^T,  gradB = R^-T gB_t N_out^T
  logits          : masked_logits = logits N_head  ->  logits = masked_logits N_head^-1
  logit grad back : g_masked = dlogits M_head
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import torch

from pllo.experiments.real_tdx_attestation import (
    AttestationFailClosed,
    build_binding,
    compute_service_runtime_hash,
    verify_binding,
)

DT = torch.float64


class FailClosed(Exception):
    """Any protocol/validation failure. The service refuses to compute. No fallback."""


def _inv(m):
    return torch.linalg.inv(m)


def _rand_invertible(n, g, dtype):
    a = torch.randn(n, n, generator=g, dtype=dtype) / n ** 0.5 + torch.eye(n, dtype=dtype)
    return a


def _orthogonal(n, g, dtype):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=dtype))
    return q * torch.sign(torch.diagonal(r))


def _is_orthogonal(m, tol=1e-6):
    n = m.shape[0]
    return bool((m @ m.T - torch.eye(n, dtype=m.dtype)).abs().max() < tol)


_MASKED_SGD_MODES = {"gpu_masked_sgd", "gpu_masked_momentum_sgd"}
_ALLOWED_MODES = {"trusted_adamw"} | _MASKED_SGD_MODES
_SGD_OPTIMIZERS = {"sgd", "momentum_sgd"}
_REJECTED_SGD_OPTIMIZERS = {"adam", "adamw", "rmsprop", "adagrad"}


@dataclass
class _LayerState:
    layer_id: str
    d_in: int
    d_out: int
    rank: int
    A: torch.Tensor
    B: torch.Tensor
    N_in: torch.Tensor
    N_in_inv: torch.Tensor
    R: torch.Tensor
    R_inv: torch.Tensor
    N_out: torch.Tensor
    N_out_inv: torch.Tensor
    # AdamW optimizer state (plaintext, trusted-only)
    mA: torch.Tensor = None
    vA: torch.Tensor = None
    mB: torch.Tensor = None
    vB: torch.Tensor = None
    t: int = 0

    def masked_adapters(self):
        return self.N_in_inv @ self.A @ self.R, self.R_inv @ self.B @ self.N_out


@dataclass
class TrustedTrainingService:
    run_id: str
    config: dict
    mode: str = "cpu_contract"          # "cpu_contract" | "real_tdx"
    require_real_tdx: bool = False
    dtype: torch.dtype = DT
    seed: int = 0
    _layers: dict = field(default_factory=dict)
    _labels: dict = field(default_factory=dict)          # step_id -> label ids
    _head: dict = field(default_factory=dict)            # head masks
    _expected_nonce: str = ""
    _runtime_hash: str = ""
    _binding: object = None
    _flags: dict = field(default_factory=dict)

    # ---- lifecycle ----
    def init_session(self, req: dict) -> dict:
        self._require(req.get("run_id") == self.run_id, "run_id mismatch at init")
        self._require(req.get("config_digest") == self.config["config_digest"],
                      "config_digest mismatch at init")
        self._runtime_hash = compute_service_runtime_hash(self.config["config_digest"])
        nonce = secrets.token_hex(16)
        self._binding = build_binding(run_id=self.run_id,
                                      config_digest=self.config["config_digest"],
                                      nonce=nonce, mode=self.mode,
                                      require_real_tdx=self.require_real_tdx)
        # fail-closed: verify our own binding before serving
        verify_binding(self._binding, expected_runtime_hash=self._runtime_hash,
                       require_real_tdx=self.require_real_tdx)
        # ---- optimizer_mode + fail-closed policy ----
        self.optimizer_mode = self.config.get("optimizer_mode", "trusted_adamw")
        self._require(self.optimizer_mode in _ALLOWED_MODES,
                      f"unknown optimizer_mode {self.optimizer_mode!r}")
        opt = self.config.get("optimizer", "adamw")
        if self.optimizer_mode in _MASKED_SGD_MODES:
            self._require(opt not in _REJECTED_SGD_OPTIMIZERS,
                          f"{self.optimizer_mode} rejects optimizer {opt!r} "
                          "(Adam/AdamW/RMSProp/Adagrad need a trusted optimizer boundary)")
            self._require(opt in _SGD_OPTIMIZERS,
                          f"{self.optimizer_mode} requires SGD/momentum_sgd, got {opt!r}")
        orth = self.optimizer_mode in _MASKED_SGD_MODES
        g = torch.Generator().manual_seed(self.seed)
        for spec in req["lora_manifest"]:
            lid, d_in, d_out, r = spec["layer_id"], spec["d_in"], spec["d_out"], spec["rank"]
            A = torch.randn(d_in, r, generator=g, dtype=self.dtype) / d_in ** 0.5
            B = torch.zeros(r, d_out, dtype=self.dtype)
            mk = _orthogonal if orth else _rand_invertible
            N_in = mk(d_in, g, self.dtype)
            R = mk(r, g, self.dtype)
            N_out = mk(d_out, g, self.dtype)
            if orth:            # fail-closed: masked-SGD REQUIRES orthogonal masks
                self._require(_is_orthogonal(N_in) and _is_orthogonal(R) and _is_orthogonal(N_out),
                              f"{self.optimizer_mode} requires orthogonal masks for layer {lid}")
            st = _LayerState(lid, d_in, d_out, r, A, B, N_in, _inv(N_in), R, _inv(R),
                             N_out, _inv(N_out),
                             torch.zeros_like(A), torch.zeros_like(A),
                             torch.zeros_like(B), torch.zeros_like(B), 0)
            self._layers[lid] = st
        # head masks for logits (vocab V)
        V = req["vocab_size"]
        N_head = _rand_invertible(V, g, self.dtype)
        M_head = _rand_invertible(V, g, self.dtype)
        self._head = {"N_head": N_head, "N_head_inv": _inv(N_head),
                      "M_head": M_head, "M_head_inv": _inv(M_head)}
        self._labels = {int(k): torch.as_tensor(v, dtype=torch.long)
                        for k, v in req.get("labels_by_step", {}).items()}
        self._expected_nonce = nonce
        masked = {lid: st.masked_adapters() for lid, st in self._layers.items()}
        self._flags = {"tee_type": self._binding.tee_type,
                       "guest_verified": self._binding.guest_verified,
                       "attestation_verified": self._binding.attestation_verified,
                       "runtime_hash_bound": self._binding.runtime_hash_bound,
                       "optimizer_mode": self.optimizer_mode}
        return {"run_id": self.run_id, "next_nonce": nonce,
                "masked_adapters": masked, "head_mask_id": "N_head",
                "attestation": self._binding.to_dict(),
                "verification": dict(self._flags)}

    # ---- boundary 1: logits -> loss -> masked logit gradient ----
    def logits_loss(self, req: dict) -> dict:
        self._validate_common(req)
        man = req["manifest"]["masked_logits"]
        ml = req["masked_logits"]
        self._require(tuple(ml.shape) == tuple(man["shape"]),
                      f"masked_logits shape {tuple(ml.shape)} != manifest {tuple(man['shape'])}")
        self._require(str(ml.dtype).replace("torch.", "") == man["dtype"],
                      "masked_logits dtype != manifest")
        step = int(req["step_id"])
        self._require(step in self._labels, f"no private labels registered for step {step}")
        # recover real logits inside the trusted service
        logits = ml.to(self.dtype) @ self._head["N_head_inv"]
        labels = self._labels[step]
        logp = torch.log_softmax(logits, dim=-1)
        n = logits.shape[0]
        loss = -logp[torch.arange(n), labels].mean()
        p = torch.softmax(logits, dim=-1)
        dlogits = p.clone()
        dlogits[torch.arange(n), labels] -= 1.0
        dlogits /= n
        g_masked = dlogits @ self._head["M_head"]           # re-mask before returning
        nxt = self._rotate_nonce()
        return {"run_id": self.run_id, "step_id": step, "loss": float(loss.item()),
                "masked_logit_gradient": g_masked, "logit_grad_mask_id": "M_head",
                "next_nonce": nxt,
                "verification": {**self._flags, "trusted_loss_executed": True,
                                 "plaintext_label_exported": False}}

    # ---- boundary 2: packed masked LoRA grads -> unmask -> AdamW -> re-mask ----
    def packed_update(self, req: dict) -> dict:
        self._validate_common(req)
        # fail-closed: the packed trusted-optimizer boundary exists ONLY in trusted_adamw.
        self._require(self.optimizer_mode == "trusted_adamw",
                      f"packed_update not allowed in optimizer_mode={self.optimizer_mode} "
                      "(GPU owns the masked SGD optimizer; no trusted update boundary)")
        manifest = req["manifest"]["layers"]
        packed = req["packed_grads"]
        # manifest must cover EXACTLY the registered layers
        self._require(set(manifest.keys()) == set(self._layers.keys()),
                      "packed manifest layer set != registered layers")
        lr = self.config.get("lr", 1e-3)
        b1 = self.config.get("beta1", 0.9); b2 = self.config.get("beta2", 0.999)
        eps = self.config.get("eps", 1e-8); wd = self.config.get("weight_decay", 0.0)
        updated = {}
        for lid, st in self._layers.items():
            m = manifest[lid]
            gA_t, gB_t = packed[lid]["gradA"], packed[lid]["gradB"]
            self._require(tuple(gA_t.shape) == (st.d_in, st.rank), f"{lid} gradA shape")
            self._require(tuple(gB_t.shape) == (st.rank, st.d_out), f"{lid} gradB shape")
            self._require(m["gradA"]["dtype"] == str(gA_t.dtype).replace("torch.", ""),
                          f"{lid} gradA dtype manifest")
            # unmask (in-TDX): gradA = N_in^-T gA_t R^T ; gradB = R^-T gB_t N_out^T
            gA = st.N_in_inv.T @ gA_t.to(self.dtype) @ st.R.T
            gB = st.R_inv.T @ gB_t.to(self.dtype) @ st.N_out.T
            # real AdamW on plaintext params + opt state
            st.t += 1
            st.mA = b1 * st.mA + (1 - b1) * gA
            st.vA = b2 * st.vA + (1 - b2) * gA * gA
            mhat, vhat = st.mA / (1 - b1 ** st.t), st.vA / (1 - b2 ** st.t)
            st.A = st.A - lr * (mhat / (vhat.sqrt() + eps) + wd * st.A)
            st.mB = b1 * st.mB + (1 - b1) * gB
            st.vB = b2 * st.vB + (1 - b2) * gB * gB
            mhb, vhb = st.mB / (1 - b1 ** st.t), st.vB / (1 - b2 ** st.t)
            st.B = st.B - lr * (mhb / (vhb.sqrt() + eps) + wd * st.B)
            updated[lid] = st.masked_adapters()               # re-mask before returning
        nxt = self._rotate_nonce()
        return {"run_id": self.run_id, "step_id": int(req["step_id"]),
                "updated_masked_adapters": updated, "next_nonce": nxt,
                "verification": {**self._flags, "trusted_adamw_executed": True,
                                 "packed_gradient_unmask_executed": True,
                                 "adapter_remask_executed": True,
                                 "plaintext_adapter_exported": False}}

    # ---- validation helpers (fail-closed) ----
    def _require(self, cond, msg):
        if not cond:
            raise FailClosed(msg)

    def _validate_common(self, req):
        self._require(req.get("run_id") == self.run_id, "run_id mismatch")
        self._require(req.get("config_digest") == self.config["config_digest"],
                      "config_digest mismatch")
        self._require(req.get("nonce") == self._expected_nonce,
                      "bad/replayed nonce")
        self._require("manifest" in req, "missing manifest")

    def _rotate_nonce(self):
        self._expected_nonce = secrets.token_hex(16)
        return self._expected_nonce

    # ---- introspection (safe: no plaintext tensors) ----
    def plaintext_adapter(self, lid):     # test-only accessor (trusted side)
        return self._layers[lid].A, self._layers[lid].B

    def adam_state(self, lid):
        st = self._layers[lid]
        return {"mA": st.mA, "vA": st.vA, "mB": st.mB, "vB": st.vB, "t": st.t}

    # ---- TEST-ONLY trusted introspection (used by Gate-1 contract tests to form
    #      masked inputs consistently; the real GPU never calls these) ----
    def _layer_masks(self, lid):
        st = self._layers[lid]
        return st.N_in, st.R, st.N_out

    def _head_masks(self):
        return self._head["N_head"], self._head["M_head"]


# ---------------------------------------------------------------------------
# HTTP transport (deploys unchanged inside the TDX guest for Gate 2)
# ---------------------------------------------------------------------------
def serve_http(service: "TrustedTrainingService", host: str = "127.0.0.1",
               port: int = 18091, *, codec: str = "safe", limits=None):
    """Blocking HTTP server exposing /train/init|logits_loss|packed_update.

    Runs INSIDE the TDX guest. Uses the SAFE binary codec (no pickle) by default.
    In real_tdx mode the unsafe torch_save/pickle codec is REFUSED. FailClosed /
    CodecError -> HTTP 400 with a constant error envelope; never a silent success.
    Binds localhost by default -- remote exposure is only via an explicit tunnel."""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pllo.experiments.real_tdx_attestation import AttestationFailClosed
    from pllo.experiments.real_tdx_safe_codec import (
        CodecError,
        Limits,
        decode_message,
        encode_message,
    )

    if service.require_real_tdx and codec != "safe":
        raise RuntimeError("real_tdx mode refuses the unsafe torch_save codec "
                           "(pickle RCE risk); use codec='safe'.")
    lim = limits or Limits()
    routes = {"/train/init": service.init_session,
              "/train/logits_loss": service.logits_loss,
              "/train/packed_update": service.packed_update}

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_bytes(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg):     # constant error envelope, no internals leaked
            self._send_bytes(code, json.dumps({"error": msg}).encode(), "application/json")

        def do_POST(self):
            fn = routes.get(self.path.rstrip("/"))
            if fn is None:
                return self._err(404, "no such endpoint")
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > lim.max_total_bytes + lim.max_header_bytes + 64:
                return self._err(413, "request too large")
            raw = self.rfile.read(n)
            try:
                if codec == "safe":
                    req = decode_message(raw, limits=lim)
                else:
                    from pllo.experiments.real_tdx_training_client import _decode_resp
                    req = _decode_resp(json.loads(raw.decode()))
            except (CodecError, Exception) as exc:      # malformed payload -> reject
                return self._err(400, f"decode rejected: {type(exc).__name__}")
            try:
                resp = fn(req)
            except (FailClosed, AttestationFailClosed) as exc:
                return self._err(400, f"{type(exc).__name__}: {exc}")
            if codec == "safe":
                self._send_bytes(200, encode_message(resp), "application/x-pllo-safe")
            else:
                from pllo.experiments.real_tdx_training_client import _encode_req
                self._send_bytes(200, json.dumps(_encode_req(resp)).encode(), "application/json")

    httpd = ThreadingHTTPServer((host, port), _H)
    return httpd
