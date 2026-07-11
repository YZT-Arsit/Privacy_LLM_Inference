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
# Gate 2 §1-B: the two documented (both-correct) masked-backward gradient
# conventions. Frozen at init and bound into config_digest + report_data. This
# service build implements the nout_dual recovery (gradB = R^-T gB_t N_out^T);
# independent_mout is a valid but distinct convention that this build does NOT
# implement -> it is accepted as a config value only to be rejected fail-closed,
# never silently coerced.
_GRADIENT_CONVENTIONS = {"nout_dual", "independent_mout"}
_IMPLEMENTED_CONVENTIONS = {"nout_dual"}


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
        # ---- optimizer_mode + gradient_convention frozen at init (Gate 2 §1-B/§1-C) ----
        # Both MUST be present in config so they are covered by config_digest (which
        # is bound into the attestation report_data); missing -> fail-closed, never
        # defaulted implicitly.
        self._require("optimizer_mode" in self.config,
                      "config missing optimizer_mode (must be in config_digest)")
        self._require("gradient_convention" in self.config,
                      "config missing gradient_convention (must be in config_digest)")
        self.optimizer_mode = self.config["optimizer_mode"]
        self.gradient_convention = self.config["gradient_convention"]
        self._require(self.optimizer_mode in _ALLOWED_MODES,
                      f"unknown optimizer_mode {self.optimizer_mode!r}")
        self._require(self.gradient_convention in _GRADIENT_CONVENTIONS,
                      f"unknown gradient_convention {self.gradient_convention!r}")
        self._require(self.gradient_convention in _IMPLEMENTED_CONVENTIONS,
                      f"gradient_convention {self.gradient_convention!r} not implemented "
                      "in this service build (only nout_dual); refusing to start")
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
        # head mask for logits. Two families:
        #   dense           : V x V invertible mask (Gate-1/2; only for TINY vocab)
        #   vocab_permutation: O(V) permutation pi (Gate-3 real vocab; NO dense matrix)
        V = req["vocab_size"]
        self.logits_mask_family = self.config.get("logits_mask_family", "dense")
        self._require(self.logits_mask_family in ("dense", "vocab_permutation"),
                      f"unknown logits_mask_family {self.logits_mask_family!r}")
        extra = {}
        if self.logits_mask_family == "vocab_permutation":
            pi = torch.randperm(V, generator=g)
            pi_inv = torch.empty_like(pi)
            pi_inv[pi] = torch.arange(V)
            # bijection + inverse verified before use (fail-closed)
            self._require(int(pi.numel()) == V and int(torch.unique(pi).numel()) == V,
                          "logit permutation is not a bijection of length vocab")
            self._require(bool((pi[pi_inv] == torch.arange(V)).all()),
                          "logit permutation inverse verification failed")
            self._head = {"family": "vocab_permutation", "pi": pi, "pi_inv": pi_inv, "V": V}
            # deliver pi to the AUTHORIZED (attested) client over the AEAD session so it
            # can form Z_tilde = Z[:, pi]. (Under a folded LM head the client would not
            # receive pi; with a plaintext base this is authorized initialization.)
            extra = {"logit_permutation": pi.to(torch.int64), "logits_mask_family": "vocab_permutation"}
        else:
            self._require(V <= 4096, "dense head mask only for tiny vocab; use vocab_permutation")
            N_head = _rand_invertible(V, g, self.dtype)
            M_head = _rand_invertible(V, g, self.dtype)
            self._head = {"family": "dense", "N_head": N_head, "N_head_inv": _inv(N_head),
                          "M_head": M_head, "M_head_inv": _inv(M_head), "V": V}
        self._labels = {int(k): torch.as_tensor(v, dtype=torch.long)
                        for k, v in req.get("labels_by_step", {}).items()}
        self._expected_nonce = nonce
        masked = {lid: st.masked_adapters() for lid, st in self._layers.items()}
        self._flags = {"tee_type": self._binding.tee_type,
                       "guest_verified": self._binding.guest_verified,
                       "attestation_verified": self._binding.attestation_verified,
                       "runtime_hash_bound": self._binding.runtime_hash_bound,
                       "optimizer_mode": self.optimizer_mode,
                       "gradient_convention": self.gradient_convention,
                       "logits_mask_family": self.logits_mask_family}
        return {"run_id": self.run_id, "next_nonce": nonce,
                "masked_adapters": masked, "head_mask_id": self.logits_mask_family,
                "attestation": self._binding.to_dict(),
                "verification": dict(self._flags), **extra}

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
        labels = self._labels[step]                          # private; never leaves TDX
        n = int(ml.shape[0])
        self._require(int(labels.numel()) == n, "labels length != masked_logits rows")
        if self._head["family"] == "vocab_permutation":
            pi, pi_inv = self._head["pi"], self._head["pi_inv"]
            self._require(int(ml.shape[1]) == self._head["V"], "masked_logits vocab != V")
            # recover Z = Z_tilde[:, pi_inv]; CE accumulation in fp32 (bf16 wire ok).
            # memory-frugal for the 14GB TD guest at vocab=151936: reuse buffers, free
            # intermediates, avoid holding log_softmax AND softmax simultaneously.
            import torch.nn.functional as _F
            Z = ml.to(torch.float32).index_select(1, pi_inv)
            del ml
            valid = labels != -100                           # ignore-index / padding mask
            nv = int(valid.sum())
            self._require(nv > 0, "no valid (non-ignore) labels in batch")
            idx = torch.arange(n)[valid]
            loss = _F.cross_entropy(Z[valid], labels[valid], reduction="mean")
            G = torch.softmax(Z, dim=-1)                      # dL/dZ = softmax - onehot
            del Z
            G[idx, labels[valid]] -= 1.0
            G[~valid] = 0.0                                  # no gradient from ignored rows
            G /= nv
            g_masked = G.index_select(1, pi).to(torch.bfloat16)   # bf16 wire back
            del G                                            # G_tilde = G[:, pi], same layout
            mask_id = "vocab_permutation"
        else:
            man = req["manifest"]["masked_logits"]           # (dense path retains manifest use)
            logits = ml.to(self.dtype) @ self._head["N_head_inv"]
            logp = torch.log_softmax(logits, dim=-1)
            loss = -logp[torch.arange(n), labels].mean()
            p = torch.softmax(logits, dim=-1)
            dlogits = p.clone(); dlogits[torch.arange(n), labels] -= 1.0; dlogits /= n
            g_masked = dlogits @ self._head["M_head"]
            mask_id = "M_head"
        nxt = self._rotate_nonce()
        return {"run_id": self.run_id, "step_id": step, "loss": float(loss.item()),
                "masked_logit_gradient": g_masked, "logit_grad_mask_id": mask_id,
                "next_nonce": nxt,
                "verification": {**self._flags, "trusted_loss_executed": True,
                                 "logits_mask_family": self._head["family"],
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
        # Gate 2 §1-B/§1-C: no implicit mode/convention switching. If the request
        # declares them they MUST equal the session's frozen values.
        if "optimizer_mode" in req:
            self._require(req["optimizer_mode"] == self.optimizer_mode,
                          f"optimizer_mode switch rejected "
                          f"({req['optimizer_mode']!r} != session {self.optimizer_mode!r})")
        if "gradient_convention" in req:
            self._require(req["gradient_convention"] == self.gradient_convention,
                          f"gradient_convention switch rejected "
                          f"({req['gradient_convention']!r} != session {self.gradient_convention!r})")

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
# Gate 2 §4 — guest-side attested session (challenge -> real quote -> ECDH -> AEAD)
# ---------------------------------------------------------------------------
class GuestAttestedSession:
    """Holds the guest ephemeral ECDH key + derived AEAD keys for one attested
    session. The ephemeral private key never leaves the process (never touches disk).

    /train/challenge : bind (guest ECDH pubkey || verifier nonce || session fields)
                       into report_data and produce a REAL TD Quote (real_tdx) that
                       the external verifier appraises.
    /train/handshake : complete ECDH with the verifier's public key -> directional
                       AEAD keys; the session becomes authenticated.
    Subsequent /train/{init,logits_loss,packed_update} bodies are AEAD-sealed."""

    def __init__(self, service: "TrustedTrainingService", *, quote_workdir=None):
        from pllo.experiments.real_tdx_kex import GuestKeyExchange
        self.service = service
        self._kex = GuestKeyExchange()
        self.guest_public_hex = self._kex.public_hex
        self._quote_workdir = quote_workdir or "/tmp/pllo_gate2_quote"
        self.runtime_hash = compute_service_runtime_hash(service.config["config_digest"])
        # frozen identity is sourced from config (challenge precedes init_session)
        if "optimizer_mode" not in service.config or "gradient_convention" not in service.config:
            raise FailClosed("config missing optimizer_mode/gradient_convention "
                             "(must be in config_digest before attestation)")
        self.optimizer_mode = service.config["optimizer_mode"]
        self.gradient_convention = service.config["gradient_convention"]
        self.report_data_hex = None
        self.quote_hash_hex = None
        self.challenge = None
        self._rx = None                 # AeadReceiver (client->service)
        self._tx = None                 # AeadSender (service->client)
        self.authenticated = False

    def on_challenge(self, req: dict) -> dict:
        from pllo.experiments.real_tdx_attestation import compute_report_data
        svc = self.service
        # the challenge fields MUST match this service's frozen identity
        if req.get("run_id") != svc.run_id:
            raise FailClosed("challenge run_id mismatch")
        if req.get("config_digest") != svc.config["config_digest"]:
            raise FailClosed("challenge config_digest mismatch")
        for f, want in (("optimizer_mode", self.optimizer_mode),
                        ("gradient_convention", self.gradient_convention)):
            if req.get(f) is not None and req.get(f) != want:
                raise FailClosed(f"challenge {f} mismatch")
        nonce = req.get("verifier_nonce")
        if not isinstance(nonce, str) or len(nonce) < 16:
            raise FailClosed("missing/short verifier_nonce")
        model_id = svc.config.get("model_id", "unknown")
        rd = compute_report_data(
            runtime_hash_hex=self.runtime_hash, config_digest=svc.config["config_digest"],
            run_id=svc.run_id, model_id=model_id,
            gradient_convention=self.gradient_convention, optimizer_mode=self.optimizer_mode,
            guest_ephemeral_public_hex=self.guest_public_hex, verifier_nonce=nonce)
        self.report_data_hex = rd
        self.challenge = dict(req)
        bundle = {"run_id": svc.run_id, "model_id": model_id,
                  "runtime_hash_hex": self.runtime_hash, "config_digest": svc.config["config_digest"],
                  "gradient_convention": self.gradient_convention,
                  "optimizer_mode": self.optimizer_mode,
                  "guest_ephemeral_public_hex": self.guest_public_hex,
                  "report_data_hex": rd, "nonce": nonce, "tee_type": svc.mode}
        if svc.mode == "real_tdx":
            from pllo.experiments.real_tdx_quote import generate_and_appraise, QuoteError
            try:
                qb = generate_and_appraise(rd, self._quote_workdir)
            except QuoteError as exc:
                raise FailClosed(f"real quote generation failed: {exc}")
            self.quote_hash_hex = qb.quote_hash
            bundle.update({
                "quote_available": True, "quote_b64": qb.quote_b64,
                "quote_hash": qb.quote_hash,
                "overall_appraisal_result": qb.overall_appraisal_result,
                "appraisal_ok": qb.appraisal_ok, "reportdata_binds": qb.reportdata_binds,
                "rp_reportdata_binds": qb.rp_reportdata_binds,
                "signature_verified": qb.signature_verified,
                "appraisal_source": qb.appraisal_source,
                "mr_td": qb.mr_td, "rtmr0": qb.rtmr0, "rtmr1": qb.rtmr1,
                "rtmr2": qb.rtmr2, "rtmr3": qb.rtmr3, "debug": qb.debug,
                "td_attributes": qb.td_attributes, "quote_version": qb.quote_version,
                "quote_tee_type": qb.tee_type, "jwt_parts": qb.jwt_parts,
                "appraisal_jwt": qb.appraisal_jwt})
        else:
            # cpu_contract: no real quote (honest); off-guest verifier keeps
            # attestation_verified=False. quote_hash salts the ECDH deterministically.
            import hashlib
            self.quote_hash_hex = hashlib.sha256(("cpu_contract:" + rd).encode()).hexdigest()
            bundle.update({"quote_available": False, "quote_hash": self.quote_hash_hex,
                           "appraisal_ok": False, "debug": None})
        return bundle

    def on_handshake(self, req: dict) -> dict:
        from pllo.experiments.real_tdx_kex import AeadReceiver, AeadSender
        svc = self.service
        vpub = req.get("verifier_public_hex")
        if not vpub:
            raise FailClosed("handshake missing verifier_public_hex")
        if self.report_data_hex is None or self.quote_hash_hex is None:
            raise FailClosed("handshake before challenge")
        res = self._kex.complete(
            verifier_public_hex=vpub, run_id=svc.run_id,
            config_digest=svc.config["config_digest"], optimizer_mode=self.optimizer_mode,
            gradient_convention=self.gradient_convention, quote_hash_hex=self.quote_hash_hex)
        timeout = float(svc.config.get("session_timeout_s", 300.0))
        self._rx = AeadReceiver(key=res.c2s_key, run_id=svc.run_id, direction=1,
                                timeout_s=timeout)
        self._tx = AeadSender(key=res.s2c_key, run_id=svc.run_id, direction=2)
        self.authenticated = True
        return {"run_id": svc.run_id, "authenticated": True}

    def open_request(self, endpoint: str, envelope: dict, *, now: float) -> bytes:
        if not self.authenticated:
            raise FailClosed("no authenticated session")
        import base64
        ct = base64.b64decode(envelope["ct_b64"])
        return self._rx.open(endpoint=endpoint, seq=int(envelope["seq"]), ciphertext=ct,
                             plaintext_len=int(envelope["plaintext_len"]), now=now)

    def seal_response(self, endpoint: str, plaintext: bytes) -> dict:
        import base64
        seq, ct = self._tx.seal(endpoint, plaintext)
        return {"seq": seq, "ct_b64": base64.b64encode(ct).decode(),
                "plaintext_len": len(plaintext)}


# ---------------------------------------------------------------------------
# HTTP transport (deploys unchanged inside the TDX guest for Gate 2)
# ---------------------------------------------------------------------------
def serve_http(service: "TrustedTrainingService", host: str = "127.0.0.1",
               port: int = 18091, *, codec: str = "safe", limits=None,
               require_authenticated_session: bool | None = None, quote_workdir=None):
    """Blocking HTTP server for the trusted training service (runs in the TDX guest).

    Endpoints:
      /train/challenge  (plaintext JSON) -> real-quote attestation bundle
      /train/handshake  (plaintext JSON) -> completes attested ECDH; authenticates
      /train/{init,logits_loss,packed_update} -> AEAD-sealed when a session is
          required (real_tdx), else plaintext SAFE binary codec (cpu contract tests).

    SAFE binary codec (no pickle) by default; real_tdx REFUSES torch_save/pickle.
    FailClosed / CodecError / KexError -> HTTP 400 with a constant error envelope;
    never a silent success. Binds localhost by default (tunnel-only exposure)."""
    import json
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pllo.experiments.real_tdx_attestation import AttestationFailClosed
    from pllo.experiments.real_tdx_kex import KexError
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
    if require_authenticated_session is None:
        require_authenticated_session = bool(service.require_real_tdx)
    session = GuestAttestedSession(service, quote_workdir=quote_workdir)
    compute_routes = {"/train/init": service.init_session,
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

        def _json(self, code, obj):
            self._send_bytes(code, json.dumps(obj).encode(), "application/json")

        def _err(self, code, msg):     # constant error envelope, no internals leaked
            self._json(code, {"error": msg})

        def _read(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > lim.max_total_bytes + lim.max_header_bytes + 64:
                return None
            return self.rfile.read(n)

        def do_POST(self):
            path = self.path.rstrip("/")
            raw = self._read()
            if raw is None:
                return self._err(413, "request too large")
            # ---- attestation handshake (plaintext control plane) ----
            if path in ("/train/challenge", "/train/handshake"):
                try:
                    req = json.loads(raw.decode())
                    resp = (session.on_challenge(req) if path == "/train/challenge"
                            else session.on_handshake(req))
                except (FailClosed, AttestationFailClosed, KexError) as exc:
                    return self._err(400, f"{type(exc).__name__}: {exc}")
                except Exception as exc:                       # malformed control msg
                    return self._err(400, f"control rejected: {type(exc).__name__}")
                return self._json(200, resp)
            # ---- compute plane ----
            fn = compute_routes.get(path)
            if fn is None:
                return self._err(404, "no such endpoint")
            try:
                if require_authenticated_session:
                    envelope = json.loads(raw.decode())
                    inner = session.open_request(path, envelope, now=time.monotonic())
                    req = decode_message(inner, limits=lim)
                elif codec == "safe":
                    req = decode_message(raw, limits=lim)
                else:
                    from pllo.experiments.real_tdx_training_client import _decode_resp
                    req = _decode_resp(json.loads(raw.decode()))
            except (CodecError, KexError, FailClosed) as exc:
                return self._err(400, f"decode rejected: {type(exc).__name__}: {exc}")
            except Exception as exc:
                return self._err(400, f"decode rejected: {type(exc).__name__}")
            try:
                resp = fn(req)
                if require_authenticated_session:
                    outer = session.seal_response(path, encode_message(resp))
                    return self._json(200, outer)
                if codec == "safe":
                    return self._send_bytes(200, encode_message(resp), "application/x-pllo-safe")
                from pllo.experiments.real_tdx_training_client import _encode_req
                return self._send_bytes(200, json.dumps(_encode_req(resp)).encode(),
                                        "application/json")
            except (FailClosed, AttestationFailClosed) as exc:
                return self._err(400, f"{type(exc).__name__}: {exc}")
            except (CodecError, KexError) as exc:       # response codec/seal failure
                return self._err(400, f"{type(exc).__name__}: {exc}")
            except Exception as exc:                     # never kill the socket silently
                return self._err(500, f"internal: {type(exc).__name__}")

    httpd = ThreadingHTTPServer((host, port), _H)
    httpd.pllo_session = session          # expose for evidence (no secrets)
    return httpd
