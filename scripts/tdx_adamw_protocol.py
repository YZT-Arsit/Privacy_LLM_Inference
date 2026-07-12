"""Trusted AdamW state protocol (O1-C L12) — runs INSIDE the TDX enclave.

Frozen Gate-0.5 split (proven exact in scripts/full_matrix_equivalence.py, worst 3.8e-13):
  TRUSTED (enclave holds theta_plain + m + v + step; A10 never sees these or gamma or plaintext grad):
    - A factors of {q,k,v,gate,up}  (input transform T_in = diag(gamma) @ Nr is a SCALED signed-perm:
      monomial but NOT unit, so AdamW's elementwise 2nd moment + eps does not cancel gamma -> enclave)
    - B factors of {q,k}            (output transform is a 2D RoPE-commuting rotation: NON-monomial)
  GPU-EXACT (A10 runs AdamW directly on the masked factor; monomial transform commutes with AdamW):
    - A of {o,down}; B of {o,down,v,gate,up}  (transforms are pure signed-perm/perm: unit monomial)

Enclave op flow per trusted factor:
  init : receive initial masked factor theta_tilde -> un-fold to theta_plain, set m=v=0, t=0
  step : receive masked grad g_tilde -> g_plain = un-fold(g_tilde); exact plaintext AdamW on
         (theta_plain, m, v); re-fold theta_plain -> theta_tilde; return theta_tilde only.
Never returns gamma / m / v / plaintext theta. Fails closed on missing/stale factor state.

Fold conventions (build_bundle, full_matrix_equivalence): A_tilde = A_plain @ Tin_iT ;
B_tilde = Tout.T @ B_plain ; gA_plain = gAc @ Tin_inv ; gB_plain = Tout_iT @ gBc.
"""
from __future__ import annotations
import hashlib, hmac, io, json, os
import torch

DT = torch.float64
TRUSTED_A = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")   # A-factor -> enclave
TRUSTED_B = ("q_proj", "k_proj")                                      # B-factor -> enclave
ATTN = ("q_proj", "k_proj", "v_proj")

SEAL_MAGIC_STD = b"L12AEADv1"       # ChaCha20-Poly1305 optimizer-state seal (paper-facing)
SEAL_MAGIC_PROTO = b"L12PROTOv0"    # SHA256-CTR+HMAC prototype (preserved history only)


# ============================ optimizer-state sealing ============================
# Standard AEAD: ChaCha20-Poly1305 (RFC 8439). Dedicated sealing key derived by HKDF-SHA256 from the
# attested session key, with DOMAIN SEPARATION from the transport HMAC/session key. Random unique 96-bit
# nonce per seal (ledgered to refuse reuse). AAD binds run_id/package/adapter/profile/version/checkpoint_seq/
# model-config-hash/service-hash. Fail closed on tamper/wrong-key/wrong-binding/stale-version/nonce-reuse.
def derive_seal_key(session_key: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=b"l12-optstate-seal-salt-v1",
                info=b"l12/optimizer-state/ChaCha20Poly1305/seal-key/domain-separated-from-transport"
                ).derive(session_key)


def canonical_aad(binding: dict, version: int, checkpoint_seq: int) -> bytes:
    # deterministic AAD over the full required binding set (order-independent via sorted keys)
    fields = {
        "run_id": binding["run_id"], "package_root_hash": binding["package_root_hash"],
        "adapter_id": binding["adapter_id"], "optimizer_profile": binding.get("optimizer_profile", ""),
        "model_config_hash": binding.get("model_config_hash", ""),
        "service_hash": binding.get("service_hash", ""),
        "optimizer_state_version": int(version), "checkpoint_sequence": int(checkpoint_seq),
        "aead": "ChaCha20Poly1305",
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()


def aead_seal_std(session_key: bytes, plaintext: bytes, aad: bytes, nonce_ledger: set | None = None) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    k = derive_seal_key(session_key)
    nonce = os.urandom(12)
    if nonce_ledger is not None:
        while nonce in nonce_ledger:                 # random 96-bit; collision astronomically unlikely
            nonce = os.urandom(12)
        nonce_ledger.add(nonce)
    ct = ChaCha20Poly1305(k).encrypt(nonce, plaintext, aad)   # ct includes the Poly1305 tag
    return SEAL_MAGIC_STD + nonce + ct


def aead_open_std(session_key: bytes, blob: bytes, aad: bytes, nonce_ledger: set | None = None) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    if not blob.startswith(SEAL_MAGIC_STD):
        raise ValueError("aead_open_std: not a standard-AEAD seal (magic mismatch)")
    body = blob[len(SEAL_MAGIC_STD):]; nonce, ct = body[:12], body[12:]
    if nonce_ledger is not None and nonce in nonce_ledger:
        raise ValueError("aead_open_std: nonce reuse detected (fail closed)")
    k = derive_seal_key(session_key)
    pt = ChaCha20Poly1305(k).decrypt(nonce, ct, aad)          # raises InvalidTag on tamper/wrong-key/wrong-AAD
    if nonce_ledger is not None:
        nonce_ledger.add(nonce)
    return pt


# ---- PRESERVED prototype (SHA256-CTR + HMAC-SHA256). prototype_checkpoint_sealing — NOT paper-facing.
def _proto_ks(k_enc: bytes, nonce: bytes, nbytes: int) -> bytes:
    out = bytearray(); ctr = 0
    while len(out) < nbytes:
        out += hashlib.sha256(k_enc + nonce + ctr.to_bytes(8, "big")).digest(); ctr += 1
    return bytes(out[:nbytes])


def prototype_aead_seal(session_key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    k_enc = hashlib.sha256(session_key + b"l12-ckpt-enc").digest()
    k_mac = hashlib.sha256(session_key + b"l12-ckpt-mac").digest()
    nonce = os.urandom(16); ks = _proto_ks(k_enc, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(k_mac, nonce + ct + aad, hashlib.sha256).digest()
    return SEAL_MAGIC_PROTO + nonce + tag + ct


def prototype_aead_open(session_key: bytes, blob: bytes, aad: bytes) -> bytes:
    blob = blob[len(SEAL_MAGIC_PROTO):] if blob.startswith(SEAL_MAGIC_PROTO) else blob
    k_enc = hashlib.sha256(session_key + b"l12-ckpt-enc").digest()
    k_mac = hashlib.sha256(session_key + b"l12-ckpt-mac").digest()
    nonce, tag, ct = blob[:16], blob[16:48], blob[48:]
    exp = hmac.new(k_mac, nonce + ct + aad, hashlib.sha256).digest()
    if not hmac.compare_digest(exp, tag):
        raise ValueError("prototype_aead_open: authentication failed")
    return bytes(a ^ b for a, b in zip(ct, _proto_ks(k_enc, nonce, len(ct))))


def orthogonal_signed_perm(n, seed, dtype=DT):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype); N[torch.arange(n), perm] = signs
    return N


def rope_rot(hd, seed, dtype=DT):
    half = hd // 2
    g = torch.Generator().manual_seed(int(seed))
    ang = torch.rand(half, generator=g, dtype=dtype) * 6.283185307179586
    B = torch.eye(hd, dtype=dtype)
    c, s = torch.cos(ang), torch.sin(ang)
    for i in range(half):
        j = i + half
        B[i, i] = c[i]; B[j, j] = c[i]; B[i, j] = -s[i]; B[j, i] = s[i]
    return B


def adamw_step(theta, g, m, v, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta = theta - lr * (mhat / (vhat.sqrt() + eps) + wd * theta)
    return theta, m, v


class TrustedAdamW:
    """Holds trusted plaintext factors + Adam moments for A of {q,k,v,gate,up} and B of {q,k}.

    gb: gamma bundle {"ga":{l:tensor}, "gm":{l:tensor}, "nr_seed", "num_layers", "hidden"}.
    """
    def __init__(self, gb, cfg, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, state_dtype=DT,
                 optimizer="adamw", mom=0.9):
        self.optimizer = optimizer   # "adamw" | "sgd" | "momentum" (plaintext-domain optimizer in enclave)
        self.mom = mom               # momentum coefficient (used only when optimizer == "momentum")
        # state_dtype: authoritative master + m + v dtype. FP32 for the frozen mixed-precision
        # deployment profile; FP64 for the exactness self-test. Transforms are precomputed in the
        # SAME dtype so the un-fold->AdamW->re-fold pipeline is self-consistent at that precision.
        self.sdt = state_dtype
        self.L = gb["num_layers"]; H = gb["hidden"]
        self.nh = cfg["num_attention_heads"]; self.nkv = cfg["num_key_value_heads"]
        self.hd = H // self.nh
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.version = 0                 # monotonic optimizer-state version (checkpoint/rollback guard)
        self.binding = None              # full binding dict — set via set_binding()
        self.checkpoint_seq = 0          # monotonic checkpoint counter (distinct from version)
        self.nonce_ledger = set()        # AEAD nonces issued/seen this enclave (nonce-reuse guard)
        Nr = orthogonal_signed_perm(H, gb["nr_seed"]).to(self.sdt)
        NrT = Nr.T
        # DEPLOYED package convention (verified against the trusted verifier + gram_inv=Nr^T diag(g^2) Nr):
        #   fold  M = diag(gamma) @ Nr        so  A_tilde = A_plain @ M
        #   grad  M^T = Nr^T diag(gamma)      so  gA_plain = gA_tilde @ M^T   (=> gA_plain@M = gA_tilde@gram_inv)
        #   unfold M^-1 = Nr^T diag(1/gamma)  so  A_plain  = A_tilde @ M^-1
        self.tinA = {}      # per (l,proj) -> (foldM, gradMT, unfoldMinv)
        self.toutB = {}     # per (l,proj) -> Tout (Bq/Bk); B is orthogonal (no gamma), already correct
        for l in range(self.L):
            ga = gb["ga"][l].to(self.sdt); gm = gb["gm"][l].to(self.sdt)
            for p in TRUSTED_A:
                g = ga if p in ATTN else gm
                foldM = torch.diag(g) @ Nr           # A_tilde = A_plain @ foldM
                gradMT = NrT @ torch.diag(g)         # gA_plain = gA_tilde @ gradMT
                unfoldMinv = NrT @ torch.diag(1.0 / g)  # A_plain = A_tilde @ unfoldMinv
                self.tinA[(l, p)] = (foldM, gradMT, unfoldMinv)
            Bl = rope_rot(self.hd, 1000 + l).to(self.sdt)
            self.toutB[(l, "q_proj")] = torch.block_diag(*([Bl] * self.nh))
            self.toutB[(l, "k_proj")] = torch.block_diag(*([Bl] * self.nkv))
        self.stateA = {}    # (l,proj) -> [theta_plain, m, v]
        self.stateB = {}
        self.t = 0

    def set_binding(self, run_id, package_root_hash, adapter_id, optimizer_profile="L12_adamw",
                    model_config_hash="", service_hash=""):
        self.binding = {"run_id": str(run_id), "package_root_hash": str(package_root_hash),
                        "adapter_id": str(adapter_id), "optimizer_profile": str(optimizer_profile),
                        "model_config_hash": str(model_config_hash), "service_hash": str(service_hash)}

    # ---- init: un-fold initial masked factors to plaintext, zero moments ----
    def init_factor(self, l, proj, A_tilde=None, B_tilde=None):
        if A_tilde is not None and proj in TRUSTED_A:
            foldM, gradMT, unfoldMinv = self.tinA[(l, proj)]
            Ap = A_tilde.to(self.sdt) @ unfoldMinv    # A_plain = A_tilde @ M^-1
            self.stateA[(l, proj)] = [Ap, torch.zeros_like(Ap), torch.zeros_like(Ap)]
        if B_tilde is not None and proj in TRUSTED_B:
            Tout = self.toutB[(l, proj)]
            # B_tilde = Tout.T @ B_plain  ->  B_plain = Tout @ B_tilde
            Bp = Tout @ B_tilde.to(self.sdt)
            self.stateB[(l, proj)] = [Bp, torch.zeros_like(Bp), torch.zeros_like(Bp)]

    def _update(self, theta, g_plain, m, v):
        # plaintext-domain optimizer step; slots: adamw uses (m,v); momentum uses m as buffer; sgd neither.
        if self.optimizer == "adamw":
            return adamw_step(theta, g_plain, m, v, self.t, self.lr, self.b1, self.b2, self.eps, self.wd)
        if self.optimizer == "momentum":
            m = self.mom * m + g_plain               # buffer accumulates PLAINTEXT-domain corrected grad
            return theta - self.lr * m, m, v
        return theta - self.lr * g_plain, m, v       # sgd

    # ---- step: masked grads in -> exact plaintext optimizer -> re-folded masked factors out ----
    def step(self, gA_tilde: dict, gB_tilde: dict):
        self.t += 1
        outA, outB = {}, {}
        expectedA = {(l, p) for l in range(self.L) for p in TRUSTED_A}
        expectedB = {(l, p) for l in range(self.L) for p in TRUSTED_B}
        gotA = set(); gotB = set()
        for k, gt in gA_tilde.items():
            l, proj = int(k.split(".", 1)[0]), k.split(".", 1)[1]
            if proj not in TRUSTED_A or (l, proj) not in self.stateA:
                raise KeyError(f"stale/missing trusted A state {k}")
            foldM, gradMT, unfoldMinv = self.tinA[(l, proj)]
            gA_plain = gt.to(self.sdt) @ gradMT       # gA_plain = gA_tilde @ M^T
            Ap, m, v = self.stateA[(l, proj)]
            Ap, m, v = self._update(Ap, gA_plain, m, v)
            self.stateA[(l, proj)] = [Ap, m, v]
            outA[k] = (Ap @ foldM)           # re-fold -> masked (A_tilde = A_plain @ M)
            gotA.add((l, proj))
        for k, gt in gB_tilde.items():
            l, proj = int(k.split(".", 1)[0]), k.split(".", 1)[1]
            if proj not in TRUSTED_B or (l, proj) not in self.stateB:
                raise KeyError(f"stale/missing trusted B state {k}")
            Tout = self.toutB[(l, proj)]
            gB_plain = Tout @ gt.to(self.sdt)  # Tout_iT @ gBc, Tout_iT == Tout (orthogonal)
            Bp, m, v = self.stateB[(l, proj)]
            Bp, m, v = self._update(Bp, gB_plain, m, v)
            self.stateB[(l, proj)] = [Bp, m, v]
            outB[k] = (Tout.T @ Bp)           # re-fold -> masked
            gotB.add((l, proj))
        missing = len(expectedA - gotA) + len(expectedB - gotB)
        if missing == 0:
            self.version += 1        # monotonic: one successful step == one version bump
        return outA, outB, missing

    def state_present(self):
        return len(self.stateA) == self.L * len(TRUSTED_A) and len(self.stateB) == self.L * len(TRUSTED_B)

    # ---- durable STANDARD-AEAD checkpoint / recovery (ChaCha20-Poly1305; multi-step restart continuity) ----
    def checkpoint(self, session_key: bytes) -> bytes:
        """ChaCha20-Poly1305-seal (theta_master, m, v, step, version, binding). Never durable plaintext.
        Dedicated HKDF sealing key (domain-separated from transport). Random unique nonce (ledgered).
        AAD binds run_id/package/adapter/profile/version/checkpoint_seq/model-cfg/service so a blob cannot
        be replayed into a different run/package/adapter/profile or rolled back to a stale version."""
        if self.binding is None:
            raise ValueError("checkpoint: binding not set (fail closed)")
        self.checkpoint_seq += 1
        buf = io.BytesIO()
        torch.save({"stateA": {f"{l}.{p}": [x.cpu() for x in s] for (l, p), s in self.stateA.items()},
                    "stateB": {f"{l}.{p}": [x.cpu() for x in s] for (l, p), s in self.stateB.items()},
                    "t": self.t, "version": self.version, "checkpoint_seq": self.checkpoint_seq,
                    "hparams": [self.lr, self.b1, self.b2, self.eps, self.wd],
                    "binding": self.binding, "sdt": str(self.sdt)}, buf)
        aad = canonical_aad(self.binding, self.version, self.checkpoint_seq)
        return aead_seal_std(session_key, buf.getvalue(), aad, nonce_ledger=self.nonce_ledger)

    def restore(self, blob: bytes, session_key: bytes, expected_binding: dict, min_version: int = 0,
                expected_checkpoint_seq: int | None = None):
        """Authenticated decrypt (ChaCha20-Poly1305) + binding + monotonic-version + nonce-reuse. Fail closed."""
        ver = int(expected_binding["version"])
        cseq = int(expected_binding.get("checkpoint_sequence", expected_checkpoint_seq or 1))
        aad = canonical_aad(expected_binding, ver, cseq)   # any mismatched bound field -> InvalidTag
        pt = aead_open_std(session_key, blob, aad, nonce_ledger=self.nonce_ledger)  # raises on tamper/key/aad/nonce
        d = torch.load(io.BytesIO(pt), map_location="cpu", weights_only=False)
        b = d["binding"]
        for f in ("run_id", "package_root_hash", "adapter_id"):
            if str(b.get(f)) != str(expected_binding[f]):
                raise ValueError(f"restore: binding[{f}] mismatch (fail closed)")
        if d["version"] != ver or d["version"] < min_version:
            raise ValueError(f"restore: version {d['version']} < min {min_version} or != expected {ver} (rollback; fail closed)")
        self.sdt = getattr(torch, str(d["sdt"]).split(".")[-1])
        self.stateA = {(int(k.split(".")[0]), k.split(".", 1)[1]): [x.to(self.sdt) for x in s]
                       for k, s in d["stateA"].items()}
        self.stateB = {(int(k.split(".")[0]), k.split(".", 1)[1]): [x.to(self.sdt) for x in s]
                       for k, s in d["stateB"].items()}
        self.t = d["t"]; self.version = d["version"]; self.binding = b
        self.checkpoint_seq = d.get("checkpoint_seq", cseq)
        return {"restored_version": self.version, "restored_t": self.t, "restored_checkpoint_seq": self.checkpoint_seq}
