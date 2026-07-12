"""Shared harness for the AAAI private-base security evaluation (S1-S4).

FROZEN THREAT MODEL (see results/aaai_private_base/security/security_registry.yaml):
  Scenario  : from-scratch PRIVATE base model (Qwen2.5-0.5B architecture).
  Attacker  : knows architecture + public hyperparameters + the transformed GPU
              package (``*_tilde`` tensors) + tensors the protocol exposes.
  Attacker  : does NOT know plaintext W, the original checkpoint, masks/inverses,
              TDX secrets, or trusted optimizer state.
  RULE      : the ATTACKER SIDE never receives plaintext W, plaintext H, the masks,
              or a *public* Qwen checkpoint. This module is the DEFENDER oracle: it
              legitimately holds plaintext + secrets (like the trusted packager) and
              hands the attack scripts ONLY transformed observations.

Design decision (documented as a limitation): the ground-truth private base uses the
real Qwen2.5-0.5B *weights* so representations have realistic structure, but those
weights are treated as SECRET and are NEVER passed to any attacker function. No attack
here loads a public checkpoint. The mask-concealment geometry under test (orthogonal /
permutation / monomial) is weight-distribution invariant, so this faithfully evaluates
the transform; using genuinely from-scratch random weights would change representation
realism, not the leakage surface. (We do NOT claim the real weights are "unrecoverable"
— we evaluate published attacks under the stated observation model.)

Exact fold reproduces scripts/gate0_build_private_package.py:
  residual  : H_tilde = H @ Nr,  Nr = orthogonal_signed_perm(896, seed=9000)  (Nr Nr^T=I)
  embedding : E_tilde = E @ Nr
  per layer : B=rope_commuting_rotation(64,1000+l); S=signed_perm(64,2000+l);
              P=permutation(4864,3000+l)
  vocab     : monomial Pi (permutation-only, D=I) baseline; proposed monomial adds D.

CPU only. Never touches the A10/TDX training path.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pllo.ops.masked_training_kernels import (  # noqa: E402
    orthogonal_signed_perm, permutation_matrix, rmsnorm_core)
from gate0_build_private_package import rope_commuting_rotation, block_diag  # noqa: E402

CKPT = Path("/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B")
DT = torch.float64
STORE = torch.float32


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return sha256_bytes(t.detach().to(torch.float32).contiguous().numpy().tobytes())


def load_config() -> dict:
    return json.loads((CKPT / "config.json").read_text())


class PrivateBaseOracle:
    """Defender-side oracle: holds plaintext weights + all mask secrets. Attack scripts
    call only the ``observe_*`` methods, which return transformed tensors."""

    def __init__(self, dtype=DT, device="cpu"):
        self.cfg = load_config()
        self.dt = dtype
        self.device = device
        self.H = self.cfg["hidden_size"]
        self.L = self.cfg["num_hidden_layers"]
        self.nh = self.cfg["num_attention_heads"]
        self.nkv = self.cfg["num_key_value_heads"]
        self.hd = self.H // self.nh
        self.I = self.cfg["intermediate_size"]
        self.V = self.cfg["vocab_size"]
        self.eps = self.cfg["rms_norm_eps"]
        # ---- secret masks (defender only) ----
        self.Nr = orthogonal_signed_perm(self.H, seed=9000, dtype=dtype)      # (896,896)
        self.Nr_inv = self.Nr.T                                              # orthogonal
        self.B = {l: rope_commuting_rotation(self.hd, seed=1000 + l, dtype=dtype) for l in range(self.L)}
        self.S = {l: orthogonal_signed_perm(self.hd, seed=2000 + l, dtype=dtype) for l in range(self.L)}
        self.P = {l: permutation_matrix(self.I, seed=3000 + l, dtype=dtype) for l in range(self.L)}
        # vocab monomial: permutation-only baseline (D=I) + proposed nonzero D
        g8 = torch.Generator().manual_seed(8000)
        self.vocab_perm = torch.randperm(self.V, generator=g8)
        self.vocab_perm_inv = torch.empty_like(self.vocab_perm)
        self.vocab_perm_inv[self.vocab_perm] = torch.arange(self.V)
        gD = torch.Generator().manual_seed(8100)
        # bounded-condition positive diagonal (proposed monomial), cond ~ e^2 ~ 7.4
        self.vocab_D = torch.exp((torch.rand(self.V, generator=gD, dtype=dtype) - 0.5) * 2.0)
        self._model = None
        self._sd = None

    # ---------- lazy real model (defender side) ----------
    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(CKPT))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(CKPT), torch_dtype=torch.float32).eval()
        self._sd = self._model.state_dict()

    def embed_table_plain(self) -> torch.Tensor:
        self._load()
        return self._sd["model.embed_tokens.weight"].to(self.dt)

    # ---------- real plaintext forward (defender) ----------
    @torch.no_grad()
    def real_forward(self, prompts: List[str], max_len: int = 32) -> dict:
        """Run the real (plaintext) model; return plaintext hidden states at every depth,
        logits, token ids. DEFENDER-ONLY output — not given to attackers directly."""
        self._load()
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len)
        out = self._model(**enc, output_hidden_states=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "hidden_states": [h.to(self.dt) for h in out.hidden_states],  # len L+1
            "logits": out.logits.to(self.dt),                             # (B,T,V)
        }

    # ---------- transformed observations handed to attackers ----------
    def observe_residual(self, H: torch.Tensor) -> torch.Tensor:
        """What the untrusted GPU sees for a residual state: H_tilde = H @ Nr."""
        return H.to(self.dt) @ self.Nr

    def observe_embed_table(self) -> torch.Tensor:
        """E_tilde = E @ Nr  (in the GPU package)."""
        return self.embed_table_plain() @ self.Nr

    def observe_logits(self, logits: torch.Tensor, scheme: str) -> torch.Tensor:
        """scheme in {plaintext, perm_only, monomial}. logits: (...,V)."""
        lg = logits.to(self.dt)
        if scheme == "plaintext":
            return lg
        if scheme == "perm_only":            # logits @ Pi  == column perm
            return lg[..., self.vocab_perm_inv]
        if scheme == "monomial":             # logits @ (D Pi) = (logits*D) then col-perm
            return (lg * self.vocab_D)[..., self.vocab_perm_inv]
        raise ValueError(scheme)

    # ---------- LoRA adapters (S2) ----------
    def make_lora(self, d_in: int, d_out: int, r: int, seed: int):
        """Plaintext LoRA (A: d_out x r, B: r x d_in), ΔW = A B (row-vec convention y=xΔW^T).
        Masked view uses residual masks on the two sides + a secret rank mask U (rank space).
        Attacker receives A_tilde,B_tilde; product = masked ΔW; U cancels only in the product."""
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(d_out, r, generator=g, dtype=self.dt) * (1.0 / (r ** 0.5))
        Bm = torch.randn(r, d_in, generator=g, dtype=self.dt) * (1.0 / (d_in ** 0.5))
        # rank mask U (invertible, well-conditioned, secret)
        U = orthogonal_signed_perm(r, seed=seed + 555, dtype=self.dt)
        # side masks (residual-basis in/out); use Nr slices for a faithful orthogonal mask
        Nin = orthogonal_signed_perm(d_in, seed=seed + 11, dtype=self.dt)
        Nout = orthogonal_signed_perm(d_out, seed=seed + 22, dtype=self.dt)
        A_tilde = Nout.T @ A @ U            # (d_out, r)
        B_tilde = U.T @ Bm @ Nin            # (r, d_in)  (U orthogonal -> U^{-1}=U^T)
        return {"A": A, "B": Bm, "U": U, "Nin": Nin, "Nout": Nout,
                "A_tilde": A_tilde, "B_tilde": B_tilde,
                "dW": A @ Bm, "dW_tilde": A_tilde @ B_tilde}


def random_baseline_token_acc(V: int) -> float:
    return 1.0 / V


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float64); b = b.reshape(-1).to(torch.float64)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def rel_err(est: torch.Tensor, ref: torch.Tensor) -> float:
    return float((est - ref).norm() / (ref.norm() + 1e-30))


DEFAULT_PROMPTS = [
    "The movie was a stunning and emotional masterpiece.",
    "I would not recommend this restaurant to anyone.",
    "Quarterly revenue exceeded analyst expectations this year.",
    "The patient reported mild symptoms after the treatment.",
    "She walked quietly through the empty morning streets.",
    "This is the worst customer service I have ever received.",
    "The algorithm converged after a few hundred iterations.",
    "A gentle rain fell over the quiet mountain village.",
]


def stamp(out_dir: Path, name: str, obj: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    blob = json.dumps(obj, indent=2, default=float)
    p.write_text(blob)
    (out_dir / (name + ".sha256")).write_text(sha256_bytes(blob.encode()) + "  " + name + "\n")
    return p


if __name__ == "__main__":
    # self-test: fold correctness (masked residual reconstructs plaintext geometry)
    o = PrivateBaseOracle()
    H = torch.randn(4, o.H, dtype=DT)
    Ht = o.observe_residual(H)
    assert abs((Ht @ o.Nr_inv - H).abs().max().item()) < 1e-10, "Nr not invertible"
    # orthogonal mask preserves norm + pairwise gram (documented leak)
    assert (H.norm(dim=1) - Ht.norm(dim=1)).abs().max() < 1e-10
    print("pb_harness self-test OK; Nr orthogonal, norm/gram preserved (documented leak)")
