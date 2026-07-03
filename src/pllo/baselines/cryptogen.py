"""CryptoGen baseline (Zhang et al., arXiv:2602.08798).

CryptoGen is a hybrid HE+MPC system for secure *autoregressive* generation with
an **encrypted KV cache**. Its threat model is pure-cryptographic (NO TEE),
client-server, semi-honest, and *dual-sided* (client prompt + generated tokens
AND model weights are hidden). Linear layers run in HE (CT x PT); every
non-linearity (GELU/LayerNorm/Softmax) runs in interactive MPC. Its contribution
is making the encrypted KV cache reusable so attention scales O(L) per sequence
instead of BOLT's O(L^2).

This is the pure-crypto opposite of our TEE-anchored, open-weight scheme (all
non-linearity on the untrusted GPU, single entry/exit, ms latency at 7B).

What this module implements *faithfully* (verified on plaintext-packed slots,
no encryption -- the packing / rotation / fold-sum ALGEBRA is exact):
  * the three ciphertext packings (outer / inner / diagonal, Fig. 2);
  * the diagonal CT x PT matrix-vector product (the HE linear-layer kernel);
  * the ARCC attention kernels (inner-inner, inner-outer, Fig. 7) with the
    O(log d) "folding-sum" reduction;
  * slot-aware encrypted KV-cache concatenation (Algorithm 1) keeping the
    ciphertext count ~constant.
What it does NOT reproduce (``requires_crypto_library=True``,
``full_system_reproduced=False``): the BFV encryption / noise budget /
bootstrapping and the EzPC MPC rounds. For those we provide a faithful COST
MODEL reproducing their Tables I / II / IV (Mult / Rot / Ct counts and the
O(L^2)->O(L) attention scaling). No key material or ciphertext is produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
)

_DECLARE = BaselineSelfDeclaration(
    name="cryptogen",
    paper="CryptoGen: Secure Transformer Generation with Encrypted KV-Cache Reuse (Zhang et al., arXiv:2602.08798, 2026)",
    exact_primitive_implemented=True,     # packing + ARCC kernels + KV concat (plaintext-faithful)
    full_system_reproduced=False,         # no BFV/MPC execution
    requires_crypto_library=True,         # real system needs SEAL (BFV) + EzPC (MPC)
    supports_static_forward=True,
    supports_decoder_generation=True,
    supports_kv_cache_append=True,
    supports_lora_training=False,
    cost_model_only=False,
    notes=(
        "Heterogeneous encoding (outer/inner/diagonal), diagonal CT x PT matvec,"
        " ARCC inner-inner/inner-outer attention with O(log d) folding-sum, and"
        " slot-aware encrypted KV-cache concatenation implemented + verified on"
        " plaintext-packed slots. BFV encryption/noise + EzPC MPC NOT executed;"
        " a cost model reproduces their Tables I/II/IV (O(L^2)->O(L))."
    ),
)


@dataclass
class CryptoGenConfig:
    poly_modulus_degree: int = 8192       # n, SEAL SIMD slots
    hidden: int = 768                     # d1 (GPT-2 base)
    n_heads: int = 12
    head_dim: int = 64                    # d2 = d1 / n_heads
    dtype: str = "float64"
    seed: int = 0

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


def _rot(ct: torch.Tensor, k: int) -> torch.Tensor:
    """Cyclic left-rotation of ciphertext slots (models an HE Galois rotation)."""
    return torch.roll(ct, shifts=-k, dims=0)


# ---------------------------------------------------------------------------
# Packings (Fig. 2). A "ciphertext" is a length-n slot vector; a packed matrix
# is a list of such vectors. These are exact data layouts, no encryption.
# ---------------------------------------------------------------------------
class CryptoGen(BaselineProtocol):
    declare = _DECLARE

    def __init__(self, config: CryptoGenConfig | None = None) -> None:
        self.config = config or CryptoGenConfig()

    # -- encodings ---------------------------------------------------------
    def outer_pack(self, A: torch.Tensor) -> list[torch.Tensor]:
        """Outer-based: one ciphertext per COLUMN A[:, j]."""
        n = self.config.poly_modulus_degree
        m = A.shape[0]
        assert m <= n, "column longer than slot count"
        out = []
        for j in range(A.shape[1]):
            ct = torch.zeros(n, dtype=A.dtype)
            ct[:m] = A[:, j]
            out.append(ct)
        return out

    def inner_pack(self, A: torch.Tensor) -> list[torch.Tensor]:
        """Inner-based: one ciphertext per ROW A[i, :]."""
        n = self.config.poly_modulus_degree
        cols = A.shape[1]
        assert cols <= n
        out = []
        for i in range(A.shape[0]):
            ct = torch.zeros(n, dtype=A.dtype)
            ct[:cols] = A[i, :]
            out.append(ct)
        return out

    def diagonal_pack(self, A: torch.Tensor) -> list[torch.Tensor]:
        """Diagonal (Halevi-Shoup): k-th diag holds A[i, (i+k) mod cols]."""
        rows, cols = A.shape
        diags = []
        for k in range(cols):
            d = torch.stack([A[i % rows, (i + k) % cols] for i in range(cols)])
            diags.append(d)
        return diags

    # -- linear layer: diagonal CT x PT matvec -----------------------------
    def diagonal_matvec(self, A: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
        """y = A x via diagonal encoding: y = sum_k diag_k (*) rot(x, k).

        A is the plaintext weight (diagonal-encoded), x the ciphertext activation
        (CT x PT). Verified equal to the plaintext product. Counts Mult/Rot."""
        rows, cols = A.shape
        assert rows == cols, "square weight for diagonal matvec"
        n = self.config.poly_modulus_degree
        assert n % cols == 0, "slot count must tile the row dim"
        # tile x with period cols so rot() wraps modulo cols (HE replication)
        xslot = x.repeat(n // cols)
        diags = self.diagonal_pack(A)
        acc = torch.zeros(n, dtype=x.dtype)
        rots = 0
        mults = 0
        for k in range(cols):
            dslot = torch.zeros(n, dtype=x.dtype)
            dslot[:cols] = diags[k]
            acc = acc + dslot * _rot(xslot, k)      # CT x PT multiply + accumulate
            mults += 1
            if k > 0:
                rots += 1
        y = acc[:rows]
        y_plain = A @ x
        return {
            "y": y, "y_plain": y_plain,
            "max_abs_err": float((y - y_plain).abs().max().item()),
            "he_multiplications": mults, "he_rotations": rots,
        }

    # -- ARCC attention (Fig. 7) with O(log d) folding-sum -----------------
    def fold_sum(self, ct: torch.Tensor, d: int) -> tuple[torch.Tensor, int]:
        """Sum the first d slots into slot 0 via log2(d) rotate-and-add."""
        acc = ct.clone()
        rots = 0
        shift = 1
        while shift < d:
            acc = acc + _rot(acc, shift)
            rots += 1
            shift *= 2
        return acc, rots

    def arcc_scores(self, q: torch.Tensor, K: torch.Tensor) -> dict[str, Any]:
        """Attention scores s = q K^T for one query over cached keys.

        Faithful to ARCC: element-wise q (*) K_row then O(log d) folding-sum
        reduction per key. Verified equal to plaintext q @ K^T. Counts Rot."""
        d = q.shape[0]
        n = self.config.poly_modulus_degree
        qslot = torch.zeros(n, dtype=q.dtype)
        qslot[:d] = q
        scores = []
        total_rots = 0
        for t in range(K.shape[0]):
            kslot = torch.zeros(n, dtype=q.dtype)
            kslot[:d] = K[t]
            prod = qslot * kslot                       # SIMD element-wise
            summed, r = self.fold_sum(prod, d)         # O(log d) reduction
            total_rots += r
            scores.append(summed[0])
        s = torch.stack(scores)
        s_plain = q @ K.transpose(0, 1)
        return {
            "scores": s, "scores_plain": s_plain,
            "max_abs_err": float((s - s_plain).abs().max().item()),
            "rotations_per_key": max(1, (d - 1).bit_length()),
            "total_rotations": total_rots,
            "reduction_complexity": "O(log d)",
        }

    # -- encrypted KV-cache concatenation (Algorithm 1) --------------------
    def kv_concat(self, tokens: torch.Tensor) -> dict[str, Any]:
        """Slot-aware append: pack B = floor(n/d2) tokens per ciphertext.

        Returns the compact cache and verifies exact reconstruction. Ciphertext
        count = ceil(L/B) (~constant vs naive one-CT-per-token)."""
        n = self.config.poly_modulus_degree
        d2 = self.config.head_dim
        L, dim = tokens.shape
        assert dim == d2, "token dim must equal head_dim d2"
        B = max(1, n // d2)                             # tokens per ciphertext
        n_ct = (L + B - 1) // B
        cache = [torch.zeros(n, dtype=tokens.dtype) for _ in range(n_ct)]
        for m in range(L):                             # slot-aware accumulation
            ci = m // B
            pos = (m % B) * d2
            cache[ci][pos:pos + d2] = cache[ci][pos:pos + d2] + tokens[m]
        # reconstruct
        rec = torch.zeros_like(tokens)
        for m in range(L):
            ci = m // B
            pos = (m % B) * d2
            rec[m] = cache[ci][pos:pos + d2]
        return {
            "cache_ciphertexts": n_ct,
            "naive_ciphertexts": L,
            "tokens_per_ciphertext": B,
            "reconstruct_max_abs_err": float((rec - tokens).abs().max().item()),
        }

    # -- cost model (Tables I / II / IV) -----------------------------------
    def cost_model(self, m: int, k: int) -> dict[str, Any]:
        """Reproduce CryptoGen vs BOLT CT x PT / attention costs.

        m = prefill length, k = generated tokens. Uses paper params by default
        (d1=768, d2=64, n=8192). Attention: BOLT O(k^2), CryptoGen O(k)."""
        d1 = self.config.hidden
        d2 = self.config.head_dim
        n = self.config.poly_modulus_degree
        import math
        # CT x PT per-step Mult (Table I): BOLT ~ m*d1*d2/n ; CryptoGen ~ d1*d2/n
        bolt_gen_mult = int(round(m * d1 * d2 / n)) * k
        cg_gen_mult = int(round(d1 * d2 / n)) * k
        # attention CT x CT (Table II): BOLT O(k^2), CryptoGen O(k) (KV reuse)
        bolt_attn = k * k
        cg_attn = k
        # per-key reduction rotations: BOLT O(d1), CryptoGen O(log d1)
        return {
            "params": {"m": m, "k": k, "d1": d1, "d2": d2, "n": n},
            "ctxpt_gen_mult_asymptotic": {"bolt": bolt_gen_mult, "cryptogen": cg_gen_mult,
                                          "formula": "BOLT O(m*d1*d2/n), CryptoGen O(d1*d2/n)",
                                          "cryptogen_independent_of_m": True},
            "attention_ctxct": {"bolt": bolt_attn, "cryptogen": cg_attn,
                                "bolt_order": "O(k^2)", "cryptogen_order": "O(k)"},
            "reduction_rotations_order": {"bolt": "O(d1)",
                                          "cryptogen": f"O(log d1) = {int(math.log2(d1))}"},
            # concrete per-stage counts cited from Table I (m=128,d1=768,d2=64,n=8192)
            "paper_table_i_cited": {
                "note": "per-token Gen counts (k=5): CryptoGen decouples from prompt length m",
                "mult_gen_per_token": {"bolt": 768, "cryptogen": 64},
                "rot_gen_per_token": {"bolt": 43, "cryptogen": 5},
                "ct_gen_per_token": {"bolt": 12, "cryptogen": 1},
            },
            # measured per-block seconds cited from their Table IV (NOT ours)
            "paper_per_block_seconds_table_iv": {
                "64": {"bolt": 11.92, "cryptogen": 9.6},
                "128": {"bolt": 28.42, "cryptogen": 16.45},
                "256": {"bolt": 81.05, "cryptogen": 33.84},
                "512": {"bolt": 225.7, "cryptogen": 65.55},
            },
            "source": "arXiv:2602.08798 Tables I, II, IV (cited, not measured here)",
        }

    # -- structural audit vs ours ------------------------------------------
    def structural_audit(self) -> dict[str, Any]:
        cfg = self.config
        # GPT-2: their fastest, L=512 -> 65.55 s/block * 12 blocks per token
        cg_per_token_gpt2_s = 65.55 * 12
        return {
            "hardware_trust_anchor": "none (pure crypto: HE + MPC)",
            "threat_model": "client-server, semi-honest, dual-sided (data + weights hidden)",
            "nonlinears": "interactive MPC (client<->server rounds per GELU/LayerNorm/Softmax)",
            "kv_cache": "encrypted, reused (O(L) attention; the paper's contribution)",
            "interactive": True,
            "largest_model_demonstrated": "GPT-2 base (12 layers, d=768)",
            "approx_per_token_latency_gpt2_L512_s": round(cg_per_token_gpt2_s, 1),
            "scales_to_7b": False,
            "contrast_ours_A_rightmul": {
                "hardware_trust_anchor": "minimal TEE (boundary + k=1 guardrail)",
                "threat_model": "open-weight (public weights); protects user tokens",
                "nonlinears": "all on untrusted GPU, non-interactive, 0 nonlinear crossings",
                "interactive": False,
                "largest_model_demonstrated": "Qwen2.5-7B",
                "approx_per_token_latency_7b_ms": 11.5,
                "scales_to_7b": True,
            },
        }


__all__ = ["CryptoGen", "CryptoGenConfig"]
