"""ObfuscaTune baseline protocol class (arXiv:2407.02960, PPAI-25 / AAAI-W).

This is the ``BaselineProtocol`` entry point for the baseline registry. It was
the original single-file ``pllo.baselines.obfuscatune`` module; the package now
splits the reusable pieces into :mod:`.random_matrices`, :mod:`.linear_obfuscation`,
:mod:`.modules`, :mod:`.metrics` and :mod:`.hf_gpt2_obfuscatune`, while this file
keeps the exact class + declaration + generator-based ``matrix_with_condition_number``
API that the registry and the legacy tests import.

ObfuscaTune protects a *proprietary* model + private data by (i) keeping the
low-parameter layers (embed / norm / softmax / activation / output ~5% of params)
inside a TEE and (ii) obfuscating the high-parameter attention+MLP linear layers
with random matrices placed OUTSIDE the TEE:

    X* = X Ra ,   W* = Ra^{-1} W     =>   X* W* = X W        (Q, K, V recovered)
    O* = H Wo Rb ,  O = O* Rb^{-1}                            (eq. 1-6 of the paper)

Two structural facts of the scheme (verified by this module and its tests):
  1. The projections cancel the mask, so the accelerator sees the TRUE (plaintext)
     Q, K, V. Security rests on the model weights being secret; it is NOT an
     open-weight defense.
  2. Every non-linearity (LayerNorm, softmax, activation) runs INSIDE the TEE on
     de-obfuscated values, so each block round-trips the TEE boundary multiple
     times -- the opposite of our A_rightmul design (all non-linearities on the
     untrusted GPU, single entry/exit, zero nonlinear crossings).

Raw obfuscation matrices are never returned/serialised.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from pllo.baselines.baseline_protocol import BaselineProtocol, BaselineSelfDeclaration

from .config import ObfuscaTuneConfig

_DECLARE = BaselineSelfDeclaration(
    name="obfuscatune",
    paper="ObfuscaTune: Obfuscated Offsite Finetuning and Inference of Proprietary LLMs on Private Datasets (Frikha et al., arXiv:2407.02960, 2025)",
    exact_primitive_implemented=True,
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=True,
    supports_kv_cache_append=True,
    supports_lora_training=True,
    notes=(
        "Obfuscated linear (X Ra, Ra^{-1} W) + TEE/GPU split block implemented"
        " from eq.1-6; exposes TRUE Q/K/V outside the TEE and runs every"
        " non-linearity INSIDE the TEE (per-block boundary round-trips). Full"
        " nanoGPT LoRA pipeline + lm-eval harness NOT reproduced."
    ),
)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _orthogonal(d: int, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    a = torch.randn(d, d, dtype=torch.float64, generator=g)
    q, _ = torch.linalg.qr(a)
    return q.to(dtype)


def matrix_with_condition_number(
    d: int, kappa: float, dtype: torch.dtype, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy generator-based API: (R, R_inv) with cond(R)=kappa (paper App. B).

    Preserved verbatim for the baseline registry and ``test_baselines_obfuscatune``.
    New code should prefer the seed-based
    :func:`pllo.baselines.obfuscatune.random_matrices.matrix_with_condition_number`.
    """
    if kappa <= 1.0:
        q = _orthogonal(d, dtype, g)
        return q, q.transpose(0, 1).contiguous()
    qa = _orthogonal(d, torch.float64, g)
    qb = _orthogonal(d, torch.float64, g)
    smax = 1.0
    smin = smax / float(kappa)
    mid = torch.rand(d - 2, dtype=torch.float64, generator=g) * (smax - smin) + smin
    sv = torch.cat([torch.tensor([smax], dtype=torch.float64), mid,
                    torch.tensor([smin], dtype=torch.float64)])
    S = torch.diag(sv)
    Sinv = torch.diag(1.0 / sv)
    R = (qa @ S @ qb).to(dtype)
    R_inv = (qb.transpose(0, 1) @ Sinv @ qa.transpose(0, 1)).to(dtype)
    return R, R_inv


class ObfuscaTune(BaselineProtocol):
    """ObfuscaTune obfuscated linear + TEE/GPU-split transformer block."""

    declare = _DECLARE

    def __init__(self, config: ObfuscaTuneConfig | None = None) -> None:
        self.config = config or ObfuscaTuneConfig()

    # ------------------------------------------------------------------
    # Core obfuscated linear primitive (eq. 1-3)
    # ------------------------------------------------------------------
    def obfuscated_linear(
        self, x: torch.Tensor, w: torch.Tensor, *, seed: int | None = None
    ) -> dict[str, Any]:
        """Y = X W computed on the untrusted side over obfuscated operands.

        TEE: sample Ra, form X* = X Ra and W* = Ra^{-1} W. GPU: Y = X* W*.
        Because Ra Ra^{-1} = I, Y equals the TRUE X W -> the GPU sees plaintext Y.
        """
        dtype = self.config.torch_dtype()
        d = x.shape[-1]
        g = _gen(self.config.seed if seed is None else seed)
        Ra, Ra_inv = matrix_with_condition_number(
            d, self.config.condition_number, dtype, g)
        t0 = time.perf_counter()
        x_star = x @ Ra                       # TEE: obfuscate embedding
        w_star = Ra_inv @ w                   # TEE: obfuscate weight
        y = x_star @ w_star                   # GPU: untrusted matmul
        ms = (time.perf_counter() - t0) * 1000.0
        y_plain = x @ w
        return {
            "y": y,
            "y_plain": y_plain,
            "max_abs_error": float((y - y_plain).abs().max().item()),
            "condition_number": float(self.config.condition_number),
            "gpu_output_is_plaintext_true_value": True,
            "runtime_ms": ms,
            "boundary_calls": 1,
            "primitive": "obfuscated_linear",
        }

    def forward(self, x: torch.Tensor, w: torch.Tensor, **kw: Any) -> dict[str, Any]:
        return self.obfuscated_linear(x, w, **kw)

    # ------------------------------------------------------------------
    # Obfuscated attention+MLP block with TEE/GPU split accounting
    # ------------------------------------------------------------------
    @staticmethod
    def _plain_block(x: torch.Tensor, w: dict[str, torch.Tensor], n_heads: int) -> torch.Tensor:
        S, d = x.shape
        hd = d // n_heads

        def ln(t):
            m = t.mean(-1, keepdim=True)
            v = t.var(-1, unbiased=False, keepdim=True)
            return (t - m) / torch.sqrt(v + 1e-5)

        h = ln(x)
        Q, K, V = h @ w["Wq"], h @ w["Wk"], h @ w["Wv"]
        Qh = Q.view(S, n_heads, hd).transpose(0, 1)
        Kh = K.view(S, n_heads, hd).transpose(0, 1)
        Vh = V.view(S, n_heads, hd).transpose(0, 1)
        att = torch.softmax(Qh @ Kh.transpose(-1, -2) / (hd ** 0.5), dim=-1)
        Hh = (att @ Vh).transpose(0, 1).reshape(S, d)
        x = x + Hh @ w["Wo"]
        inter = torch.nn.functional.gelu(ln(x) @ w["W1"])
        return x + inter @ w["W2"]

    def attention_mlp_block(
        self,
        x: torch.Tensor,
        weights: dict[str, torch.Tensor],
        *,
        n_heads: int = 4,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """One GPT-2-style block under ObfuscaTune. weights: Wq,Wk,Wv,Wo,W1,W2.

        Returns the block output + a structural audit: which tensors are exposed
        in plaintext to the untrusted side and how many TEE boundary crossings the
        scheme requires (every non-linearity forces a de-obfuscate/re-obfuscate).
        """
        dtype = self.config.torch_dtype()
        x = x.to(dtype)
        x0 = x.clone()
        S, d = x.shape
        hd = d // n_heads
        g = _gen(self.config.seed if seed is None else seed)
        crossings = 0
        exposed: list[str] = []

        def ln(t: torch.Tensor) -> torch.Tensor:
            nonlocal crossings
            crossings += 2   # de-obfuscate into TEE + re-obfuscate out
            m = t.mean(-1, keepdim=True)
            v = t.var(-1, unbiased=False, keepdim=True)
            return (t - m) / torch.sqrt(v + 1e-5)

        # --- attention ---
        h = ln(x)                                             # LayerNorm in TEE
        Ra, Ra_inv = matrix_with_condition_number(d, self.config.condition_number, dtype, g)
        h_star = h @ Ra                                       # obfuscate, leave TEE
        Q = (h_star @ (Ra_inv @ weights["Wq"]))               # GPU: true Q
        K = (h_star @ (Ra_inv @ weights["Wk"]))               # GPU: true K
        V = (h_star @ (Ra_inv @ weights["Wv"]))               # GPU: true V
        exposed += ["Q_plaintext", "K_plaintext", "V_plaintext"]
        # softmax attention (non-linear) runs in TEE on the true Q,K,V
        crossings += 2
        Qh = Q.view(S, n_heads, hd).transpose(0, 1)
        Kh = K.view(S, n_heads, hd).transpose(0, 1)
        Vh = V.view(S, n_heads, hd).transpose(0, 1)
        att = torch.softmax(Qh @ Kh.transpose(-1, -2) / (hd ** 0.5), dim=-1)
        Hh = (att @ Vh).transpose(0, 1).reshape(S, d)
        # output projection obfuscated by Rb
        Rb, Rb_inv = matrix_with_condition_number(d, self.config.condition_number, dtype, g)
        O_star = (Hh @ Ra) @ (Ra_inv @ weights["Wo"]) @ Rb    # GPU
        O = O_star @ Rb_inv                                   # de-obfuscate in TEE
        x = x + O

        # --- MLP ---
        h2 = ln(x)                                            # LayerNorm in TEE
        Rc, Rc_inv = matrix_with_condition_number(d, self.config.condition_number, dtype, g)
        h2s = h2 @ Rc
        inter = h2s @ (Rc_inv @ weights["W1"])                # GPU: true intermediate
        exposed += ["mlp_intermediate_plaintext"]
        crossings += 2                                        # GELU in TEE
        inter = torch.nn.functional.gelu(inter)
        di = inter.shape[-1]
        Rd, Rd_inv = matrix_with_condition_number(di, self.config.condition_number, dtype, g)
        out_star = (inter @ Rd) @ (Rd_inv @ weights["W2"])
        x = x + out_star

        x_plain = self._plain_block(x0, weights, n_heads)
        return {
            "output": x,
            "output_plain": x_plain,
            "max_abs_error": float((x - x_plain).abs().max().item()),
            "boundary_crossings": crossings,
            "exposed_plaintext_tensors": exposed,
            "tee_nonlinear_ops": ["layernorm", "softmax", "layernorm", "gelu"],
            "condition_number": float(self.config.condition_number),
            "primitive": "obfuscatune_block",
        }

    # ------------------------------------------------------------------
    # Structural audit vs our A_rightmul design
    # ------------------------------------------------------------------
    def structural_audit(self, n_layers: int, total_params: int,
                         tee_params: int) -> dict[str, Any]:
        return {
            "true_qkv_exposed_to_untrusted": True,
            "security_requires_secret_weights": True,
            "open_weight_secure": False,
            "nonlinears_in_tee": True,
            "tee_boundary_crossings_per_block": 8,   # 2 ln + softmax + gelu, de/re-obf
            "tee_boundary_crossings_total": 8 * n_layers,
            "pct_params_in_tee": round(100.0 * tee_params / max(total_params, 1), 3),
            "contrast_ours_A_rightmul": {
                "true_qkv_exposed": False,
                "nonlinears_in_tee": False,
                "tee_boundary_crossings_total": 2,  # single entry + single exit
                "note": "k=1 guardrail moves only layer-0 nonlinears into TEE",
            },
        }


__all__ = [
    "ObfuscaTune",
    "matrix_with_condition_number",
]
