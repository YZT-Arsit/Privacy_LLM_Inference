"""Stage 7.5c - CryptoNets-style arithmetic skeleton (plaintext only).

The CryptoNets paper (Dowlin et al., ICML 2016) replaces ReLU with the
square activation ``phi(x) = x^2`` so that the entire forward pass is a
polynomial of the input and can therefore be evaluated under a leveled
homomorphic encryption scheme. We implement the *arithmetic skeleton*:

* a polynomial-only feed-forward network using ``x^2`` activations,
* a static feed-forward inference pass over plaintext,
* explicit accounting of which model features become unsupported once
  ReLU/GELU/SiLU are replaced by squares.

What this module does NOT do:

* No homomorphic encryption (no SEAL, no Pyfhel, no PALISADE). The
  ``exact_crypto_protocol_implemented = False`` field is the experimental
  result for the cryptographic side of the comparison.
* No autoregressive decoder support: the LLM building blocks (attention
  softmax, RoPE rotation, etc.) are not polynomial under finite-degree
  bounds.
* No KV cache append. No LoRA training.

Reports publish summary metrics only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from pllo.baselines.baseline_protocol import (
    BaselineProtocol,
    BaselineSelfDeclaration,
    UnsupportedResult,
)


_DECLARE = BaselineSelfDeclaration(
    name="cryptonets_polynomial_skeleton",
    paper="CryptoNets (ICML 2016) -- polynomial-activation skeleton only",
    exact_primitive_implemented=True,  # x^2 activation skeleton
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=False,
    supports_kv_cache_append=False,
    supports_lora_training=False,
    arithmetic_skeleton_only=True,
    cost_model_only=False,
    notes=(
        "Plaintext arithmetic skeleton only; no HE library, no real"
        " encryption, no CKKS / BFV / TFHE. We do not claim direct"
        " implementation of the full CryptoNets cryptographic protocol."
    ),
)


@dataclass
class CryptoNetsConfig:
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


class CryptoNetsArithmeticSkeleton(BaselineProtocol):
    declare = _DECLARE

    exact_crypto_protocol_implemented = False
    requires_he_library = False

    def __init__(self, config: CryptoNetsConfig | None = None) -> None:
        self.config = config or CryptoNetsConfig()

    def polynomial_forward(
        self,
        x: torch.Tensor,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor | None] | None = None,
    ) -> dict[str, Any]:
        """One forward pass with ``x^2`` between each Linear.

        Reports the final output, the per-layer activation magnitude
        (a coarse proxy for the multiplicative depth that an HE scheme
        would need to support), and runtime.
        """
        h = x
        if biases is None:
            biases = [None] * len(weights)
        mags: list[float] = []
        t0 = time.perf_counter()
        for w, b in zip(weights, biases):
            h = h @ w
            if b is not None:
                h = h + b
            # Square activation.
            h = h * h
            mags.append(float(h.abs().max().item()))
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "output": h,
            "per_layer_max_magnitude": mags,
            "approx_multiplicative_depth": int(len(weights)),
            "runtime_ms": runtime_ms,
            "primitive": "polynomial_x2_activation",
        }

    def forward(
        self, x: torch.Tensor, weights: list[torch.Tensor], **kw: Any,
    ) -> dict[str, Any]:
        return self.polynomial_forward(x, weights, **kw)

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="CryptoNets polynomial skeleton cannot evaluate LLM decode",
            mathematical_reason=(
                "softmax, RoPE, RMSNorm, and SwiGLU are not polynomial"
                " under bounded multiplicative depth; the CryptoNets"
                " skeleton restricts the activation to x^2 and cannot"
                " evaluate these operators directly."
            ),
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="CryptoNets is inference-only; no training is implemented.",
            paper_scope_reason="The CryptoNets paper is inference-only.",
        )


__all__ = [
    "CryptoNetsConfig",
    "CryptoNetsArithmeticSkeleton",
]
