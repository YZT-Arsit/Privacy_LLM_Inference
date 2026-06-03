"""Stage 7.5c - DarKnight-style data-blinding primitive (direct implementation
of the linear-coding skeleton; full system is NOT reproduced).

DarKnight (Hashemi et al., USENIX Security 2021) blinds a batch of inputs
with a linear *coding* scheme: the trusted side draws ``k`` random shares
of each input such that any ``k-1`` of them is independent of the
plaintext, and the accelerator computes linear ops on the coded shares;
the trusted side decodes by inverting the coding matrix.

What this module implements:

* The *additive sharing* skeleton over a single linear layer for ``k = 2``
  shares: trusted samples ``r``; sends ``(x + r, r)``; the accelerator
  computes ``Y_1 = (x + r) W`` and ``Y_2 = r W``; trusted recovers
  ``Y = Y_1 - Y_2``. This recovers the same algebraic identity as the
  paper's ``k = 2`` instantiation; we mark this as the directly
  implemented primitive.

What this module deliberately does NOT implement:

* The full DarKnight system: GPU integration, malicious-acclerator
  verification with side-channel-aware coding, batched mini-batch
  scheduling, and SGX-side cost model. ``full_system_reproduced = False``.
* Larger ``k`` (k >= 3) -- the paper's general coding matrix is well
  defined, but we do not ship a generalised solver here.

Generation, KV cache append, and LoRA training return
``UnsupportedResult`` with explicit reasons.
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
    name="darknight_blinding_primitive",
    paper="DarKnight: An Accelerated Framework for Privacy and Integrity Preserving Deep Learning (USENIX Security 2021)",
    exact_primitive_implemented=True,  # k=2 additive sharing only
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=False,
    supports_kv_cache_append=False,
    supports_lora_training=False,
    notes=(
        "k=2 additive-sharing skeleton implemented; general k>=3 coding,"
        " integrity-verifying coding, and full SGX pipeline are NOT"
        " reproduced."
    ),
)


@dataclass
class DarKnightConfig:
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


class DarKnightBlindingPrimitive(BaselineProtocol):
    declare = _DECLARE

    def __init__(self, config: DarKnightConfig | None = None) -> None:
        self.config = config or DarKnightConfig()

    def linear_two_share(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """k=2 additive sharing: ship (x + r, r), recover Y = Y1 - Y2."""
        dtype = self.config.torch_dtype()
        device = torch.device(self.config.device)
        if generator is None:
            r = torch.randn(*x.shape, dtype=dtype, device=device)
        else:
            r = torch.randn(*x.shape, dtype=dtype, device=device, generator=generator)
        t0 = time.perf_counter()
        share_1 = x + r
        share_2 = r
        # Accelerator side.
        y1 = share_1 @ w
        y2 = share_2 @ w
        y_recovered = y1 - y2
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        y_plain = x @ w
        err = float((y_recovered - y_plain).abs().max().item())
        return {
            "y_plain": y_plain,
            "y_recovered": y_recovered,
            "max_abs_error": err,
            "allclose": bool(torch.allclose(y_recovered, y_plain, atol=1e-9, rtol=1e-9)),
            "runtime_ms": runtime_ms,
            "boundary_calls": 2,  # one per share
            "k": 2,
            "primitive": "linear_two_share_additive",
        }

    def forward(
        self, x: torch.Tensor, w: torch.Tensor,
        *, generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        return self.linear_two_share(x, w, generator=generator)

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="autoregressive decode is not implemented in DarKnight",
            paper_scope_reason="DarKnight targets CNN inference / training; not LLM generation.",
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="LoRA training under the DarKnight coding scheme is not implemented",
            implementation_scope_reason="k=2 skeleton only; coded backward not shipped here.",
        )


__all__ = [
    "DarKnightConfig",
    "DarKnightBlindingPrimitive",
]
