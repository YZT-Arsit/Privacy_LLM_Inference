"""Stage 7.5c - Slalom-style delegated linear primitive (direct implementation).

Slalom delegates dense matrix multiplications to an untrusted accelerator
while the trusted side blinds the input with a fresh additive mask and
optionally verifies the returned matmul with a Freivalds-style randomised
check.

What this module implements directly from the paper formula:

    Trusted side:   r is sampled fresh on the trusted side; ship X + r to GPU.
    Accelerator:    compute Y' = (X + r) @ W.
    Trusted side:   recover Y = Y' - r @ W (the trusted side knows W and r).
    Verification:   sample s ~ {-1, +1}; check |Y s - (X) @ (W s)| < tol.

What this module does NOT implement:

* The Slalom paper's full CNN-deployment system (quantisation, model
  partitioning, batching policy, real SGX integration) is not reproduced;
  ``full_system_reproduced = False``.
* Slalom does not address autoregressive generation, KV cache append, or
  LoRA training; those return ``UnsupportedResult`` with a paper-scope
  reason.

Reports publish summary metrics only; the raw blinding tensor ``r`` is
never written to disk.
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
    name="slalom_delegated_linear",
    paper="Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware (ICLR 2019)",
    exact_primitive_implemented=True,
    full_system_reproduced=False,
    requires_crypto_library=False,
    supports_static_forward=True,
    supports_decoder_generation=False,
    supports_kv_cache_append=False,
    supports_lora_training=False,
    notes=(
        "Delegated linear blinding + Freivalds verification implemented"
        " from the paper formula; CNN partitioning / quantisation / full"
        " SGX integration is NOT reproduced."
    ),
)


@dataclass
class SlalomConfig:
    dtype: str = "float64"
    device: str = "cpu"
    num_freivalds_rounds: int = 3
    tol: float = 1e-9

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


class SlalomDelegatedLinear(BaselineProtocol):
    """Delegated linear ``Y = X @ W`` with additive input blinding."""

    declare = _DECLARE

    def __init__(self, config: SlalomConfig | None = None) -> None:
        self.config = config or SlalomConfig()

    # ------------------------------------------------------------------
    # Primitive
    # ------------------------------------------------------------------

    def delegated_linear(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """One Slalom-style delegated linear call.

        Returns ``{y, y_recovered, max_abs_error, verification_passed, ...}``.
        """
        dtype = self.config.torch_dtype()
        device = torch.device(self.config.device)
        if generator is None:
            r = torch.randn(*x.shape, dtype=dtype, device=device)
        else:
            r = torch.randn(*x.shape, dtype=dtype, device=device, generator=generator)
        t0 = time.perf_counter()
        x_blinded = x + r                   # trusted side
        # Accelerator side: Y' = (X + r) @ W.
        y_blinded = x_blinded @ w
        # Trusted side: subtract the precomputed r @ W to recover Y.
        rw = r @ w                          # trusted-side precompute
        y_recovered = y_blinded - rw
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        y_plain = x @ w
        err = float((y_recovered - y_plain).abs().max().item())
        verification = self._freivalds_verify(
            x, w, y_recovered, generator=generator,
        )
        return {
            "y_plain": y_plain,
            "y_recovered": y_recovered,
            "max_abs_error": err,
            "allclose": bool(torch.allclose(y_recovered, y_plain, atol=1e-9, rtol=1e-9)),
            "verification_passed": verification["passed"],
            "verification_residual": verification["residual"],
            "verification_rounds": int(self.config.num_freivalds_rounds),
            "runtime_ms": runtime_ms,
            "boundary_calls": 1,
            "primitive": "delegated_linear",
        }

    def forward(
        self, x: torch.Tensor, w: torch.Tensor,
        *, generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        return self.delegated_linear(x, w, generator=generator)

    # ------------------------------------------------------------------
    # Freivalds-style verification
    # ------------------------------------------------------------------

    def _freivalds_verify(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y_recovered: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> dict[str, float]:
        """Sample random ``s`` in {-1, +1} and check ``Y s == X (W s)``.

        The check passes with high probability over random ``s`` if the
        recovered ``Y`` is correct; one round fails with probability
        roughly 2^{-1} for an adversarial perturbation, so we average
        over ``num_freivalds_rounds`` rounds to make the false-accept
        probability very small.
        """
        dtype = self.config.torch_dtype()
        device = torch.device(self.config.device)
        residuals: list[float] = []
        for _ in range(int(self.config.num_freivalds_rounds)):
            if generator is None:
                s = torch.randint(0, 2, (w.shape[1],), device=device).to(dtype) * 2 - 1
            else:
                s = (
                    torch.randint(
                        0, 2, (w.shape[1],), device=device, generator=generator,
                    ).to(dtype) * 2 - 1
                )
            lhs = y_recovered @ s
            rhs = x @ (w @ s)
            residuals.append(float((lhs - rhs).abs().max().item()))
        worst = max(residuals)
        return {
            "passed": bool(worst < self.config.tol),
            "residual": float(worst),
        }

    # ------------------------------------------------------------------
    # Out-of-scope ops
    # ------------------------------------------------------------------

    def decode_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="autoregressive decode is not implemented in Slalom",
            paper_scope_reason="Slalom targets feed-forward CNN inference; not generative decoding.",
        )

    def prefill(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="LLM prefill not implemented in Slalom",
            paper_scope_reason="Slalom targets feed-forward CNN inference.",
        )

    def train_step(self, *args: Any, **kwargs: Any) -> UnsupportedResult:
        return UnsupportedResult(
            reason="Slalom does not address training; it is an inference-only delegation scheme.",
            paper_scope_reason="Slalom is an inference-only delegation scheme.",
        )


__all__ = [
    "SlalomConfig",
    "SlalomDelegatedLinear",
]
