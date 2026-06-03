"""Stage 7.5c - trusted controller (TEE-like, locally emulated).

The controller owns every secret in the protocol: the user input, the
LoRA adapter, the optimizer state, the loss closure, the mask sampler,
and the boundary pad. It dispatches dense matmuls and (operator-
compatible) nonlinear ops to an accelerator backend, but it never lets
the backend see plaintext input, plaintext adapter, plaintext gradients,
or raw masks.

This is **not** a real TEE in Stage 7.5c. The controller is a normal
Python object whose only role is to make the trust boundary explicit so
that a future TEE deployment can drop in without rewriting the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.runtime.interfaces import AcceleratorBackend
from pllo.runtime.local_cpu_backend import LocalCPUBackend


@dataclass
class TrustedControllerConfig:
    dtype: str = "float64"
    device: str = "cpu"
    use_pad: bool = True
    pad_scale: float = 1.0


class TrustedController:
    """Stage 7.5c trusted-side controller.

    The controller owns: mask sampler, pad sampler, adapter trusted state,
    optimizer (trusted), loss closure (trusted), sampler (trusted), and
    the un-mask / pad-compensation arithmetic. It dispatches the linear /
    nonlinear / KV cache / LoRA forward / LoRA backward ops to a
    backend that implements :class:`AcceleratorBackend`.
    """

    def __init__(
        self,
        *,
        backend: AcceleratorBackend | None = None,
        config: TrustedControllerConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config or TrustedControllerConfig()
        if backend is None:
            backend = LocalCPUBackend(dtype=self._dtype())
        self.backend: AcceleratorBackend = backend
        if generator is None:
            generator = torch.Generator(device=self._device())
            generator.manual_seed(2026)
        self._generator = generator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dtype(self) -> torch.dtype:
        return torch.float64 if self.config.dtype == "float64" else torch.float32

    def _device(self) -> torch.device:
        return torch.device(self.config.device)

    # ------------------------------------------------------------------
    # Mask + pad sampling (trusted side only)
    # ------------------------------------------------------------------

    def sample_mask(self, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a fresh invertible mask ``N`` and return ``(N, N^{-1})``.

        The raw ``N`` is trusted-only and MUST NOT cross the backend
        boundary; the backend only ever sees ``X @ N``.
        """
        n, n_inv = generate_invertible_matrix(dim, self._dtype(), self._device())
        return n, n_inv

    def sample_pad(self, shape: tuple[int, int]) -> torch.Tensor:
        """Sample a fresh trusted-side pad tensor."""
        if not self.config.use_pad:
            return torch.zeros(shape, dtype=self._dtype(), device=self._device())
        return generate_pad(
            shape, self._dtype(), self._device(), self.config.pad_scale,
        )

    # ------------------------------------------------------------------
    # Linear path
    # ------------------------------------------------------------------

    def transform_linear(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        n_in: torch.Tensor,
        n_in_inv: torch.Tensor,
        n_out: torch.Tensor,
        pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build ``(X_tilde, W_tilde, bias_tilde, pad_compensation)``."""
        if pad is None:
            x_tilde = x @ n_in
        else:
            x_tilde = (x - pad) @ n_in
        w_tilde = n_in_inv @ w @ n_out
        bias_tilde = None if bias is None else bias @ n_out
        compensation = None if pad is None else pad @ w @ n_out
        return x_tilde, w_tilde, bias_tilde, compensation

    def recover_output(
        self, y_tilde: torch.Tensor, n_out_inv: torch.Tensor,
    ) -> torch.Tensor:
        """``Y = Y_tilde @ N_out^{-1}`` on the trusted side."""
        return y_tilde @ n_out_inv

    def recover_logits(
        self, logits_tilde: torch.Tensor, n_out_inv: torch.Tensor,
    ) -> torch.Tensor:
        return self.recover_output(logits_tilde, n_out_inv)

    # ------------------------------------------------------------------
    # LoRA adapter path
    # ------------------------------------------------------------------

    def transform_lora_adapter(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        n_in_inv: torch.Tensor,
        n_out: torch.Tensor,
        u: torch.Tensor,
        u_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return n_in_inv @ a @ u, u_inv @ b @ n_out

    def recover_lora_gradients(
        self,
        grad_a_tilde: torch.Tensor,
        grad_b_tilde: torch.Tensor,
        *,
        n_in: torch.Tensor,
        n_out_inv: torch.Tensor,
        u: torch.Tensor,
        u_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-mask grad_A_tilde / grad_B_tilde back into plain space."""
        grad_a = n_in @ grad_a_tilde @ u_inv
        grad_b = u @ grad_b_tilde @ n_out_inv
        return grad_a, grad_b

    # ------------------------------------------------------------------
    # Optimizer step (trusted side only)
    # ------------------------------------------------------------------

    def optimizer_step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        lr: float,
    ) -> dict[str, torch.Tensor]:
        """Trivial SGD step on the trusted side. The optimizer state stays
        inside this object; the GPU / TEE backend never sees it.
        """
        updated = {}
        for key, p in params.items():
            g = grads.get(key)
            if g is None:
                updated[key] = p
                continue
            updated[key] = p - lr * g
        return updated

    # ------------------------------------------------------------------
    # Transcript sanitisation
    # ------------------------------------------------------------------

    def sanitize_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Strip any field the trusted side does not allow to be published.

        Backends shouldn't put raw secrets in the transcript -- but if a
        future backend mis-implements the contract, the controller has the
        final say over what reaches disk.
        """
        forbidden = {
            "raw_input", "raw_x", "raw_w", "raw_a", "raw_b", "raw_pad",
            "raw_n_in", "raw_n_out", "raw_u", "private_data", "labels",
            "raw_grad_a", "raw_grad_b", "optimizer_state",
        }
        cleaned = dict(summary)
        for k in list(cleaned.keys()):
            if k in forbidden:
                cleaned.pop(k)
        cleaned["contains_raw_secret"] = False
        cleaned["sanitized_by_trusted_controller"] = True
        return cleaned


__all__ = ["TrustedController", "TrustedControllerConfig"]
