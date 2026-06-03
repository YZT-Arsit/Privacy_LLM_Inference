"""Stage 7.5c - deployable runtime boundary interfaces.

This module defines the *protocol* boundary between the trusted controller
and an accelerator backend. The current artifact ships exactly one backend
implementation, ``LocalCPUBackend``; the interface is intentionally written
so that a future ``TEEBackend`` / ``GPUBackend`` only has to replace the
backend object, not the protocol logic.

**This is NOT a real TEE deployment and NOT a real GPU deployment.** The
interface is *backend-ready*; the wire-up to confidential-compute hardware
is future work and is explicitly out of scope for Stage 7.5c.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class AcceleratorBackend(Protocol):
    """The protocol every accelerator implementation must satisfy.

    The current concrete implementation is :class:`LocalCPUBackend`.
    Future TEE / GPU backends should implement the same surface and be
    registered via :func:`pllo.runtime.backend_registry.register_backend`.
    """

    name: str

    def linear(
        self,
        x_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
        bias_tilde: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked dense Linear: ``Y_tilde = X_tilde @ W_tilde + bias_tilde``."""
        ...

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Generic ``A @ B`` matmul on the accelerator."""
        ...

    def attention_scores(
        self, q_tilde: torch.Tensor, k_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """``Q_tilde @ K_tilde^T`` with ``N_Q N_K^T = I`` preserved."""
        ...

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically-stable softmax over ``dim``."""
        ...

    def activation(self, kind: str, x_tilde: torch.Tensor) -> torch.Tensor:
        """Apply a pointwise activation kernel (``gelu`` / ``relu`` / ``silu``).

        The mask family on ``x_tilde`` must be the permutation island family
        for which ``phi(Z @ P) = phi(Z) @ P``; the backend never inspects it.
        """
        ...

    def rmsnorm_core(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """Orthogonally-invariant RMSNorm core: ``X / RMS(X)`` row-wise."""
        ...

    def layernorm_core(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """Mean-preserving orthogonally-invariant LayerNorm core."""
        ...

    def append_kv_cache(
        self,
        cache_k_tilde: torch.Tensor | None,
        cache_v_tilde: torch.Tensor | None,
        new_k_tilde: torch.Tensor,
        new_v_tilde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate one or more new masked KV rows onto the token axis."""
        ...

    def lora_forward(
        self,
        x_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
        a_tilde: torch.Tensor,
        b_tilde: torch.Tensor,
        bias_tilde: torch.Tensor | None,
        pad_compensation: torch.Tensor | None,
        alpha: float,
    ) -> torch.Tensor:
        """One masked LoRA forward at a single Linear layer."""
        ...

    def lora_backward(
        self,
        x_tilde: torch.Tensor,
        a_tilde: torch.Tensor,
        b_tilde: torch.Tensor,
        grad_y_tilde: torch.Tensor,
        alpha: float,
        w_tilde: torch.Tensor | None = None,
        recover_grad_x: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Returns masked gradients ``{grad_a_tilde, grad_b_tilde[, grad_x_tilde]}``."""
        ...

    def measure_runtime(
        self, fn, *, num_warmup: int = 1, num_repeats: int = 3,
    ) -> dict[str, float]:
        """Local-emulation latency statistics for one accelerator call.

        Implementations MUST NOT call ``time.sleep``. The default
        ``LocalCPUBackend.measure_runtime`` uses ``time.perf_counter``.
        """
        ...

    def collect_transcript_summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary of what the backend has observed.

        The summary MUST NOT contain raw tensors, masks, adapters,
        gradients, or any plaintext input the trusted side considers
        private. Backends should redact at the source.
        """
        ...


class UnsupportedBackendOp(NotImplementedError):
    """A backend was asked to perform an op it has not implemented.

    Examples: a future ``GPUBackend`` that does not yet ship masked LoRA
    backward, or a ``TEEBackend`` that has not yet wired up RMSNorm.
    """


__all__ = ["AcceleratorBackend", "UnsupportedBackendOp"]
