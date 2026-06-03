"""Stage 7.5c - local CPU implementation of :class:`AcceleratorBackend`.

This is the only backend that ships in Stage 7.5c. It performs every
accelerator-side op on the local CPU using ordinary ``torch`` matmuls.
**No real TEE. No GPU. No network. No ``time.sleep``.**

The backend never inspects the trusted-side mask state: every input is
treated as opaque masked data. The ``RuntimeTranscript`` it builds carries
only shapes, op names, boundary-call counts, and timing -- never raw
tensors.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch

from pllo.runtime.transcript import RuntimeTranscript


class LocalCPUBackend:
    """The local CPU implementation of :class:`AcceleratorBackend`.

    Future TEE / GPU backends only need to re-implement the body of each
    method; the trusted-side protocol logic does not change.
    """

    name = "local_cpu"

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.dtype = dtype
        self._transcript = RuntimeTranscript(
            notes=(
                "Local CPU emulation backend; not real TEE wall-time and"
                " not GPU throughput."
            ),
        )

    # ------------------------------------------------------------------
    # Linear / matmul
    # ------------------------------------------------------------------

    def linear(
        self,
        x_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
        bias_tilde: torch.Tensor | None,
    ) -> torch.Tensor:
        self._transcript.record_op(
            "linear", shapes=[tuple(x_tilde.shape), tuple(w_tilde.shape)],
        )
        y = x_tilde @ w_tilde
        if bias_tilde is not None:
            y = y + bias_tilde
        return y

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._transcript.record_op(
            "matmul", shapes=[tuple(a.shape), tuple(b.shape)],
        )
        return a @ b

    # ------------------------------------------------------------------
    # Attention / softmax / nonlinear cores
    # ------------------------------------------------------------------

    def attention_scores(
        self, q_tilde: torch.Tensor, k_tilde: torch.Tensor,
    ) -> torch.Tensor:
        self._transcript.record_op(
            "attention_scores",
            shapes=[tuple(q_tilde.shape), tuple(k_tilde.shape)],
        )
        return q_tilde @ k_tilde.transpose(-2, -1)

    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._transcript.record_op("softmax", shapes=[tuple(x.shape)])
        return torch.softmax(x, dim=dim)

    def activation(self, kind: str, x_tilde: torch.Tensor) -> torch.Tensor:
        self._transcript.record_op(
            f"activation:{kind}", shapes=[tuple(x_tilde.shape)],
        )
        if kind == "gelu":
            return torch.nn.functional.gelu(x_tilde)
        if kind == "relu":
            return torch.nn.functional.relu(x_tilde)
        if kind == "silu":
            return torch.nn.functional.silu(x_tilde)
        raise ValueError(f"unknown activation kind {kind!r}")

    def rmsnorm_core(self, x_tilde: torch.Tensor) -> torch.Tensor:
        self._transcript.record_op("rmsnorm_core", shapes=[tuple(x_tilde.shape)])
        rms = x_tilde.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        return x_tilde / rms

    def layernorm_core(self, x_tilde: torch.Tensor) -> torch.Tensor:
        self._transcript.record_op("layernorm_core", shapes=[tuple(x_tilde.shape)])
        mean = x_tilde.mean(dim=-1, keepdim=True)
        centered = x_tilde - mean
        rms = centered.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        return centered / rms

    # ------------------------------------------------------------------
    # KV cache append
    # ------------------------------------------------------------------

    def append_kv_cache(
        self,
        cache_k_tilde: torch.Tensor | None,
        cache_v_tilde: torch.Tensor | None,
        new_k_tilde: torch.Tensor,
        new_v_tilde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._transcript.record_op(
            "append_kv_cache",
            shapes=[tuple(new_k_tilde.shape), tuple(new_v_tilde.shape)],
        )
        k = new_k_tilde if cache_k_tilde is None else torch.cat(
            [cache_k_tilde, new_k_tilde], dim=0,
        )
        v = new_v_tilde if cache_v_tilde is None else torch.cat(
            [cache_v_tilde, new_v_tilde], dim=0,
        )
        return k, v

    # ------------------------------------------------------------------
    # LoRA forward / backward
    # ------------------------------------------------------------------

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
        self._transcript.record_op(
            "lora_forward",
            shapes=[tuple(x_tilde.shape), tuple(a_tilde.shape), tuple(b_tilde.shape)],
        )
        rank = a_tilde.shape[1]
        scale = float(alpha) / max(rank, 1)
        y = x_tilde @ w_tilde + scale * (x_tilde @ a_tilde) @ b_tilde
        if bias_tilde is not None:
            y = y + bias_tilde
        if pad_compensation is not None:
            y = y + pad_compensation
        return y

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
        self._transcript.record_op(
            "lora_backward",
            shapes=[
                tuple(x_tilde.shape),
                tuple(a_tilde.shape),
                tuple(b_tilde.shape),
                tuple(grad_y_tilde.shape),
            ],
        )
        rank = a_tilde.shape[1]
        scale = float(alpha) / max(rank, 1)
        grad_a_tilde = scale * x_tilde.transpose(0, 1) @ (
            grad_y_tilde @ b_tilde.transpose(0, 1)
        )
        grad_b_tilde = scale * (x_tilde @ a_tilde).transpose(0, 1) @ grad_y_tilde
        out: dict[str, torch.Tensor | None] = {
            "grad_a_tilde": grad_a_tilde,
            "grad_b_tilde": grad_b_tilde,
        }
        if recover_grad_x:
            if w_tilde is None:
                raise ValueError("recover_grad_x=True requires w_tilde")
            out["grad_x_tilde"] = grad_y_tilde @ w_tilde.transpose(0, 1) + scale * (
                grad_y_tilde @ b_tilde.transpose(0, 1)
            ) @ a_tilde.transpose(0, 1)
        return out

    # ------------------------------------------------------------------
    # Measurement / transcript
    # ------------------------------------------------------------------

    def measure_runtime(
        self,
        fn: Callable[[], Any],
        *,
        num_warmup: int = 1,
        num_repeats: int = 3,
    ) -> dict[str, float]:
        """``time.perf_counter`` only -- never ``time.sleep``."""
        for _ in range(num_warmup):
            fn()
        times_ms: list[float] = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        result = {
            "mean_ms": float(statistics.mean(times_ms)),
            "median_ms": float(statistics.median(times_ms)),
            "std_ms": float(statistics.pstdev(times_ms)) if num_repeats > 1 else 0.0,
            "min_ms": float(min(times_ms)),
            "max_ms": float(max(times_ms)),
            "num_warmup": int(num_warmup),
            "num_repeats": int(num_repeats),
        }
        self._transcript.runtime_ms = float(result["mean_ms"])
        return result

    def collect_transcript_summary(self) -> dict[str, Any]:
        # By construction the transcript only carries shapes / op names /
        # boundary counts / runtime / redacted mask ids -- no raw tensors,
        # no plaintext input, no masks, no adapters, no gradients.
        return self._transcript.to_summary()


__all__ = ["LocalCPUBackend"]
