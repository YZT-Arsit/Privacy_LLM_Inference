"""Trainable split-runtime primitives for the adapted baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch
from torch import nn

from .lora_layers import lora_scale
from .transforms import transform_base, transform_factors


@dataclass
class RuntimeCounters:
    transformed_linear_calls: int = 0
    trusted_norm_calls: int = 0
    trusted_rope_calls: int = 0
    trusted_softmax_calls: int = 0
    trusted_nonlinear_calls: int = 0
    trusted_residual_calls: int = 0
    trusted_embedding_calls: int = 0
    trusted_output_calls: int = 0
    trusted_loss_calls: int = 0
    backward_boundary_calls: int = 0
    optimizer_updates: int = 0
    physical_messages: int = 0
    bytes_a10_to_tdx: int = 0
    bytes_tdx_to_a10: int = 0
    trusted_compute_ns: int = 0
    transport_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


class TrustedRuntime:
    """Local oracle for operations assigned to TDX by the frozen contract.

    A remote transport can implement this same interface.  Counting is kept
    here so a local test cannot accidentally be reported as a real crossing.
    """

    def __init__(self, counters: RuntimeCounters | None = None, *, real_transport: bool = False):
        self.counters = counters or RuntimeCounters()
        self.real_transport = bool(real_transport)

    def _call(self, name: str, inputs: tuple[torch.Tensor, ...], fn):
        start = time.perf_counter_ns()
        result = fn()
        self.counters.trusted_compute_ns += time.perf_counter_ns() - start
        setattr(self.counters, name, getattr(self.counters, name) + 1)
        if self.real_transport:
            self.counters.physical_messages += 2
            self.counters.bytes_a10_to_tdx += sum(_bytes(x) for x in inputs)
            outs = result if isinstance(result, tuple) else (result,)
            self.counters.bytes_tdx_to_a10 += sum(_bytes(x) for x in outs if isinstance(x, torch.Tensor))
        return result

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        def op():
            x32 = x.float()
            return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
        return self._call("trusted_norm_calls", (x, weight), op)

    def rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        def op():
            c, s = cos, sin
            while c.ndim < q.ndim:
                c, s = c.unsqueeze(0), s.unsqueeze(0)
            return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        return self._call("trusted_rope_calls", (q, k, cos, sin), op)

    def softmax(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self._call("trusted_softmax_calls", (scores,), lambda: torch.softmax(scores.float(), dim=dim).to(scores.dtype))

    def swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self._call("trusted_nonlinear_calls", (gate, up), lambda: torch.nn.functional.silu(gate) * up)

    def residual(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return self._call("trusted_residual_calls", (residual, update), lambda: residual + update)

    def embedding(self, input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._call("trusted_embedding_calls", (input_ids, weight), lambda: torch.nn.functional.embedding(input_ids, weight))

    def output(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._call("trusted_output_calls", (hidden, weight), lambda: hidden @ weight.transpose(0, 1))

    def cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        return self._call("trusted_loss_calls", (logits, labels), lambda: torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]), labels[:, 1:].contiguous().view(-1),
            ignore_index=ignore_index))


class ExternalLoRALinear(nn.Module):
    """Frozen transformed base plus trainable transformed-coordinate factors.

    Mathematical weight layout is ``(d_in, d_out)``.  Only ``a_star`` and
    ``b_star`` are parameters; neither canonical factors nor transform secrets
    are included in ``state_dict``.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        rotation: torch.Tensor,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        direction: str = "input",
        bias: torch.Tensor | None = None,
        counters: RuntimeCounters | None = None,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or rank <= 0:
            raise ValueError("invalid weight or rank")
        self.direction = direction
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = lora_scale(alpha, rank)
        self.counters = counters or RuntimeCounters()
        gen = torch.Generator(device="cpu").manual_seed(int(init_seed))
        a = torch.randn(weight.shape[0], rank, generator=gen, dtype=torch.float64).to(weight) * 0.01
        b = torch.zeros(rank, weight.shape[1], dtype=weight.dtype, device=weight.device)
        w_star = transform_base(weight, rotation, direction)
        a_star, b_star = transform_factors(a, b, rotation, direction)
        self.register_buffer("weight_star", w_star.detach().clone())
        self.register_buffer("rotation", rotation.detach().clone(), persistent=False)
        self.register_buffer("trusted_bias", None if bias is None else bias.detach().clone(), persistent=False)
        self.a_star = nn.Parameter(a_star)
        self.b_star = nn.Parameter(b_star)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counters.transformed_linear_calls += 1
        merged = self.weight_star + self.scale * (self.a_star @ self.b_star)
        if self.direction == "input":
            output = (x @ self.rotation) @ merged
        elif self.direction == "output":
            output = (x @ merged) @ self.rotation.transpose(0, 1)
        else:
            raise ValueError("invalid direction")
        if self.trusted_bias is not None:
            output = output + self.trusted_bias
        if output.requires_grad:
            output.register_hook(self._record_backward_boundary)
        return output

    def _record_backward_boundary(self, gradient: torch.Tensor) -> torch.Tensor:
        self.counters.backward_boundary_calls += 1
        return gradient

    @classmethod
    def from_torch_linear(cls, linear: nn.Linear, rotation: torch.Tensor, **kwargs):
        return cls(linear.weight.detach().transpose(0, 1), rotation, bias=linear.bias, **kwargs)
