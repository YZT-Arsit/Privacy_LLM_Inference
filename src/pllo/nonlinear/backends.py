"""Selectable nonlinear-handling backends (Line A vs Line B).

The repo handles transformer nonlinear "islands" (GELU/SiLU, Softmax,
LayerNorm/RMSNorm, and the trusted-softmax shortcut) two ways:

* **current** -- the nonlinearity is evaluated in the *trusted boundary*
  (trusted island / trusted shortcut); see :mod:`pllo.ops.nonlinear_islands`.
* **amulet_migrated** -- the nonlinearity is *migrated* off the trusted boundary
  onto the untrusted accelerator via an Amulet-style lifted/transformed view;
  see :mod:`pllo.ops.amulet_lifted_islands`.

This module defines the common backend interface + a uniform per-op accounting
record so the two lines can be compared on identical inputs for correctness and
efficiency. torch is used (consistent with the rest of the repo).

Security note: the Amulet migration's security is **not formally claimed** here
(status ``not_formally_claimed`` / ``under_discussion``); see
:mod:`pllo.nonlinear.amulet_backend`. This module measures correctness +
efficiency only and makes no security claim for either backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "NonlinearOpResult",
    "NonlinearBackend",
    "tensor_bytes",
    "reference_gelu",
    "reference_silu",
    "reference_softmax",
    "reference_layernorm",
    "reference_rmsnorm",
    "OP_NAMES",
]

OP_NAMES = ("gelu", "silu", "softmax", "layernorm", "rmsnorm")


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


# ---------------------------------------------------------------------------
# Float64 references (ground truth for correctness)
# ---------------------------------------------------------------------------


def reference_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.to(torch.float64))                  # exact erf GELU


def reference_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x.to(torch.float64))


def reference_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x.to(torch.float64), dim=dim)


def reference_layernorm(x: torch.Tensor, weight: torch.Tensor | None = None,
                        bias: torch.Tensor | None = None,
                        eps: float = 1e-5) -> torch.Tensor:
    xd = x.to(torch.float64)
    mean = xd.mean(-1, keepdim=True)
    centered = xd - mean
    var = centered.pow(2).mean(-1, keepdim=True)
    out = centered * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.to(torch.float64)
    if bias is not None:
        out = out + bias.to(torch.float64)
    return out


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None = None,
                      eps: float = 1e-6) -> torch.Tensor:
    xd = x.to(torch.float64)
    out = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.to(torch.float64)
    return out


# ---------------------------------------------------------------------------
# Per-op result / accounting
# ---------------------------------------------------------------------------


@dataclass
class NonlinearOpResult:
    """Output of one nonlinear op + where the work ran (trust accounting).

    ``trusted_calls`` / ``trusted_bytes`` -- work done inside the trusted
    boundary (ECALL-like). ``gpu_bytes`` -- masked/lifted payload transferred to
    the untrusted accelerator. ``tee_used_on_gpu`` -- always False: the model
    nonlinear never runs *inside* a TEE on the GPU."""
    output: torch.Tensor
    trusted_calls: int
    trusted_bytes: int
    gpu_bytes: int
    tee_used_on_gpu: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class NonlinearBackend(ABC):
    """Common interface for a nonlinear-handling backend.

    Subclasses set ``name`` and ``security_status`` and implement the ops. Each
    op returns a :class:`NonlinearOpResult` so the microbench can compare
    correctness and the trusted/accelerator cost split across backends."""

    name: str = "abstract"
    # one of: "trusted_boundary" | "not_formally_claimed" | "under_discussion"
    security_status: str = "not_formally_claimed"
    # paper-facing claim status (E3): "established" | "under_discussion".
    # Amulet stays "under_discussion" until the advisor confirms the formal
    # security boundary; no security is proven here.
    security_claim_status: str = "under_discussion"
    security_note: str = ""

    @abstractmethod
    def gelu(self, x: torch.Tensor) -> NonlinearOpResult: ...

    @abstractmethod
    def silu(self, x: torch.Tensor) -> NonlinearOpResult: ...

    @abstractmethod
    def softmax(self, x: torch.Tensor, dim: int = -1) -> NonlinearOpResult: ...

    @abstractmethod
    def layernorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None,
                  eps: float = 1e-5) -> NonlinearOpResult: ...

    @abstractmethod
    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-6) -> NonlinearOpResult: ...

    def run(self, op_name: str, x: torch.Tensor, **kw) -> NonlinearOpResult:
        """Dispatch by op name (used by the microbench)."""
        if op_name == "gelu":
            return self.gelu(x)
        if op_name == "silu":
            return self.silu(x)
        if op_name == "softmax":
            return self.softmax(x, dim=kw.get("dim", -1))
        if op_name == "layernorm":
            return self.layernorm(x, weight=kw.get("weight"),
                                  bias=kw.get("bias"), eps=kw.get("eps", 1e-5))
        if op_name == "rmsnorm":
            return self.rmsnorm(x, weight=kw.get("weight"),
                                eps=kw.get("eps", 1e-6))
        raise ValueError(f"unknown op {op_name!r}; expected one of {OP_NAMES}")

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "security_status": self.security_status,
                "security_claim_status": self.security_claim_status,
                "security_note": self.security_note, "ops": list(OP_NAMES)}
