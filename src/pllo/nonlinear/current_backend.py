"""Current nonlinear handling: trusted islands / trusted shortcut (Line A).

Each nonlinearity is evaluated inside the **trusted boundary** (the TEE island /
trusted shortcut already used by the repo, see
:mod:`pllo.ops.nonlinear_islands` and the ``trusted_softmax_attention`` path in
the decoder wrappers). The computation is exact in the working dtype; the cost is
charged to the trusted boundary (``trusted_calls`` / ``trusted_bytes``) and no
nonlinear payload is sent to the untrusted accelerator (``gpu_bytes = 0``).

Security here rests on the trusted boundary itself (the established design); this
module makes no *new* security claim. ``tee_used_on_gpu`` is always False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pllo.nonlinear.backends import NonlinearBackend, NonlinearOpResult, tensor_bytes

__all__ = ["CurrentNonlinearBackend"]


class CurrentNonlinearBackend(NonlinearBackend):
    name = "current"
    security_status = "trusted_boundary"
    security_note = ("Nonlinearity evaluated inside the trusted boundary "
                     "(trusted island / trusted shortcut); security rests on the "
                     "established TEE-boundary design, not on a new construction.")

    def _trusted(self, x: torch.Tensor, out: torch.Tensor) -> NonlinearOpResult:
        # One trusted-boundary crossing; bytes = input copied in + output out.
        return NonlinearOpResult(
            output=out, trusted_calls=1,
            trusted_bytes=tensor_bytes(x) + tensor_bytes(out),
            gpu_bytes=0, tee_used_on_gpu=False,
            extra={"location": "trusted_boundary"})

    def gelu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._trusted(x, F.gelu(x))

    def silu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._trusted(x, F.silu(x))

    def softmax(self, x: torch.Tensor, dim: int = -1) -> NonlinearOpResult:
        return self._trusted(x, torch.softmax(x, dim=dim))

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None,
                  eps: float = 1e-5) -> NonlinearOpResult:
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(-1, keepdim=True)
        out = centered * torch.rsqrt(var + eps)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return self._trusted(x, out)

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-6) -> NonlinearOpResult:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            out = out * weight
        return self._trusted(x, out)
