"""A_rightmul nonlinear handling: compatible right-multiply islands (Line A_rightmul).

This is the paper's **A_rightmul** design. Every transformer nonlinear island
(SiLU/SwiGLU MLP, attention softmax, RMSNorm/LayerNorm core) is evaluated
**directly on the untrusted accelerator** over the compatible right-multiply /
permutation-masked state, with **no trusted-boundary crossing** for the
nonlinearity. The TEE is entered exactly once (input embedding + mask) and exited
exactly once (final logits recovery); no nonlinear op runs inside the TEE.

Why this is exact (the "compatible" masks):

* **RMSNorm/LayerNorm core** -- the residual mask is a signed permutation
  ``N_res`` (an orthogonal monomial matrix). RMSNorm's per-row mean-square is
  invariant under a signed permutation, so
  ``rmsnorm_core(x @ N_res) == rmsnorm_core(x) @ N_res`` -- the masked core can be
  computed in place on the accelerator and the output stays in the masked basis.
* **Attention softmax** -- the per-head Q/K/V masks satisfy
  ``Q_tilde K_tilde^T == Q K^T`` so the scores fed to softmax are already the true
  scores; ``softmax`` runs on the accelerator and the (masked) value projection
  carries the result back into the masked basis.
* **SiLU / SwiGLU** -- the gate/up projections use a shared channel permutation
  ``P`` so ``SiLU(gate @ P) * (up @ P) == (SiLU(gate) * up) @ P`` -- the
  activation is applied directly to the masked channels on the accelerator.

Because the op is applied to exactly the tensor the folded worker already holds
(same ``F.silu`` / ``torch.softmax`` / ``x * rsqrt(mean(x^2)+eps)`` formula as the
``current`` backend), the numerics are **bit-identical to ``current``** -- the
only difference is *where* the work is accounted: on the accelerator with
``trusted_calls == 0`` (vs. ``current`` charging it to the trusted boundary).

SECURITY STATUS: ``under_development``. The paper's compatible right-multiply
security argument is not yet a completed proof in this repo; we therefore mark it
``under_development`` (NOT ``established``) until the proofs are added. No formal
security is claimed here. ``tee_used_on_gpu`` is always False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pllo.nonlinear.backends import NonlinearBackend, NonlinearOpResult, tensor_bytes

__all__ = ["RightMultiplyNonlinearBackend"]


class RightMultiplyNonlinearBackend(NonlinearBackend):
    name = "compatible_right_multiply"
    security_status = "under_development"
    security_claim_status = "under_development"   # proofs not yet added
    security_note = (
        "A_rightmul compatible right-multiply nonlinear islands: every "
        "nonlinearity runs on the untrusted accelerator over the "
        "permutation/right-multiply-masked state with no trusted crossing "
        "(single TEE entry/exit). Numerically identical to the 'current' "
        "trusted-island backend. Security is under development -- the "
        "compatible right-multiply proof is not yet completed in this repo and "
        "is NOT formally claimed.")

    def __init__(self, **_ignored) -> None:
        # Accepts (and ignores) lift_k/seed so it is interchangeable with the
        # other registry backends via make_nonlinear_backend(**kwargs).
        pass

    def _on_accelerator(self, x: torch.Tensor, out: torch.Tensor,
                        op: str) -> NonlinearOpResult:
        """One accelerator-side right-multiply island: zero trusted crossings."""
        return NonlinearOpResult(
            output=out, trusted_calls=0, trusted_bytes=0,
            gpu_bytes=tensor_bytes(x) + tensor_bytes(out), tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator",
                   "right_multiply_executed": True,
                   "masking_mode": "compatible_right_multiply_or_permutation",
                   "op": op})

    def gelu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._on_accelerator(x, F.gelu(x), "gelu")

    def silu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._on_accelerator(x, F.silu(x), "silu")

    def softmax(self, x: torch.Tensor, dim: int = -1) -> NonlinearOpResult:
        return self._on_accelerator(x, torch.softmax(x, dim=dim), "softmax")

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
        return self._on_accelerator(x, out, "layernorm")

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-6) -> NonlinearOpResult:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            out = out * weight
        return self._on_accelerator(x, out, "rmsnorm")
