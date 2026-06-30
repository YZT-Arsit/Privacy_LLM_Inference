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

SECURITY STATUS: ``claimed_under_compatible_mask_assumption``. The A_rightmul
security claim holds *only when the masks are in the compatible family* -- the
residual/RMSNorm/LayerNorm mask is a signed permutation (orthogonal monomial),
the attention Q/K masks are orthogonal + score-preserving (``Nq Nk^T == I``), and
SwiGLU uses a shared channel permutation. That assumption is made CHECKABLE by
:mod:`pllo.ops.compatible_mask_verify` and is enforced in the real build/worker
path (``compatible_masks_verified == True``): an arbitrary dense /
``pairwise_complex_scaling`` mask is REJECTED, never silently accepted. The claim
is therefore *conditional*, not a completed unconditional proof, and is NOT
claimed for arbitrary dense masks. ``tee_used_on_gpu`` is always False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pllo.nonlinear.backends import NonlinearBackend, NonlinearOpResult, tensor_bytes

__all__ = ["RightMultiplyNonlinearBackend"]


class RightMultiplyNonlinearBackend(NonlinearBackend):
    name = "compatible_right_multiply"
    security_status = "claimed_under_compatible_mask_assumption"
    security_claim_status = "claimed_under_assumption"
    security_note = (
        "A_rightmul compatible right-multiply nonlinear islands: every "
        "nonlinearity runs on the untrusted accelerator over the "
        "permutation/right-multiply-masked state with no trusted crossing "
        "(single TEE entry/exit). Numerically identical to the 'current' "
        "trusted-island backend. Security is CLAIMED ONLY UNDER THE COMPATIBLE-"
        "MASK ASSUMPTION -- residual mask = signed permutation, attention Q/K "
        "orthogonal + score-preserving (Nq Nk^T == I), SwiGLU shared channel "
        "permutation -- which pllo.ops.compatible_mask_verify makes checkable and "
        "the real build/worker path enforces (compatible_masks_verified). It is "
        "NOT claimed for arbitrary dense / pairwise_complex_scaling masks (those "
        "are rejected, not silently accepted).")

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
