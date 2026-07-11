"""Shared, autograd-friendly masked operator kernels (inference == training).

These are the operator kernels the paper-facing A_rightmul path uses. They are
written once here so the masked *inference* forward and the masked *training*
forward call the SAME implementation (spec E: "extract shared operator kernels,
let inference/training share one implementation" -- do not fork a parallel
training-only copy).

Key properties, all under the *compatible-mask* family:

* residual / hidden mask ``N`` is an **orthogonal signed permutation** (monomial):
  ``N N^T = I``. RMSNorm's per-row mean-square is invariant under a signed
  permutation, so ``rmsnorm_core(x @ N) == rmsnorm_core(x) @ N``.
* attention post-RoPE Q/K mask ``R`` (per head) is **orthogonal** so
  ``(q @ R)(k @ R)^T == q k^T`` -- scores are preserved, softmax runs on the true
  scores directly over the masked state.
* SwiGLU channel mask ``P`` is a **pure permutation** (no signs -- SiLU is not
  odd), so ``SiLU(gate @ P) * (up @ P) == (SiLU(gate) * up) @ P``.

Every nonlinearity here runs directly on the (masked) accelerator state with
``trusted_calls == 0`` -- there is no unmask-to-plaintext. The kernels are plain
differentiable torch ops, so a standard ``loss.backward()`` differentiates
through the masked graph (no custom backward needed for the linear/residual
transport; the masks are constants w.r.t. the LoRA leaves).

CPU/fp64 contract math. NOT a real-model or GPU result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F

__all__ = [
    "NonlinearAccounting",
    "orthogonal_signed_perm",
    "permutation_matrix",
    "rmsnorm_core",
    "silu_island",
    "softmax_island",
    "rope_cos_sin",
    "apply_rope",
    "repeat_kv",
]


@dataclass
class NonlinearAccounting:
    """Counts where nonlinear work ran. paper_safe requires trusted_calls == 0
    and plaintext_hidden_materializations == 0."""
    nonlinear_trusted_calls: int = 0
    trusted_nonlinear_ops_count: int = 0
    plaintext_hidden_materializations: int = 0
    accelerator_nonlinear_ops: int = 0
    per_op: Dict[str, int] = field(default_factory=dict)

    def note_accelerator(self, op: str):
        self.accelerator_nonlinear_ops += 1
        self.per_op[op] = self.per_op.get(op, 0) + 1


def orthogonal_signed_perm(n: int, seed: int, dtype=torch.float64) -> torch.Tensor:
    """Dense (n, n) orthogonal signed permutation (monomial): exactly one +/-1
    per row/column. ``N N^T == I``. Used for residual/hidden and per-head Q/K/V
    masks (RMSNorm-invariant, score-preserving)."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype)
    N[torch.arange(n), perm] = signs
    return N


def permutation_matrix(n: int, seed: int, dtype=torch.float64) -> torch.Tensor:
    """Dense (n, n) pure permutation (no signs). Used for the SwiGLU shared
    channel mask (SiLU is not odd, so signs would break the identity)."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    P = torch.zeros(n, n, dtype=dtype)
    P[torch.arange(n), perm] = 1.0
    return P


def rmsnorm_core(x: torch.Tensor, eps: float = 1e-6,
                 acct: "NonlinearAccounting | None" = None) -> torch.Tensor:
    """RMSNorm core (no elementwise gamma; gamma is folded into the next
    projection). Invariant under a signed permutation on the last dim."""
    if acct is not None:
        acct.note_accelerator("rmsnorm")
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def silu_island(gate: torch.Tensor, up: torch.Tensor,
                acct: "NonlinearAccounting | None" = None) -> torch.Tensor:
    """SwiGLU activation on the (channel-permuted) masked state."""
    if acct is not None:
        acct.note_accelerator("silu")
    return F.silu(gate) * up


def softmax_island(scores: torch.Tensor, dim: int = -1,
                   acct: "NonlinearAccounting | None" = None) -> torch.Tensor:
    if acct is not None:
        acct.note_accelerator("softmax")
    return torch.softmax(scores, dim=dim)


def rope_cos_sin(seq_len: int, head_dim: int, base: float = 1e6,
                 dtype=torch.float64):
    """Standard RoPE cos/sin tables (Qwen uses base 1e6)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=dtype) / half))
    t = torch.arange(seq_len, dtype=dtype)
    freqs = torch.outer(t, inv_freq)                     # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)              # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
               ) -> torch.Tensor:
    """x: (..., T, head_dim); cos/sin: (T, head_dim)."""
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: expand (n_kv, T, hd) -> (n_kv*n_rep, T, hd). Differentiable
    (repeat_interleave has a correct backward)."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)
