"""O(V) monomial logit mask for the private cross-entropy boundary.

The LM head emits *masked* logits ``Z_tilde = Z M`` where ``M = D Pi`` is a
**monomial** matrix (each column exactly one nonzero): a permutation ``Pi`` times a
nonzero diagonal ``D``. This is strictly O(V) in memory/compute (a permutation
index + a scale vector) -- there is NO dense ``V x V`` matrix. The dense form is
built only inside the unit tests, at tiny V, to check exactness against the O(V)
kernels.

Index algebra (columns of an ``(n, V)`` logit tensor; ``d`` nonzero scales,
``pi`` a permutation, ``pi_inv`` its inverse):

* mask (accelerator, on true logits Z):   ``Z_tilde[:, k] = d[k] * Z[:, pi[k]]``
* recover (trusted boundary):             ``Z_rec = (Z_tilde / d)[:, pi_inv]``  (== Z)
* grad remap (trusted boundary -> wire):  ``G_tilde[:, k] = G[:, pi[k]] / d[k]``

where ``G = dL/dZ`` is the true-logit gradient (softmax - onehot, averaged over
valid positions). ``G_tilde = dL/dZ_tilde`` is exactly the gradient the
accelerator needs to keep back-propagating through the folded LM head -- verified
against autograd in the tests.

Permutation-only baseline: ``d == 1`` (a pure permutation; the existing Gate-3
vocab-permutation mask). The monomial adds the nonzero diagonal, which also hides
the logit **value multiset** (a permutation alone leaks it). ``D`` is drawn with a
bounded condition number and away from extreme bf16 scales.

A dense vocab mask is intentionally NOT implemented (it would be ~92 GB at
V=151936 and is unnecessary: the monomial is exact and O(V)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "MonomialLogitMask",
    "make_monomial_logit_mask",
    "PrivateCEResult",
]


@dataclass
class PrivateCEResult:
    """Output of the trusted-boundary private CE over masked logits."""
    loss: torch.Tensor            # scalar mean CE over valid positions
    masked_logit_gradient: torch.Tensor  # G_tilde = dL/dZ_tilde  (n, V)
    n_valid: int


class MonomialLogitMask:
    """An O(V) monomial logit mask ``M = D Pi`` (permutation + nonzero diagonal).

    Holds only ``pi`` (perm), ``pi_inv`` (inverse perm) and ``d`` (scales). The
    permutation-only baseline sets ``d == 1``.
    """

    def __init__(self, pi: torch.Tensor, d: torch.Tensor):
        assert pi.dim() == 1 and d.dim() == 1 and pi.numel() == d.numel()
        V = pi.numel()
        pi = pi.to(torch.long)
        # validate bijection
        assert int(torch.unique(pi).numel()) == V, "pi is not a permutation"
        assert torch.all(d != 0), "D must be nonzero (invertible monomial)"
        self.V = V
        self.pi = pi
        self.pi_inv = torch.empty_like(pi)
        self.pi_inv[pi] = torch.arange(V, device=pi.device)
        self.d = d

    # -- device / dtype helpers --------------------------------------------
    def to(self, device=None, dtype=None) -> "MonomialLogitMask":
        pi = self.pi.to(device=device) if device is not None else self.pi
        d = self.d
        if device is not None:
            d = d.to(device=device)
        if dtype is not None:
            d = d.to(dtype=dtype)
        m = MonomialLogitMask.__new__(MonomialLogitMask)
        m.V = self.V
        m.pi = pi
        m.pi_inv = self.pi_inv.to(device=device) if device is not None else self.pi_inv
        m.d = d
        return m

    @property
    def is_permutation_only(self) -> bool:
        return bool(torch.all(self.d == 1))

    def condition_number(self) -> float:
        a = self.d.abs()
        return float(a.max() / a.min())

    # -- O(V) kernels -------------------------------------------------------
    def mask_logits(self, Z: torch.Tensor) -> torch.Tensor:
        """Z_tilde[:, k] = d[k] * Z[:, pi[k]] -- applied on the accelerator."""
        d = self.d.to(Z.dtype)
        return Z.index_select(-1, self.pi.to(Z.device)) * d.to(Z.device)

    def recover_logits(self, Z_tilde: torch.Tensor) -> torch.Tensor:
        """Z_rec = (Z_tilde / d)[:, pi_inv] == Z -- trusted boundary only."""
        d = self.d.to(Z_tilde.dtype).to(Z_tilde.device)
        tmp = Z_tilde / d
        return tmp.index_select(-1, self.pi_inv.to(Z_tilde.device))

    def mask_gradient(self, G: torch.Tensor) -> torch.Tensor:
        """G_tilde[:, k] = G[:, pi[k]] / d[k] -- remap true-logit grad to wire."""
        d = self.d.to(G.dtype).to(G.device)
        return G.index_select(-1, self.pi.to(G.device)) / d

    def dense_matrix(self, dtype=torch.float64) -> torch.Tensor:
        """Explicit (V, V) monomial M with M[pi[k], k] = d[k]. TESTS ONLY."""
        M = torch.zeros(self.V, self.V, dtype=dtype)
        M[self.pi, torch.arange(self.V)] = self.d.to(dtype)
        return M

    # -- trusted-boundary private CE ---------------------------------------
    def private_ce(self, Z_tilde: torch.Tensor, labels: torch.Tensor,
                   *, ignore_index: int = -100,
                   compute_dtype: torch.dtype = torch.float32) -> PrivateCEResult:
        """Recover true logits, compute private CE with private labels, return
        masked dlogits ``G_tilde`` for the wire. Runs entirely at the trusted
        boundary. ``labels`` are the already-shifted next-token ids; padding /
        non-target positions carry ``ignore_index``.

        The mean is over valid (non-ignored) positions -- matching
        ``F.cross_entropy(reduction='mean')`` so the loss equals the plaintext CE.
        """
        Zt = Z_tilde.to(compute_dtype)
        Z = self.recover_logits(Zt)                       # exact true logits
        valid = labels != ignore_index
        nv = int(valid.sum())
        if nv == 0:
            zero = Z.new_zeros(())
            return PrivateCEResult(loss=zero,
                                   masked_logit_gradient=torch.zeros_like(Z_tilde),
                                   n_valid=0)
        lab_v = labels[valid]
        Zv = Z[valid]
        loss = F.cross_entropy(Zv, lab_v, reduction="mean")
        # analytic grad wrt true logits: (softmax - onehot)/nv on valid rows
        G = torch.zeros_like(Z)
        P = torch.softmax(Zv, dim=-1)
        idx = torch.arange(nv, device=Z.device)
        P[idx, lab_v] -= 1.0
        G[valid] = P / nv
        G_tilde = self.mask_gradient(G).to(Z_tilde.dtype)
        return PrivateCEResult(loss=loss, masked_logit_gradient=G_tilde,
                               n_valid=nv)

    # -- leakage proxies (analysis only) -----------------------------------
    def value_multiset_preserved(self, Z: torch.Tensor,
                                 atol: float = 1e-5) -> bool:
        """True iff the *sorted* logit values survive masking (i.e. an attacker
        who sees Z_tilde recovers the exact value multiset). A permutation alone
        preserves it; a nontrivial monomial does NOT."""
        Zt = self.mask_logits(Z)
        a = torch.sort(Z.flatten().to(torch.float64)).values
        b = torch.sort(Zt.flatten().to(torch.float64)).values
        return bool(torch.allclose(a, b, atol=atol))


def make_monomial_logit_mask(vocab_size: int, *, seed: int = 0,
                             permutation_only: bool = False,
                             scale_low: float = 0.5, scale_high: float = 2.0,
                             device=None) -> MonomialLogitMask:
    """Draw an O(V) monomial mask with a bounded condition number.

    ``permutation_only=True`` gives ``d == 1`` (the permutation baseline). The
    scales are drawn uniformly in ``[scale_low, scale_high]`` with a random sign,
    so ``|d|`` in ``[scale_low, scale_high]`` bounds the condition number by
    ``scale_high / scale_low`` and keeps values well inside bf16 range.
    """
    assert scale_low > 0 and scale_high >= scale_low
    g = torch.Generator().manual_seed(int(seed))
    pi = torch.randperm(vocab_size, generator=g)
    if permutation_only:
        d = torch.ones(vocab_size, dtype=torch.float64)
    else:
        mag = torch.rand(vocab_size, generator=g, dtype=torch.float64)
        mag = scale_low + (scale_high - scale_low) * mag
        sign = torch.where(torch.rand(vocab_size, generator=g) < 0.5,
                           -1.0, 1.0).to(torch.float64)
        d = mag * sign
    m = MonomialLogitMask(pi=pi, d=d)
    if device is not None:
        m = m.to(device=device)
    return m
