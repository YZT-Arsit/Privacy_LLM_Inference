"""Amulet-migrated nonlinear handling (Line B).

Each nonlinearity is *migrated* off the trusted boundary onto the untrusted
accelerator, consistent with :mod:`pllo.ops.amulet_lifted_islands`:

* **GELU / SiLU** -- a selector-lift view ``lift_R(U)`` expands each feature into
  ``k`` columns (valid column scaled by 1, decoys by positive scales); the
  activation runs on the accelerator over the lifted columns and an offline,
  trusted-folded squeeze selects the valid column. Exact for non-homogeneous
  activations. The activation no longer crosses the trusted boundary online
  (``trusted_calls = 0``) at the cost of ~``k``x accelerator bytes.
* **Softmax / LayerNorm / RMSNorm** -- the elementwise nonlinear part is migrated
  to the accelerator; only the small reduction statistic (row max / mean /
  mean-square) is kept as a trusted shortcut (``trusted_calls = 1``, a few bytes).

Correctness is preserved (the migration is functionally exact up to fp); the
measured difference vs the ``current`` backend is *where* the work runs (trusted
vs accelerator bytes) and the latency, not the numerics.

SECURITY STATUS: ``not_formally_claimed`` (``under_discussion`` with the advisor).
As noted in :mod:`pllo.ops.amulet_lifted_islands`, if a lifted/squeeze view is
visible to the accelerator and decoy rows are exactly zero, selector positions
could be inferred -- this is a correctness/efficiency prototype, **not** a proven
secure construction. ``tee_used_on_gpu`` is always False.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from pllo.nonlinear.backends import NonlinearBackend, NonlinearOpResult, tensor_bytes

__all__ = ["AmuletMigratedNonlinearBackend"]


class AmuletMigratedNonlinearBackend(NonlinearBackend):
    name = "amulet_migrated"
    security_status = "not_formally_claimed"
    security_claim_status = "under_discussion"     # NOT proven; advisor to confirm
    security_note = ("Amulet migration is a correctness/efficiency prototype; "
                     "security is under discussion with the advisor and NOT "
                     "formally claimed. Selector-lift positions could leak if the "
                     "lifted/squeeze view is observable with zero decoys "
                     "(see pllo.ops.amulet_lifted_islands).")

    def __init__(self, lift_k: int = 2, seed: int = 0) -> None:
        if lift_k < 2:
            raise ValueError("lift_k must be >= 2 (1 valid + >=1 decoy column)")
        self.lift_k = int(lift_k)
        self.seed = int(seed)
        # Cache of the deterministic selector-lift factors (``valid``, ``R``),
        # keyed by (h, device, dtype). They depend ONLY on (h, lift_k, seed) with
        # a fixed seed, so they are identical on every call and every layer --
        # regenerating them per call (CPU RNG + host->device copy, x28 layers x
        # every decoded token) was pure waste that starved the GPU. Caching is
        # bit-identical (same seed -> same factors) and does NOT change the
        # security posture: ``R`` was never per-call-randomized, and it lives in
        # the untrusted worker where the lifted view is materialized each call
        # regardless (see class-level SECURITY STATUS).
        self._lift_factors_cache: dict[
            tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _lift_factors(self, h: int, device: torch.device,
                      dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (``valid``, ``R``) selector-lift factors for width ``h`` on
        ``(device, dtype)``. Generated once (CPU RNG, fixed seed) then moved and
        cached; every subsequent call reuses the exact same tensors, so the
        activation output is bit-identical to regenerating per call."""
        key = (int(h), str(device), str(dtype))
        cached = self._lift_factors_cache.get(key)
        if cached is not None:
            return cached
        # Generated on CPU with a fixed seed (the torch CPU generator cannot
        # target CUDA directly) then moved onto the compute device. Same op order
        # as the historical per-call path -> identical bytes.
        gen = torch.Generator().manual_seed(self.seed)
        valid = torch.randint(0, self.lift_k, (h,), generator=gen).to(device)
        R = (torch.rand(h, self.lift_k, generator=gen) + 0.5).to(
            device=device, dtype=dtype)
        R[torch.arange(h, device=device), valid] = 1.0   # valid column scale = 1
        self._lift_factors_cache[key] = (valid, R)
        return valid, R

    # -- selector-lift migrated activation (exact) ----------------------------
    def _selector_lift(self, x: torch.Tensor,
                       act: Callable[[torch.Tensor], torch.Tensor]
                       ) -> NonlinearOpResult:
        lead = x.shape[:-1]
        h = x.shape[-1]
        U = x.reshape(-1, h)                              # [m, h]
        m = U.shape[0]
        # Deterministic, seed-fixed factors -> cached across calls/layers. The
        # lift runs on whatever accelerator holds ``x`` (CPU for prototypes, CUDA
        # on the H800); ``valid``/``R`` are read-only and shared, ``lift``/``Af``
        # are still materialized each call (the real accelerator cost + counters).
        valid, R = self._lift_factors(h, U.device, U.dtype)
        lift = U.unsqueeze(-1) * R.unsqueeze(0)           # [m, h, k]  (accelerator)
        Af = act(lift)                                    # activation on accelerator
        idx = valid.view(1, h, 1).expand(m, h, 1)
        out = Af.gather(2, idx).squeeze(-1)              # squeeze (folded offline)
        return NonlinearOpResult(
            output=out.reshape(*lead, h),
            trusted_calls=0,                              # migrated off the boundary
            trusted_bytes=0,
            gpu_bytes=tensor_bytes(lift) + tensor_bytes(Af),
            tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator", "lift_k": self.lift_k})

    def gelu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._selector_lift(x, F.gelu)

    def silu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._selector_lift(x, F.silu)

    # -- migrated elementwise reductions (trusted stat shortcut) --------------
    def softmax(self, x: torch.Tensor, dim: int = -1) -> NonlinearOpResult:
        mx = x.max(dim=dim, keepdim=True).values         # trusted reduction stat
        shifted = x - mx                                 # -> accelerator
        e = torch.exp(shifted)
        out = e / e.sum(dim=dim, keepdim=True)           # accelerator
        return NonlinearOpResult(
            output=out, trusted_calls=1, trusted_bytes=tensor_bytes(mx),
            gpu_bytes=tensor_bytes(shifted) + tensor_bytes(out),
            tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator",
                   "trusted_shortcut": "row_max"})

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None,
                  eps: float = 1e-5) -> NonlinearOpResult:
        mean = x.mean(-1, keepdim=True)                  # trusted reduction stats
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + eps)        # normalize on accelerator
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return NonlinearOpResult(
            output=out, trusted_calls=1,
            trusted_bytes=tensor_bytes(mean) + tensor_bytes(var),
            gpu_bytes=tensor_bytes(x) + tensor_bytes(out), tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator",
                   "trusted_shortcut": "mean+var"})

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-6) -> NonlinearOpResult:
        ms = x.pow(2).mean(-1, keepdim=True)             # trusted reduction stat
        out = x * torch.rsqrt(ms + eps)                  # normalize on accelerator
        if weight is not None:
            out = out * weight
        return NonlinearOpResult(
            output=out, trusted_calls=1, trusted_bytes=tensor_bytes(ms),
            gpu_bytes=tensor_bytes(x) + tensor_bytes(out), tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator",
                   "trusted_shortcut": "mean_square"})
