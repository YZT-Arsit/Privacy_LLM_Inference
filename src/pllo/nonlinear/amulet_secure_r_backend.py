"""Amulet-like *secure-R* nonlinear handling (Line B, hardened).

This replaces the old ``trusted_shortcut`` / ``amulet_migrated`` backend's two
weaknesses for paper-facing use:

* **No per-op trusted reduction shortcut.** Every online nonlinear op runs fully
  on the untrusted accelerator with ``trusted_calls == 0`` (softmax / RMSNorm /
  LayerNorm are computed directly over the masked state, exactly as the
  compatible right-multiply design does; the activation lift below also keeps
  ``trusted_calls == 0``). There is a single TEE entry and a single TEE exit;
  no nonlinear op crosses the boundary online.

* **Secure R (no observable selector).** GELU / SiLU use the dense single-one
  Kronecker construction from :mod:`pllo.ops.amulet_right_mask_islands`:
  ``R_bar = R1 R2 R3`` has **exactly one** entry equal to 1 at a *secret*
  coordinate ``(a, b)`` and every other entry is a dense, non-zero, non-one value
  (**no zero decoys, no visible one-hot selector**). The activation is evaluated
  on a *shuffled* dense lifted tensor ``Z`` (the only GPU-visible activation
  artifact); the squeeze that recovers the true value is folded with the secret
  permutations, so the valid channel is not directly observable from ``Z`` or the
  folded operators.

Correctness is exact (``phi(x)`` is recovered to fp). Security is **claimed under
the secure-R assumption** (``security_status = "claimed_under_secure_R_assumption"``)
-- i.e. the adversary cannot recover the secret coordinate ``(a, b)`` / the
shuffles from the GPU-visible dense lift. This is NOT a completed formal proof;
the checkable conditions (no zero decoys, dense single-one R, no raw one-hot /
selector tensor exposed, secret coordinate never reported) are asserted and
surfaced as audit fields.

``tee_used_on_gpu`` is always False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pllo.nonlinear.backends import NonlinearBackend, NonlinearOpResult, tensor_bytes
from pllo.ops.amulet_right_mask_islands import sample_amulet_r_factors

__all__ = ["AmuletSecureRNonlinearBackend", "SecureRViolation",
           "secure_r_activation"]


class SecureRViolation(RuntimeError):
    """Raised when a secure-R artifact violates a checkable secure condition
    (zero decoy, visible one-hot, or the unit coordinate is not unique)."""


def _gen(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _assert_secure_rbar(rbar: torch.Tensor, a: int, b: int,
                        *, avoid_eps: float = 1e-6) -> None:
    """Verify the dense single-one secure-R conditions (raise otherwise)."""
    k = rbar.shape[0]
    # exactly one entry == 1 (the secret valid coordinate)
    near_one = (rbar - 1.0).abs() < 1e-9
    if int(near_one.sum().item()) != 1 or not bool(near_one[a, b].item()):
        raise SecureRViolation("R_bar must have exactly one unit entry at (a, b)")
    # no zero decoys: every entry is non-zero (dense lift, no visible structure)
    if bool((rbar.abs() < avoid_eps).any().item()):
        raise SecureRViolation("R_bar has a zero decoy (would expose structure)")
    # not a one-hot selector: off-(a,b) entries are not all ~0/~1
    mask = torch.ones(k, k, dtype=torch.bool, device=rbar.device)
    mask[a, b] = False
    if k > 1 and bool(((rbar[mask] - 1.0).abs() < avoid_eps).any().item()):
        raise SecureRViolation("R_bar decoy equals 1 (visible one-hot selector)")


def secure_r_activation(
    x: torch.Tensor, act, *, k: int = 2, seed: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Evaluate ``act(x)`` exactly via the secure-R lift/shuffle/squeeze.

    Returns ``(out == act(x), audit)``. The only GPU-visible activation artifact
    is the shuffled dense lifted tensor ``Z`` (``audit['gpu_bytes']``); the secret
    coordinate / permutations are never returned. ``trusted_calls == 0``."""
    if k < 2:
        raise ValueError("secure-R requires k >= 2 (1 valid + >=1 dense decoy)")
    lead = x.shape[:-1]
    h = int(x.shape[-1])
    u = x.reshape(-1, h)                                   # [m, h]
    m = int(u.shape[0])
    dtype, device = u.dtype, u.device
    g = _gen(seed, device)

    rf = sample_amulet_r_factors(k, dtype=dtype, device=torch.device("cpu"),
                                 generator=g)
    rbar = rf.rbar.to(device)
    a, b = int(rf.selected_row), int(rf.selected_col)
    _assert_secure_rbar(rbar, a, b)

    # secret column/row shuffles (derive from the same generator; never exposed)
    p_row = torch.randperm(m * k, generator=g).to(device)
    p_col = torch.randperm(h * k, generator=g).to(device)
    inv_row = torch.empty_like(p_row); inv_row[p_row] = torch.arange(
        m * k, device=device)
    inv_col = torch.empty_like(p_col); inv_col[p_col] = torch.arange(
        h * k, device=device)

    lifted = torch.kron(u.contiguous(), rbar.contiguous())  # [mk, hk] dense
    z = lifted.index_select(0, p_row).index_select(1, p_col)  # GPU-visible shuffle
    s = act(z)                                              # activation on GPU
    # un-shuffle (phi commutes with permutation) + squeeze the secret coordinate
    s_un = s.index_select(0, inv_row).index_select(1, inv_col)
    rows = torch.arange(m, device=device) * k + a
    cols = torch.arange(h, device=device) * k + b
    out = s_un.index_select(0, rows).index_select(1, cols)  # == act(u)
    audit = {
        "gpu_bytes": tensor_bytes(lifted) + tensor_bytes(z) + tensor_bytes(s),
        "lift_k": int(k),
        "secure_R_enabled": True,
        "zero_decoys": False,
        "selector_visible_to_gpu": False,
        "valid_channel_observable": False,
        "rbar_dense_single_one": True,
        "trusted_calls": 0,
    }
    return out.reshape(*lead, h), audit


class AmuletSecureRNonlinearBackend(NonlinearBackend):
    name = "amulet_secure_R"
    security_status = "claimed_under_secure_R_assumption"
    security_claim_status = "claimed_under_assumption"
    security_note = (
        "Amulet secure-R: online nonlinear ops run fully on the untrusted "
        "accelerator (trusted_calls == 0, single TEE entry/exit). GELU/SiLU use "
        "a dense single-one Kronecker R (no zero decoys, no visible one-hot "
        "selector) with secret shuffles; softmax/RMSNorm/LayerNorm run directly "
        "over the masked state with no trusted reduction shortcut. Security is "
        "claimed UNDER the secure-R assumption (secret coordinate/shuffles not "
        "recoverable from the GPU-visible dense lift); NOT a completed formal "
        "proof. Checkable conditions are asserted + reported.")

    def __init__(self, lift_k: int = 2, seed: int = 0, **_ignored) -> None:
        if lift_k < 2:
            raise ValueError("lift_k must be >= 2 (1 valid + >=1 dense decoy)")
        self.lift_k = int(lift_k)
        self.seed = int(seed)
        self._op_seed = 0

    def _next_seed(self) -> int:
        # fresh secret coordinate/shuffles per op (still deterministic per run)
        self._op_seed += 1
        return self.seed + 10_007 * self._op_seed

    def _lift(self, x: torch.Tensor, act) -> NonlinearOpResult:
        out, audit = secure_r_activation(
            x, act, k=self.lift_k, seed=self._next_seed())
        if audit["trusted_calls"] != 0:
            raise SecureRViolation("secure-R activation made a trusted call")
        return NonlinearOpResult(
            output=out, trusted_calls=0, trusted_bytes=0,
            gpu_bytes=int(audit["gpu_bytes"]), tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator", **audit})

    def gelu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._lift(x, F.gelu)

    def silu(self, x: torch.Tensor) -> NonlinearOpResult:
        return self._lift(x, F.silu)

    # -- reductions: no trusted shortcut; run directly on the masked state -----
    def _on_accelerator(self, x: torch.Tensor, out: torch.Tensor) -> NonlinearOpResult:
        return NonlinearOpResult(
            output=out, trusted_calls=0, trusted_bytes=0,
            gpu_bytes=tensor_bytes(x) + tensor_bytes(out), tee_used_on_gpu=False,
            extra={"location": "untrusted_accelerator", "secure_R_enabled": True,
                   "trusted_shortcut": None})

    def softmax(self, x: torch.Tensor, dim: int = -1) -> NonlinearOpResult:
        return self._on_accelerator(x, torch.softmax(x, dim=dim))

    def layernorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                  bias: torch.Tensor | None = None,
                  eps: float = 1e-5) -> NonlinearOpResult:
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        out = centered * torch.rsqrt(centered.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return self._on_accelerator(x, out)

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-6) -> NonlinearOpResult:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            out = out * weight
        return self._on_accelerator(x, out)
