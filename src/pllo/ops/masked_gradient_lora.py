"""Stage 7.6 — Masked-gradient LoRA training primitives.

Cleanroom module focused on the rank-space orthogonal mixer ``M``.
The construction is

    A_tilde = N_x^T A M
    B_tilde = M^T B N_y
    X_tilde = X N_x
    Y_tilde = X_tilde A_tilde B_tilde = X A B N_y

with orthogonal ``N_x in R^{d_in x d_in}``, ``N_y in R^{d_out x d_out}``
and ``M in R^{r_pad x r_pad}`` (``M^T M = I``).

Masked SGD / masked momentum SGD on ``(A_tilde, B_tilde)`` are
algebraically equivalent to plaintext SGD / momentum SGD on
``(A, B)`` because right-multiplication by a constant orthogonal mask
commutes with the linear update rule. AdamW does NOT enjoy this
property: coordinate-wise second moments are not invariant under a
dense orthogonal mixer (``(g Q)_ij^2 != g_ij^2`` when ``Q`` is not a
signed permutation). This module therefore exposes SGD + momentum
masked-update functions only, and the experiment companion records
the AdamW limitation explicitly.

Rank padding integrates a cancellation block via
``A_pad = [A_real, R, -R]``, ``B_pad = vstack(B_real, S, S)`` so that
``A_pad B_pad = A_real B_real`` and the dummy contribution is
identically zero in the plain forward. The mixer ``M`` is applied
*after* the padding, so the visible adapter rank is ``r_pad`` while
the true rank ``r_real`` remains hidden from any shape inspection.

CPU local emulation only. No formal cryptographic / semantic /
differential-privacy security is claimed. JSON / CSV / Markdown
emitters in the companion experiment file carry only summary
scalars, shapes, and short fingerprints — raw tensors, masks, and
gradients are never exported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedGradientLoRAConfig:
    d_in: int = 16
    d_out: int = 8
    true_rank: int = 2
    padded_rank: int = 4
    batch_size: int = 8
    lr: float = 1e-2
    momentum: float = 0.9
    use_momentum: bool = False
    use_rank_padding: bool = True
    dummy_strategy: str = "paired_cancellation"  # paired_cancellation / none
    mixer_kind: str = "orthogonal"  # orthogonal / signed_permutation
    seed: int = 0
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------


def create_orthogonal_matrix(
    dim: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample a uniformly random orthogonal matrix via QR."""
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def create_signed_permutation(
    dim: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample a signed-permutation matrix ``P S`` with ``+/-1`` diagonal.

    Signed permutations are the only orthogonal masks that commute with
    coordinate-wise second-moment statistics (used by Adam / AdamW);
    they are provided here only for the AdamW-compatibility ablation.
    """
    perm = torch.randperm(dim, generator=generator)
    P = torch.zeros(dim, dim, dtype=dtype, device=device)
    P[torch.arange(dim), perm] = 1.0
    signs = (torch.randint(0, 2, (dim,), generator=generator) * 2 - 1).to(dtype)
    return P * signs.unsqueeze(0)


# ---------------------------------------------------------------------------
# Cancellation rank padding
# ---------------------------------------------------------------------------


def create_cancellation_padded_lora(
    A_real: torch.Tensor, B_real: torch.Tensor,
    *,
    padded_rank: int,
    strategy: str = "paired_cancellation",
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Pad ``(A_real, B_real)`` to ``(A_pad, B_pad)`` with
    ``A_pad B_pad = A_real B_real`` and ``padded_rank >= 2 * true_rank
    + true_rank`` (so the pair-cancellation block fits).

    Strategy ``"paired_cancellation"``: split the dummy budget
    ``r_dummy = padded_rank - true_rank`` into two halves ``r_half =
    r_dummy // 2`` (must be even on the column / row side).
    ``A_pad = [A_real, R, -R]`` and ``B_pad = vstack(B_real, S, S)`` so
    that ``A_pad B_pad = A_real B_real + R S - R S = A_real B_real``.
    """
    if A_real.ndim != 2 or B_real.ndim != 2:
        raise ValueError("A_real and B_real must be 2-D")
    d_in, r_real = A_real.shape
    r_real_b, d_out = B_real.shape
    if r_real != r_real_b:
        raise ValueError("rank mismatch between A_real and B_real")
    r_pad = int(padded_rank)
    r_dummy = r_pad - r_real
    if r_dummy < 0:
        raise ValueError("padded_rank must be >= true_rank")
    metadata: dict = {
        "strategy": strategy,
        "true_rank": int(r_real),
        "padded_rank": int(r_pad),
        "dummy_columns_in_A": int(r_dummy),
        "dummy_rows_in_B": int(r_dummy),
    }
    if r_dummy == 0 or strategy == "none":
        return (
            A_real.clone(), B_real.clone(),
            {**metadata, "strategy": "none"},
        )
    if strategy != "paired_cancellation":
        raise ValueError(
            f"unsupported dummy strategy: {strategy!r}; "
            "supported: paired_cancellation, none",
        )
    if r_dummy % 2 != 0:
        raise ValueError(
            "padded_rank - true_rank must be even for paired_cancellation",
        )
    r_half = r_dummy // 2
    dtype = A_real.dtype
    device = A_real.device
    if generator is None:
        R = torch.randn(d_in, r_half, dtype=dtype, device=device)
        S = torch.randn(r_half, d_out, dtype=dtype, device=device)
    else:
        R = torch.randn(
            d_in, r_half, dtype=dtype, device=device, generator=generator,
        )
        S = torch.randn(
            r_half, d_out, dtype=dtype, device=device, generator=generator,
        )
    A_pad = torch.cat([A_real, R, -R], dim=1)
    B_pad = torch.cat([B_real, S, S], dim=0)
    metadata["dummy_pair_norm_R"] = float(R.norm().item())
    metadata["dummy_pair_norm_S"] = float(S.norm().item())
    return A_pad, B_pad, metadata


def dummy_contribution_norm(
    A_pad: torch.Tensor, B_pad: torch.Tensor,
    *,
    true_rank: int,
    A_real_view: Optional[torch.Tensor] = None,
    B_real_view: Optional[torch.Tensor] = None,
) -> float:
    """``|| A_pad B_pad - A_real B_real ||_F`` -- must be 0 for the
    paired cancellation pad, modulo float round-off."""
    if A_real_view is None:
        A_real_view = A_pad[:, :true_rank]
    if B_real_view is None:
        B_real_view = B_pad[:true_rank, :]
    return float((A_pad @ B_pad - A_real_view @ B_real_view).norm().item())


# ---------------------------------------------------------------------------
# Masked state construction
# ---------------------------------------------------------------------------


@dataclass
class MaskedLoRAState:
    """Container for accelerator-visible masked LoRA tensors.

    Field naming mirrors the construction:
        A_tilde = N_x^T A M
        B_tilde = M^T B N_y
    """

    A_tilde: torch.Tensor
    B_tilde: torch.Tensor
    N_x: torch.Tensor
    N_x_inv: torch.Tensor  # = N_x^T for orthogonal N_x
    N_y: torch.Tensor
    N_y_inv: torch.Tensor
    M: torch.Tensor
    M_inv: torch.Tensor    # = M^T for orthogonal M
    padded_rank: int
    true_rank: int


def create_masked_lora_state(
    A: torch.Tensor, B: torch.Tensor,
    *,
    N_x: torch.Tensor, N_y: torch.Tensor, M: torch.Tensor,
    padded_rank: int, true_rank: int,
) -> MaskedLoRAState:
    A_tilde = N_x.transpose(-2, -1) @ A @ M
    B_tilde = M.transpose(-2, -1) @ B @ N_y
    return MaskedLoRAState(
        A_tilde=A_tilde, B_tilde=B_tilde,
        N_x=N_x, N_x_inv=N_x.transpose(-2, -1),
        N_y=N_y, N_y_inv=N_y.transpose(-2, -1),
        M=M, M_inv=M.transpose(-2, -1),
        padded_rank=padded_rank, true_rank=true_rank,
    )


def masked_lora_forward(
    X_tilde: torch.Tensor, A_tilde: torch.Tensor, B_tilde: torch.Tensor,
) -> torch.Tensor:
    """``Y_tilde = X_tilde A_tilde B_tilde``."""
    return X_tilde @ A_tilde @ B_tilde


# ---------------------------------------------------------------------------
# Recovery helpers (test-only — NEVER call inside the training loop)
# ---------------------------------------------------------------------------


def recover_lora_from_masked(
    A_tilde: torch.Tensor, B_tilde: torch.Tensor,
    *,
    N_x: torch.Tensor, N_y: torch.Tensor, M: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover plaintext ``(A, B)`` from masked ``(A_tilde, B_tilde)``.

    Uses the orthogonality identities ``N^{-1} = N^T``, ``M^{-1} = M^T``:
        A = N_x A_tilde M^T
        B = M B_tilde N_y^T
    Test-only utility; the protocol does NOT call this on the
    accelerator path.
    """
    A = N_x @ A_tilde @ M.transpose(-2, -1)
    B = M @ B_tilde @ N_y.transpose(-2, -1)
    return A, B


# ---------------------------------------------------------------------------
# Masked optimizer updates
# ---------------------------------------------------------------------------


def masked_sgd_step(
    A_tilde: torch.Tensor, B_tilde: torch.Tensor,
    grad_A_tilde: torch.Tensor, grad_B_tilde: torch.Tensor,
    *, lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla SGD in masked space.

    For orthogonal ``N_x, N_y, M`` we have
        grad_A_tilde = N_x^T grad_A M
        grad_B_tilde = M^T grad_B N_y
    so
        A_tilde - lr * grad_A_tilde
            = N_x^T (A - lr * grad_A) M
    i.e. the masked update mirrors the plaintext update term-by-term.
    """
    return (
        A_tilde - lr * grad_A_tilde,
        B_tilde - lr * grad_B_tilde,
    )


def masked_momentum_sgd_step(
    A_tilde: torch.Tensor, B_tilde: torch.Tensor,
    grad_A_tilde: torch.Tensor, grad_B_tilde: torch.Tensor,
    V_A_tilde: torch.Tensor, V_B_tilde: torch.Tensor,
    *, lr: float, momentum: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Heavy-ball momentum SGD in masked space.

    Update rule:
        V <- momentum * V + grad
        param <- param - lr * V
    Right-multiplication by an orthogonal mask distributes over the
    linear combination, so the masked update equals the plaintext
    update transformed by ``N_x^T ... M`` / ``M^T ... N_y``.
    """
    V_A_next = momentum * V_A_tilde + grad_A_tilde
    V_B_next = momentum * V_B_tilde + grad_B_tilde
    A_next = A_tilde - lr * V_A_next
    B_next = B_tilde - lr * V_B_next
    return A_next, B_next, V_A_next, V_B_next


# ---------------------------------------------------------------------------
# AdamW limitation — explicit raise
# ---------------------------------------------------------------------------


class DenseMaskedAdamWUnsupported(Exception):
    """Raised when AdamW is requested under a dense orthogonal mixer.

    AdamW's per-coordinate second moment ``v <- beta_2 v + (1 - beta_2)
    g^2`` is not invariant under right-multiplication by a dense
    orthogonal ``Q`` (``(g Q)^2 != g^2 Q``); therefore the masked
    update does not match the plaintext update term-by-term.

    Stage 7.6 deliberately raises rather than silently approximating.
    """


def masked_adamw_step_unsupported(*_args, **_kwargs):  # pragma: no cover
    raise DenseMaskedAdamWUnsupported(
        "Dense masked AdamW exactness is NOT supported. Coordinate-"
        "wise second moments are not invariant under a dense "
        "orthogonal mixer. Options: (i) trusted-assisted update "
        "(recover, AdamW on plain, re-mask), (ii) signed-permutation "
        "masks (the only orthogonal class that commutes with "
        "coordinate-wise squaring), (iii) a specialised masked "
        "optimiser. None of these is in scope for Stage 7.6."
    )


# ---------------------------------------------------------------------------
# Fingerprinting (only ever returns short hex digests + shapes)
# ---------------------------------------------------------------------------


def _tensor_fingerprint(t: torch.Tensor) -> str:
    buf = t.detach().to(torch.float64).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def masked_lora_state_fingerprint(state: MaskedLoRAState) -> dict:
    """Publish only shapes + fingerprints. NEVER export raw tensors.

    Used by the JSON / Markdown emitters in the companion experiment
    to demonstrate that the masked-state values are stable across calls
    (or fresh under a fresh-mask policy) without leaking the values
    themselves.
    """
    return {
        "A_tilde_shape": list(state.A_tilde.shape),
        "B_tilde_shape": list(state.B_tilde.shape),
        "padded_rank": int(state.padded_rank),
        "true_rank": int(state.true_rank),
        "A_tilde_fingerprint": _tensor_fingerprint(state.A_tilde),
        "B_tilde_fingerprint": _tensor_fingerprint(state.B_tilde),
        # Do NOT publish N_x / N_y / M fingerprints — these are trusted
        # secrets. Publishing their fingerprint enables membership
        # linkability across sessions.
    }


def visible_grad_fingerprint(
    grad_A_tilde: torch.Tensor, grad_B_tilde: torch.Tensor,
) -> dict:
    """Short fingerprints of GPU-visible gradients only."""
    return {
        "grad_A_tilde_shape": list(grad_A_tilde.shape),
        "grad_B_tilde_shape": list(grad_B_tilde.shape),
        "grad_A_tilde_fingerprint": _tensor_fingerprint(grad_A_tilde),
        "grad_B_tilde_fingerprint": _tensor_fingerprint(grad_B_tilde),
    }


__all__ = [
    "MaskedGradientLoRAConfig",
    "MaskedLoRAState",
    "DenseMaskedAdamWUnsupported",
    "create_orthogonal_matrix",
    "create_signed_permutation",
    "create_cancellation_padded_lora",
    "dummy_contribution_norm",
    "create_masked_lora_state",
    "masked_lora_forward",
    "recover_lora_from_masked",
    "masked_sgd_step",
    "masked_momentum_sgd_step",
    "masked_adamw_step_unsupported",
    "masked_lora_state_fingerprint",
    "visible_grad_fingerprint",
]
