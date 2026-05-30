"""Stage 5.1 — Trusted LayerNorm / RMSNorm primitive.

Wraps the existing trusted-LayerNorm shortcut behind one named primitive,
adds RMSNorm coverage, and emits structured correctness output that the
norm probe / experiment harness can consume. The primitive deliberately
still runs the actual normalisation inside the simulated trusted side —
right-multiply masks do not commute with the mean / variance over the
hidden dimension, so a GPU-side norm would require a different protocol
(restricted orthogonal masks, see ``norm_probe.py``).

Surface:

    @dataclass
    class TrustedNormConfig: ...

    def trusted_norm_forward(...) -> dict:
        # x_tilde = X N_in  (no pad) or  (X - T_in) N_in  (with pad)
        # → recover X
        # → LayerNorm(X) or RMSNorm(X)
        # → y_tilde = Y N_out  (no pad) or  (Y - T_out) N_out  (with pad)

The returned dict carries ``y_plain``, ``y_tilde``, ``y_recovered``, and
the standard headline metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from pllo.utils.tensor_compare import compare_tensors


NormType = Literal["layernorm", "rmsnorm"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrustedNormConfig:
    norm_type: NormType
    hidden_size: int
    eps: float = 1e-5
    use_pad: bool = True
    dtype: str = "float32"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Plain-side primitives (reference implementations)
# ---------------------------------------------------------------------------


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """LayerNorm reference matching ``torch.nn.LayerNorm`` semantics."""
    return F.layer_norm(
        x,
        (x.shape[-1],),
        weight=weight,
        bias=bias,
        eps=eps,
    )


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """RMSNorm: ``y = gamma * x / sqrt(mean(x^2) + eps)``.

    Matches the LLaMA / T5 family ``RMSNorm`` semantics. ``weight=None``
    behaves as ``gamma=1`` (pure normalisation, no affine).
    """
    norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        norm = norm * weight
    return norm


# ---------------------------------------------------------------------------
# Trusted-side wrapper
# ---------------------------------------------------------------------------


def _recover(
    x_tilde: torch.Tensor,
    n_in_inv: torch.Tensor,
    pad_in: torch.Tensor | None,
) -> torch.Tensor:
    x = x_tilde @ n_in_inv
    if pad_in is not None:
        x = x + pad_in
    return x


def _remask(
    y: torch.Tensor,
    n_out: torch.Tensor,
    pad_out: torch.Tensor | None,
) -> torch.Tensor:
    if pad_out is not None:
        return (y - pad_out) @ n_out
    return y @ n_out


def trusted_norm_forward(
    x_tilde: torch.Tensor,
    n_in_inv: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    n_out: torch.Tensor,
    norm_type: str,
    eps: float,
    pad_in: torch.Tensor | None = None,
    pad_out: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run the trusted-norm primitive over a masked input and return metrics.

    ``norm_type`` is ``"layernorm"`` or ``"rmsnorm"``. ``norm_bias`` is
    accepted for LayerNorm; for RMSNorm it must be ``None`` (RMSNorm has
    no bias term). Both ``norm_weight`` and ``norm_bias`` may be ``None``
    to fall back to gamma=1 / beta=0.

    Math:

        X       = x_tilde N_in_inv          (no pad)
        X       = x_tilde N_in_inv + T_in   (with pad)
        Y       = LayerNorm(X) or RMSNorm(X)
        y_tilde = Y N_out                   (no pad)
        y_tilde = (Y - T_out) N_out         (with pad)

    The returned dict includes ``y_plain`` (the trusted-side normalisation
    output, which is the *recovered* domain answer), ``y_tilde`` (what the
    GPU side now consumes), and ``y_recovered`` (``y_tilde N_out_inv +
    T_out`` reapplied for verification — must equal ``y_plain`` up to
    ``atol``).
    """
    norm_type_l = norm_type.lower()
    if norm_type_l not in {"layernorm", "rmsnorm"}:
        raise ValueError(
            f"norm_type must be 'layernorm' or 'rmsnorm', got {norm_type!r}"
        )
    if norm_type_l == "rmsnorm" and norm_bias is not None:
        raise ValueError(
            "RMSNorm does not take a bias term; pass norm_bias=None."
        )

    # 1. Recover plaintext input.
    x_plain = _recover(x_tilde, n_in_inv, pad_in)

    # 2. Apply norm in the trusted side.
    if norm_type_l == "layernorm":
        y_plain = layer_norm_reference(x_plain, norm_weight, norm_bias, eps)
    else:
        y_plain = rms_norm_reference(x_plain, norm_weight, eps)

    # 3. Re-mask output (and add output pad if requested).
    y_tilde = _remask(y_plain, n_out, pad_out)

    # 4. Verifier: recover y_tilde back and compare to y_plain.
    n_out_inv = torch.linalg.inv(n_out.to(torch.float64)).to(n_out.dtype)
    y_recovered = y_tilde @ n_out_inv
    if pad_out is not None:
        y_recovered = y_recovered + pad_out

    metrics = compare_tensors(y_plain, y_recovered, atol=1e-4, rtol=1e-4)

    return {
        "y_plain": y_plain,
        "y_tilde": y_tilde,
        "y_recovered": y_recovered,
        "max_abs_error": metrics["max_abs_error"],
        "mean_abs_error": metrics["mean_abs_error"],
        "relative_l2_error": metrics["relative_l2_error"],
        "cosine_similarity": metrics["cosine_similarity"],
        "allclose": metrics["allclose"],
    }


__all__ = [
    "TrustedNormConfig",
    "layer_norm_reference",
    "rms_norm_reference",
    "trusted_norm_forward",
]
