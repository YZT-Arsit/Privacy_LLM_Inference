"""Stage 7.1 — Masked LoRA backward primitive.

Forward (Stage 7.0 — unchanged):
    Y = X W + (alpha / r) X A B + bias
    Y_tilde = Y N_out, X_tilde = (X - T_in) N_in (use_pad) or X N_in,
    W_tilde = N_in^{-1} W N_out, A_tilde = N_in^{-1} A U,
    B_tilde = U^{-1} B N_out, bias_tilde = bias N_out.

Stage 7.1 — backward analytic chain rule, numerically verified.

Let s = alpha / r and let L be a scalar loss with plain-space upstream
gradient G = ∂L/∂Y. Define the masked upstream gradient

    G_tilde = G N_out^{-T}

so that for orthogonal N_out, G_tilde = G N_out. Then the chain rule is
preserved on the boundary:

    tr(G^T dY) = tr(G_tilde^T dY_tilde)         ⟺      G_tilde = G N_out^{-T}

The GPU-side masked backward operates ONLY on masked tensors:

    grad_A_tilde   = s X_tilde^T (G_tilde B_tilde^T)
    grad_B_tilde   = s (X_tilde A_tilde)^T G_tilde
    grad_X_tilde   = G_tilde W_tilde^T + s G_tilde B_tilde^T A_tilde^T     (optional)

Trusted-side recovery (no_pad):

    grad_A     = N_in^{-T} grad_A_tilde U^T
    grad_B     = U^{-T}    grad_B_tilde N_out^T
    grad_X     = grad_X_tilde N_in^T                                       (optional)

Pad compensation (use_pad=True). With X_tilde = (X - T_in) N_in the GPU
backward only sees ``(X - T_in)``, so the naive recovered gradients are
``s (X - T_in)^T G B^T`` and ``s A^T (X - T_in)^T G``. Adding the trusted
plaintext compensations

    grad_A_pad_compensation = s T_in^T G B^T
    grad_B_pad_compensation = s A^T T_in^T G

returns the plain-space gradients exactly. ``grad_X`` does NOT need a pad
compensation — the chain rule ``X_tilde = (X - T_in) N_in`` has
``∂X_tilde/∂X = N_in`` regardless of T_in.

Everything published in JSON/CSV/Markdown reports MUST be summary
metrics + fingerprints. The raw gradients, adapters, private inputs,
masks, and optimizer state stay trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.utils.validation import (
    require_rank2,
    require_same_dtype_device,
    require_shape,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_DTYPE_TABLE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_TABLE[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dtype {name!r}; expected one of {tuple(_DTYPE_TABLE)}"
        ) from exc


@dataclass
class MaskedLoRABackwardConfig:
    """Runtime knobs for one masked LoRA backward call.

    ``recover_grad_x=True`` also recovers ``grad_X`` (plain space). It costs
    an extra ``N_in^T`` multiply and exposes ``grad_X_tilde`` to the GPU; it
    is opt-in because most LoRA prototypes only need ``grad_A`` / ``grad_B``.
    """

    use_pad: bool = True
    fresh_u_per_step: bool = True
    recover_grad_x: bool = False
    dtype: str = "float64"
    device: str = "cpu"

    def torch_dtype(self) -> torch.dtype:
        return _torch_dtype(self.dtype)

    def torch_device(self) -> torch.device:
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Plain reference (analytic, no autograd)
# ---------------------------------------------------------------------------


def plain_lora_backward_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Analytic plain-space LoRA backward.

    Returns ``{"grad_x": ..., "grad_a": ..., "grad_b": ...}``. ``w`` is
    only used for ``grad_x = G W^T + s G B^T A^T``.
    """
    require_rank2("x", x)
    require_rank2("w", w)
    require_rank2("a", a)
    require_rank2("b", b)
    require_rank2("upstream_grad", upstream_grad)
    if upstream_grad.shape != (x.shape[0], w.shape[1]):
        raise ValueError(
            f"upstream_grad must have shape (batch, d_out)="
            f"({x.shape[0]}, {w.shape[1]}), got {tuple(upstream_grad.shape)}"
        )
    require_same_dtype_device("x", x, w=w, a=a, b=b, upstream_grad=upstream_grad)

    rank = a.shape[1]
    scale = float(alpha) / max(rank, 1)
    grad_b = scale * (x @ a).transpose(0, 1) @ upstream_grad
    grad_a = scale * x.transpose(0, 1) @ upstream_grad @ b.transpose(0, 1)
    grad_x = upstream_grad @ w.transpose(0, 1) + scale * (
        upstream_grad @ b.transpose(0, 1)
    ) @ a.transpose(0, 1)
    return {"grad_x": grad_x, "grad_a": grad_a, "grad_b": grad_b}


# ---------------------------------------------------------------------------
# Upstream gradient transform: G ⇄ G_tilde
# ---------------------------------------------------------------------------


def transform_upstream_gradient(
    grad_y: torch.Tensor,
    n_out: torch.Tensor,
    *,
    convention: str = "row_vector",
) -> torch.Tensor:
    """Map plain-space upstream gradient G to masked-space G_tilde.

    With ``Y_tilde = Y N_out`` (row-vector convention),
    ``G_tilde = G N_out^{-T}``. For orthogonal ``N_out`` this is
    ``G N_out``.
    """
    if convention != "row_vector":
        raise ValueError(
            f"only row_vector convention is supported, got {convention!r}"
        )
    require_rank2("grad_y", grad_y)
    require_rank2("n_out", n_out)
    if n_out.shape != (grad_y.shape[1], grad_y.shape[1]):
        raise ValueError(
            f"n_out must be ({grad_y.shape[1]}, {grad_y.shape[1]}), got"
            f" {tuple(n_out.shape)}"
        )
    require_same_dtype_device("grad_y", grad_y, n_out=n_out)
    # G_tilde = G @ N_out^{-T} = G @ (N_out^T)^{-1}
    n_out_t_inv = torch.linalg.inv(n_out.transpose(0, 1))
    return grad_y @ n_out_t_inv


def invert_upstream_gradient_mask(
    grad_y_tilde: torch.Tensor,
    n_out: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`transform_upstream_gradient`: G = G_tilde N_out^T."""
    require_rank2("grad_y_tilde", grad_y_tilde)
    require_rank2("n_out", n_out)
    if n_out.shape != (grad_y_tilde.shape[1], grad_y_tilde.shape[1]):
        raise ValueError(
            f"n_out must be ({grad_y_tilde.shape[1]}, {grad_y_tilde.shape[1]}),"
            f" got {tuple(n_out.shape)}"
        )
    require_same_dtype_device("grad_y_tilde", grad_y_tilde, n_out=n_out)
    return grad_y_tilde @ n_out.transpose(0, 1)


# ---------------------------------------------------------------------------
# GPU-side masked backward
# ---------------------------------------------------------------------------


def masked_lora_backward(
    x_tilde: torch.Tensor,
    a_tilde: torch.Tensor,
    b_tilde: torch.Tensor,
    grad_y_tilde: torch.Tensor,
    *,
    alpha: float = 1.0,
    w_tilde: torch.Tensor | None = None,
    recover_grad_x: bool = False,
) -> dict[str, torch.Tensor | None]:
    """GPU-side masked backward of one LoRA-augmented linear.

    The GPU only sees masked / transformed tensors; no plaintext A / B /
    X / G ever enters here. Returns a dict with keys
    ``"grad_a_tilde"`` / ``"grad_b_tilde"`` and (when
    ``recover_grad_x=True``) ``"grad_x_tilde"``.
    """
    require_rank2("x_tilde", x_tilde)
    require_rank2("a_tilde", a_tilde)
    require_rank2("b_tilde", b_tilde)
    require_rank2("grad_y_tilde", grad_y_tilde)
    if a_tilde.shape[0] != x_tilde.shape[1]:
        raise ValueError(
            f"a_tilde d_in must be {x_tilde.shape[1]}, got {tuple(a_tilde.shape)}"
        )
    if b_tilde.shape[0] != a_tilde.shape[1]:
        raise ValueError(
            f"b_tilde rank must match a_tilde rank={a_tilde.shape[1]},"
            f" got {tuple(b_tilde.shape)}"
        )
    if grad_y_tilde.shape != (x_tilde.shape[0], b_tilde.shape[1]):
        raise ValueError(
            f"grad_y_tilde must be {(x_tilde.shape[0], b_tilde.shape[1])},"
            f" got {tuple(grad_y_tilde.shape)}"
        )
    require_same_dtype_device(
        "x_tilde", x_tilde,
        a_tilde=a_tilde, b_tilde=b_tilde, grad_y_tilde=grad_y_tilde,
        w_tilde=w_tilde,
    )
    if recover_grad_x and w_tilde is None:
        raise ValueError(
            "recover_grad_x=True requires w_tilde for grad_X_tilde = "
            "G_tilde W_tilde^T + s G_tilde B_tilde^T A_tilde^T"
        )

    rank = a_tilde.shape[1]
    scale = float(alpha) / max(rank, 1)

    # grad_A_tilde = s X_tilde^T (G_tilde B_tilde^T)
    grad_a_tilde = scale * x_tilde.transpose(0, 1) @ (
        grad_y_tilde @ b_tilde.transpose(0, 1)
    )
    # grad_B_tilde = s (X_tilde A_tilde)^T G_tilde
    grad_b_tilde = scale * (x_tilde @ a_tilde).transpose(0, 1) @ grad_y_tilde

    grad_x_tilde: torch.Tensor | None = None
    if recover_grad_x:
        # grad_X_tilde = G_tilde W_tilde^T + s G_tilde B_tilde^T A_tilde^T
        assert w_tilde is not None  # for type checker
        grad_x_tilde = grad_y_tilde @ w_tilde.transpose(0, 1) + scale * (
            grad_y_tilde @ b_tilde.transpose(0, 1)
        ) @ a_tilde.transpose(0, 1)
    return {
        "grad_a_tilde": grad_a_tilde,
        "grad_b_tilde": grad_b_tilde,
        "grad_x_tilde": grad_x_tilde,
    }


# ---------------------------------------------------------------------------
# Trusted-side recovery
# ---------------------------------------------------------------------------


def recover_lora_gradients(
    grad_a_tilde: torch.Tensor,
    grad_b_tilde: torch.Tensor,
    n_in: torch.Tensor,
    n_out: torch.Tensor,
    u: torch.Tensor,
    *,
    grad_x_tilde: torch.Tensor | None = None,
    grad_a_pad_compensation: torch.Tensor | None = None,
    grad_b_pad_compensation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Recover plain-space ``grad_A`` / ``grad_B`` / optional ``grad_X``.

    ``grad_A = N_in^{-T} grad_A_tilde U^T``,
    ``grad_B = U^{-T} grad_B_tilde N_out^T``,
    ``grad_X = grad_X_tilde N_in^T``.

    Pad compensations (if non-None) are added in plain space:
    ``grad_A += s T_in^T G B^T``, ``grad_B += s A^T T_in^T G``.
    """
    require_rank2("grad_a_tilde", grad_a_tilde)
    require_rank2("grad_b_tilde", grad_b_tilde)
    require_rank2("n_in", n_in)
    require_rank2("n_out", n_out)
    require_rank2("u", u)
    require_same_dtype_device(
        "grad_a_tilde", grad_a_tilde, grad_b_tilde=grad_b_tilde,
        n_in=n_in, n_out=n_out, u=u,
        grad_x_tilde=grad_x_tilde,
        grad_a_pad_compensation=grad_a_pad_compensation,
        grad_b_pad_compensation=grad_b_pad_compensation,
    )

    n_in_inv_t = torch.linalg.inv(n_in.transpose(0, 1))
    u_inv_t = torch.linalg.inv(u.transpose(0, 1))
    grad_a = n_in_inv_t @ grad_a_tilde @ u.transpose(0, 1)
    grad_b = u_inv_t @ grad_b_tilde @ n_out.transpose(0, 1)
    if grad_a_pad_compensation is not None:
        require_shape("grad_a_pad_compensation", grad_a_pad_compensation,
                      tuple(grad_a.shape))
        grad_a = grad_a + grad_a_pad_compensation
    if grad_b_pad_compensation is not None:
        require_shape("grad_b_pad_compensation", grad_b_pad_compensation,
                      tuple(grad_b.shape))
        grad_b = grad_b + grad_b_pad_compensation

    grad_x: torch.Tensor | None = None
    if grad_x_tilde is not None:
        require_rank2("grad_x_tilde", grad_x_tilde)
        grad_x = grad_x_tilde @ n_in.transpose(0, 1)
    return {"grad_a": grad_a, "grad_b": grad_b, "grad_x": grad_x}


# ---------------------------------------------------------------------------
# Pad compensation builders (trusted side, plain-space)
# ---------------------------------------------------------------------------


def make_lora_grad_pad_compensation(
    a: torch.Tensor,
    b: torch.Tensor,
    pad: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute trusted-side pad compensations for ``grad_A`` and ``grad_B``.

    ``grad_A_pad = s T_in^T G B^T``       (shape: d_in × rank)
    ``grad_B_pad = s A^T T_in^T G``       (shape: rank × d_out)

    The compensation requires plain ``A`` and ``B`` so it MUST stay on the
    trusted side; the GPU never sees these tensors.
    """
    require_rank2("a", a)
    require_rank2("b", b)
    require_rank2("pad", pad)
    require_rank2("upstream_grad", upstream_grad)
    if pad.shape[1] != a.shape[0]:
        raise ValueError(
            f"pad d_in must match a d_in={a.shape[0]}, got {tuple(pad.shape)}"
        )
    if upstream_grad.shape[1] != b.shape[1]:
        raise ValueError(
            f"upstream_grad d_out must match b d_out={b.shape[1]},"
            f" got {tuple(upstream_grad.shape)}"
        )
    require_same_dtype_device(
        "a", a, b=b, pad=pad, upstream_grad=upstream_grad,
    )
    rank = a.shape[1]
    scale = float(alpha) / max(rank, 1)
    grad_a_pad = scale * pad.transpose(0, 1) @ upstream_grad @ b.transpose(0, 1)
    grad_b_pad = scale * a.transpose(0, 1) @ pad.transpose(0, 1) @ upstream_grad
    return {
        "grad_a_pad_compensation": grad_a_pad,
        "grad_b_pad_compensation": grad_b_pad,
    }


# ---------------------------------------------------------------------------
# Convenience: one-shot trusted-orchestrated masked LoRA backward
# ---------------------------------------------------------------------------


def run_masked_lora_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    alpha: float,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    u: torch.Tensor,
    u_inv: torch.Tensor,
    pad: torch.Tensor | None = None,
    recover_grad_x: bool = False,
) -> dict[str, torch.Tensor | None]:
    """End-to-end trusted-orchestrated masked LoRA backward.

    The flow inside this helper is:
      1. trusted side computes ``X_tilde`` / ``W_tilde`` / ``A_tilde`` /
         ``B_tilde``;
      2. trusted side maps ``G → G_tilde``;
      3. *GPU side* runs :func:`masked_lora_backward`;
      4. trusted side calls :func:`recover_lora_gradients` (with pad
         compensation when ``pad`` is given) to get plain-space
         ``grad_A`` / ``grad_B`` / optional ``grad_X``.
    """
    # 1. Mask the forward operands. These mirror the Stage 7.0 forward.
    if pad is None:
        x_tilde = x @ n_in
    else:
        x_tilde = (x - pad) @ n_in
    w_tilde = n_in_inv @ w @ n_out
    a_tilde = n_in_inv @ a @ u
    b_tilde = u_inv @ b @ n_out
    # 2. Map upstream gradient into masked space.
    grad_y_tilde = transform_upstream_gradient(upstream_grad, n_out)
    # 3. GPU-side masked backward (no plaintext A/B/X/G/N/U).
    masked = masked_lora_backward(
        x_tilde, a_tilde, b_tilde, grad_y_tilde,
        alpha=alpha, w_tilde=w_tilde if recover_grad_x else None,
        recover_grad_x=recover_grad_x,
    )
    # 4. Pad compensations (trusted side).
    pad_compensation: dict[str, torch.Tensor] | None = None
    if pad is not None:
        pad_compensation = make_lora_grad_pad_compensation(
            a, b, pad, upstream_grad, alpha=alpha,
        )
    return recover_lora_gradients(
        masked["grad_a_tilde"], masked["grad_b_tilde"],
        n_in, n_out, u,
        grad_x_tilde=masked["grad_x_tilde"],
        grad_a_pad_compensation=(
            pad_compensation["grad_a_pad_compensation"]
            if pad_compensation is not None else None
        ),
        grad_b_pad_compensation=(
            pad_compensation["grad_b_pad_compensation"]
            if pad_compensation is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "MaskedLoRABackwardConfig",
    "invert_upstream_gradient_mask",
    "make_lora_grad_pad_compensation",
    "masked_lora_backward",
    "plain_lora_backward_reference",
    "recover_lora_gradients",
    "run_masked_lora_backward",
    "transform_upstream_gradient",
]
