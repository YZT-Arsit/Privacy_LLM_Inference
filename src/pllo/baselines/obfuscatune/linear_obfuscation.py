"""Obfuscated-linear primitives for ObfuscaTune (eq. 1-6 of arXiv:2407.02960).

Everything here operates on the *math layout* weight ``W`` of shape
``(d_in, d_out)`` so that a plaintext linear is simply ``Y = X @ W`` with
``X`` of shape ``(..., d_in)``. Two obfuscation directions are provided:

Input projection (attention Q/K/V, MLP first linear)::

    X* = X @ Ra                 # obfuscate the (embedding) input, Ra: (d_in,d_in)
    W* = Ra^{-1} @ W            # obfuscate the weight
    Y  = X* @ W* = X @ W        # cancels -> untrusted side sees the TRUE Y

Output projection (attention output proj, MLP second linear)::

    W* = W @ Rb                 # obfuscate the weight on the right, Rb:(d_out,d_out)
    Y* = H @ W* = H @ W @ Rb    # untrusted side sees an obfuscated output
    Y  = Y* @ Rb^{-1} = H @ W   # de-obfuscate back inside the TEE

The two module layouts we care about are handled by :func:`math_weight`:

  * ``torch.nn.Linear``: stored weight is ``(d_out, d_in)`` and the layer computes
    ``x @ weight.T`` -> math weight is ``weight.T``.
  * HF GPT-2 ``Conv1D``: stored weight is already ``(d_in, d_out)`` and the layer
    computes ``x @ weight`` -> math weight is ``weight`` itself.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Core obfuscation primitives (operate on math-layout weights (d_in, d_out))
# ---------------------------------------------------------------------------
def obfuscate_input_right(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """X* = X @ Ra  (obfuscate the input embedding, Ra: (d_in, d_in))."""
    return x @ r


def obfuscate_weight_left(w: torch.Tensor, r_inv: torch.Tensor) -> torch.Tensor:
    """W* = Ra^{-1} @ W  (obfuscate an input-projection weight, W: (d_in, d_out))."""
    return r_inv @ w


def obfuscate_weight_output_right(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """W* = W @ Rb  (obfuscate an output-projection weight, Rb: (d_out, d_out))."""
    return w @ r


def deobfuscate_output_right(y_star: torch.Tensor, r_inv: torch.Tensor) -> torch.Tensor:
    """Y = Y* @ Rb^{-1}  (de-obfuscate an output inside the TEE)."""
    return y_star @ r_inv


def obfuscated_input_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    r: torch.Tensor,
    r_inv: torch.Tensor,
) -> torch.Tensor:
    """Full input-projection path: (X @ Ra) @ (Ra^{-1} @ W). Equals X @ W."""
    return obfuscate_input_right(x, r) @ obfuscate_weight_left(w, r_inv)


def obfuscated_output_linear(
    h: torch.Tensor,
    w: torch.Tensor,
    r: torch.Tensor,
    r_inv: torch.Tensor,
) -> torch.Tensor:
    """Full output-projection path: (H @ (W @ Rb)) @ Rb^{-1}. Equals H @ W."""
    y_star = h @ obfuscate_weight_output_right(w, r)
    return deobfuscate_output_right(y_star, r_inv)


# ---------------------------------------------------------------------------
# Module-layout adapters
# ---------------------------------------------------------------------------
def is_conv1d(module: torch.nn.Module) -> bool:
    """True for HF GPT-2 ``Conv1D`` (weight stored as (d_in, d_out))."""
    cls = type(module)
    return cls.__name__ == "Conv1D"


def math_weight(module: torch.nn.Module) -> torch.Tensor:
    """Return the ``(d_in, d_out)`` math-layout weight for a linear module.

    Supports ``torch.nn.Linear`` (weight (d_out, d_in) -> transpose) and HF
    GPT-2 ``Conv1D`` (weight already (d_in, d_out)).
    """
    if is_conv1d(module):
        return module.weight
    if isinstance(module, torch.nn.Linear):
        return module.weight.transpose(0, 1)
    raise TypeError(
        f"unsupported linear module type {type(module).__name__!r}; "
        "expected torch.nn.Linear or HF GPT-2 Conv1D"
    )


def module_bias(module: torch.nn.Module) -> torch.Tensor | None:
    return getattr(module, "bias", None)


def plain_linear(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reference plaintext ``Y = X @ W (+ b)`` using the module's math weight."""
    y = x @ math_weight(module)
    b = module_bias(module)
    if b is not None:
        y = y + b
    return y


__all__ = [
    "obfuscate_input_right",
    "obfuscate_weight_left",
    "obfuscate_weight_output_right",
    "deobfuscate_output_right",
    "obfuscated_input_linear",
    "obfuscated_output_linear",
    "is_conv1d",
    "math_weight",
    "module_bias",
    "plain_linear",
]
