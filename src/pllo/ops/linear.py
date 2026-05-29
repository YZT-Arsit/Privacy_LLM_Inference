"""Standard linear operation helpers."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.utils.validation import validate_linear_inputs


def linear_plain(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the plaintext row-vector linear layer Y = XW + b."""
    validate_linear_inputs(x, w, bias)
    y = x @ w
    if bias is not None:
        y = y + bias
    return y


def linear_obfuscated(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_pad: bool = False,
    pad_scale: float = 1.0,
    tee: SimulatedTEE | None = None,
    executor: UntrustedGPUExecutor | None = None,
) -> torch.Tensor:
    """Execute a standard linear layer through trusted/untrusted separation."""
    validate_linear_inputs(x, w, bias)
    tee = tee or SimulatedTEE(dtype=x.dtype, device=x.device)
    executor = executor or UntrustedGPUExecutor()

    state = tee.create_linear_mask_state(x, w.shape[1], use_pad=use_pad, pad_scale=pad_scale)
    x_tilde = tee.obfuscate_input(x, state)
    w_tilde, b_tilde = tee.transform_linear_weight(w, bias, state)
    compensation = tee.make_linear_pad_compensation(w, state)
    y_tilde = executor.linear_forward(x_tilde, w_tilde, b_tilde, compensation)
    return tee.recover_output(y_tilde, state)
