"""LoRA linear operation helpers."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.utils.validation import validate_lora_inputs


def lora_linear_plain(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute plaintext LoRA linear Y = XW + XAB + b."""
    validate_lora_inputs(x, w, a, b, bias)
    y = x @ w + (x @ a) @ b
    if bias is not None:
        y = y + bias
    return y


def lora_linear_obfuscated(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_pad: bool = False,
    pad_scale: float = 1.0,
    tee: SimulatedTEE | None = None,
    executor: UntrustedGPUExecutor | None = None,
) -> torch.Tensor:
    """Execute LoRA linear through trusted/untrusted separation."""
    validate_lora_inputs(x, w, a, b, bias)
    tee = tee or SimulatedTEE(dtype=x.dtype, device=x.device)
    executor = executor or UntrustedGPUExecutor()

    state = tee.create_linear_mask_state(x, w.shape[1], use_pad=use_pad, pad_scale=pad_scale)
    tee.add_lora_rank_mask(state, a.shape[1])
    x_tilde = tee.obfuscate_input(x, state)
    w_tilde, bias_tilde = tee.transform_linear_weight(w, bias, state)
    a_tilde, b_tilde = tee.transform_lora_adapters(a, b, state)
    compensation = tee.make_lora_pad_compensation(w, a, b, state)
    y_tilde = executor.lora_linear_forward(
        x_tilde,
        w_tilde,
        a_tilde,
        b_tilde,
        bias_tilde=bias_tilde,
        compensation=compensation,
    )
    return tee.recover_output(y_tilde, state)
