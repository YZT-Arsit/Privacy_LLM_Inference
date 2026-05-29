"""LM head helpers."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.masks.mask_state import MaskState
from pllo.ops.linear import linear_plain


def lm_head_plain(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute plaintext logits from hidden states."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = linear_plain(flat, weight, bias)
    return logits.reshape(*hidden.shape[:-1], weight.shape[1])


def lm_head_obfuscated(
    hidden_plain: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
) -> torch.Tensor:
    """Compute logits through an obfuscated LM head and recover them."""
    flat = hidden_plain.reshape(-1, hidden_plain.shape[-1])
    state = tee.create_linear_mask_state(flat, weight.shape[1], use_pad=use_pad, pad_scale=pad_scale)
    state.n_out = vocab_state.n_out
    state.n_out_inv = vocab_state.n_out_inv

    x_tilde = tee.obfuscate_input(flat, state)
    w_tilde, b_tilde = tee.transform_linear_weight(weight, bias, state)
    compensation = tee.make_linear_pad_compensation(weight, state)
    logits_tilde = executor.linear_forward(x_tilde, w_tilde, b_tilde, compensation)
    logits_hat = tee.recover_output(logits_tilde, state)
    return logits_hat.reshape(*hidden_plain.shape[:-1], weight.shape[1])
