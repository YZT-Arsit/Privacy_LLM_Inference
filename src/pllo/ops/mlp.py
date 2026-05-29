"""MLP helpers for the tiny Transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.masks.mask_state import MaskState
from pllo.ops.linear import linear_plain


def mlp_plain(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Compute MLP(X) = Linear2(GELU(Linear1(X)))."""
    flat = x.reshape(-1, x.shape[-1])
    hidden = F.gelu(linear_plain(flat, w1, b1))
    out = linear_plain(hidden, w2, b2)
    return out.reshape(*x.shape[:-1], w2.shape[1])


def mlp_obfuscated(
    x_plain: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    output_state: MaskState,
    tee: SimulatedTEE,
    executor: UntrustedGPUExecutor,
    use_pad: bool = False,
    pad_scale: float = 1.0,
) -> torch.Tensor:
    """Run an MLP with obfuscated linear layers and trusted GELU.

    Stage 2 uses trusted GELU as an engineering simplification. Linear1 is
    recovered in the simulated TEE, GELU is applied there, and Linear2 returns
    to the residual branch's output mask space.
    """
    flat = x_plain.reshape(-1, x_plain.shape[-1])

    state1 = tee.create_linear_mask_state(flat, w1.shape[1], use_pad=use_pad, pad_scale=pad_scale)
    x_tilde = tee.obfuscate_input(flat, state1)
    w1_tilde, b1_tilde = tee.transform_linear_weight(w1, b1, state1)
    comp1 = tee.make_linear_pad_compensation(w1, state1)
    hidden_tilde = executor.linear_forward(x_tilde, w1_tilde, b1_tilde, comp1)
    hidden_plain = F.gelu(tee.recover_output(hidden_tilde, state1))

    state2 = tee.create_linear_mask_state(
        hidden_plain,
        w2.shape[1],
        use_pad=use_pad,
        pad_scale=pad_scale,
    )
    state2.n_out = output_state.n_out
    state2.n_out_inv = output_state.n_out_inv

    hidden2_tilde = tee.obfuscate_input(hidden_plain, state2)
    w2_tilde, b2_tilde = tee.transform_linear_weight(w2, b2, state2)
    comp2 = tee.make_linear_pad_compensation(w2, state2)
    out_tilde = executor.linear_forward(hidden2_tilde, w2_tilde, b2_tilde, comp2)
    return out_tilde.reshape(*x_plain.shape[:-1], w2.shape[1])
