"""Tests for tiny Transformer MLP helpers."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.ops.mlp import mlp_obfuscated, mlp_plain
from pllo.utils.seed import set_seed


def test_plain_and_obfuscated_mlp_match() -> None:
    set_seed(1101)
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    w1 = torch.randn(16, 32, dtype=torch.float64) * 0.02
    b1 = torch.randn(32, dtype=torch.float64) * 0.02
    w2 = torch.randn(32, 16, dtype=torch.float64) * 0.02
    b2 = torch.randn(16, dtype=torch.float64) * 0.02
    tee = SimulatedTEE()
    executor = UntrustedGPUExecutor()
    output_state = tee.create_linear_mask_state(x.reshape(-1, 16), 16, use_pad=False)
    output_state.n_out = output_state.n_in
    output_state.n_out_inv = output_state.n_in_inv

    out_tilde = mlp_obfuscated(x, w1, b1, w2, b2, output_state, tee, executor, use_pad=True)
    recovered = tee.recover_output(out_tilde.reshape(-1, 16), output_state).reshape_as(x)
    assert torch.allclose(mlp_plain(x, w1, b1, w2, b2), recovered, atol=1e-8, rtol=1e-6)


def test_trusted_gelu_path_runs() -> None:
    set_seed(1102)
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    w1 = torch.randn(8, 16, dtype=torch.float64) * 0.02
    b1 = torch.zeros(16, dtype=torch.float64)
    w2 = torch.randn(16, 8, dtype=torch.float64) * 0.02
    b2 = torch.zeros(8, dtype=torch.float64)
    tee = SimulatedTEE()
    output_state = tee.create_linear_mask_state(x.reshape(-1, 8), 8, use_pad=False)
    output_state.n_out = output_state.n_in
    output_state.n_out_inv = output_state.n_in_inv
    out_tilde = mlp_obfuscated(x, w1, b1, w2, b2, output_state, tee, UntrustedGPUExecutor())
    assert out_tilde.shape == x.shape


def test_mlp_residual_output_mask_is_consistent() -> None:
    set_seed(1103)
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    w1 = torch.randn(8, 16, dtype=torch.float64) * 0.02
    b1 = torch.zeros(16, dtype=torch.float64)
    w2 = torch.randn(16, 8, dtype=torch.float64) * 0.02
    b2 = torch.zeros(8, dtype=torch.float64)
    tee = SimulatedTEE()
    output_state = tee.create_linear_mask_state(x.reshape(-1, 8), 8, use_pad=False)
    output_state.n_out = output_state.n_in
    output_state.n_out_inv = output_state.n_in_inv
    out_tilde = mlp_obfuscated(x, w1, b1, w2, b2, output_state, tee, UntrustedGPUExecutor())
    recovered = tee.recover_output(out_tilde.reshape(-1, 8), output_state).reshape_as(x)
    assert torch.allclose(recovered, mlp_plain(x, w1, b1, w2, b2), atol=1e-8, rtol=1e-6)
