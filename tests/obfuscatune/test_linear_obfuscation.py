"""Tests for the ObfuscaTune obfuscated-linear primitive (eq. 1-6)."""

from __future__ import annotations

import pytest
import torch

from pllo.baselines.obfuscatune.config import ObfuscaTuneConfig
from pllo.baselines.obfuscatune.linear_obfuscation import (
    math_weight,
    obfuscated_input_linear,
    obfuscated_output_linear,
    plain_linear,
)
from pllo.baselines.obfuscatune.modules import ObfuscaTuneLinearSimulator
from pllo.baselines.obfuscatune.random_matrices import orthogonal_matrix


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_projection_equivalence(dtype):
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 16, generator=g, dtype=dtype)
    w = torch.randn(16, 24, generator=g, dtype=dtype)   # math layout (d_in, d_out)
    r, r_inv = orthogonal_matrix(16, seed=1, dtype=dtype)
    y = obfuscated_input_linear(x, w, r, r_inv)
    assert torch.allclose(y, x @ w, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_projection_equivalence(dtype):
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    g = torch.Generator().manual_seed(0)
    h = torch.randn(8, 16, generator=g, dtype=dtype)
    w = torch.randn(16, 16, generator=g, dtype=dtype)
    r, r_inv = orthogonal_matrix(16, seed=2, dtype=dtype)
    o = obfuscated_output_linear(h, w, r, r_inv)
    assert torch.allclose(o, h @ w, atol=tol)


def test_nn_linear_layout_math_weight():
    lin = torch.nn.Linear(16, 24, bias=True).double()
    x = torch.randn(5, 16, dtype=torch.float64)
    # math_weight is (d_in, d_out) and reproduces the module forward
    assert math_weight(lin).shape == (16, 24)
    assert torch.allclose(plain_linear(lin, x), lin(x), atol=1e-10)


def test_conv1d_layout_math_weight():
    Conv1D = pytest.importorskip(
        "transformers.models.gpt2.modeling_gpt2").Conv1D
    conv = Conv1D(24, 16)                                 # nf=d_out=24, nx=d_in=16
    conv.double()
    x = torch.randn(5, 16, dtype=torch.float64)
    assert math_weight(conv).shape == (16, 24)           # (d_in, d_out)
    assert torch.allclose(plain_linear(conv, x), conv(x), atol=1e-10)


def test_simulator_input_direction_is_exact_orthogonal():
    lin = torch.nn.Linear(16, 16, bias=True).double()
    cfg = ObfuscaTuneConfig(dtype="float64", random_matrix_type="orthogonal")
    sim = ObfuscaTuneLinearSimulator(lin, cfg, direction="input")
    x = torch.randn(6, 16, dtype=torch.float64)
    y, m = sim.forward(x)
    assert m.max_abs_err < 1e-9
    assert m.outside_matmul_count == 1 and m.tee_matmul_count == 1


def test_naive_random_error_at_least_orthogonal_in_chain():
    # multi-layer chain in float32: naive random accumulates more error than orthogonal
    torch.manual_seed(0)
    d = 32
    x0 = torch.randn(8, d, dtype=torch.float32)
    layers = [torch.nn.Linear(d, d, bias=False).float() for _ in range(6)]

    def run(kind: str) -> float:
        cfg = ObfuscaTuneConfig(dtype="float32", random_matrix_type=kind, seed=0)
        x = x0.clone()
        ref = x0.clone()
        for i, lin in enumerate(layers):
            sim = ObfuscaTuneLinearSimulator(lin, cfg, direction="input", seed=i)
            x, _ = sim.forward(x)
            ref = plain_linear(lin, ref)
        return float((x - ref).abs().max().item())

    err_orth = run("orthogonal")
    err_rand = run("random")
    assert err_rand >= err_orth
