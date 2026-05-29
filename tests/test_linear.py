"""Tests for standard linear obfuscated execution."""

from __future__ import annotations

import torch
import pytest

from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.evaluation import compute_correctness_metrics
from pllo.ops.linear import linear_obfuscated, linear_plain
from pllo.utils.seed import set_seed


def _make_case(
    s: int = 4,
    d_in: int = 16,
    d_out: int = 32,
    with_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = torch.randn(s, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    bias = torch.randn(d_out, dtype=torch.float64) if with_bias else None
    return x, w, bias


def _assert_correct(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    metrics = compute_correctness_metrics(reference, candidate)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-8
    assert metrics["relative_l2_error"] < 1e-8


def test_linear_without_pad() -> None:
    set_seed(1)
    x, w, bias = _make_case()
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=False))


def test_linear_with_pad() -> None:
    set_seed(2)
    x, w, bias = _make_case()
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=True))


def test_linear_different_sequence_length() -> None:
    set_seed(3)
    x, w, bias = _make_case(s=3)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=True))


def test_linear_different_input_output_dimensions() -> None:
    set_seed(4)
    x, w, bias = _make_case(s=3, d_in=8, d_out=12)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=True))


def test_linear_with_bias() -> None:
    set_seed(5)
    x, w, bias = _make_case(with_bias=True)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=False))


def test_linear_without_bias() -> None:
    set_seed(6)
    x, w, bias = _make_case(with_bias=False)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=True))


@pytest.mark.parametrize(
    ("s", "d_in", "d_out"),
    [
        (1, 4, 4),
        (3, 8, 12),
        (5, 16, 32),
        (2, 32, 16),
    ],
)
@pytest.mark.parametrize("use_pad", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_parameterized_dimensions(
    s: int,
    d_in: int,
    d_out: int,
    use_pad: bool,
    with_bias: bool,
) -> None:
    set_seed(100 + s + d_in + d_out + int(use_pad) + int(with_bias))
    x, w, bias = _make_case(s=s, d_in=d_in, d_out=d_out, with_bias=with_bias)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=use_pad))


def test_linear_shape_mismatch_w_dimension() -> None:
    x, _, bias = _make_case(s=2, d_in=4, d_out=6)
    bad_w = torch.randn(5, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="w must have shape"):
        linear_plain(x, bad_w, bias)


def test_linear_shape_mismatch_bias_dimension() -> None:
    x, w, _ = _make_case(s=2, d_in=4, d_out=6)
    bad_bias = torch.randn(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="bias must have shape"):
        linear_obfuscated(x, w, bad_bias)


def test_linear_compensation_shape_mismatch() -> None:
    executor = UntrustedGPUExecutor()
    x_tilde = torch.randn(2, 4, dtype=torch.float64)
    w_tilde = torch.randn(4, 6, dtype=torch.float64)
    bad_compensation = torch.randn(2, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="compensation must have shape"):
        executor.linear_forward(x_tilde, w_tilde, compensation=bad_compensation)


def test_linear_dtype_mismatch() -> None:
    x = torch.randn(2, 4, dtype=torch.float64)
    w = torch.randn(4, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype must match"):
        linear_plain(x, w)


def test_linear_pad_and_no_pad_match_same_reference() -> None:
    set_seed(200)
    x, w, bias = _make_case(s=3, d_in=8, d_out=12, with_bias=True)
    reference = linear_plain(x, w, bias)
    no_pad = linear_obfuscated(x, w, bias, use_pad=False)
    with_pad = linear_obfuscated(x, w, bias, use_pad=True)
    _assert_correct(reference, no_pad)
    _assert_correct(reference, with_pad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_linear_with_pad_cuda() -> None:
    set_seed(300)
    device = torch.device("cuda")
    x = torch.randn(3, 8, dtype=torch.float64, device=device)
    w = torch.randn(8, 12, dtype=torch.float64, device=device)
    bias = torch.randn(12, dtype=torch.float64, device=device)
    _assert_correct(linear_plain(x, w, bias), linear_obfuscated(x, w, bias, use_pad=True))
