"""Tests for LoRA linear obfuscated execution."""

from __future__ import annotations

import torch
import pytest

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.evaluation import compute_correctness_metrics
from pllo.ops.lora_linear import lora_linear_obfuscated, lora_linear_plain
from pllo.utils.seed import set_seed


def _make_case(
    s: int = 4,
    d_in: int = 16,
    d_out: int = 32,
    rank: int = 4,
    with_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = torch.randn(s, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    a = torch.randn(d_in, rank, dtype=torch.float64)
    b = torch.randn(rank, d_out, dtype=torch.float64)
    bias = torch.randn(d_out, dtype=torch.float64) if with_bias else None
    return x, w, a, b, bias


def _assert_correct(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    metrics = compute_correctness_metrics(reference, candidate)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-8
    assert metrics["relative_l2_error"] < 1e-8


def test_lora_without_pad() -> None:
    set_seed(11)
    x, w, a, b, bias = _make_case()
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=False)
    _assert_correct(reference, candidate)


def test_lora_with_pad() -> None:
    set_seed(12)
    x, w, a, b, bias = _make_case()
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=True)
    _assert_correct(reference, candidate)


def test_lora_different_rank() -> None:
    set_seed(13)
    x, w, a, b, bias = _make_case(s=3, d_in=8, d_out=12, rank=2)
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=True)
    _assert_correct(reference, candidate)


def test_lora_with_bias() -> None:
    set_seed(14)
    x, w, a, b, bias = _make_case(with_bias=True)
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=False)
    _assert_correct(reference, candidate)


def test_lora_without_bias() -> None:
    set_seed(15)
    x, w, a, b, bias = _make_case(with_bias=False)
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=True)
    _assert_correct(reference, candidate)


def test_lora_adapter_not_merged_into_w() -> None:
    set_seed(16)
    x, w, a, b, _ = _make_case(rank=4)
    tee = SimulatedTEE()
    state = tee.create_linear_mask_state(x, w.shape[1], use_pad=False)
    tee.add_lora_rank_mask(state, a.shape[1])
    w_tilde, _ = tee.transform_linear_weight(w, None, state)
    a_tilde, b_tilde = tee.transform_lora_adapters(a, b, state)

    assert w_tilde.shape == w.shape
    assert a_tilde.shape == a.shape
    assert b_tilde.shape == b.shape
    assert a_tilde.shape[1] == a.shape[1]
    assert b_tilde.shape[0] == b.shape[0]


@pytest.mark.parametrize(
    ("s", "d_in", "d_out"),
    [
        (1, 4, 4),
        (3, 8, 12),
        (5, 16, 32),
        (2, 32, 16),
    ],
)
@pytest.mark.parametrize("rank", [1, 2, 4, 8])
@pytest.mark.parametrize("use_pad", [False, True])
def test_lora_parameterized_dimensions_and_ranks(
    s: int,
    d_in: int,
    d_out: int,
    rank: int,
    use_pad: bool,
) -> None:
    if rank > min(d_in, d_out):
        pytest.skip("rank must be <= min(d_in, d_out)")
    set_seed(400 + s + d_in + d_out + rank + int(use_pad))
    x, w, a, b, bias = _make_case(s=s, d_in=d_in, d_out=d_out, rank=rank)
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=use_pad)
    _assert_correct(reference, candidate)


def test_lora_shape_mismatch_w_dimension() -> None:
    x, _, a, b, bias = _make_case(s=2, d_in=4, d_out=6, rank=2)
    bad_w = torch.randn(5, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="w must have shape"):
        lora_linear_plain(x, bad_w, a, b, bias)


def test_lora_shape_mismatch_bias_dimension() -> None:
    x, w, a, b, _ = _make_case(s=2, d_in=4, d_out=6, rank=2)
    bad_bias = torch.randn(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="bias must have shape"):
        lora_linear_obfuscated(x, w, a, b, bad_bias)


def test_lora_shape_mismatch_adapter_rank() -> None:
    x, w, a, _, bias = _make_case(s=2, d_in=4, d_out=6, rank=2)
    bad_b = torch.randn(3, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="b rank dimension"):
        lora_linear_plain(x, w, a, bad_b, bias)


def test_lora_rank_mask_shape_mismatch() -> None:
    x, w, a, b, _ = _make_case(s=2, d_in=4, d_out=6, rank=2)
    tee = SimulatedTEE()
    state = tee.create_linear_mask_state(x, w.shape[1], use_pad=False)
    state.rank_mask = torch.eye(3, dtype=torch.float64)
    state.rank_mask_inv = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="state.rank_mask must have shape"):
        tee.transform_lora_adapters(a, b, state)


def test_lora_compensation_shape_mismatch() -> None:
    executor = UntrustedGPUExecutor()
    x_tilde = torch.randn(2, 4, dtype=torch.float64)
    w_tilde = torch.randn(4, 6, dtype=torch.float64)
    a_tilde = torch.randn(4, 2, dtype=torch.float64)
    b_tilde = torch.randn(2, 6, dtype=torch.float64)
    bad_compensation = torch.randn(2, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="compensation must have shape"):
        executor.lora_linear_forward(x_tilde, w_tilde, a_tilde, b_tilde, compensation=bad_compensation)


def test_lora_pad_and_no_pad_match_same_reference() -> None:
    set_seed(500)
    x, w, a, b, bias = _make_case(s=3, d_in=8, d_out=12, rank=2, with_bias=True)
    reference = lora_linear_plain(x, w, a, b, bias)
    no_pad = lora_linear_obfuscated(x, w, a, b, bias, use_pad=False)
    with_pad = lora_linear_obfuscated(x, w, a, b, bias, use_pad=True)
    _assert_correct(reference, no_pad)
    _assert_correct(reference, with_pad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_lora_with_pad_cuda() -> None:
    set_seed(600)
    device = torch.device("cuda")
    x = torch.randn(3, 8, dtype=torch.float64, device=device)
    w = torch.randn(8, 12, dtype=torch.float64, device=device)
    a = torch.randn(8, 2, dtype=torch.float64, device=device)
    b = torch.randn(2, 12, dtype=torch.float64, device=device)
    bias = torch.randn(12, dtype=torch.float64, device=device)
    reference = lora_linear_plain(x, w, a, b, bias)
    candidate = lora_linear_obfuscated(x, w, a, b, bias, use_pad=True)
    _assert_correct(reference, candidate)
