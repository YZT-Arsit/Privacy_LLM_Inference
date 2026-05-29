"""Tests for GPT-2 Conv1D adapter utilities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from pllo.model_zoo import ExternalModelConfig, get_model_loader
from pllo.model_zoo.gpt2_conv1d_adapter import (
    compare_c_attn_split_equivalence,
    compare_conv1d_equivalence,
    conv1d_forward_as_linear,
    extract_conv1d_as_linear,
    is_hf_gpt2_conv1d,
    split_gpt2_c_attn_weights,
)


def _load_tiny_gpt2_model():
    config = ExternalModelConfig(
        source="huggingface",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        dtype="float32",
    )
    try:
        _, model = get_model_loader("hf").load(config)
        return model
    except Exception as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable in this environment: {exc}")


def test_is_hf_gpt2_conv1d_recognizes_c_attn() -> None:
    model = _load_tiny_gpt2_model()
    assert is_hf_gpt2_conv1d(model.transformer.h[0].attn.c_attn) is True


def test_is_hf_gpt2_conv1d_does_not_match_torch_conv1d() -> None:
    assert is_hf_gpt2_conv1d(nn.Conv1d(2, 4, kernel_size=1)) is False


def test_extract_conv1d_as_linear_shapes() -> None:
    model = _load_tiny_gpt2_model()
    hidden = model.config.n_embd
    weight, bias = extract_conv1d_as_linear(model.transformer.h[0].attn.c_attn)
    assert tuple(weight.shape) == (hidden, 3 * hidden)
    assert bias is not None
    assert tuple(bias.shape) == (3 * hidden,)


def test_conv1d_forward_as_linear_matches_hf_output() -> None:
    model = _load_tiny_gpt2_model()
    hidden = model.config.n_embd
    x = torch.randn(2, 4, hidden)
    module = model.transformer.h[0].attn.c_proj
    assert torch.allclose(module(x), conv1d_forward_as_linear(x, module), atol=1e-6, rtol=1e-5)
    metrics = compare_conv1d_equivalence(module, x)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] == 0.0


def test_split_gpt2_c_attn_weights_shapes() -> None:
    model = _load_tiny_gpt2_model()
    hidden = model.config.n_embd
    pieces = split_gpt2_c_attn_weights(model.transformer.h[0].attn.c_attn, hidden)
    for name in ("q", "k", "v"):
        assert tuple(pieces[name]["weight"].shape) == (hidden, hidden)
        assert pieces[name]["bias"] is not None
        assert tuple(pieces[name]["bias"].shape) == (hidden,)


def test_compare_c_attn_split_equivalence_allclose() -> None:
    model = _load_tiny_gpt2_model()
    hidden = model.config.n_embd
    x = torch.randn(2, 4, hidden)
    metrics = compare_c_attn_split_equivalence(model.transformer.h[0].attn.c_attn, x, hidden)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] == 0.0
