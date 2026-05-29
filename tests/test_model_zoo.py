"""Tests for model-zoo registry and base helpers."""

from __future__ import annotations

import pytest
import torch

from pllo.model_zoo import HuggingFaceModelLoader, get_model_loader, torch_dtype_from_string


def test_registry_returns_huggingface_loader() -> None:
    loader = get_model_loader("hf")
    assert isinstance(loader, HuggingFaceModelLoader)
    assert isinstance(get_model_loader("huggingface"), HuggingFaceModelLoader)


def test_registry_modelscope_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Stage 5"):
        get_model_loader("modelscope")


def test_dtype_string_mapping() -> None:
    assert torch_dtype_from_string("float32") is torch.float32
    assert torch_dtype_from_string("float64") is torch.float64


def test_float16_requires_cuda() -> None:
    if torch.cuda.is_available():
        assert torch_dtype_from_string("float16", "cuda") is torch.float16
    else:
        with pytest.raises(ValueError, match="only allowed on CUDA"):
            torch_dtype_from_string("float16", "cpu")


def test_invalid_dtype_raises() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        torch_dtype_from_string("bfloat16")
