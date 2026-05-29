"""Tests for Stage 4.6 GPT-2 single-block wrapper."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader


def _load_model(dtype: str = "float32"):
    config = ExternalModelConfig(
        source="huggingface",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        dtype=dtype,
    )
    try:
        _, model = get_model_loader("hf").load(config)
        return model
    except Exception as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable in this environment: {exc}")


def _first_hidden(output):
    return output[0] if isinstance(output, tuple) else output


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_gpt2_block_wrapper_float32_matches_plain(batch_size: int, seq_len: int) -> None:
    model = _load_model("float32")
    block = model.transformer.h[0]
    hidden_states = torch.randn(batch_size, seq_len, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        plain = _first_hidden(block(hidden_states, use_cache=False))
        wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32, use_pad=False)
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-4
    assert metrics["relative_l2_error"] < 1e-4


def test_gpt2_block_wrapper_float64_matches_plain_when_loaded_float64() -> None:
    model = _load_model("float64")
    block = model.transformer.h[0]
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float64)
    with torch.no_grad():
        plain = _first_hidden(block(hidden_states, use_cache=False))
        wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float64, use_pad=False)
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-8, rtol=1e-6)
    assert metrics["allclose"] is True


def test_gpt2_block_wrapper_does_not_replace_hf_modules() -> None:
    model = _load_model("float32")
    block = model.transformer.h[0]
    before = block.attn.c_attn.__class__.__name__
    wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32)
    _ = wrapper.forward(torch.randn(1, 4, model.config.n_embd, dtype=torch.float32))
    assert block.attn.c_attn.__class__.__name__ == before
