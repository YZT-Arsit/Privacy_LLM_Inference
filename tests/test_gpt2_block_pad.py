"""Pad compensation audit tests for the GPT-2 block wrapper."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.evaluation import compute_correctness_metrics
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader


class RecordingExecutor(UntrustedGPUExecutor):
    """Executor that records whether compensation is supplied."""

    def __init__(self) -> None:
        super().__init__()
        self.compensation_seen: list[bool] = []

    def linear_forward(self, x_tilde, w_tilde, b_tilde=None, compensation=None):
        self.compensation_seen.append(compensation is not None)
        return super().linear_forward(x_tilde, w_tilde, b_tilde, compensation)


def _load_model():
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


def _first_hidden(output):
    return output[0] if isinstance(output, tuple) else output


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_gpt2_block_use_pad_true_matches_plain(batch_size: int, seq_len: int) -> None:
    model = _load_model()
    block = model.transformer.h[0]
    hidden_states = torch.randn(batch_size, seq_len, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        plain = _first_hidden(block(hidden_states, use_cache=False))
        wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32, use_pad=True)
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-4
    assert wrapper.pad_report["attn_c_attn_pad"] is True
    assert wrapper.pad_report["attn_c_proj_pad"] is True
    assert wrapper.pad_report["mlp_c_fc_pad"] is True
    assert wrapper.pad_report["mlp_c_proj_pad"] is True
    assert wrapper.pad_report["fresh_pad_count"] == 4
    assert wrapper.pad_report["fresh_pad_unique"] is True


def test_gpt2_block_use_pad_false_still_matches_plain() -> None:
    model = _load_model()
    block = model.transformer.h[0]
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        plain = _first_hidden(block(hidden_states, use_cache=False))
        wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32, use_pad=False)
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert wrapper.pad_report["attn_c_attn_pad"] is False
    assert wrapper.pad_report["attn_c_proj_pad"] is False
    assert wrapper.pad_report["mlp_c_fc_pad"] is False
    assert wrapper.pad_report["mlp_c_proj_pad"] is False


def test_gpt2_block_untrusted_executor_receives_compensation_not_plain_pad() -> None:
    model = _load_model()
    block = model.transformer.h[0]
    wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32, use_pad=True)
    recorder = RecordingExecutor()
    wrapper.executor = recorder
    _ = wrapper.forward(torch.randn(1, 4, model.config.n_embd, dtype=torch.float32))
    assert recorder.compensation_seen == [True, True, True, True]
    assert wrapper.pad_report["untrusted_receives_plain_pad"] is False


def test_gpt2_block_pad_does_not_modify_hf_model() -> None:
    model = _load_model()
    block = model.transformer.h[0]
    before = block.attn.c_attn.__class__.__name__
    wrapper = ObfuscatedGPT2BlockWrapper(block, model.config, dtype=torch.float32, use_pad=True)
    _ = wrapper.forward(torch.randn(1, 4, model.config.n_embd, dtype=torch.float32))
    assert block.attn.c_attn.__class__.__name__ == before
