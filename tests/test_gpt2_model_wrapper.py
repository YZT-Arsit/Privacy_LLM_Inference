"""Tests for Stage 4.7 GPT-2 model-level obfuscated wrapper."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import top1_match_rate
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper
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


# ---------------------------------------------------------------------------
# HF model integrity guards (must hold before and after wrapper construction)
# ---------------------------------------------------------------------------

def test_gpt2_model_hf_c_attn_not_replaced() -> None:
    model = _load_model()
    before = type(model.transformer.h[0].attn.c_attn).__name__
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    _ = wrapper.forward(torch.randint(0, model.config.vocab_size, (1, 4)))
    assert type(model.transformer.h[0].attn.c_attn).__name__ == before == "Conv1D"


def test_gpt2_model_tied_embedding_not_broken_by_wrapper() -> None:
    """Wrapper must not change lm_head.weight tying status (tied or untied)."""
    model = _load_model()
    was_tied = model.lm_head.weight is model.transformer.wte.weight
    _ = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    is_tied_after = model.lm_head.weight is model.transformer.wte.weight
    assert was_tied == is_tied_after, (
        "ObfuscatedGPT2ModelWrapper changed lm_head.weight tying status"
    )


# ---------------------------------------------------------------------------
# Logits correctness (use_pad=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_gpt2_model_wrapper_use_pad_true(batch_size: int, seq_len: int) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        plain_logits = model(input_ids).logits
        wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
        recovered_logits = wrapper.forward(input_ids)

    assert plain_logits.shape == recovered_logits.shape, (
        f"shape mismatch: plain={plain_logits.shape}, recovered={recovered_logits.shape}"
    )
    metrics = compute_correctness_metrics(plain_logits, recovered_logits, atol=1e-4, rtol=1e-4)
    rate = top1_match_rate(plain_logits, recovered_logits)
    assert metrics["allclose"] is True, f"allclose=False, max_abs_error={metrics['max_abs_error']}"
    assert metrics["max_abs_error"] < 1e-4, f"max_abs_error={metrics['max_abs_error']}"
    assert rate == 1.0, f"top1_match_rate={rate}"


# ---------------------------------------------------------------------------
# Logits correctness (use_pad=False)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_gpt2_model_wrapper_use_pad_false(batch_size: int, seq_len: int) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        plain_logits = model(input_ids).logits
        wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=False)
        recovered_logits = wrapper.forward(input_ids)

    assert plain_logits.shape == recovered_logits.shape
    metrics = compute_correctness_metrics(plain_logits, recovered_logits, atol=1e-4, rtol=1e-4)
    rate = top1_match_rate(plain_logits, recovered_logits)
    assert metrics["allclose"] is True, f"allclose=False, max_abs_error={metrics['max_abs_error']}"
    assert metrics["max_abs_error"] < 1e-4, f"max_abs_error={metrics['max_abs_error']}"
    assert rate == 1.0, f"top1_match_rate={rate}"


# ---------------------------------------------------------------------------
# Output shape sanity
# ---------------------------------------------------------------------------

def test_gpt2_model_wrapper_output_shape() -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    batch, seq = 2, 6
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    with torch.no_grad():
        logits = wrapper.forward(input_ids)
    assert logits.shape == (batch, seq, vocab_size), (
        f"expected ({batch}, {seq}, {vocab_size}), got {tuple(logits.shape)}"
    )


# ---------------------------------------------------------------------------
# Pad reports populated after forward
# ---------------------------------------------------------------------------

def test_gpt2_model_wrapper_cosine_similarity_in_valid_range() -> None:
    """cosine_similarity must never exceed 1.0 (floating-point clamp guard)."""
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    for use_pad in (True, False):
        input_ids = torch.randint(0, vocab_size, (2, 8))
        with torch.no_grad():
            plain_logits = model(input_ids).logits
            wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=use_pad)
            recovered_logits = wrapper.forward(input_ids)
        metrics = compute_correctness_metrics(plain_logits, recovered_logits, atol=1e-4, rtol=1e-4)
        cs = metrics["cosine_similarity"]
        assert -1.0 <= cs <= 1.0, f"cosine_similarity={cs} out of [-1, 1] (use_pad={use_pad})"


def test_gpt2_model_wrapper_pad_reports_populated() -> None:
    model = _load_model("float32")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    with torch.no_grad():
        wrapper.forward(input_ids)
    for i, report in enumerate(wrapper.pad_reports):
        assert report["attn_c_attn_pad"] is True, f"block {i}: attn_c_attn_pad not set"
        assert report["attn_c_proj_pad"] is True, f"block {i}: attn_c_proj_pad not set"
        assert report["mlp_c_fc_pad"] is True, f"block {i}: mlp_c_fc_pad not set"
        assert report["mlp_c_proj_pad"] is True, f"block {i}: mlp_c_proj_pad not set"
