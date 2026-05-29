"""Tests for Stage 4.8 GPT-2 prefill/decode KV cache correctness."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import top1_match_rate
from pllo.hf_wrappers import (
    ObfuscatedGPT2KVCache,
    ObfuscatedGPT2ModelWrapper,
    gpt2_cache_invariant_metrics,
)
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


def _run_plain_prefill_decode(
    model,
    prompt_ids: torch.Tensor,
    decode_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run HF reference path with use_cache=True and return logits."""
    with torch.no_grad():
        prefill_out = model(prompt_ids, use_cache=True)
        past = prefill_out.past_key_values
        decode_logits = []
        for step in range(decode_token_ids.shape[1]):
            step_out = model(
                decode_token_ids[:, step : step + 1],
                past_key_values=past,
                use_cache=True,
            )
            decode_logits.append(step_out.logits)
            past = step_out.past_key_values
    return prefill_out.logits, decode_logits


def _run_obfuscated_prefill_decode(
    wrapper: ObfuscatedGPT2ModelWrapper,
    prompt_ids: torch.Tensor,
    decode_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], ObfuscatedGPT2KVCache]:
    """Run obfuscated prefill + multi-step decode and return logits + cache."""
    with torch.no_grad():
        prefill_logits, cache = wrapper.prefill(prompt_ids)
        decode_logits = []
        for step in range(decode_token_ids.shape[1]):
            step_logits, cache = wrapper.decode_step(
                decode_token_ids[:, step : step + 1], cache
            )
            decode_logits.append(step_logits)
    return prefill_logits, decode_logits, cache


# ---------------------------------------------------------------------------
# HF model integrity guards
# ---------------------------------------------------------------------------


def test_gpt2_cache_hf_model_not_replaced() -> None:
    model = _load_model()
    before = type(model.transformer.h[0].attn.c_attn).__name__
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    new_token = torch.randint(0, model.config.vocab_size, (1, 1))
    _, cache = wrapper.prefill(prompt)
    _, _ = wrapper.decode_step(new_token, cache)
    assert type(model.transformer.h[0].attn.c_attn).__name__ == before == "Conv1D"


# ---------------------------------------------------------------------------
# Logits correctness — prefill + multi-step decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "prompt_len", "decode_steps"),
    [(1, 4, 1), (2, 8, 2)],
)
def test_gpt2_cache_use_pad_true_logits(
    batch_size: int, prompt_len: int, decode_steps: int
) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))
    decode_tokens = torch.randint(0, vocab_size, (batch_size, decode_steps))

    plain_prefill, plain_decode_list = _run_plain_prefill_decode(model, prompt, decode_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    rec_prefill, rec_decode_list, cache = _run_obfuscated_prefill_decode(
        wrapper, prompt, decode_tokens
    )

    pm = compute_correctness_metrics(plain_prefill, rec_prefill, atol=1e-4, rtol=1e-4)
    assert pm["allclose"] is True, f"prefill allclose=False, max={pm['max_abs_error']}"
    assert pm["max_abs_error"] < 1e-4
    assert top1_match_rate(plain_prefill, rec_prefill) == 1.0

    for step, (plain_step, rec_step) in enumerate(zip(plain_decode_list, rec_decode_list)):
        sm = compute_correctness_metrics(plain_step, rec_step, atol=1e-4, rtol=1e-4)
        assert sm["allclose"] is True, f"decode step {step} allclose=False, max={sm['max_abs_error']}"
        assert sm["max_abs_error"] < 1e-4, f"decode step {step}: {sm['max_abs_error']}"
        assert top1_match_rate(plain_step, rec_step) == 1.0


@pytest.mark.parametrize(
    ("batch_size", "prompt_len", "decode_steps"),
    [(1, 4, 1), (2, 8, 2)],
)
def test_gpt2_cache_use_pad_false_logits(
    batch_size: int, prompt_len: int, decode_steps: int
) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))
    decode_tokens = torch.randint(0, vocab_size, (batch_size, decode_steps))

    plain_prefill, plain_decode_list = _run_plain_prefill_decode(model, prompt, decode_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=False)
    rec_prefill, rec_decode_list, _cache = _run_obfuscated_prefill_decode(
        wrapper, prompt, decode_tokens
    )

    pm = compute_correctness_metrics(plain_prefill, rec_prefill, atol=1e-4, rtol=1e-4)
    assert pm["allclose"] is True
    assert pm["max_abs_error"] < 1e-4
    assert top1_match_rate(plain_prefill, rec_prefill) == 1.0

    for step, (plain_step, rec_step) in enumerate(zip(plain_decode_list, rec_decode_list)):
        sm = compute_correctness_metrics(plain_step, rec_step, atol=1e-4, rtol=1e-4)
        assert sm["allclose"] is True
        assert sm["max_abs_error"] < 1e-4
        assert top1_match_rate(plain_step, rec_step) == 1.0


# ---------------------------------------------------------------------------
# Cache invariant (K_tilde = K N_K, V_tilde = V N_V)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_gpt2_cache_invariant_after_prefill_and_decode(use_pad: bool) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (2, 8))
    decode_tokens = torch.randint(0, vocab_size, (2, 2))

    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=use_pad)
    _, _, cache = _run_obfuscated_prefill_decode(wrapper, prompt, decode_tokens)

    metrics = gpt2_cache_invariant_metrics(cache, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True, f"cache invariant failed: {metrics}"
    assert metrics["max_key_error"] < 1e-4
    assert metrics["max_value_error"] < 1e-4
    expected_seq = prompt.shape[1] + decode_tokens.shape[1]
    assert cache.seq_len == expected_seq
    for layer in cache.layers:
        assert layer.key_tilde.shape[2] == expected_seq
        assert layer.value_tilde.shape[2] == expected_seq


# ---------------------------------------------------------------------------
# Session masks must be reused across decode steps
# ---------------------------------------------------------------------------


def test_gpt2_cache_masks_unchanged_within_session() -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    with torch.no_grad():
        _, cache = wrapper.prefill(torch.randint(0, vocab_size, (1, 4)))
        prefill_ids = [
            (id(l.key_masks), id(l.value_masks), id(l.key_mask_inverses), id(l.value_mask_inverses))
            for l in cache.layers
        ]
        for _ in range(3):
            _, cache = wrapper.decode_step(torch.randint(0, vocab_size, (1, 1)), cache)
    after_ids = [
        (id(l.key_masks), id(l.value_masks), id(l.key_mask_inverses), id(l.value_mask_inverses))
        for l in cache.layers
    ]
    assert prefill_ids == after_ids, "session masks must be reused across decode steps"


# ---------------------------------------------------------------------------
# Position IDs in decode must not restart from 0
# ---------------------------------------------------------------------------


def test_gpt2_cache_decode_position_id_advances() -> None:
    """Decode logits depend on position; a position-0 bug would mismatch HF."""
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (1, 6))
    decode_tokens = torch.randint(0, vocab_size, (1, 3))

    # If decode reused position id 0 instead of advancing past the prompt, the
    # logits would diverge from HF's reference (which uses past_length onward).
    plain_prefill, plain_decode_list = _run_plain_prefill_decode(model, prompt, decode_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    _, rec_decode_list, cache = _run_obfuscated_prefill_decode(
        wrapper, prompt, decode_tokens
    )
    assert cache.seq_len == prompt.shape[1] + decode_tokens.shape[1]
    for plain_step, rec_step in zip(plain_decode_list, rec_decode_list):
        assert torch.allclose(plain_step, rec_step, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_gpt2_cache_output_shapes() -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    prompt = torch.randint(0, vocab_size, (2, 5))
    with torch.no_grad():
        prefill_logits, cache = wrapper.prefill(prompt)
        decode_logits, cache = wrapper.decode_step(
            torch.randint(0, vocab_size, (2, 1)), cache
        )
    assert prefill_logits.shape == (2, 5, vocab_size)
    assert decode_logits.shape == (2, 1, vocab_size)
    assert cache.seq_len == 6
    for layer in cache.layers:
        assert layer.key_tilde.shape[0] == 2
        assert layer.key_tilde.shape[2] == 6
