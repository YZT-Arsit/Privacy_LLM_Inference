"""Tests for Stage 3 greedy generation correctness."""

from __future__ import annotations

import torch
import pytest

from pllo.evaluation import compute_correctness_metrics, top1_match_rate
from pllo.models import ObfuscatedTinyDecoderOnlyTransformer, PlainTinyDecoderOnlyTransformer, TinyTransformerConfig
from pllo.utils.seed import set_seed


def _models(
    batch_size: int,
    prompt_len: int,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> tuple[PlainTinyDecoderOnlyTransformer, ObfuscatedTinyDecoderOnlyTransformer, torch.Tensor]:
    config = TinyTransformerConfig(
        vocab_size=64,
        max_seq_len=prompt_len + max_new_tokens,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dtype=dtype,
    )
    plain = PlainTinyDecoderOnlyTransformer(config)
    obf = ObfuscatedTinyDecoderOnlyTransformer.from_plain(plain, config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    return plain, obf, input_ids


@pytest.mark.parametrize(("batch_size", "prompt_len", "max_new_tokens"), [(1, 4, 4), (2, 8, 4)])
def test_greedy_generation_tokens_match(batch_size: int, prompt_len: int, max_new_tokens: int) -> None:
    set_seed(2201 + batch_size + prompt_len)
    plain, obf, input_ids = _models(batch_size, prompt_len, max_new_tokens, torch.float64)
    assert torch.equal(
        plain.generate_greedy(input_ids, max_new_tokens),
        obf.generate_greedy(input_ids, max_new_tokens),
    )


def test_each_step_logits_top1_match_float64() -> None:
    set_seed(2202)
    plain, obf, input_ids = _models(2, 8, 4, torch.float64)
    plain_logits, plain_cache = plain.prefill(input_ids)
    obf_logits, obf_cache = obf.prefill(input_ids)
    assert compute_correctness_metrics(plain_logits, obf_logits)["allclose"] is True
    assert top1_match_rate(plain_logits, obf_logits) == 1.0
    next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(4):
        plain_logits, plain_cache = plain.decode_step(next_token, plain_cache)
        obf_logits, obf_cache = obf.decode_step(next_token, obf_cache)
        assert compute_correctness_metrics(plain_logits, obf_logits)["allclose"] is True
        assert top1_match_rate(plain_logits, obf_logits) == 1.0
        next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)


def test_float32_generation_top1_and_tokens_match() -> None:
    set_seed(2203)
    plain, obf, input_ids = _models(2, 8, 4, torch.float32)
    assert torch.equal(plain.generate_greedy(input_ids, 4), obf.generate_greedy(input_ids, 4))
    plain_logits, _ = plain.prefill(input_ids)
    obf_logits, _ = obf.prefill(input_ids)
    assert top1_match_rate(plain_logits, obf_logits) == 1.0
    assert compute_correctness_metrics(plain_logits, obf_logits, atol=1e-4, rtol=1e-4)["allclose"] is True
