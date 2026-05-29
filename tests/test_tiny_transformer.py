"""End-to-end tests for the Stage 2 tiny Transformer."""

from __future__ import annotations

import torch
import pytest

from pllo.evaluation import compute_correctness_metrics
from pllo.models import (
    ObfuscatedTinyDecoderOnlyTransformer,
    PlainTinyDecoderOnlyTransformer,
    TinyTransformerConfig,
)
from pllo.utils.seed import set_seed


def _make_models(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> tuple[PlainTinyDecoderOnlyTransformer, ObfuscatedTinyDecoderOnlyTransformer, torch.Tensor]:
    config = TinyTransformerConfig(
        vocab_size=64,
        max_seq_len=seq_len,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dtype=dtype,
    )
    plain = PlainTinyDecoderOnlyTransformer(config)
    obfuscated = ObfuscatedTinyDecoderOnlyTransformer.from_plain(plain, config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    return plain, obfuscated, input_ids


def test_plain_model_forward_shape() -> None:
    set_seed(1301)
    plain, _, input_ids = _make_models(batch_size=1, seq_len=8, dtype=torch.float64)
    assert plain(input_ids).shape == (1, 8, 64)


def test_obfuscated_model_forward_shape() -> None:
    set_seed(1302)
    _, obfuscated, input_ids = _make_models(batch_size=1, seq_len=8, dtype=torch.float64)
    assert obfuscated(input_ids).shape == (1, 8, 64)


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 8), (2, 16)])
def test_tiny_transformer_float64_logits_match(batch_size: int, seq_len: int) -> None:
    set_seed(1303 + batch_size + seq_len)
    plain, obfuscated, input_ids = _make_models(batch_size=batch_size, seq_len=seq_len, dtype=torch.float64)
    reference = plain(input_ids)
    candidate = obfuscated(input_ids)
    metrics = compute_correctness_metrics(reference, candidate)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-8
    assert metrics["relative_l2_error"] < 1e-8
    assert float((reference.argmax(dim=-1) == candidate.argmax(dim=-1)).double().mean().item()) == 1.0


def test_tiny_transformer_float32_logits_match_with_wider_tolerance() -> None:
    set_seed(1304)
    plain, obfuscated, input_ids = _make_models(batch_size=2, seq_len=8, dtype=torch.float32)
    reference = plain(input_ids)
    candidate = obfuscated(input_ids)
    metrics = compute_correctness_metrics(reference, candidate, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-4
    assert float((reference.argmax(dim=-1) == candidate.argmax(dim=-1)).double().mean().item()) == 1.0
