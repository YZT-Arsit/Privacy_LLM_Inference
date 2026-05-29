"""Tests for Stage 3 decode attention and cache invariants."""

from __future__ import annotations

import torch

from pllo.cache import cache_invariant_metrics
from pllo.evaluation import compute_correctness_metrics
from pllo.models import ObfuscatedTinyDecoderOnlyTransformer, PlainTinyDecoderOnlyTransformer, TinyTransformerConfig
from pllo.ops.attention import qk_head_mask_constraint_error
from pllo.utils.seed import set_seed


def _models() -> tuple[PlainTinyDecoderOnlyTransformer, ObfuscatedTinyDecoderOnlyTransformer, torch.Tensor]:
    config = TinyTransformerConfig(
        vocab_size=64,
        max_seq_len=12,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dtype=torch.float64,
    )
    plain = PlainTinyDecoderOnlyTransformer(config)
    obf = ObfuscatedTinyDecoderOnlyTransformer.from_plain(plain, config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    return plain, obf, input_ids


def test_prefill_logits_align_with_full_forward() -> None:
    set_seed(2101)
    plain, _, input_ids = _models()
    logits, _ = plain.prefill(input_ids)
    assert torch.allclose(logits, plain(input_ids), atol=1e-10, rtol=1e-10)


def test_decode_step_matches_full_sequence_last_position() -> None:
    set_seed(2102)
    plain, _, input_ids = _models()
    logits, cache = plain.prefill(input_ids)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_logits, _ = plain.decode_step(next_token, cache)
    full_logits = plain(torch.cat([input_ids, next_token], dim=1))
    assert torch.allclose(decode_logits[:, -1, :], full_logits[:, -1, :], atol=1e-10, rtol=1e-10)


def test_obfuscated_decode_logits_and_cache_invariant() -> None:
    set_seed(2103)
    plain, obf, input_ids = _models()
    plain_logits, plain_cache = plain.prefill(input_ids)
    obf_logits, obf_cache = obf.prefill(input_ids)
    assert compute_correctness_metrics(plain_logits, obf_logits)["allclose"] is True
    assert cache_invariant_metrics(plain_cache, obf_cache)["allclose"] is True
    next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    plain_logits, plain_cache = plain.decode_step(next_token, plain_cache)
    obf_logits, obf_cache = obf.decode_step(next_token, obf_cache)
    assert compute_correctness_metrics(plain_logits, obf_logits)["allclose"] is True
    assert cache_invariant_metrics(plain_cache, obf_cache)["allclose"] is True


def test_qk_cache_mask_constraint_still_holds() -> None:
    set_seed(2104)
    _, obf, input_ids = _models()
    _, cache = obf.prefill(input_ids)
    for key_masks, key_inverses in zip(cache.key_masks, cache.key_mask_inverses):
        assert qk_head_mask_constraint_error(key_masks, key_inverses) < 1e-10
