"""Stage 6.4c — prefill / decode_step / greedy generation tests."""

from __future__ import annotations

import pytest
import torch

from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    ModernDecoderModelWeights,
    ObfuscatedModernDecoderModelWrapper,
    plain_decode_step,
    plain_prefill,
)


@pytest.fixture
def synthetic_weights() -> ModernDecoderModelWeights:
    torch.manual_seed(0)
    return ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_layers=2, seed=0,
    )


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 32, (1, 5))


def _wrapper(synthetic_weights, *, bundle, use_pad):
    return ObfuscatedModernDecoderModelWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle=bundle,
        use_pad=use_pad,
    )


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def test_prefill_top1_match(synthetic_weights, input_ids) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True)
    out = w.prefill(input_ids)
    assert out["report"]["logits_metrics"]["allclose"] is True
    assert out["report"]["logits_metrics"]["top1_match_rate"] == 1.0


def test_prefill_cache_has_all_layers(synthetic_weights, input_ids) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle="fresh_perm_only", use_pad=False)
    out = w.prefill(input_ids)
    cache = out["kv_cache"]
    assert len(cache.layers) == 2
    assert cache.total_seq_len == int(input_ids.shape[-1])
    for layer in cache.layers:
        assert layer.seq_len == int(input_ids.shape[-1])
        assert layer.cache_status == "filled_after_prefill"


# ---------------------------------------------------------------------------
# Decode step
# ---------------------------------------------------------------------------


def test_decode_step_updates_cache_length(synthetic_weights, input_ids) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True)
    pf = w.prefill(input_ids)
    pre_len = pf["kv_cache"].layers[0].seq_len
    next_t = pf["logits_recovered"][:, -1, :].argmax(dim=-1)
    step = w.decode_step(
        next_t, pf["kv_cache"],
        position=int(input_ids.shape[-1]),
        plain_layer_caches=pf["plain_layer_caches"],
    )
    assert step["kv_cache"].layers[0].seq_len == pre_len + 1
    assert step["kv_cache"].total_seq_len == int(input_ids.shape[-1]) + 1


def test_decode_step_next_token_top1_match(synthetic_weights, input_ids) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True)
    pf = w.prefill(input_ids)
    next_t = pf["logits_recovered"][:, -1, :].argmax(dim=-1)
    step = w.decode_step(
        next_t, pf["kv_cache"],
        position=int(input_ids.shape[-1]),
        plain_layer_caches=pf["plain_layer_caches"],
    )
    metrics = step["report"]["logits_metrics"]
    assert metrics is not None
    assert metrics["allclose"] is True
    assert metrics["top1_match_rate"] == 1.0


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bundle", [
    "fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad",
])
@pytest.mark.parametrize("use_pad", [False, True])
def test_greedy_generation_sequence_match(
    synthetic_weights, input_ids, bundle, use_pad,
) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle=bundle, use_pad=use_pad)
    g = w.greedy_generate(input_ids, max_new_tokens=3)
    assert g["report"]["sequence_exact_match"] is True
    assert g["report"]["token_match_rate"] == 1.0


# ---------------------------------------------------------------------------
# RoPE position metadata
# ---------------------------------------------------------------------------


def test_rope_position_metadata_correct(synthetic_weights, input_ids) -> None:
    torch.manual_seed(7)
    w = _wrapper(synthetic_weights, bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True)
    pf = w.prefill(input_ids)
    next_t = pf["logits_recovered"][:, -1, :].argmax(dim=-1)
    step = w.decode_step(
        next_t, pf["kv_cache"],
        position=int(input_ids.shape[-1]),
        plain_layer_caches=pf["plain_layer_caches"],
    )
    assert step["report"]["rope_position_used"] == int(input_ids.shape[-1])
    assert step["report"]["rope_position_increment"] is True


# ---------------------------------------------------------------------------
# GQA cache metadata
# ---------------------------------------------------------------------------


def test_gqa_cache_metadata_correct() -> None:
    torch.manual_seed(0)
    weights = ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_layers=2, seed=0,
    )
    w = ObfuscatedModernDecoderModelWrapper(
        weights, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
    )
    input_ids = torch.randint(0, 32, (1, 5))
    torch.manual_seed(7)
    pf = w.prefill(input_ids)
    assert pf["kv_cache"].attention_variant == "gqa"
    for layer in pf["kv_cache"].layers:
        assert layer.num_kv_heads == 2
        assert layer.head_dim == 8


def test_mha_cache_metadata_correct() -> None:
    torch.manual_seed(0)
    weights = ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
        num_layers=2, seed=0,
    )
    w = ObfuscatedModernDecoderModelWrapper(
        weights, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
    )
    input_ids = torch.randint(0, 32, (1, 4))
    torch.manual_seed(7)
    pf = w.prefill(input_ids)
    assert pf["kv_cache"].attention_variant == "mha"


def test_mqa_cache_metadata_correct() -> None:
    torch.manual_seed(0)
    weights = ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8,
        num_layers=2, seed=0,
    )
    w = ObfuscatedModernDecoderModelWrapper(
        weights, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
    )
    input_ids = torch.randint(0, 32, (1, 4))
    torch.manual_seed(7)
    pf = w.prefill(input_ids)
    assert pf["kv_cache"].attention_variant == "mqa"
    for layer in pf["kv_cache"].layers:
        assert layer.num_kv_heads == 1
