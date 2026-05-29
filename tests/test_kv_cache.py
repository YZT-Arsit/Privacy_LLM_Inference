"""Tests for Stage 3 KV cache containers and invariants."""

from __future__ import annotations

import torch

from pllo.cache import ObfuscatedKVCache, PlainKVCache, apply_head_masks, cache_invariant_metrics
from pllo.ops.attention import generate_head_masks
from pllo.utils.seed import set_seed


def test_empty_cache_initialization() -> None:
    cache = PlainKVCache.empty(num_layers=2)
    assert cache.length() == 0
    assert len(cache.keys) == 2


def test_append_single_step_kv() -> None:
    cache = PlainKVCache.empty(num_layers=1)
    key = torch.randn(2, 4, 1, 8, dtype=torch.float64)
    value = torch.randn(2, 4, 1, 8, dtype=torch.float64)
    cache.append(0, key, value)
    assert cache.length() == 1
    assert cache.keys[0].shape == (2, 4, 1, 8)


def test_append_multi_step_kv() -> None:
    cache = PlainKVCache.empty(num_layers=1)
    for _ in range(3):
        cache.append(0, torch.randn(2, 4, 1, 8), torch.randn(2, 4, 1, 8))
    assert cache.length() == 3
    assert cache.values[0].shape == (2, 4, 3, 8)


def test_obfuscated_cache_shape_and_invariant() -> None:
    set_seed(2001)
    key = torch.randn(2, 4, 3, 8, dtype=torch.float64)
    value = torch.randn(2, 4, 3, 8, dtype=torch.float64)
    key_masks, key_inverses = generate_head_masks(4, 8, torch.float64, "cpu")
    value_masks, value_inverses = generate_head_masks(4, 8, torch.float64, "cpu")
    plain = PlainKVCache.empty(1)
    plain.append(0, key, value)
    obf = ObfuscatedKVCache.empty(1, [key_masks], [value_masks], [key_inverses], [value_inverses])
    obf.append(0, apply_head_masks(key, key_masks), apply_head_masks(value, value_masks))
    metrics = cache_invariant_metrics(plain, obf)
    assert metrics["allclose"] is True
    assert metrics["max_key_error"] < 1e-10
    assert metrics["max_value_error"] < 1e-10
