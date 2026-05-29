"""Cache correctness metrics."""

from __future__ import annotations

import torch

from pllo.cache.kv_cache import ObfuscatedKVCache, PlainKVCache


def apply_head_masks(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Apply per-head right masks to a tensor shaped [batch, heads, seq, dim]."""
    return torch.einsum("bhsd,hde->bhse", x, masks)


def cache_invariant_metrics(
    plain_cache: PlainKVCache,
    obf_cache: ObfuscatedKVCache,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, float | bool]:
    """Check K_tilde = K N_K and V_tilde = V N_V for all cached layers."""
    max_key_error = 0.0
    max_value_error = 0.0
    key_error_sum = 0.0
    value_error_sum = 0.0
    count = 0
    allclose = True
    for layer_idx, (key, value) in enumerate(zip(plain_cache.keys, plain_cache.values)):
        if key is None or value is None:
            continue
        expected_key = apply_head_masks(key, obf_cache.key_masks[layer_idx])
        expected_value = apply_head_masks(value, obf_cache.value_masks[layer_idx])
        key_diff = (expected_key - obf_cache.keys_tilde[layer_idx]).abs()
        value_diff = (expected_value - obf_cache.values_tilde[layer_idx]).abs()
        max_key_error = max(max_key_error, float(key_diff.max().item()))
        max_value_error = max(max_value_error, float(value_diff.max().item()))
        key_error_sum += float(key_diff.mean().item())
        value_error_sum += float(value_diff.mean().item())
        count += 1
        allclose = allclose and bool(
            torch.allclose(expected_key, obf_cache.keys_tilde[layer_idx], atol=atol, rtol=rtol)
        )
        allclose = allclose and bool(
            torch.allclose(expected_value, obf_cache.values_tilde[layer_idx], atol=atol, rtol=rtol)
        )
    denom = max(count, 1)
    return {
        "max_key_error": max_key_error,
        "max_value_error": max_value_error,
        "mean_key_error": key_error_sum / denom,
        "mean_value_error": value_error_sum / denom,
        "allclose": allclose,
    }
