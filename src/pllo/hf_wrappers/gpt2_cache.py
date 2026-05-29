"""KV cache data structures for Stage 4.8 GPT-2 prefill/decode correctness."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ObfuscatedGPT2LayerCache:
    """Per-layer cache for one obfuscated GPT-2 attention block.

    The ``*_tilde`` tensors are the GPU-visible obfuscated cache entries with
    shape ``[batch, num_heads, seq, head_dim]``. The ``*_masks`` and inverse
    tensors are trusted-side state that lives logically inside the simulated
    TEE (they must not be exposed to the untrusted executor). The
    ``*_plain_for_test`` tensors are only populated for correctness
    invariant checks and would not exist in a real deployment.
    """

    key_tilde: torch.Tensor
    value_tilde: torch.Tensor
    key_masks: torch.Tensor
    key_mask_inverses: torch.Tensor
    value_masks: torch.Tensor
    value_mask_inverses: torch.Tensor
    key_plain_for_test: torch.Tensor | None = None
    value_plain_for_test: torch.Tensor | None = None


@dataclass
class ObfuscatedGPT2KVCache:
    """Session-level multi-layer obfuscated KV cache.

    ``layers[i]`` is the per-layer cache for transformer block ``i``.
    ``seq_len`` is the number of tokens cached so far (used to derive the
    next decode position id).
    """

    layers: list[ObfuscatedGPT2LayerCache] = field(default_factory=list)
    seq_len: int = 0


def gpt2_cache_invariant_metrics(
    cache: ObfuscatedGPT2KVCache,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, float | bool]:
    """Verify ``K_tilde = K N_K`` and ``V_tilde = V N_V`` per head per layer."""
    max_key_error = 0.0
    max_value_error = 0.0
    key_error_sum = 0.0
    value_error_sum = 0.0
    count = 0
    allclose = True
    for layer in cache.layers:
        if layer.key_plain_for_test is None or layer.value_plain_for_test is None:
            continue
        expected_key = torch.einsum(
            "bhsd,hde->bhse", layer.key_plain_for_test, layer.key_masks
        )
        expected_value = torch.einsum(
            "bhsd,hde->bhse", layer.value_plain_for_test, layer.value_masks
        )
        key_diff = (expected_key - layer.key_tilde).abs()
        value_diff = (expected_value - layer.value_tilde).abs()
        max_key_error = max(max_key_error, float(key_diff.max().item()))
        max_value_error = max(max_value_error, float(value_diff.max().item()))
        key_error_sum += float(key_diff.mean().item())
        value_error_sum += float(value_diff.mean().item())
        count += 1
        allclose = allclose and bool(
            torch.allclose(expected_key, layer.key_tilde, atol=atol, rtol=rtol)
        )
        allclose = allclose and bool(
            torch.allclose(expected_value, layer.value_tilde, atol=atol, rtol=rtol)
        )
    denom = max(count, 1)
    return {
        "max_key_error": max_key_error,
        "max_value_error": max_value_error,
        "mean_key_error": key_error_sum / denom,
        "mean_value_error": value_error_sum / denom,
        "allclose": allclose,
    }
