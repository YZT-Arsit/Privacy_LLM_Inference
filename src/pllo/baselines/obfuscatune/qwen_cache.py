"""Simulated KV cache for ObfuscaTune-Qwen + cache correctness metrics.

Under ObfuscaTune the untrusted side sees the TRUE plaintext K/V (the input
obfuscation cancels at q/k/v projection), so the cached key/value tensors are
**plaintext intermediates exposed outside the TEE**. This cache reflects that
honestly: ``cache_obfuscation = "plain_intermediate_exposed"``. It is NOT an
obfuscated / protected cache -- do not label ``protected_kv_cache = true``.
"""

from __future__ import annotations

from typing import Any

import torch


class QwenObfCache:
    """Per-layer plaintext K/V cache (post-RoPE, pre-GQA-repeat).

    Layout per layer matches HF DynamicCache entries: ``(B, n_kv, seq, head_dim)``.
    """

    cache_obfuscation = "plain_intermediate_exposed"

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers

    def get(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._k[layer], self._v[layer]

    def append(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self._k[layer] is None:
            self._k[layer], self._v[layer] = k_new, v_new
        else:
            self._k[layer] = torch.cat([self._k[layer], k_new], dim=2)
            self._v[layer] = torch.cat([self._v[layer], v_new], dim=2)

    def length(self) -> int:
        return 0 if self._k[0] is None else self._k[0].shape[2]

    def shapes(self) -> list[tuple[int, ...]]:
        return [tuple(k.shape) for k in self._k if k is not None]

    def dtype_device(self) -> dict[str, str]:
        if self._k[0] is None:
            return {"dtype": "none", "device": "none"}
        return {"dtype": str(self._k[0].dtype), "device": str(self._k[0].device)}


def hf_cache_shapes(past_key_values) -> list[tuple[int, ...]]:
    """Extract per-layer key shapes from an HF cache (DynamicCache or legacy)."""
    shapes = []
    if past_key_values is None:
        return shapes
    if hasattr(past_key_values, "layers"):          # newer DynamicCache
        for lyr in past_key_values.layers:
            shapes.append(tuple(lyr.keys.shape))
    else:                                            # legacy tuple-of-tuples
        for entry in past_key_values:
            shapes.append(tuple(entry[0].shape))
    return shapes


def cache_metrics(
    *,
    enabled: bool,
    prefill_cache_len: int,
    decode_cache_len: int,
    obf_shapes: list[tuple[int, ...]],
    ref_shapes: list[tuple[int, ...]],
    cache_obfuscation: str = QwenObfCache.cache_obfuscation,
) -> dict[str, Any]:
    shape_match = (len(obf_shapes) == len(ref_shapes)) and all(
        a == b for a, b in zip(obf_shapes, ref_shapes))
    return {
        "enabled": enabled,
        "prefill_cache_len": prefill_cache_len,
        "decode_cache_len": decode_cache_len,
        "cache_shape_match": bool(shape_match),
        "cache_obfuscation": cache_obfuscation,
        "obf_cache_shapes": obf_shapes,
        "ref_cache_shapes": ref_shapes,
    }


__all__ = ["QwenObfCache", "hf_cache_shapes", "cache_metrics"]
