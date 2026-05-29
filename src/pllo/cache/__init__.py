"""KV cache utilities."""

from pllo.cache.cache_state import apply_head_masks, cache_invariant_metrics
from pllo.cache.kv_cache import ObfuscatedKVCache, PlainKVCache

__all__ = ["ObfuscatedKVCache", "PlainKVCache", "apply_head_masks", "cache_invariant_metrics"]
