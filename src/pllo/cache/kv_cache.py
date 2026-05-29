"""KV cache containers for Stage 3 prefill/decode."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PlainKVCache:
    """Plain K/V cache with one tensor per layer."""

    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def empty(cls, num_layers: int) -> "PlainKVCache":
        """Create an empty cache with placeholders for all layers."""
        return cls(keys=[None] * num_layers, values=[None] * num_layers)  # type: ignore[list-item]

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one or more sequence positions to a layer cache."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for a layer."""
        key = self.keys[layer_idx]
        return 0 if key is None else int(key.shape[2])


@dataclass
class ObfuscatedKVCache:
    """Obfuscated K/V cache plus trusted-side mask metadata.

    Mask inverses are trusted-side information. They are stored here so the
    simulated TEE and tests can verify invariants, but they must not be passed
    to the untrusted executor.
    """

    keys_tilde: list[torch.Tensor] = field(default_factory=list)
    values_tilde: list[torch.Tensor] = field(default_factory=list)
    key_masks: list[torch.Tensor] = field(default_factory=list)
    value_masks: list[torch.Tensor] = field(default_factory=list)
    key_mask_inverses: list[torch.Tensor] | None = None
    value_mask_inverses: list[torch.Tensor] | None = None

    @classmethod
    def empty(
        cls,
        num_layers: int,
        key_masks: list[torch.Tensor],
        value_masks: list[torch.Tensor],
        key_mask_inverses: list[torch.Tensor] | None,
        value_mask_inverses: list[torch.Tensor] | None,
    ) -> "ObfuscatedKVCache":
        """Create an empty obfuscated cache with fixed per-layer masks."""
        return cls(
            keys_tilde=[None] * num_layers,  # type: ignore[list-item]
            values_tilde=[None] * num_layers,  # type: ignore[list-item]
            key_masks=key_masks,
            value_masks=value_masks,
            key_mask_inverses=key_mask_inverses,
            value_mask_inverses=value_mask_inverses,
        )

    def append(
        self,
        layer_idx: int,
        key_tilde: torch.Tensor,
        value_tilde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append obfuscated key/value tensors to a layer cache."""
        if self.keys_tilde[layer_idx] is None:
            self.keys_tilde[layer_idx] = key_tilde
            self.values_tilde[layer_idx] = value_tilde
        else:
            self.keys_tilde[layer_idx] = torch.cat([self.keys_tilde[layer_idx], key_tilde], dim=2)
            self.values_tilde[layer_idx] = torch.cat([self.values_tilde[layer_idx], value_tilde], dim=2)
        return self.keys_tilde[layer_idx], self.values_tilde[layer_idx]

    def length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for a layer."""
        key = self.keys_tilde[layer_idx]
        return 0 if key is None else int(key.shape[2])
