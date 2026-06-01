"""Stage 6.4c — KV cache for modern decoder-only model wrapper.

Holds per-layer masked ``K_tilde`` / ``V_tilde`` plus the per-kv-head mask
material (``N_K`` / ``N_V`` / ``N_V_inv``) needed to (a) decode the next
token with the same mask space and (b) recover plain attention output on
the trusted side. Per-token decode appends ``k_new @ N_K`` and
``v_new @ N_V`` along the sequence axis — ``N_K`` and ``N_V`` are
constant for the lifetime of one generation session so the append is a
plain concat.

GQA / MQA compatibility: ``K_tilde`` / ``V_tilde`` are stored with
``num_kv_heads`` head dimension. ``repeat_kv`` is applied lazily inside
the attention math (post-mask), so the cache itself stores the minimal
layout. ``num_query_heads`` is recorded for downstream invariants but is
NOT used to shape the cache.

JSON-safe summary: the cache exposes a ``summary_dict()`` that publishes
shapes, layer counts, attention variant and mask fingerprints; never the
mask material, never the raw K/V tensors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch


def _fingerprint(t: torch.Tensor) -> str:
    """SHA-256 prefix over a tensor's float32 bytes (16 hex chars)."""
    buf = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-layer cache
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderLayerKVCache:
    """One layer's masked KV cache.

    ``key_tilde`` shape: ``[batch, num_kv_heads, seq, head_dim]`` — already
    post-RoPE and post-N_K masked. Same layout for ``value_tilde``.

    ``n_k_stack`` / ``n_v_stack`` are ``[num_kv_heads, head_dim, head_dim]``
    dense per-kv-head masks used so the decoder can mask new tokens the
    same way prefill did. ``n_v_inv_stack`` is the trusted-side inverse
    cached for the attention output recovery.

    ``key_plain`` / ``value_plain`` are optional — populated only when the
    caller passes ``debug_keep_plain=True`` (test / probe mode). Production
    must keep them ``None``.
    """

    key_tilde: torch.Tensor
    value_tilde: torch.Tensor
    n_k_stack: torch.Tensor
    n_v_stack: torch.Tensor
    n_v_inv_stack: torch.Tensor
    seq_len: int
    num_kv_heads: int
    head_dim: int
    layer_index: int
    cache_status: str
    key_plain: torch.Tensor | None = None
    value_plain: torch.Tensor | None = None

    # ------------------------------------------------------------------ ops
    def append(
        self,
        key_tilde_new: torch.Tensor,
        value_tilde_new: torch.Tensor,
        key_plain_new: torch.Tensor | None = None,
        value_plain_new: torch.Tensor | None = None,
    ) -> None:
        """Append per-token masked K/V along the sequence axis.

        Shapes:
          ``key_tilde_new`` / ``value_tilde_new``: ``[batch, num_kv_heads, S_new, head_dim]``.

        The append is a plain concat — ``N_K`` / ``N_V`` were applied
        upstream by the caller using the cached mask material.
        """
        if key_tilde_new.shape[1] != self.num_kv_heads:
            raise ValueError(
                f"key_tilde_new num_kv_heads {key_tilde_new.shape[1]}"
                f" != cached {self.num_kv_heads}"
            )
        if key_tilde_new.shape[-1] != self.head_dim:
            raise ValueError(
                f"key_tilde_new head_dim {key_tilde_new.shape[-1]}"
                f" != cached {self.head_dim}"
            )
        self.key_tilde = torch.cat([self.key_tilde, key_tilde_new], dim=-2)
        self.value_tilde = torch.cat([self.value_tilde, value_tilde_new], dim=-2)
        if (
            self.key_plain is not None
            and key_plain_new is not None
            and value_plain_new is not None
        ):
            self.key_plain = torch.cat([self.key_plain, key_plain_new], dim=-2)
            self.value_plain = torch.cat([self.value_plain, value_plain_new], dim=-2)
        else:
            # Drop debug shadows once the cache outgrows the debug seed.
            self.key_plain = None
            self.value_plain = None
        self.seq_len = int(self.key_tilde.shape[-2])

    def summary_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "seq_len": int(self.seq_len),
            "num_kv_heads": int(self.num_kv_heads),
            "head_dim": int(self.head_dim),
            "key_tilde_shape": list(self.key_tilde.shape),
            "value_tilde_shape": list(self.value_tilde.shape),
            "n_k_stack_fingerprint": _fingerprint(self.n_k_stack),
            "n_v_stack_fingerprint": _fingerprint(self.n_v_stack),
            "cache_status": self.cache_status,
            "debug_plain_present": (
                self.key_plain is not None and self.value_plain is not None
            ),
        }


# ---------------------------------------------------------------------------
# Whole-model cache
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderKVCache:
    layers: list[ModernDecoderLayerKVCache]
    batch_size: int
    total_seq_len: int
    cache_type: str = "autoregressive_kv_cache"
    attention_variant: str = "unknown"   # mha / gqa / mqa

    def append_layer(
        self,
        layer_index: int,
        key_tilde_new: torch.Tensor,
        value_tilde_new: torch.Tensor,
        key_plain_new: torch.Tensor | None = None,
        value_plain_new: torch.Tensor | None = None,
    ) -> None:
        layer = self.layers[layer_index]
        layer.append(
            key_tilde_new, value_tilde_new,
            key_plain_new=key_plain_new,
            value_plain_new=value_plain_new,
        )
        # After all layers append for one token the model wrapper bumps
        # total_seq_len; we don't auto-bump here to avoid double counting
        # when the wrapper iterates layers.

    def bump_seq_len(self, n_new_tokens: int) -> None:
        self.total_seq_len = int(self.total_seq_len + n_new_tokens)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "cache_type": self.cache_type,
            "attention_variant": self.attention_variant,
            "batch_size": int(self.batch_size),
            "total_seq_len": int(self.total_seq_len),
            "num_layers": int(len(self.layers)),
            "layers": [l.summary_dict() for l in self.layers],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def init_empty_modern_decoder_kv_cache(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    attention_variant: str,
    dtype: torch.dtype,
    device: torch.device,
) -> ModernDecoderKVCache:
    """Build an empty cache shell (no entries yet) for one decode session.

    Mask material is filled in per-layer by the model wrapper during
    prefill; this helper just allocates layer placeholders.
    """
    layers: list[ModernDecoderLayerKVCache] = []
    for i in range(num_layers):
        layers.append(
            ModernDecoderLayerKVCache(
                key_tilde=torch.empty(
                    batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device,
                ),
                value_tilde=torch.empty(
                    batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device,
                ),
                n_k_stack=torch.empty(
                    num_kv_heads, head_dim, head_dim, dtype=dtype, device=device,
                ),
                n_v_stack=torch.empty(
                    num_kv_heads, head_dim, head_dim, dtype=dtype, device=device,
                ),
                n_v_inv_stack=torch.empty(
                    num_kv_heads, head_dim, head_dim, dtype=dtype, device=device,
                ),
                seq_len=0,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_index=i,
                cache_status="initialised_empty",
            )
        )
    return ModernDecoderKVCache(
        layers=layers,
        batch_size=batch_size,
        total_seq_len=0,
        attention_variant=attention_variant,
    )


__all__ = [
    "ModernDecoderKVCache",
    "ModernDecoderLayerKVCache",
    "init_empty_modern_decoder_kv_cache",
]
