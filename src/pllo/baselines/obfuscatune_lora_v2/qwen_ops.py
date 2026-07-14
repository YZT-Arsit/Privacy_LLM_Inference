"""Qwen2.5 trusted operators and cache-aware GQA attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .runtime import ExternalLoRALinear, TrustedRuntime


@dataclass
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key = key if self.key is None else torch.cat((self.key, key), dim=-2)
        self.value = value if self.value is None else torch.cat((self.value, value), dim=-2)

    @property
    def length(self) -> int:
        return 0 if self.key is None else int(self.key.shape[-2])


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups <= 0:
        raise ValueError("groups must be positive")
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


def causal_mask(query_length: int, key_length: int, *, offset: int, device) -> torch.Tensor:
    q = torch.arange(query_length, device=device)[:, None] + offset
    k = torch.arange(key_length, device=device)[None, :]
    return k <= q


def gqa_attention(
    hidden: torch.Tensor,
    *,
    q_proj: ExternalLoRALinear,
    k_proj: ExternalLoRALinear,
    v_proj: ExternalLoRALinear,
    o_proj: ExternalLoRALinear,
    num_heads: int,
    num_kv_heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    trusted: TrustedRuntime,
    cache: KVCache | None = None,
) -> tuple[torch.Tensor, KVCache, torch.Tensor]:
    b, s, hidden_size = hidden.shape
    if hidden_size % num_heads or num_heads % num_kv_heads:
        raise ValueError("invalid GQA dimensions")
    head_dim = hidden_size // num_heads
    q = q_proj(hidden).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = k_proj(hidden).view(b, s, num_kv_heads, head_dim).transpose(1, 2)
    v = v_proj(hidden).view(b, s, num_kv_heads, head_dim).transpose(1, 2)
    q, k = trusted.rope(q, k, cos, sin)
    cache = cache or KVCache()
    offset = cache.length
    cache.append(k, v)
    kr = repeat_kv(cache.key, num_heads // num_kv_heads)
    vr = repeat_kv(cache.value, num_heads // num_kv_heads)
    scores = q @ kr.transpose(-1, -2) / (head_dim**0.5)
    mask = causal_mask(s, cache.length, offset=offset, device=hidden.device)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = trusted.softmax(scores)
    context = (probs @ vr).transpose(1, 2).reshape(b, s, hidden_size)
    return o_proj(context), cache, probs


def swiglu_mlp(
    hidden: torch.Tensor,
    *,
    gate_proj: ExternalLoRALinear,
    up_proj: ExternalLoRALinear,
    down_proj: ExternalLoRALinear,
    trusted: TrustedRuntime,
) -> torch.Tensor:
    return down_proj(trusted.swiglu(gate_proj(hidden), up_proj(hidden)))
