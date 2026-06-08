"""Tiny synthetic modern-decoder LLM used by the padded full-generation
correctness wrapper.

This module is the *plain reference* implementation. It is deliberately
small (vocab=97, hidden=64, 2 layers, GQA 4/2 heads) so that CPU-only
float64 correctness verification stays fast.

Architecture (LLaMA / Qwen-style, simplified):

    embed_tokens
      |
    repeat (num_layers):
      residual = h
      x = RMSNorm(h)
      Q = x W_q;  K = x W_k;  V = x W_v
      (reshape into per-head, apply RoPE to Q,K, optionally append KV cache)
      repeat_kv for GQA  ->  causal-mask softmax  ->  attn_out
      attn_out = attn_out W_o
      h = residual + attn_out
      residual = h
      x = RMSNorm(h)
      gate = silu(x W_gate);  up = x W_up
      h = residual + (gate * up) W_down
    h = RMSNorm(h)
    logits = h W_lm   (W_lm = embed_tokens.weight.T  -- *not* tied here)

The plain reference has no obfuscation logic in it. The padded masked
wrapper in :mod:`pllo.wrappers.padded_modern_decoder_generation_wrapper`
re-implements the same forward path with boundary pads and per-head
right-masks, then verifies the recovered output matches this reference
to float64 precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TinyModernDecoderConfig:
    """Hyperparameters for the tiny CPU-only modern decoder."""

    vocab_size: int = 97
    hidden_size: int = 64
    intermediate_size: int = 176
    num_layers: int = 2
    num_query_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 128
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_query_heads

    @property
    def group_size(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.hidden_size % self.num_query_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_query_heads; "
                f"got hidden_size={self.hidden_size}, "
                f"num_query_heads={self.num_query_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got head_dim={self.head_dim}"
            )
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads; "
                f"got num_query_heads={self.num_query_heads}, "
                f"num_kv_heads={self.num_kv_heads}"
            )
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, got "
                f"{self.max_position_embeddings}"
            )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _rope_cos_sin(
    head_dim: int,
    positions: torch.Tensor,
    base: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LLaMA / Qwen-style RoPE frequency table for the given absolute positions.

    Returns ``(cos, sin)`` each of shape ``[len(positions), head_dim]``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    theta = positions.to(torch.float64).unsqueeze(-1) * inv_freq.unsqueeze(0)  # [S, half]
    cos = theta.cos().repeat(1, 2).to(dtype)
    sin = theta.sin().repeat(1, 2).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float
) -> torch.Tensor:
    """Apply LLaMA / Qwen rotate-half RoPE.

    ``x`` is ``[B, H, S, D]`` with even ``D``. ``positions`` is a 1-D
    LongTensor of length ``S`` giving the absolute position of each token
    along the sequence axis.
    """
    head_dim = x.shape[-1]
    cos, sin = _rope_cos_sin(head_dim, positions, base, x.dtype, x.device)
    # Broadcast cos/sin against [B, H, S, D].
    return (x * cos) + (_rotate_half(x) * sin)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """LLaMA-style RMSNorm: ``weight * x / sqrt(mean(x^2) + eps)``.

    ``weight`` is the learned affine scale, shape ``[hidden_size]``. Run
    in float64 internally to keep the masked-vs-plain comparison at
    machine precision.
    """
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    norm = x * torch.rsqrt(mean_sq + eps)
    return norm * weight


def repeat_kv(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat each KV head ``group_size`` times along the head axis."""
    if group_size == 1:
        return x
    b, h, s, d = x.shape
    return (
        x.unsqueeze(2)
        .expand(b, h, group_size, s, d)
        .reshape(b, h * group_size, s, d)
    )


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_len: int,
) -> torch.Tensor:
    """Plain causal multi-head attention over already-repeated K/V.

    ``q`` is ``[B, H, S_new, D]`` (new-token queries).
    ``k`` and ``v`` are ``[B, H, S_total, D]`` (full cached + new).
    ``past_len`` is the number of cached tokens BEFORE the new ones.

    Returns ``[B, H, S_new, D]``.
    """
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, S_new, S_total]
    # Causal mask: query position p (absolute = past_len + p) can attend to
    # key positions 0..past_len+p inclusive.
    s_new = q.shape[-2]
    s_total = k.shape[-2]
    device = q.device
    q_abs = torch.arange(s_new, device=device) + past_len
    k_abs = torch.arange(s_total, device=device)
    mask = k_abs.unsqueeze(0) > q_abs.unsqueeze(-1)   # [S_new, S_total]
    scores = scores.masked_fill(mask, float("-inf"))
    # softmax with float64 input is already stable; subtract max for safety.
    scores = scores - scores.amax(dim=-1, keepdim=True)
    probs = scores.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs @ v


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class TinyModernDecoderAttention(nn.Module):
    def __init__(self, cfg: TinyModernDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(
            cfg.hidden_size, cfg.num_query_heads * cfg.head_dim, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )
        self.o_proj = nn.Linear(
            cfg.num_query_heads * cfg.head_dim, cfg.hidden_size, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, cfg.num_query_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, positions, cfg.rope_base)
        k = apply_rope(k, positions, cfg.rope_base)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        new_past = (k, v)

        k_rep = repeat_kv(k, cfg.group_size)
        v_rep = repeat_kv(v, cfg.group_size)

        out = causal_attention(q, k_rep, v_rep, past_len)
        out = out.transpose(1, 2).reshape(b, s, cfg.num_query_heads * cfg.head_dim)
        out = self.o_proj(out)
        return out, new_past


class TinyModernDecoderMLP(nn.Module):
    def __init__(self, cfg: TinyModernDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.up_proj = nn.Linear(
            cfg.hidden_size, cfg.intermediate_size, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )
        self.gate_proj = nn.Linear(
            cfg.hidden_size, cfg.intermediate_size, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.hidden_size, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.up_proj(x)
        b = self.gate_proj(x)
        return self.down_proj(a * torch.nn.functional.silu(b))


class TinyModernDecoderLayer(nn.Module):
    def __init__(self, cfg: TinyModernDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_norm_weight = nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=cfg.dtype, device=cfg.device)
        )
        self.post_attn_norm_weight = nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=cfg.dtype, device=cfg.device)
        )
        self.attn = TinyModernDecoderAttention(cfg)
        self.mlp = TinyModernDecoderMLP(cfg)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = h
        x = rmsnorm(h, self.input_norm_weight, self.cfg.rms_norm_eps)
        attn_out, new_past = self.attn(x, positions, past_kv)
        h = residual + attn_out
        residual = h
        x = rmsnorm(h, self.post_attn_norm_weight, self.cfg.rms_norm_eps)
        h = residual + self.mlp(x)
        return h, new_past


# ---------------------------------------------------------------------------
# Top-level plain model
# ---------------------------------------------------------------------------


class TinyModernDecoderForCausalLM(nn.Module):
    """Plain CPU-only reference of the tiny modern decoder."""

    def __init__(self, cfg: TinyModernDecoderConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, dtype=cfg.dtype, device=cfg.device
        )
        self.layers = nn.ModuleList(
            [TinyModernDecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.final_norm_weight = nn.Parameter(
            torch.ones(cfg.hidden_size, dtype=cfg.dtype, device=cfg.device)
        )
        self.lm_head = nn.Linear(
            cfg.hidden_size, cfg.vocab_size, bias=False,
            dtype=cfg.dtype, device=cfg.device,
        )

    def init_random_weights(self, generator: torch.Generator) -> None:
        """Initialise every parameter from a fixed-seed Generator.

        The training-mode default initialiser is non-deterministic across
        torch versions; we use this hook in tests + experiments so the
        plain-vs-masked numbers are reproducible.
        """
        with torch.no_grad():
            self.embed_tokens.weight.normal_(generator=generator)
            self.final_norm_weight.uniform_(0.9, 1.1, generator=generator)
            self.lm_head.weight.normal_(generator=generator)
            self.lm_head.weight.mul_(0.02)
            for layer in self.layers:
                layer.input_norm_weight.uniform_(0.9, 1.1, generator=generator)
                layer.post_attn_norm_weight.uniform_(0.9, 1.1, generator=generator)
                for lin in (
                    layer.attn.q_proj,
                    layer.attn.k_proj,
                    layer.attn.v_proj,
                    layer.attn.o_proj,
                    layer.mlp.up_proj,
                    layer.mlp.gate_proj,
                    layer.mlp.down_proj,
                ):
                    lin.weight.normal_(generator=generator)
                    lin.weight.mul_(0.05)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Plain forward returning ``(logits, past_key_values)``."""
        cfg = self.cfg
        b, s = input_ids.shape
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[-2]
        positions = torch.arange(past_len, past_len + s, device=cfg.device)

        h = self.embed_tokens(input_ids)
        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[layer_idx]
            h, layer_past = layer(h, positions, past_kv)
            new_past.append(layer_past)

        h = rmsnorm(h, self.final_norm_weight, cfg.rms_norm_eps)
        logits = self.lm_head(h)
        return logits, new_past

    @torch.no_grad()
    def greedy_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Plain greedy generation; returns ``[B, prompt_len + max_new_tokens]``."""
        all_ids = input_ids
        past: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(self.cfg.num_layers)
        ]

        # Prefill on the full prompt.
        logits, past_list = self.forward(input_ids, past_key_values=None)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([all_ids, next_token], dim=-1)
        past = list(past_list)

        for _ in range(max_new_tokens - 1):
            logits, past_list = self.forward(next_token, past_key_values=past)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_ids = torch.cat([all_ids, next_token], dim=-1)
            past = list(past_list)
        return all_ids


__all__ = [
    "TinyModernDecoderConfig",
    "TinyModernDecoderForCausalLM",
    "TinyModernDecoderLayer",
    "TinyModernDecoderAttention",
    "TinyModernDecoderMLP",
    "apply_rope",
    "rmsnorm",
    "repeat_kv",
    "causal_attention",
]
