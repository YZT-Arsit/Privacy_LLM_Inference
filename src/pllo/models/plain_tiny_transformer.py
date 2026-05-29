"""Plain tiny decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import nn

from pllo.cache.kv_cache import PlainKVCache
from pllo.models.tiny_config import TinyTransformerConfig
from pllo.ops.attention import (
    causal_self_attention_decode_plain,
    causal_self_attention_plain,
    causal_self_attention_prefill_plain,
)
from pllo.ops.layernorm import layer_norm_plain
from pllo.ops.lm_head import lm_head_plain
from pllo.ops.mlp import mlp_plain


class PlainTransformerBlock(nn.Module):
    """Pre-LN decoder-only Transformer block."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.ln1_weight = nn.Parameter(torch.ones(h, dtype=config.dtype))
        self.ln1_bias = nn.Parameter(torch.zeros(h, dtype=config.dtype))
        self.w_q = nn.Parameter(torch.empty(h, h, dtype=config.dtype))
        self.w_k = nn.Parameter(torch.empty(h, h, dtype=config.dtype))
        self.w_v = nn.Parameter(torch.empty(h, h, dtype=config.dtype))
        self.w_o = nn.Parameter(torch.empty(h, h, dtype=config.dtype))
        self.ln2_weight = nn.Parameter(torch.ones(h, dtype=config.dtype))
        self.ln2_bias = nn.Parameter(torch.zeros(h, dtype=config.dtype))
        self.w_mlp_1 = nn.Parameter(torch.empty(h, config.ffn_dim, dtype=config.dtype))
        self.b_mlp_1 = nn.Parameter(torch.zeros(config.ffn_dim, dtype=config.dtype))
        self.w_mlp_2 = nn.Parameter(torch.empty(config.ffn_dim, h, dtype=config.dtype))
        self.b_mlp_2 = nn.Parameter(torch.zeros(h, dtype=config.dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with a small normal distribution."""
        for weight in (self.w_q, self.w_k, self.w_v, self.w_o, self.w_mlp_1, self.w_mlp_2):
            nn.init.normal_(weight, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the Transformer block."""
        attn_in = layer_norm_plain(hidden, self.ln1_weight, self.ln1_bias)
        hidden = hidden + causal_self_attention_plain(
            attn_in,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
        )
        mlp_in = layer_norm_plain(hidden, self.ln2_weight, self.ln2_bias)
        hidden = hidden + mlp_plain(mlp_in, self.w_mlp_1, self.b_mlp_1, self.w_mlp_2, self.b_mlp_2)
        return hidden

    def prefill(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run prefill block forward and return K/V for cache initialization."""
        attn_in = layer_norm_plain(hidden, self.ln1_weight, self.ln1_bias)
        attn_out, key, value = causal_self_attention_prefill_plain(
            attn_in,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
        )
        hidden = hidden + attn_out
        mlp_in = layer_norm_plain(hidden, self.ln2_weight, self.ln2_bias)
        hidden = hidden + mlp_plain(mlp_in, self.w_mlp_1, self.b_mlp_1, self.w_mlp_2, self.b_mlp_2)
        return hidden, key, value

    def decode_step(
        self,
        hidden: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one-token decode block forward and return newly projected K/V."""
        attn_in = layer_norm_plain(hidden, self.ln1_weight, self.ln1_bias)
        attn_out, key, value = causal_self_attention_decode_plain(
            attn_in,
            past_key,
            past_value,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
        )
        hidden = hidden + attn_out
        mlp_in = layer_norm_plain(hidden, self.ln2_weight, self.ln2_bias)
        hidden = hidden + mlp_plain(mlp_in, self.w_mlp_1, self.b_mlp_1, self.w_mlp_2, self.b_mlp_2)
        return hidden, key, value


class PlainTinyDecoderOnlyTransformer(nn.Module):
    """Small plain decoder-only Transformer for correctness experiments."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_size, dtype=config.dtype)
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config.max_seq_len, config.hidden_size, dtype=config.dtype)
        )
        self.blocks = nn.ModuleList([PlainTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_ln_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=config.dtype))
        self.final_ln_bias = nn.Parameter(torch.zeros(config.hidden_size, dtype=config.dtype))
        self.lm_head_weight = nn.Parameter(
            torch.empty(config.hidden_size, config.vocab_size, dtype=config.dtype)
        )
        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size, dtype=config.dtype))
        self.reset_parameters()
        self.to(device=torch.device(config.device), dtype=config.dtype)

    def reset_parameters(self) -> None:
        """Initialize embedding and LM head weights."""
        nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head_weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run full-sequence decoder-only forward and return logits."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (batch, seq), got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding[input_ids] + self.position_embedding[positions].reshape(1, seq_len, -1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = layer_norm_plain(hidden, self.final_ln_weight, self.final_ln_bias)
        return lm_head_plain(hidden, self.lm_head_weight, self.lm_head_bias)

    def _embed(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Embed tokens using absolute position ids starting at start_pos."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (batch, seq), got {tuple(input_ids.shape)}")
        seq_len = input_ids.shape[1]
        if start_pos + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"positions [{start_pos}, {start_pos + seq_len}) exceed max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
        return self.token_embedding[input_ids] + self.position_embedding[positions].reshape(1, seq_len, -1)

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, PlainKVCache]:
        """Run prompt prefill and return full prompt logits plus plain K/V cache."""
        hidden = self._embed(input_ids, start_pos=0)
        cache = PlainKVCache.empty(len(self.blocks))
        for layer_idx, block in enumerate(self.blocks):
            hidden, key, value = block.prefill(hidden)
            cache.append(layer_idx, key, value)
        hidden = layer_norm_plain(hidden, self.final_ln_weight, self.final_ln_bias)
        return lm_head_plain(hidden, self.lm_head_weight, self.lm_head_bias), cache

    def decode_step(self, input_ids: torch.Tensor, cache: PlainKVCache) -> tuple[torch.Tensor, PlainKVCache]:
        """Run one-token decode using a plain K/V cache."""
        if input_ids.shape[1] != 1:
            raise ValueError(f"decode_step expects input_ids shape (batch, 1), got {tuple(input_ids.shape)}")
        hidden = self._embed(input_ids, start_pos=cache.length(0))
        for layer_idx, block in enumerate(self.blocks):
            hidden, key, value = block.decode_step(hidden, cache.keys[layer_idx], cache.values[layer_idx])
            cache.append(layer_idx, key, value)
        hidden = layer_norm_plain(hidden, self.final_ln_weight, self.final_ln_bias)
        return lm_head_plain(hidden, self.lm_head_weight, self.lm_head_bias), cache

    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens with greedy decoding."""
        logits, cache = self.prefill(input_ids)
        generated = input_ids
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for step in range(max_new_tokens):
            generated = torch.cat([generated, next_token], dim=1)
            if step == max_new_tokens - 1:
                break
            logits, cache = self.decode_step(next_token, cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return generated
