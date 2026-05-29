"""Obfuscated tiny decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import nn

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.cache.kv_cache import ObfuscatedKVCache
from pllo.masks.mask_state import MaskState
from pllo.models.plain_tiny_transformer import PlainTinyDecoderOnlyTransformer, PlainTransformerBlock
from pllo.models.tiny_config import TinyTransformerConfig
from pllo.ops.attention import (
    causal_self_attention_decode_obfuscated,
    causal_self_attention_obfuscated,
    causal_self_attention_prefill_obfuscated,
    generate_head_masks,
)
from pllo.ops.layernorm import trusted_layer_norm
from pllo.ops.lm_head import lm_head_obfuscated
from pllo.ops.mlp import mlp_obfuscated


class ObfuscatedTransformerBlock(PlainTransformerBlock):
    """Transformer block with obfuscated linear attention/MLP paths."""

    def forward_obfuscated(
        self,
        hidden_tilde: torch.Tensor,
        hidden_state: MaskState,
        tee: SimulatedTEE,
        executor: UntrustedGPUExecutor,
        use_pad: bool,
        pad_scale: float,
    ) -> torch.Tensor:
        """Run a block while keeping residual tensors in hidden mask space."""
        batch, seq_len, hidden_size = hidden_tilde.shape
        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        attn_in = trusted_layer_norm(hidden_plain, self.ln1_weight, self.ln1_bias)
        attn_tilde = causal_self_attention_obfuscated(
            attn_in,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        hidden_tilde = hidden_tilde + attn_tilde

        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        mlp_in = trusted_layer_norm(hidden_plain, self.ln2_weight, self.ln2_bias)
        mlp_tilde = mlp_obfuscated(
            mlp_in,
            self.w_mlp_1,
            self.b_mlp_1,
            self.w_mlp_2,
            self.b_mlp_2,
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        return hidden_tilde + mlp_tilde

    def prefill_obfuscated(
        self,
        hidden_tilde: torch.Tensor,
        hidden_state: MaskState,
        layer_idx: int,
        cache: ObfuscatedKVCache,
        tee: SimulatedTEE,
        executor: UntrustedGPUExecutor,
        use_pad: bool,
        pad_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run block prefill and return obfuscated K/V tensors for cache."""
        batch, seq_len, hidden_size = hidden_tilde.shape
        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        attn_in = trusted_layer_norm(hidden_plain, self.ln1_weight, self.ln1_bias)
        attn_tilde, key_tilde, value_tilde = causal_self_attention_prefill_obfuscated(
            attn_in,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
            cache.key_masks[layer_idx],
            cache.key_mask_inverses[layer_idx],  # type: ignore[index]
            cache.value_masks[layer_idx],
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        hidden_tilde = hidden_tilde + attn_tilde
        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        mlp_in = trusted_layer_norm(hidden_plain, self.ln2_weight, self.ln2_bias)
        mlp_tilde = mlp_obfuscated(
            mlp_in,
            self.w_mlp_1,
            self.b_mlp_1,
            self.w_mlp_2,
            self.b_mlp_2,
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        return hidden_tilde + mlp_tilde, key_tilde, value_tilde

    def decode_step_obfuscated(
        self,
        hidden_tilde: torch.Tensor,
        hidden_state: MaskState,
        layer_idx: int,
        cache: ObfuscatedKVCache,
        tee: SimulatedTEE,
        executor: UntrustedGPUExecutor,
        use_pad: bool,
        pad_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one-token block decode with obfuscated cached K/V."""
        batch, seq_len, hidden_size = hidden_tilde.shape
        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        attn_in = trusted_layer_norm(hidden_plain, self.ln1_weight, self.ln1_bias)
        attn_tilde, key_tilde, value_tilde = causal_self_attention_decode_obfuscated(
            attn_in,
            cache.keys_tilde[layer_idx],
            cache.values_tilde[layer_idx],
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.num_heads,
            cache.key_masks[layer_idx],
            cache.key_mask_inverses[layer_idx],  # type: ignore[index]
            cache.value_masks[layer_idx],
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        hidden_tilde = hidden_tilde + attn_tilde
        hidden_plain = tee.recover_output(hidden_tilde.reshape(-1, hidden_size), hidden_state).reshape(
            batch,
            seq_len,
            hidden_size,
        )
        mlp_in = trusted_layer_norm(hidden_plain, self.ln2_weight, self.ln2_bias)
        mlp_tilde = mlp_obfuscated(
            mlp_in,
            self.w_mlp_1,
            self.b_mlp_1,
            self.w_mlp_2,
            self.b_mlp_2,
            hidden_state,
            tee,
            executor,
            use_pad=use_pad,
            pad_scale=pad_scale,
        )
        return hidden_tilde + mlp_tilde, key_tilde, value_tilde


class ObfuscatedTinyDecoderOnlyTransformer(PlainTinyDecoderOnlyTransformer):
    """Tiny decoder-only Transformer with Stage 2 obfuscated full forward.

    Stage 2 keeps LayerNorm and GELU on the simulated trusted side as an
    engineering simplification. Linear projections, MLP linears, and the LM
    head use mask/pad obfuscated execution through the untrusted executor.
    """

    def __init__(
        self,
        config: TinyTransformerConfig,
        use_pad: bool = True,
        pad_scale: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.use_pad = use_pad
        self.pad_scale = pad_scale
        self.tee = SimulatedTEE(dtype=config.dtype, device=config.device)
        self.executor = UntrustedGPUExecutor()
        self.token_embedding = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_size, dtype=config.dtype)
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config.max_seq_len, config.hidden_size, dtype=config.dtype)
        )
        self.blocks = nn.ModuleList([ObfuscatedTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_ln_weight = nn.Parameter(torch.ones(config.hidden_size, dtype=config.dtype))
        self.final_ln_bias = nn.Parameter(torch.zeros(config.hidden_size, dtype=config.dtype))
        self.lm_head_weight = nn.Parameter(
            torch.empty(config.hidden_size, config.vocab_size, dtype=config.dtype)
        )
        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size, dtype=config.dtype))
        self.reset_parameters()
        self.to(device=torch.device(config.device), dtype=config.dtype)

    @classmethod
    def from_plain(
        cls,
        plain_model: PlainTinyDecoderOnlyTransformer,
        config: TinyTransformerConfig | None = None,
        use_pad: bool = True,
        pad_scale: float = 1.0,
    ) -> "ObfuscatedTinyDecoderOnlyTransformer":
        """Create an obfuscated model with weights copied from a plain model."""
        model = cls(config or plain_model.config, use_pad=use_pad, pad_scale=pad_scale)
        model.load_state_dict(plain_model.state_dict())
        return model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run full-sequence obfuscated forward and return recovered logits."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (batch, seq), got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_plain = self.token_embedding[input_ids] + self.position_embedding[positions].reshape(1, seq_len, -1)
        flat_hidden = hidden_plain.reshape(-1, self.config.hidden_size)
        hidden_state = self.tee.create_linear_mask_state(
            flat_hidden,
            self.config.hidden_size,
            use_pad=False,
        )
        hidden_state.n_out = hidden_state.n_in
        hidden_state.n_out_inv = hidden_state.n_in_inv
        hidden_tilde = (flat_hidden @ hidden_state.n_out).reshape(batch, seq_len, self.config.hidden_size)

        for block in self.blocks:
            hidden_tilde = block.forward_obfuscated(
                hidden_tilde,
                hidden_state,
                self.tee,
                self.executor,
                use_pad=self.use_pad,
                pad_scale=self.pad_scale,
            )

        hidden_plain = self.tee.recover_output(
            hidden_tilde.reshape(-1, self.config.hidden_size),
            hidden_state,
        ).reshape(batch, seq_len, self.config.hidden_size)
        hidden_plain = trusted_layer_norm(hidden_plain, self.final_ln_weight, self.final_ln_bias)

        vocab_state = self.tee.create_linear_mask_state(
            hidden_plain.reshape(-1, self.config.hidden_size),
            self.config.vocab_size,
            use_pad=False,
        )
        return lm_head_obfuscated(
            hidden_plain,
            self.lm_head_weight,
            self.lm_head_bias,
            vocab_state,
            self.tee,
            self.executor,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
        )

    def _new_hidden_state(self, hidden_plain: torch.Tensor) -> tuple[MaskState, torch.Tensor]:
        """Create a hidden residual mask state and obfuscate hidden states."""
        flat_hidden = hidden_plain.reshape(-1, self.config.hidden_size)
        state = self.tee.create_linear_mask_state(flat_hidden, self.config.hidden_size, use_pad=False)
        state.n_out = state.n_in
        state.n_out_inv = state.n_in_inv
        hidden_tilde = (flat_hidden @ state.n_out).reshape_as(hidden_plain)
        return state, hidden_tilde

    def _new_obfuscated_cache(self) -> ObfuscatedKVCache:
        """Create a new generation-session cache with persistent per-layer masks."""
        d_head = self.config.hidden_size // self.config.num_heads
        key_masks = []
        key_mask_inverses = []
        value_masks = []
        value_mask_inverses = []
        for _ in range(self.config.num_layers):
            k_mask, k_inv = generate_head_masks(
                self.config.num_heads,
                d_head,
                self.config.dtype,
                self.config.device,
            )
            v_mask, v_inv = generate_head_masks(
                self.config.num_heads,
                d_head,
                self.config.dtype,
                self.config.device,
            )
            key_masks.append(k_mask)
            key_mask_inverses.append(k_inv)
            value_masks.append(v_mask)
            value_mask_inverses.append(v_inv)
        return ObfuscatedKVCache.empty(
            self.config.num_layers,
            key_masks,
            value_masks,
            key_mask_inverses,
            value_mask_inverses,
        )

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

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ObfuscatedKVCache]:
        """Run obfuscated prompt prefill and return recovered logits plus obfuscated K/V cache."""
        hidden_plain = self._embed(input_ids, start_pos=0)
        hidden_state, hidden_tilde = self._new_hidden_state(hidden_plain)
        cache = self._new_obfuscated_cache()
        for layer_idx, block in enumerate(self.blocks):
            hidden_tilde, key_tilde, value_tilde = block.prefill_obfuscated(
                hidden_tilde,
                hidden_state,
                layer_idx,
                cache,
                self.tee,
                self.executor,
                self.use_pad,
                self.pad_scale,
            )
            cache.append(layer_idx, key_tilde, value_tilde)
        hidden_plain = self.tee.recover_output(
            hidden_tilde.reshape(-1, self.config.hidden_size),
            hidden_state,
        ).reshape_as(hidden_plain)
        hidden_plain = trusted_layer_norm(hidden_plain, self.final_ln_weight, self.final_ln_bias)
        vocab_state = self.tee.create_linear_mask_state(
            hidden_plain.reshape(-1, self.config.hidden_size),
            self.config.vocab_size,
            use_pad=False,
        )
        logits = lm_head_obfuscated(
            hidden_plain,
            self.lm_head_weight,
            self.lm_head_bias,
            vocab_state,
            self.tee,
            self.executor,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
        )
        return logits, cache

    def decode_step(self, input_ids: torch.Tensor, cache: ObfuscatedKVCache) -> tuple[torch.Tensor, ObfuscatedKVCache]:
        """Run one-token obfuscated decode using persistent cache masks."""
        if input_ids.shape[1] != 1:
            raise ValueError(f"decode_step expects input_ids shape (batch, 1), got {tuple(input_ids.shape)}")
        hidden_plain = self._embed(input_ids, start_pos=cache.length(0))
        hidden_state, hidden_tilde = self._new_hidden_state(hidden_plain)
        for layer_idx, block in enumerate(self.blocks):
            hidden_tilde, key_tilde, value_tilde = block.decode_step_obfuscated(
                hidden_tilde,
                hidden_state,
                layer_idx,
                cache,
                self.tee,
                self.executor,
                self.use_pad,
                self.pad_scale,
            )
            cache.append(layer_idx, key_tilde, value_tilde)
        hidden_plain = self.tee.recover_output(
            hidden_tilde.reshape(-1, self.config.hidden_size),
            hidden_state,
        ).reshape_as(hidden_plain)
        hidden_plain = trusted_layer_norm(hidden_plain, self.final_ln_weight, self.final_ln_bias)
        vocab_state = self.tee.create_linear_mask_state(
            hidden_plain.reshape(-1, self.config.hidden_size),
            self.config.vocab_size,
            use_pad=False,
        )
        logits = lm_head_obfuscated(
            hidden_plain,
            self.lm_head_weight,
            self.lm_head_bias,
            vocab_state,
            self.tee,
            self.executor,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
        )
        return logits, cache

    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens with greedy decoding from recovered logits."""
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
