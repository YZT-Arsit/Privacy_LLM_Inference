"""End-to-end Qwen2.5 wrapper for the adapted all-seven LoRA baseline."""

from __future__ import annotations

import torch
from torch import nn

from .qwen_block import AdaptedQwenBlock
from .qwen_ops import KVCache
from .runtime import TrustedRuntime


class AdaptedQwenCausalLM(nn.Module):
    """Paper-partitioned model with transformed large linears.

    In the local oracle the trusted tensors remain in this process.  The real
    runner must provide a remote ``TrustedRuntime`` and place embedding, norm,
    RoPE, softmax, nonlinearities, residual, output, and loss on TDX.
    """

    def __init__(self, hf_model: nn.Module, *, trusted: TrustedRuntime, rank: int = 8,
                 alpha: float = 16.0, transform_seed: int = 1234):
        super().__init__()
        self.config = hf_model.config
        self.trusted = trusted
        self.register_buffer("embedding_weight", hf_model.model.embed_tokens.weight.detach().clone(), persistent=False)
        self.register_buffer("final_norm_weight", hf_model.model.norm.weight.detach().clone(), persistent=False)
        self.register_buffer("lm_head_weight", hf_model.lm_head.weight.detach().clone(), persistent=False)
        self.rotary_emb = hf_model.model.rotary_emb
        self.blocks = nn.ModuleList([
            AdaptedQwenBlock.from_hf_layer(layer, config=hf_model.config, trusted=trusted,
                                           rank=rank, alpha=alpha,
                                           transform_seed=transform_seed + 100 * index)
            for index, layer in enumerate(hf_model.model.layers)
        ])

    def forward(self, input_ids: torch.Tensor, *, caches: list[KVCache] | None = None,
                labels: torch.Tensor | None = None):
        hidden = self.trusted.embedding(input_ids, self.embedding_weight)
        offset = 0 if not caches else caches[0].length
        positions = torch.arange(offset, offset + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, positions)
        caches = caches or [KVCache() for _ in self.blocks]
        next_caches = []
        for block, cache in zip(self.blocks, caches):
            hidden, cache, _ = block(hidden, cos, sin, cache)
            next_caches.append(cache)
        hidden = self.trusted.rmsnorm(hidden, self.final_norm_weight, self.config.rms_norm_eps)
        logits = self.trusted.output(hidden, self.lm_head_weight)
        loss = None if labels is None else self.trusted.cross_entropy(logits, labels)
        return logits, next_caches, loss

    def transformed_parameters(self) -> dict[str, torch.Tensor]:
        result = {}
        for index, block in enumerate(self.blocks):
            for name, value in block.transformed_parameters().items():
                result[f"layers.{index}.{name}"] = value
        return result

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        logits, caches, _ = self(prompt_ids)
        token = logits[:, -1].argmax(-1, keepdim=True)
        generated = [token]
        for _ in range(max_new_tokens - 1):
            logits, caches, _ = self(token, caches=caches)
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
        return torch.cat(generated, dim=1)
