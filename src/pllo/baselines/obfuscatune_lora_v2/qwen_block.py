"""One trainable Qwen decoder block under the selected partition."""

from __future__ import annotations

import torch
from torch import nn

from .qwen_ops import KVCache, gqa_attention, swiglu_mlp
from .runtime import ExternalLoRALinear, TrustedRuntime
from .transforms import orthogonal_matrix


class AdaptedQwenBlock(nn.Module):
    TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotations: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
        biases: dict[str, torch.Tensor | None] | None = None,
        trusted: TrustedRuntime,
        rank: int = 8,
        alpha: float = 16.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if set(weights) != set(self.TARGETS) or set(rotations) != set(self.TARGETS):
            raise ValueError("all-seven target inventory required")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        self.trusted = trusted
        self.input_norm = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.post_norm = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        directions = {name: ("output" if name in {"o_proj", "down_proj"} else "input") for name in self.TARGETS}
        biases = biases or {name: None for name in self.TARGETS}
        for idx, name in enumerate(self.TARGETS):
            setattr(self, name, ExternalLoRALinear(weights[name], rotations[name], rank=rank, alpha=alpha,
                                                   direction=directions[name], bias=biases.get(name), counters=trusted.counters,
                                                   init_seed=100 + idx))

    @classmethod
    def from_hf_layer(cls, layer: nn.Module, *, config, trusted: TrustedRuntime,
                      rank: int = 8, alpha: float = 16.0, transform_seed: int = 0):
        modules = {
            "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        weights = {name: module.weight.detach().transpose(0, 1) for name, module in modules.items()}
        biases = {name: module.bias for name, module in modules.items()}
        directions = {"o_proj": "output", "down_proj": "output"}
        rotations = {}
        for index, (name, weight) in enumerate(weights.items()):
            dim = weight.shape[1] if directions.get(name) == "output" else weight.shape[0]
            rotations[name] = orthogonal_matrix(dim, transform_seed + index, dtype=weight.dtype,
                                                 device=weight.device)
        block = cls(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads,
                    rotations=rotations, weights=weights, biases=biases, trusted=trusted,
                    rank=rank, alpha=alpha, eps=config.rms_norm_eps)
        with torch.no_grad():
            block.input_norm.copy_(layer.input_layernorm.weight)
            block.post_norm.copy_(layer.post_attention_layernorm.weight)
        return block

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None = None):
        normed = self.trusted.rmsnorm(hidden, self.input_norm, self.eps)
        attn, cache, probs = gqa_attention(normed, q_proj=self.q_proj, k_proj=self.k_proj,
                                           v_proj=self.v_proj, o_proj=self.o_proj,
                                           num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                                           cos=cos, sin=sin, trusted=self.trusted, cache=cache)
        hidden = self.trusted.residual(hidden, attn)
        normed = self.trusted.rmsnorm(hidden, self.post_norm, self.eps)
        mlp = swiglu_mlp(normed, gate_proj=self.gate_proj, up_proj=self.up_proj,
                         down_proj=self.down_proj, trusted=self.trusted)
        return self.trusted.residual(hidden, mlp), cache, probs

    def transformed_parameters(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.TARGETS:
            layer = getattr(self, name)
            result[f"{name}.a_star"] = layer.a_star
            result[f"{name}.b_star"] = layer.b_star
        return result
