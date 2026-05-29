"""GPT-2 linear mapping reports."""

from __future__ import annotations

from typing import Any

from torch import nn

from pllo.model_zoo.gpt2_conv1d_adapter import extract_conv1d_as_linear, is_hf_gpt2_conv1d


def _shape_or_none(tensor) -> list[int] | None:
    return None if tensor is None else list(tensor.shape)


def _conv1d_shape(module) -> list[int] | None:
    if not is_hf_gpt2_conv1d(module):
        return None
    weight, _ = extract_conv1d_as_linear(module)
    return list(weight.shape)


def build_gpt2_linear_mapping_report(model) -> dict[str, Any]:
    """Build a report mapping GPT-2 Conv1D modules to internal linear conventions."""
    config = getattr(model, "config", None)
    transformer = getattr(model, "transformer", None)
    blocks = list(getattr(transformer, "h", [])) if transformer is not None else []
    hidden_size = getattr(config, "n_embd", None)
    layers = []
    for idx, block in enumerate(blocks):
        c_attn = block.attn.c_attn
        c_proj = block.attn.c_proj
        c_fc = block.mlp.c_fc
        mlp_c_proj = block.mlp.c_proj
        c_attn_weight, c_attn_bias = extract_conv1d_as_linear(c_attn)
        layer = {
            "layer": idx,
            "c_attn": {
                "path": f"transformer.h.{idx}.attn.c_attn",
                "is_conv1d": is_hf_gpt2_conv1d(c_attn),
                "weight_shape": list(c_attn_weight.shape),
                "bias_shape": _shape_or_none(c_attn_bias),
                "qkv_split_shapes": {
                    "q": [hidden_size, hidden_size],
                    "k": [hidden_size, hidden_size],
                    "v": [hidden_size, hidden_size],
                },
                "requires_fused_qkv_handling": True,
            },
            "c_proj": {
                "path": f"transformer.h.{idx}.attn.c_proj",
                "is_conv1d": is_hf_gpt2_conv1d(c_proj),
                "weight_shape": _conv1d_shape(c_proj),
            },
            "mlp_c_fc": {
                "path": f"transformer.h.{idx}.mlp.c_fc",
                "is_conv1d": is_hf_gpt2_conv1d(c_fc),
                "weight_shape": _conv1d_shape(c_fc),
            },
            "mlp_c_proj": {
                "path": f"transformer.h.{idx}.mlp.c_proj",
                "is_conv1d": is_hf_gpt2_conv1d(mlp_c_proj),
                "weight_shape": _conv1d_shape(mlp_c_proj),
            },
        }
        layers.append(layer)

    lm_head = getattr(model, "lm_head", None)
    token_embedding = getattr(transformer, "wte", None) if transformer is not None else None
    tied = False
    if lm_head is not None and token_embedding is not None:
        tied = lm_head.weight.data_ptr() == token_embedding.weight.data_ptr()

    return {
        "model_class": model.__class__.__name__,
        "num_layers": len(blocks),
        "hidden_size": hidden_size,
        "vocab_size": getattr(config, "vocab_size", None),
        "layers": layers,
        "lm_head": {
            "path": "lm_head",
            "is_nn_linear": isinstance(lm_head, nn.Linear),
            "weight_shape": _shape_or_none(getattr(lm_head, "weight", None)),
        },
        "token_embedding": {
            "path": "transformer.wte",
            "weight_shape": _shape_or_none(getattr(token_embedding, "weight", None)),
        },
        "tied_embedding": tied,
        "mapping_limitations": [
            "GPT-2 Conv1D must be adapted before using internal ObfuscatedLinear.",
            "Fused c_attn must be split into Q/K/V or specially wrapped.",
            "HF past_key_values format is not handled in this stage.",
            "No module replacement is performed in this stage.",
        ],
    }
