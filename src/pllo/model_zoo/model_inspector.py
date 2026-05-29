"""Utilities for inspecting external model structure."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn


def _is_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D"


def _module_record(name: str, module: nn.Module) -> dict[str, Any]:
    return {
        "name": name,
        "type": module.__class__.__name__,
        "num_parameters": sum(param.numel() for param in module.parameters(recurse=False)),
    }


def inspect_model_modules(model) -> dict[str, Any]:
    """Inspect module types and GPT-style structural groups."""
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    type_counts: Counter[str] = Counter()
    linear_like = []
    layernorm = []
    embeddings = []
    attention_names = []
    mlp_names = []
    lm_head = None

    for name, module in model.named_modules():
        if name == "":
            continue
        type_counts[module.__class__.__name__] += 1
        lowered = name.lower()
        if isinstance(module, nn.Linear) or _is_conv1d(module):
            linear_like.append(_module_record(name, module))
        if isinstance(module, nn.LayerNorm):
            layernorm.append(_module_record(name, module))
        if isinstance(module, nn.Embedding):
            embeddings.append(_module_record(name, module))
        if "attn" in lowered or "attention" in lowered:
            attention_names.append(name)
        if "mlp" in lowered or "ffn" in lowered or "feed_forward" in lowered:
            mlp_names.append(name)
        if name == "lm_head" or lowered.endswith(".lm_head"):
            lm_head = _module_record(name, module)

    return {
        "model_class": model.__class__.__name__,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "module_type_counts": dict(sorted(type_counts.items())),
        "linear_like_modules": linear_like,
        "layernorm_modules": layernorm,
        "embedding_modules": embeddings,
        "lm_head_module": lm_head,
        "attention_related_module_names": attention_names,
        "mlp_related_module_names": mlp_names,
        "recognizes_hf_conv1d": any(record["type"] == "Conv1D" for record in linear_like),
    }
