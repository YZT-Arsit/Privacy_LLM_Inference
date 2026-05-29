"""GPT-2 structure specification helpers."""

from __future__ import annotations

from typing import Any


def get_gpt2_module_spec(model) -> dict[str, Any]:
    """Return a JSON-serializable GPT-2 module path specification."""
    config = getattr(model, "config", None)
    transformer = getattr(model, "transformer", None)
    blocks = list(getattr(transformer, "h", [])) if transformer is not None else []
    num_layers = len(blocks)
    block_paths = [f"transformer.h.{idx}" for idx in range(num_layers)]
    tied = False
    if hasattr(model, "lm_head") and transformer is not None and hasattr(transformer, "wte"):
        tied = model.lm_head.weight.data_ptr() == transformer.wte.weight.data_ptr()

    return {
        "model_class": model.__class__.__name__,
        "num_layers": num_layers,
        "hidden_size": getattr(config, "n_embd", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "n_positions", None),
        "embedding_paths": {
            "token_embedding": "transformer.wte",
            "position_embedding": "transformer.wpe",
        },
        "block_paths": block_paths,
        "attention_projection_paths": [
            {
                "layer": idx,
                "qkv_fused": f"transformer.h.{idx}.attn.c_attn",
                "output": f"transformer.h.{idx}.attn.c_proj",
                "projection_type": "transformers.pytorch_utils.Conv1D",
            }
            for idx in range(num_layers)
        ],
        "mlp_projection_paths": [
            {
                "layer": idx,
                "fc": f"transformer.h.{idx}.mlp.c_fc",
                "proj": f"transformer.h.{idx}.mlp.c_proj",
                "projection_type": "transformers.pytorch_utils.Conv1D",
            }
            for idx in range(num_layers)
        ],
        "layernorm_paths": {
            "per_block": [
                {
                    "layer": idx,
                    "ln_1": f"transformer.h.{idx}.ln_1",
                    "ln_2": f"transformer.h.{idx}.ln_2",
                }
                for idx in range(num_layers)
            ],
            "final": "transformer.ln_f",
        },
        "lm_head_path": "lm_head",
        "lm_head_tied_with_token_embedding": tied,
    }


def is_gpt2_like(model) -> bool:
    """Return True when a model exposes the standard GPT-2 module layout."""
    return hasattr(model, "transformer") and hasattr(model.transformer, "h") and hasattr(model, "lm_head")
