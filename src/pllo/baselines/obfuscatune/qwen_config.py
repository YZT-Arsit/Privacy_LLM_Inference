"""Qwen2/Qwen2.5 config + robust module discovery for ObfuscaTune (offline).

Loads a *local* Qwen checkpoint (ModelScope/HF layout) or builds a tiny random
Qwen2 for tests -- never downloads. Provides robust discovery of the attention
projections (q/k/v/o), the SwiGLU MLP projections (gate/up/down) and the
RMSNorm modules, raising an explicit error when a module cannot be found rather
than silently falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Tiny random config + offline loading
# ---------------------------------------------------------------------------
def _require_qwen():
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2Config, Qwen2ForCausalLM
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "ObfuscaTune-Qwen requires the 'transformers' package (Qwen2). "
            "Install the optional 'hf' extra. No model is downloaded: use "
            "make_tiny_qwen_config() / --tiny-random-qwen, or pass a local "
            "--model-name-or-path."
        ) from exc


def make_tiny_qwen_config(
    *,
    vocab_size: int = 64,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,     # GQA by default (kv < heads)
    head_dim: int = 8,
    max_position_embeddings: int = 128,
):
    """Return a tiny random ``Qwen2Config`` (GQA by default; no download)."""
    Qwen2Config, _ = _require_qwen()
    return Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )


def load_qwen(
    *,
    model_name_or_path: str | None = None,
    tiny_random_qwen: Any | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """Load a Qwen2 CausalLM in eval mode. Exactly one source; never downloads.

    ``tiny_random_qwen`` may be ``True`` (build a default tiny config) or a
    ``Qwen2Config`` instance. ``model_name_or_path`` must be a local path.
    """
    Qwen2Config, Qwen2ForCausalLM = _require_qwen()
    if (model_name_or_path is None) == (tiny_random_qwen is None):
        raise ValueError(
            "provide exactly one of model_name_or_path (local path) or "
            "tiny_random_qwen"
        )
    if tiny_random_qwen is not None:
        cfg = make_tiny_qwen_config() if tiny_random_qwen is True else tiny_random_qwen
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(cfg)
    else:
        # Real checkpoint: use AutoModelForCausalLM so the loader is architecture-
        # agnostic (Qwen2 for Qwen, Llama for Llama, etc.). The attack hooks and
        # ops only rely on the shared decoder layout (model.embed_tokens,
        # layers[i].self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj) which Qwen2
        # and Llama share, so cross-family reproduction needs no other change.
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, local_files_only=True
        )
    return model.to(dtype).eval()


# ---------------------------------------------------------------------------
# Robust module discovery
# ---------------------------------------------------------------------------
_ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_KEYS = ("gate_proj", "up_proj", "down_proj")


def discover_attention_projections(attn: torch.nn.Module) -> dict[str, torch.nn.Module]:
    out = {}
    for key in _ATTN_KEYS:
        mod = getattr(attn, key, None)
        if not isinstance(mod, torch.nn.Linear):
            raise AttributeError(
                f"ObfuscaTune-Qwen: attention module {type(attn).__name__} is "
                f"missing an nn.Linear '{key}'. Found children: "
                f"{[n for n, _ in attn.named_children()]}"
            )
        out[key] = mod
    return out


def discover_mlp_projections(mlp: torch.nn.Module) -> dict[str, torch.nn.Module]:
    out = {}
    for key in _MLP_KEYS:
        mod = getattr(mlp, key, None)
        if not isinstance(mod, torch.nn.Linear):
            raise AttributeError(
                f"ObfuscaTune-Qwen: MLP module {type(mlp).__name__} is missing "
                f"an nn.Linear '{key}' (expected SwiGLU gate/up/down). Found: "
                f"{[n for n, _ in mlp.named_children()]}"
            )
        out[key] = mod
    return out


def is_rmsnorm(module: torch.nn.Module) -> bool:
    return "rmsnorm" in type(module).__name__.lower()


def discover_layer_norms(layer: torch.nn.Module) -> dict[str, torch.nn.Module]:
    out = {}
    for key in ("input_layernorm", "post_attention_layernorm"):
        mod = getattr(layer, key, None)
        if mod is None or not is_rmsnorm(mod):
            raise AttributeError(
                f"ObfuscaTune-Qwen: decoder layer {type(layer).__name__} is "
                f"missing an RMSNorm '{key}'. Found: "
                f"{[n for n, _ in layer.named_children()]}"
            )
        out[key] = mod
    return out


@dataclass
class QwenArch:
    """Architecture facts extracted from a Qwen2 config (for schema output)."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float

    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads != self.num_attention_heads


def extract_arch(model) -> QwenArch:
    c = model.config
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    return QwenArch(
        hidden_size=c.hidden_size,
        num_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
        head_dim=hd,
        intermediate_size=c.intermediate_size,
        vocab_size=c.vocab_size,
        rms_norm_eps=getattr(c, "rms_norm_eps", 1e-6),
    )


__all__ = [
    "make_tiny_qwen_config",
    "load_qwen",
    "discover_attention_projections",
    "discover_mlp_projections",
    "discover_layer_norms",
    "is_rmsnorm",
    "extract_arch",
    "QwenArch",
]
