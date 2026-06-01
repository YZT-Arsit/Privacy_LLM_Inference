"""Stage 6.4b — Modern decoder-only block inspector.

Identifies the first transformer block of a HuggingFace LLaMA / TinyLlama /
Qwen / Qwen2 model, extracts the spec needed for block-level obfuscated
forward, and surfaces it as a JSON-safe dataclass. Designed to fail
gracefully (``ValueError`` with explicit reason) when a model layout does
not match the expected modern-decoder pattern; the wider system then
falls back to a synthetic block.

Scope is intentionally narrow: only the *first* block, and only the eight
load-bearing submodules (input RMSNorm, q/k/v/o projections, post-attn
RMSNorm, gate/up/down projections). No model-level wrapper, no
generation, no tokenizer, no LM head. The block-wrapper consumes this
spec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderBlockSpec:
    """Identifies one modern-decoder block for block-level obfuscation.

    All ``*_path`` fields are dotted attribute paths *relative to the root
    model* and may be empty strings when a probe is supplied directly as a
    submodule (no root walk needed).
    """

    model_family: str
    model_class: str
    block_path: str
    block_index: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    norm_type: str
    activation_type: str
    position_encoding_type: str
    attention_variant: str
    q_proj_path: str
    k_proj_path: str
    v_proj_path: str
    o_proj_path: str
    gate_proj_path: str
    up_proj_path: str
    down_proj_path: str
    input_norm_path: str
    post_attention_norm_path: str
    rope_base: float | None = None
    rope_scaling_kind: str | None = None
    notes: list[str] = field(default_factory=list)


def spec_to_dict(spec: ModernDecoderBlockSpec) -> dict[str, Any]:
    """JSON-safe dict view of the spec (used for output artifacts)."""
    return asdict(spec)


# ---------------------------------------------------------------------------
# Family heuristics
# ---------------------------------------------------------------------------


_LLAMA_CLASS_HINTS = ("Llama", "LLaMA")
_TINYLLAMA_HINTS = ("TinyLlama",)
_QWEN_CLASS_HINTS = ("Qwen2", "Qwen")


def _classify_modern_decoder_family(model, model_id: str | None) -> str:
    """Return one of ``llama_like`` / ``qwen_like`` / ``tinyllama`` / ``unknown``."""
    name = type(model).__name__
    cfg_type = str(getattr(model.config, "model_type", "")).lower()
    mid = (model_id or "").lower()
    if any(hint in name for hint in _QWEN_CLASS_HINTS) or "qwen" in cfg_type or "qwen" in mid:
        return "qwen_like"
    if "tinyllama" in mid or "tinyllama" in name.lower():
        return "tinyllama"
    if any(hint in name for hint in _LLAMA_CLASS_HINTS) or "llama" in cfg_type or "llama" in mid:
        return "llama_like"
    return "unknown"


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------


def _resolve_block_container(model):
    """Locate ``model.model.layers`` (LLaMA / Qwen2) or fall back to ``model.layers``.

    Returns ``(block_container, block_path_prefix)``. Raises ``ValueError``
    if no candidate is found.
    """
    candidates = (
        ("model.layers", lambda m: getattr(m.model, "layers", None) if hasattr(m, "model") else None),
        ("layers", lambda m: getattr(m, "layers", None)),
        ("transformer.h", lambda m: (
            getattr(getattr(m, "transformer", None), "h", None)
            if hasattr(m, "transformer") else None
        )),
    )
    for prefix, getter in candidates:
        layers = getter(model)
        if layers is not None and len(layers) > 0:
            return layers, prefix
    raise ValueError(
        "modern decoder block container not found: expected model.model.layers"
        " or model.layers; got "
        f"{type(model).__name__} with attrs {dir(model)[:10]}..."
    )


# Per-family submodule path conventions (LLaMA / Qwen2 share the same names).
_BLOCK_SUBPATHS: dict[str, dict[str, str]] = {
    "llama_like": {
        "input_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    },
    "qwen_like": {
        "input_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    },
    "tinyllama": {
        "input_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    },
}


def _resolve_attr_path(root, path: str):
    """Walk a dotted attribute path; return None if any segment is missing."""
    obj = root
    for segment in path.split("."):
        if not hasattr(obj, segment):
            return None
        obj = getattr(obj, segment)
    return obj


def _resolve_module_with_fallback(root, primary: str, family: str) -> tuple[str | None, Any]:
    """Try the primary dotted path first; if missing, scan module names."""
    found = _resolve_attr_path(root, primary)
    if found is not None:
        return primary, found
    # Fallback: walk one level and search by leaf name.
    leaf = primary.rsplit(".", 1)[-1]
    for name, mod in root.named_modules():
        if name.endswith(leaf):
            return name, mod
    return None, None


# ---------------------------------------------------------------------------
# Inspector entry point
# ---------------------------------------------------------------------------


def inspect_modern_decoder_block(
    model,
    *,
    block_index: int = 0,
    model_id: str | None = None,
) -> ModernDecoderBlockSpec:
    """Inspect ``block_index`` (default: 0) of a modern decoder model.

    Raises ``ValueError`` with an explicit reason if the model is not a
    recognised modern decoder layout, so the orchestrator can fall back
    to synthetic.
    """
    family = _classify_modern_decoder_family(model, model_id)
    if family == "unknown":
        raise ValueError(
            f"modern decoder family not recognised for model class"
            f" {type(model).__name__!r} (model_id={model_id!r})"
        )
    subpaths = _BLOCK_SUBPATHS[family]

    block_container, block_path_prefix = _resolve_block_container(model)
    if block_index < 0 or block_index >= len(block_container):
        raise ValueError(
            f"block_index {block_index} out of range [0,"
            f" {len(block_container)})"
        )
    block = block_container[block_index]
    block_path = f"{block_path_prefix}.{block_index}"

    # Walk submodules within the block.
    notes: list[str] = []
    resolved_paths: dict[str, str] = {}
    for key, rel in subpaths.items():
        full_rel, mod = _resolve_module_with_fallback(block, rel, family)
        if mod is None:
            raise ValueError(
                f"submodule {rel!r} not found inside block at {block_path}"
                f" for family {family!r}"
            )
        if full_rel != rel:
            notes.append(
                f"submodule {key} resolved via fallback search at {full_rel!r}"
                f" (primary {rel!r} missing)"
            )
        resolved_paths[key] = f"{block_path}.{full_rel}"

    cfg = model.config
    hidden_size = int(
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "d_model", None)
    )
    intermediate_size = int(
        getattr(cfg, "intermediate_size", None)
        or getattr(cfg, "ffn_dim", None)
        or 4 * hidden_size
    )
    num_attention_heads = int(
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "num_heads", None)
    )
    num_key_value_heads = int(
        getattr(cfg, "num_key_value_heads", None)
        or num_attention_heads
    )
    head_dim = int(
        getattr(cfg, "head_dim", None)
        or (hidden_size // num_attention_heads)
    )
    rope_base = getattr(cfg, "rope_theta", None) or getattr(cfg, "rope_base", None)
    if rope_base is not None:
        rope_base = float(rope_base)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    rope_scaling_kind: str | None
    if rope_scaling is None:
        rope_scaling_kind = None
    elif isinstance(rope_scaling, dict):
        kind = rope_scaling.get("rope_type") or rope_scaling.get("type")
        rope_scaling_kind = str(kind) if kind is not None else "unknown"
    else:
        rope_scaling_kind = type(rope_scaling).__name__
    if rope_scaling_kind not in (None, "default", "linear", "ntk"):
        notes.append(
            f"rope_scaling kind {rope_scaling_kind!r} is not implemented;"
            " probe uses default LLaMA-style RoPE — limitation logged."
        )

    if num_key_value_heads == num_attention_heads:
        attention_variant = "mha"
    elif num_key_value_heads == 1:
        attention_variant = "mqa"
    else:
        attention_variant = "gqa"

    return ModernDecoderBlockSpec(
        model_family=family,
        model_class=type(model).__name__,
        block_path=block_path,
        block_index=int(block_index),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        norm_type="rmsnorm",
        activation_type="swiglu",
        position_encoding_type="rotary",
        attention_variant=attention_variant,
        q_proj_path=resolved_paths["q_proj"],
        k_proj_path=resolved_paths["k_proj"],
        v_proj_path=resolved_paths["v_proj"],
        o_proj_path=resolved_paths["o_proj"],
        gate_proj_path=resolved_paths["gate_proj"],
        up_proj_path=resolved_paths["up_proj"],
        down_proj_path=resolved_paths["down_proj"],
        input_norm_path=f"{block_path}.{subpaths['input_norm']}",
        post_attention_norm_path=f"{block_path}.{subpaths['post_attention_norm']}",
        rope_base=rope_base,
        rope_scaling_kind=rope_scaling_kind,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Weight extraction helpers (row-vector convention)
# ---------------------------------------------------------------------------


def extract_linear_row_weights(
    linear: torch.nn.Linear | torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(W_row, b)`` where ``Y = X @ W_row + b`` in row-vector form.

    PyTorch ``nn.Linear`` stores ``weight`` as ``[out_features, in_features]``
    and computes ``Y = X @ weight.T``; we transpose so callers can use the
    row-vector convention. ``bias`` may be ``None`` (LLaMA / Qwen often
    disable bias on attention and MLP projections).
    """
    weight = linear.weight
    bias = getattr(linear, "bias", None)
    W_row = weight.t().contiguous()
    if bias is not None:
        bias = bias.detach().clone()
    return W_row, bias


def extract_rmsnorm_params(
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, float]:
    """Return ``(weight, eps)`` for an RMSNorm-style module.

    Compatible with HF ``LlamaRMSNorm`` / ``Qwen2RMSNorm`` (use
    ``variance_epsilon``) and torch ``RMSNorm`` (uses ``eps``).
    """
    if not hasattr(norm, "weight"):
        raise ValueError(
            f"RMSNorm module {type(norm).__name__!r} has no .weight"
        )
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    if eps is None:
        raise ValueError(
            f"RMSNorm module {type(norm).__name__!r} has neither"
            " .variance_epsilon nor .eps"
        )
    return norm.weight.detach().clone(), float(eps)


__all__ = [
    "ModernDecoderBlockSpec",
    "extract_linear_row_weights",
    "extract_rmsnorm_params",
    "inspect_modern_decoder_block",
    "spec_to_dict",
]
