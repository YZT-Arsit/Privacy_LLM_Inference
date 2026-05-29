"""Architecture inspection for HuggingFace Transformer models (Stage 6.0).

Loads a checkpoint with the appropriate ``AutoModelFor*`` class, classifies
it as decoder-only / encoder-only / encoder-decoder, and emits an
``ArchitectureModelSpec``. Designed to skip gracefully when network or HF
hub access is unavailable — the script that drives it records the skip in
the coverage report rather than failing the whole run.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict
from typing import Any

from pllo.architectures.architecture_registry import (
    ARCH_KEY_TO_TYPE,
    AUTO_MODEL_HINTS,
    DEFAULT_ARCHITECTURE_MODELS,
)
from pllo.architectures.architecture_types import ArchitectureModelSpec, ArchitectureType


# ---------------------------------------------------------------------------
# Loader with fallbacks (handles tiny BERT / T5 quirks)
# ---------------------------------------------------------------------------


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover — surfaced by tests as skip.
        raise ImportError(
            "transformers is required for architecture inspection. "
            "Install with pip install -e '.[hf]'."
        ) from exc
    return transformers


def _resolve_auto_class(auto_class_name: str):
    transformers = _import_transformers()
    return getattr(transformers, auto_class_name)


def _try_load(model_id: str, auto_class_names: tuple[str, ...]) -> Any:
    """Try each auto-class in order; raise the final error if none succeeds."""
    last_exc: Exception | None = None
    for name in auto_class_names:
        try:
            cls = _resolve_auto_class(name)
            return cls.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"No auto-class succeeded for {model_id!r}")


def load_for_architecture(
    architecture_key: str,
    candidates: tuple[str, ...] | None = None,
) -> tuple[str, Any]:
    """Try the registry candidates for ``architecture_key`` until one loads.

    Returns ``(model_id, model)``. Raises the last underlying exception if
    every candidate fails.
    """
    if architecture_key not in DEFAULT_ARCHITECTURE_MODELS:
        raise KeyError(f"unknown architecture key {architecture_key!r}")
    if candidates is None:
        candidates = DEFAULT_ARCHITECTURE_MODELS[architecture_key]
    auto_hints = AUTO_MODEL_HINTS[architecture_key]
    last_exc: Exception | None = None
    for model_id in candidates:
        try:
            model = _try_load(model_id, auto_hints)
            model.eval()
            return model_id, model
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------


_CAUSAL_LM_CLASS_HINTS = ("CausalLM", "LMHeadModel", "ForCausalLM")
_MASKED_LM_CLASS_HINTS = ("ForMaskedLM", "ForPreTraining")
_SEQ2SEQ_CLASS_HINTS = ("Seq2SeqLM", "ForConditionalGeneration")


def _classify_architecture_type(model, cfg) -> ArchitectureType:
    """Decide which of the three Transformer families ``model`` belongs to."""
    if bool(getattr(cfg, "is_encoder_decoder", False)):
        return ArchitectureType.ENCODER_DECODER
    name = type(model).__name__
    if any(hint in name for hint in _CAUSAL_LM_CLASS_HINTS):
        return ArchitectureType.DECODER_ONLY
    # GPT-2 style: outer model has ``.transformer.h`` block list.
    inner = getattr(model, "transformer", None)
    if inner is not None and hasattr(inner, "h"):
        return ArchitectureType.DECODER_ONLY
    if any(hint in name for hint in _MASKED_LM_CLASS_HINTS):
        return ArchitectureType.ENCODER_ONLY
    if getattr(cfg, "model_type", None) in {"bert", "roberta", "albert", "electra", "distilbert"}:
        return ArchitectureType.ENCODER_ONLY
    # Last-resort: anything with ``.encoder`` but no ``.decoder`` is encoder-only.
    if hasattr(model, "encoder") and not hasattr(model, "decoder"):
        return ArchitectureType.ENCODER_ONLY
    return ArchitectureType.UNKNOWN


def _supports_past_key_values(model) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "past_key_values" in sig.parameters


def _has_lm_head(model, cfg) -> bool:
    if hasattr(model, "lm_head") and getattr(model, "lm_head", None) is not None:
        return True
    # Some T5/BART variants expose ``lm_head`` only on the for-conditional class.
    return any(hint in type(model).__name__ for hint in _CAUSAL_LM_CLASS_HINTS + _SEQ2SEQ_CLASS_HINTS)


def _has_mlm_head(model, cfg) -> bool:
    name = type(model).__name__
    if any(hint in name for hint in _MASKED_LM_CLASS_HINTS):
        return True
    cls = getattr(model, "cls", None)
    return cls is not None and hasattr(cls, "predictions")


def _has_classification_head(model, cfg) -> bool:
    return "ForSequenceClassification" in type(model).__name__ or hasattr(model, "classifier")


def _resolve_hidden_size(cfg) -> int | None:
    return (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "d_model", None)
    )


def _resolve_num_layers(cfg) -> int | None:
    return (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
        or getattr(cfg, "num_decoder_layers", None)
    )


def _resolve_num_heads(cfg) -> int | None:
    return (
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or getattr(cfg, "num_heads", None)
    )


# ---------------------------------------------------------------------------
# Module-path discovery
# ---------------------------------------------------------------------------


def _module_paths(model, arch_type: ArchitectureType) -> dict[str, str | None]:
    """Discover key module paths inside ``model``.

    Returns a string-valued dict keyed by ``embedding`` / ``self_attention`` /
    ``cross_attention`` / ``ffn`` / ``layernorm`` / ``output_head`` /
    ``cache_supported`` — paths use dotted attribute notation rooted at
    ``model``. ``None`` means "not present on this model".
    """
    paths: dict[str, str | None] = {
        "embedding": None,
        "self_attention": None,
        "cross_attention": None,
        "ffn": None,
        "layernorm": None,
        "output_head": None,
    }
    name = type(model).__name__
    # GPT-2 style
    if arch_type is ArchitectureType.DECODER_ONLY and hasattr(model, "transformer"):
        if hasattr(model.transformer, "wte"):
            paths["embedding"] = "transformer.wte"
        if hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
            paths["self_attention"] = "transformer.h.0.attn"
            paths["ffn"] = "transformer.h.0.mlp"
            paths["layernorm"] = "transformer.h.0.ln_1"
        if hasattr(model, "lm_head"):
            paths["output_head"] = "lm_head"
        return paths
    # BERT style
    if arch_type is ArchitectureType.ENCODER_ONLY:
        base = model.bert if hasattr(model, "bert") else model
        base_prefix = "bert." if hasattr(model, "bert") else ""
        if hasattr(base, "embeddings"):
            paths["embedding"] = f"{base_prefix}embeddings.word_embeddings"
        encoder = getattr(base, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer") and len(encoder.layer) > 0:
            paths["self_attention"] = f"{base_prefix}encoder.layer.0.attention.self"
            paths["ffn"] = f"{base_prefix}encoder.layer.0.intermediate"
            paths["layernorm"] = f"{base_prefix}encoder.layer.0.attention.output.LayerNorm"
        if hasattr(model, "cls"):
            paths["output_head"] = "cls.predictions"
        elif hasattr(model, "classifier"):
            paths["output_head"] = "classifier"
        elif hasattr(base, "pooler"):
            paths["output_head"] = f"{base_prefix}pooler"
        return paths
    # T5 / BART style
    if arch_type is ArchitectureType.ENCODER_DECODER:
        if hasattr(model, "shared"):
            paths["embedding"] = "shared"
        encoder = getattr(model, "encoder", None)
        decoder = getattr(model, "decoder", None)
        # T5 layout
        if encoder is not None and hasattr(encoder, "block") and len(encoder.block) > 0:
            paths["self_attention"] = "encoder.block.0.layer.0.SelfAttention"
            paths["ffn"] = "encoder.block.0.layer.1.DenseReluDense"
            paths["layernorm"] = "encoder.block.0.layer.0.layer_norm"
            if decoder is not None and hasattr(decoder, "block") and len(decoder.block) > 0:
                paths["cross_attention"] = "decoder.block.0.layer.1.EncDecAttention"
        # BART layout
        elif encoder is not None and hasattr(encoder, "layers") and len(encoder.layers) > 0:
            paths["self_attention"] = "encoder.layers.0.self_attn"
            paths["ffn"] = "encoder.layers.0.fc1"
            paths["layernorm"] = "encoder.layers.0.self_attn_layer_norm"
            if decoder is not None and hasattr(decoder, "layers") and len(decoder.layers) > 0:
                paths["cross_attention"] = "decoder.layers.0.encoder_attn"
        if hasattr(model, "lm_head"):
            paths["output_head"] = "lm_head"
        return paths
    return paths


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inspect_architecture(
    model,
    tokenizer=None,  # noqa: ARG001 — accepted for API parity; not used by current heuristics.
    model_id: str | None = None,
) -> ArchitectureModelSpec:
    """Classify ``model`` and emit a populated ``ArchitectureModelSpec``."""
    cfg = model.config
    model_class = type(model).__name__
    arch_type = _classify_architecture_type(model, cfg)
    is_enc_dec = arch_type is ArchitectureType.ENCODER_DECODER
    notes: list[str] = []
    if arch_type is ArchitectureType.UNKNOWN:
        notes.append(
            f"Could not classify model class {model_class!r}; falling back to"
            f" UNKNOWN with config.model_type={getattr(cfg, 'model_type', None)!r}."
        )

    has_encoder = arch_type in {ArchitectureType.ENCODER_ONLY, ArchitectureType.ENCODER_DECODER}
    has_decoder = arch_type in {ArchitectureType.DECODER_ONLY, ArchitectureType.ENCODER_DECODER}
    has_cross = arch_type is ArchitectureType.ENCODER_DECODER

    spec = ArchitectureModelSpec(
        model_id=model_id or "",
        architecture_type=arch_type,
        model_class=model_class,
        has_encoder=has_encoder,
        has_decoder=has_decoder,
        has_cross_attention=has_cross,
        has_causal_self_attention=has_decoder,
        has_bidirectional_self_attention=has_encoder,
        supports_past_key_values=_supports_past_key_values(model),
        has_lm_head=_has_lm_head(model, cfg),
        has_mlm_head=_has_mlm_head(model, cfg),
        has_classification_head=_has_classification_head(model, cfg),
        vocab_size=getattr(cfg, "vocab_size", None),
        hidden_size=_resolve_hidden_size(cfg),
        num_layers=_resolve_num_layers(cfg),
        num_heads=_resolve_num_heads(cfg),
        notes=notes,
    )
    return spec


def spec_to_dict(spec: ArchitectureModelSpec) -> dict[str, Any]:
    """Convert an ``ArchitectureModelSpec`` to a JSON-serialisable dict."""
    out = asdict(spec)
    # Enum values come out as the enum; serialise to its string value.
    out["architecture_type"] = spec.architecture_type.value
    return out


__all__ = [
    "inspect_architecture",
    "spec_to_dict",
    "load_for_architecture",
    "_module_paths",
]
