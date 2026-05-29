"""Architecture type taxonomy + model spec dataclass (Stage 6.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArchitectureType(str, Enum):
    """High-level Transformer architecture classification used by the
    paper-experiment coverage scaffold.

    The string values match the keys used in ``architecture_registry`` and
    the JSON output, so the enum can be serialized directly.
    """

    DECODER_ONLY = "decoder_only"
    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"
    UNKNOWN = "unknown"


@dataclass
class ArchitectureModelSpec:
    """Per-model architecture inspection record.

    Captures the structural facts the paper experiments need to reason about
    when comparing decoder-only, encoder-only, and encoder-decoder models.
    Stage 6.0 only fills this struct; it does not implement obfuscated
    wrappers for non-GPT-2 architectures.
    """

    model_id: str
    architecture_type: ArchitectureType
    model_class: str
    has_encoder: bool
    has_decoder: bool
    has_cross_attention: bool
    has_causal_self_attention: bool
    has_bidirectional_self_attention: bool
    supports_past_key_values: bool
    has_lm_head: bool
    has_mlm_head: bool
    has_classification_head: bool
    vocab_size: int | None
    hidden_size: int | None
    num_layers: int | None
    num_heads: int | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AttentionKindSpec:
    """Static description of one attention variant used across architectures."""

    name: str
    architecture_type: ArchitectureType
    q_source: str
    k_source: str
    v_source: str
    mask_type: str
    cache_type: str
    required_invariant: str
