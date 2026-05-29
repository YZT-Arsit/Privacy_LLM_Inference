"""Architecture coverage scaffold for paper experiments (Stage 6.0)."""

from pllo.architectures.architecture_inspector import (
    inspect_architecture,
    load_for_architecture,
    spec_to_dict,
)
from pllo.architectures.architecture_registry import (
    ARCH_KEY_TO_TYPE,
    AUTO_MODEL_HINTS,
    DEFAULT_ARCHITECTURE_MODELS,
)
from pllo.architectures.architecture_types import (
    ArchitectureModelSpec,
    ArchitectureType,
    AttentionKindSpec,
)
from pllo.architectures.attention_taxonomy import (
    ATTENTION_BY_NAME,
    ATTENTION_TAXONOMY,
    BIDIRECTIONAL_SELF_ATTENTION,
    CAUSAL_SELF_ATTENTION,
    CROSS_ATTENTION,
    attention_kinds_for,
)
from pllo.architectures.encoder_decoder_spec import (
    BART_MODULE_PATHS,
    EncoderDecoderModulePaths,
    T5_MODULE_PATHS,
)
from pllo.architectures.encoder_only_spec import (
    BERT_MODULE_PATHS,
    EncoderOnlyModulePaths,
)

__all__ = [
    "ArchitectureModelSpec",
    "ArchitectureType",
    "AttentionKindSpec",
    "ATTENTION_BY_NAME",
    "ATTENTION_TAXONOMY",
    "BIDIRECTIONAL_SELF_ATTENTION",
    "CAUSAL_SELF_ATTENTION",
    "CROSS_ATTENTION",
    "attention_kinds_for",
    "ARCH_KEY_TO_TYPE",
    "AUTO_MODEL_HINTS",
    "DEFAULT_ARCHITECTURE_MODELS",
    "inspect_architecture",
    "load_for_architecture",
    "spec_to_dict",
    "BERT_MODULE_PATHS",
    "EncoderOnlyModulePaths",
    "T5_MODULE_PATHS",
    "BART_MODULE_PATHS",
    "EncoderDecoderModulePaths",
]
