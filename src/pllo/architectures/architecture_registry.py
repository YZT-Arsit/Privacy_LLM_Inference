"""Registry of default models per architecture type (Stage 6.0)."""

from __future__ import annotations

from pllo.architectures.architecture_types import ArchitectureType


# Primary default + ordered fallback list per architecture.
# Each model_id is tried in order; the first one that loads cleanly wins.
DEFAULT_ARCHITECTURE_MODELS: dict[str, tuple[str, ...]] = {
    "decoder_only": (
        "sshleifer/tiny-gpt2",
    ),
    "encoder_only": (
        # `prajjwal1/bert-tiny` is widely cited as the canonical tiny BERT,
        # but its older checkpoint config lacks ``model_type`` so it cannot
        # be loaded via AutoConfig without an explicit BertConfig fallback.
        # We list HF's own tiny BERT checkpoints first because they work with
        # the standard Auto* pathway, then fall back to prajjwal1/bert-tiny.
        "hf-internal-testing/tiny-bert",
        "hf-internal-testing/tiny-random-BertModel",
        "prajjwal1/bert-tiny",
    ),
    "encoder_decoder": (
        "hf-internal-testing/tiny-random-t5",
    ),
}


# Mapping from string keys to enum (handy when emitting structured output).
ARCH_KEY_TO_TYPE: dict[str, ArchitectureType] = {
    "decoder_only": ArchitectureType.DECODER_ONLY,
    "encoder_only": ArchitectureType.ENCODER_ONLY,
    "encoder_decoder": ArchitectureType.ENCODER_DECODER,
}


# HuggingFace AutoModel class hints. The architecture inspector uses these to
# load each candidate checkpoint with the appropriate task head when present.
AUTO_MODEL_HINTS: dict[str, tuple[str, ...]] = {
    "decoder_only": ("AutoModelForCausalLM",),
    "encoder_only": ("AutoModelForMaskedLM", "AutoModel"),
    "encoder_decoder": ("AutoModelForSeq2SeqLM", "AutoModel"),
}
