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
    # Stage 6.4 — modern decoder-only models with RMSNorm + SwiGLU + RoPE
    # (TinyLlama / Qwen-style). Tried in order; the runner falls back to a
    # synthetic-tensor probe if every candidate fails to load locally.
    "modern_decoder_only": (
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ),
}


# Stage 6.4 — family metadata for modern decoder-only models.
MODERN_DECODER_FAMILY_MAP: dict[str, str] = {
    "hf-internal-testing/tiny-random-LlamaForCausalLM": "llama_like",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama",
    "Qwen/Qwen2.5-0.5B": "qwen",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen",
}


# Mapping from string keys to enum (handy when emitting structured output).
ARCH_KEY_TO_TYPE: dict[str, ArchitectureType] = {
    "decoder_only": ArchitectureType.DECODER_ONLY,
    "encoder_only": ArchitectureType.ENCODER_ONLY,
    "encoder_decoder": ArchitectureType.ENCODER_DECODER,
    # Stage 6.4 — modern decoder-only models are a *subtype* of
    # decoder_only; the top-level architecture_type remains the same.
    "modern_decoder_only": ArchitectureType.DECODER_ONLY,
}


# HuggingFace AutoModel class hints. The architecture inspector uses these to
# load each candidate checkpoint with the appropriate task head when present.
AUTO_MODEL_HINTS: dict[str, tuple[str, ...]] = {
    "decoder_only": ("AutoModelForCausalLM",),
    "encoder_only": ("AutoModelForMaskedLM", "AutoModel"),
    "encoder_decoder": ("AutoModelForSeq2SeqLM", "AutoModel"),
    "modern_decoder_only": ("AutoModelForCausalLM",),
}
