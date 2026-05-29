"""Reference module paths for BERT-style encoder-only models (Stage 6.0).

These constants document where each submodule lives inside a BERT-style
``transformers`` model so future obfuscated-wrapper stages know exactly which
linears to extract. Stage 6.0 only records this metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderOnlyModulePaths:
    embedding: str
    encoder_layer_list: str
    self_attention_path: str
    self_attention_q: str
    self_attention_k: str
    self_attention_v: str
    self_attention_output: str
    ffn_intermediate: str
    ffn_output: str
    layernorm_attn: str
    layernorm_output: str
    pooler: str | None
    mlm_head: str | None
    classification_head: str | None


# Paths for canonical BERT (``BertModel`` / ``BertForMaskedLM``).
BERT_MODULE_PATHS = EncoderOnlyModulePaths(
    embedding="embeddings.word_embeddings",
    encoder_layer_list="encoder.layer",
    self_attention_path="attention.self",
    self_attention_q="attention.self.query",
    self_attention_k="attention.self.key",
    self_attention_v="attention.self.value",
    self_attention_output="attention.output.dense",
    ffn_intermediate="intermediate.dense",
    ffn_output="output.dense",
    layernorm_attn="attention.output.LayerNorm",
    layernorm_output="output.LayerNorm",
    pooler="pooler",
    mlm_head="cls.predictions",
    classification_head="classifier",
)
