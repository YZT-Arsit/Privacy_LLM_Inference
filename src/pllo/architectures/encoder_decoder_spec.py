"""Reference module paths for encoder-decoder models (T5 / BART, Stage 6.0).

T5 and BART expose subtly different layouts. The constants below document
both so future stages can write per-family extraction adapters analogous to
``pllo.model_zoo.gpt2_conv1d_adapter`` without re-deriving the structure each
time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderDecoderModulePaths:
    shared_embedding: str
    encoder_layer_list: str
    decoder_layer_list: str
    encoder_self_attention_path: str
    encoder_self_attention_q: str
    encoder_self_attention_k: str
    encoder_self_attention_v: str
    decoder_self_attention_path: str
    decoder_self_attention_q: str
    decoder_self_attention_k: str
    decoder_self_attention_v: str
    cross_attention_path: str
    cross_attention_q: str
    cross_attention_k: str
    cross_attention_v: str
    encoder_ffn_path: str
    decoder_ffn_path: str
    encoder_layernorm_self_attn: str
    decoder_layernorm_self_attn: str
    decoder_layernorm_cross_attn: str
    lm_head: str | None


# T5 (HuggingFace ``T5Model`` / ``T5ForConditionalGeneration``).
T5_MODULE_PATHS = EncoderDecoderModulePaths(
    shared_embedding="shared",
    encoder_layer_list="encoder.block",
    decoder_layer_list="decoder.block",
    encoder_self_attention_path="layer.0.SelfAttention",
    encoder_self_attention_q="layer.0.SelfAttention.q",
    encoder_self_attention_k="layer.0.SelfAttention.k",
    encoder_self_attention_v="layer.0.SelfAttention.v",
    decoder_self_attention_path="layer.0.SelfAttention",
    decoder_self_attention_q="layer.0.SelfAttention.q",
    decoder_self_attention_k="layer.0.SelfAttention.k",
    decoder_self_attention_v="layer.0.SelfAttention.v",
    cross_attention_path="layer.1.EncDecAttention",
    cross_attention_q="layer.1.EncDecAttention.q",
    cross_attention_k="layer.1.EncDecAttention.k",
    cross_attention_v="layer.1.EncDecAttention.v",
    encoder_ffn_path="layer.1.DenseReluDense",
    decoder_ffn_path="layer.2.DenseReluDense",
    encoder_layernorm_self_attn="layer.0.layer_norm",
    decoder_layernorm_self_attn="layer.0.layer_norm",
    decoder_layernorm_cross_attn="layer.1.layer_norm",
    lm_head="lm_head",
)


# BART (HuggingFace ``BartModel`` / ``BartForConditionalGeneration``).
BART_MODULE_PATHS = EncoderDecoderModulePaths(
    shared_embedding="shared",
    encoder_layer_list="encoder.layers",
    decoder_layer_list="decoder.layers",
    encoder_self_attention_path="self_attn",
    encoder_self_attention_q="self_attn.q_proj",
    encoder_self_attention_k="self_attn.k_proj",
    encoder_self_attention_v="self_attn.v_proj",
    decoder_self_attention_path="self_attn",
    decoder_self_attention_q="self_attn.q_proj",
    decoder_self_attention_k="self_attn.k_proj",
    decoder_self_attention_v="self_attn.v_proj",
    cross_attention_path="encoder_attn",
    cross_attention_q="encoder_attn.q_proj",
    cross_attention_k="encoder_attn.k_proj",
    cross_attention_v="encoder_attn.v_proj",
    encoder_ffn_path="fc1",
    decoder_ffn_path="fc1",
    encoder_layernorm_self_attn="self_attn_layer_norm",
    decoder_layernorm_self_attn="self_attn_layer_norm",
    decoder_layernorm_cross_attn="encoder_attn_layer_norm",
    lm_head="lm_head",
)
