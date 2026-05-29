"""Tests for the Stage 6.0 architecture registry + attention taxonomy."""

from __future__ import annotations

import pytest

from pllo.architectures import (
    ARCH_KEY_TO_TYPE,
    ATTENTION_BY_NAME,
    ATTENTION_TAXONOMY,
    AUTO_MODEL_HINTS,
    BIDIRECTIONAL_SELF_ATTENTION,
    CAUSAL_SELF_ATTENTION,
    CROSS_ATTENTION,
    DEFAULT_ARCHITECTURE_MODELS,
    ArchitectureModelSpec,
    ArchitectureType,
    attention_kinds_for,
)
from pllo.architectures.encoder_decoder_spec import (
    BART_MODULE_PATHS,
    T5_MODULE_PATHS,
)
from pllo.architectures.encoder_only_spec import BERT_MODULE_PATHS


# ---------------------------------------------------------------------------
# Architecture type enum + registry coverage
# ---------------------------------------------------------------------------


def test_architecture_type_enum_has_three_families_plus_unknown() -> None:
    members = {member.value for member in ArchitectureType}
    assert members == {"decoder_only", "encoder_only", "encoder_decoder", "unknown"}


def test_registry_covers_three_architecture_keys() -> None:
    assert set(DEFAULT_ARCHITECTURE_MODELS.keys()) == {
        "decoder_only",
        "encoder_only",
        "encoder_decoder",
    }


def test_registry_includes_gpt2_for_decoder_only() -> None:
    assert "sshleifer/tiny-gpt2" in DEFAULT_ARCHITECTURE_MODELS["decoder_only"]


def test_registry_includes_a_tiny_bert_candidate() -> None:
    bert_candidates = DEFAULT_ARCHITECTURE_MODELS["encoder_only"]
    assert any("bert" in c.lower() for c in bert_candidates), bert_candidates


def test_registry_includes_a_tiny_t5_candidate() -> None:
    encdec_candidates = DEFAULT_ARCHITECTURE_MODELS["encoder_decoder"]
    assert any(
        "t5" in c.lower() or "bart" in c.lower() for c in encdec_candidates
    ), encdec_candidates


def test_arch_key_to_type_mapping_complete() -> None:
    assert ARCH_KEY_TO_TYPE["decoder_only"] is ArchitectureType.DECODER_ONLY
    assert ARCH_KEY_TO_TYPE["encoder_only"] is ArchitectureType.ENCODER_ONLY
    assert ARCH_KEY_TO_TYPE["encoder_decoder"] is ArchitectureType.ENCODER_DECODER


def test_auto_model_hints_cover_three_architecture_keys() -> None:
    assert set(AUTO_MODEL_HINTS.keys()) == set(DEFAULT_ARCHITECTURE_MODELS.keys())
    assert "AutoModelForCausalLM" in AUTO_MODEL_HINTS["decoder_only"]
    assert "AutoModelForSeq2SeqLM" in AUTO_MODEL_HINTS["encoder_decoder"]


# ---------------------------------------------------------------------------
# Attention taxonomy
# ---------------------------------------------------------------------------


def test_attention_taxonomy_has_all_three_kinds() -> None:
    names = {kind.name for kind in ATTENTION_TAXONOMY}
    assert names == {
        "causal_self_attention",
        "bidirectional_self_attention",
        "cross_attention",
    }


def test_attention_by_name_lookup() -> None:
    assert ATTENTION_BY_NAME["causal_self_attention"] is CAUSAL_SELF_ATTENTION
    assert ATTENTION_BY_NAME["bidirectional_self_attention"] is BIDIRECTIONAL_SELF_ATTENTION
    assert ATTENTION_BY_NAME["cross_attention"] is CROSS_ATTENTION


def test_causal_attention_uses_decoder_state_for_qkv() -> None:
    assert CAUSAL_SELF_ATTENTION.q_source == "decoder_hidden_states"
    assert CAUSAL_SELF_ATTENTION.k_source == "decoder_hidden_states"
    assert CAUSAL_SELF_ATTENTION.v_source == "decoder_hidden_states"
    assert CAUSAL_SELF_ATTENTION.mask_type == "causal"
    assert CAUSAL_SELF_ATTENTION.cache_type == "autoregressive_kv_cache"


def test_bidirectional_attention_has_no_cache() -> None:
    assert BIDIRECTIONAL_SELF_ATTENTION.cache_type == "none"
    assert BIDIRECTIONAL_SELF_ATTENTION.mask_type == "padding_bidirectional"
    assert BIDIRECTIONAL_SELF_ATTENTION.q_source == "encoder_hidden_states"
    assert BIDIRECTIONAL_SELF_ATTENTION.architecture_type is ArchitectureType.ENCODER_ONLY


def test_cross_attention_q_is_decoder_kv_is_encoder() -> None:
    assert CROSS_ATTENTION.q_source == "decoder_hidden_states"
    assert CROSS_ATTENTION.k_source == "encoder_memory"
    assert CROSS_ATTENTION.v_source == "encoder_memory"
    assert CROSS_ATTENTION.cache_type == "encoder_memory_cache"
    assert CROSS_ATTENTION.mask_type == "encoder_padding_mask"
    assert CROSS_ATTENTION.architecture_type is ArchitectureType.ENCODER_DECODER


def test_required_invariants_cover_kv_cache_and_qk_constraint() -> None:
    causal = CAUSAL_SELF_ATTENTION.required_invariant
    assert "Q_tilde K_tilde^T = Q K^T" in causal
    assert "K_cache_tilde" in causal and "V_cache_tilde" in causal

    bidir = BIDIRECTIONAL_SELF_ATTENTION.required_invariant
    assert "Q_tilde K_tilde^T" in bidir
    assert "no autoregressive cache" in bidir.lower()

    cross = CROSS_ATTENTION.required_invariant
    assert "Q_dec_tilde K_enc_tilde^T" in cross
    assert "K_enc_tilde" in cross and "V_enc_tilde" in cross


# ---------------------------------------------------------------------------
# attention_kinds_for() helper
# ---------------------------------------------------------------------------


def test_attention_kinds_for_decoder_only_is_causal_only() -> None:
    kinds = attention_kinds_for(ArchitectureType.DECODER_ONLY)
    assert kinds == (CAUSAL_SELF_ATTENTION,)


def test_attention_kinds_for_encoder_only_is_bidirectional_only() -> None:
    kinds = attention_kinds_for(ArchitectureType.ENCODER_ONLY)
    assert kinds == (BIDIRECTIONAL_SELF_ATTENTION,)


def test_attention_kinds_for_encoder_decoder_covers_all_three() -> None:
    kinds = attention_kinds_for(ArchitectureType.ENCODER_DECODER)
    names = {k.name for k in kinds}
    assert {"bidirectional_self_attention", "causal_self_attention", "cross_attention"} <= names


def test_attention_kinds_for_unknown_is_empty() -> None:
    assert attention_kinds_for(ArchitectureType.UNKNOWN) == ()


# ---------------------------------------------------------------------------
# Module-path metadata (BERT / T5 / BART)
# ---------------------------------------------------------------------------


def test_bert_module_paths_have_qkv_under_self_attention() -> None:
    assert BERT_MODULE_PATHS.self_attention_q.endswith(".query")
    assert BERT_MODULE_PATHS.self_attention_k.endswith(".key")
    assert BERT_MODULE_PATHS.self_attention_v.endswith(".value")
    assert BERT_MODULE_PATHS.mlm_head == "cls.predictions"


def test_t5_module_paths_have_cross_attention_in_decoder() -> None:
    assert "EncDecAttention" in T5_MODULE_PATHS.cross_attention_path
    assert T5_MODULE_PATHS.encoder_layer_list == "encoder.block"
    assert T5_MODULE_PATHS.decoder_layer_list == "decoder.block"


def test_bart_module_paths_use_layers_not_block() -> None:
    assert BART_MODULE_PATHS.encoder_layer_list == "encoder.layers"
    assert BART_MODULE_PATHS.decoder_layer_list == "decoder.layers"
    assert BART_MODULE_PATHS.cross_attention_path == "encoder_attn"


# ---------------------------------------------------------------------------
# ArchitectureModelSpec defaults
# ---------------------------------------------------------------------------


def test_architecture_model_spec_constructs_with_minimal_args() -> None:
    spec = ArchitectureModelSpec(
        model_id="dummy",
        architecture_type=ArchitectureType.DECODER_ONLY,
        model_class="DummyLM",
        has_encoder=False,
        has_decoder=True,
        has_cross_attention=False,
        has_causal_self_attention=True,
        has_bidirectional_self_attention=False,
        supports_past_key_values=True,
        has_lm_head=True,
        has_mlm_head=False,
        has_classification_head=False,
        vocab_size=100,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
    )
    # Default notes is empty list, not None.
    assert spec.notes == []
