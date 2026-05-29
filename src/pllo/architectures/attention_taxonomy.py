"""Static attention-kind taxonomy for the three Transformer architecture types."""

from __future__ import annotations

from pllo.architectures.architecture_types import ArchitectureType, AttentionKindSpec


CAUSAL_SELF_ATTENTION = AttentionKindSpec(
    name="causal_self_attention",
    architecture_type=ArchitectureType.DECODER_ONLY,
    q_source="decoder_hidden_states",
    k_source="decoder_hidden_states",
    v_source="decoder_hidden_states",
    mask_type="causal",
    cache_type="autoregressive_kv_cache",
    required_invariant=(
        "Q_tilde K_tilde^T = Q K^T (per-head N_Q N_K^T = I); "
        "K_cache_tilde = K_cache N_K; V_cache_tilde = V_cache N_V."
    ),
)


BIDIRECTIONAL_SELF_ATTENTION = AttentionKindSpec(
    name="bidirectional_self_attention",
    architecture_type=ArchitectureType.ENCODER_ONLY,
    q_source="encoder_hidden_states",
    k_source="encoder_hidden_states",
    v_source="encoder_hidden_states",
    mask_type="padding_bidirectional",
    cache_type="none",
    required_invariant=(
        "Q_tilde K_tilde^T = Q K^T (per-head N_Q N_K^T = I); "
        "no autoregressive cache (the whole sequence is seen at once)."
    ),
)


CROSS_ATTENTION = AttentionKindSpec(
    name="cross_attention",
    architecture_type=ArchitectureType.ENCODER_DECODER,
    q_source="decoder_hidden_states",
    k_source="encoder_memory",
    v_source="encoder_memory",
    mask_type="encoder_padding_mask",
    cache_type="encoder_memory_cache",
    required_invariant=(
        "Q_dec_tilde K_enc_tilde^T = Q_dec K_enc^T; "
        "K_enc_tilde = K_enc N_K; V_enc_tilde = V_enc N_V "
        "(encoder memory cached once per generation)."
    ),
)


ATTENTION_TAXONOMY: tuple[AttentionKindSpec, ...] = (
    CAUSAL_SELF_ATTENTION,
    BIDIRECTIONAL_SELF_ATTENTION,
    CROSS_ATTENTION,
)


ATTENTION_BY_NAME: dict[str, AttentionKindSpec] = {kind.name: kind for kind in ATTENTION_TAXONOMY}


def attention_kinds_for(arch_type: ArchitectureType) -> tuple[AttentionKindSpec, ...]:
    """Return the attention kinds that should appear inside a given architecture."""
    if arch_type is ArchitectureType.DECODER_ONLY:
        return (CAUSAL_SELF_ATTENTION,)
    if arch_type is ArchitectureType.ENCODER_ONLY:
        return (BIDIRECTIONAL_SELF_ATTENTION,)
    if arch_type is ArchitectureType.ENCODER_DECODER:
        return (BIDIRECTIONAL_SELF_ATTENTION, CAUSAL_SELF_ATTENTION, CROSS_ATTENTION)
    return ()
