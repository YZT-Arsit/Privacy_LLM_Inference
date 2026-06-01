"""Stage 5.6 extension — tests for inter_block_mask_mode in the model wrapper."""

from __future__ import annotations

import pytest
import torch

from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    DEFAULT_INTER_BLOCK_MASK_MODE,
    VALID_INTER_BLOCK_MASK_MODES,
    ModernDecoderModelWeights,
    ObfuscatedModernDecoderModelWrapper,
    normalize_inter_block_mask_mode,
)


def _weights():
    return ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        num_layers=2, seed=2026,
    )


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_normalize_none_returns_plain_boundary() -> None:
    assert normalize_inter_block_mask_mode(None) == "plain_boundary"
    assert DEFAULT_INTER_BLOCK_MASK_MODE == "plain_boundary"


def test_normalize_valid_pass_through() -> None:
    for m in VALID_INTER_BLOCK_MASK_MODES:
        assert normalize_inter_block_mask_mode(m) == m


def test_normalize_invalid_raises() -> None:
    with pytest.raises(ValueError):
        normalize_inter_block_mask_mode("not_a_mode")


def test_default_wrapper_plain_boundary_unchanged() -> None:
    """The default wrapper (no mode argument) must keep plain_boundary."""
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(w)
    assert wrapper.inter_block_mask_mode == "plain_boundary"
    assert wrapper.nonlinear_mode == "trusted"
    assert wrapper.mitigation_bundle == "fresh_perm_only"
    assert wrapper.use_pad is False


def test_masked_boundary_requires_compatible_islands() -> None:
    w = _weights()
    with pytest.raises(ValueError):
        ObfuscatedModernDecoderModelWrapper(
            w, nonlinear_mode="trusted",
            inter_block_mask_mode="masked_boundary_experimental",
        )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bundle", "use_pad"),
    [
        ("fresh_perm_only", False),
        ("fresh_perm_only", True),
        ("fresh_perm_plus_sandwich_plus_pad", False),
        ("fresh_perm_plus_sandwich_plus_pad", True),
    ],
)
def test_masked_full_forward_allclose(bundle: str, use_pad: bool) -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle=bundle, use_pad=use_pad,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    torch.manual_seed(2026)
    input_ids = torch.randint(0, 32, (1, 6))
    logits, report = wrapper.full_forward(input_ids)
    assert report["logits_metrics"]["allclose"] is True
    assert report["logits_metrics"]["top1_match_rate"] == 1.0


def test_masked_prefill_allclose() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    input_ids = torch.randint(0, 32, (1, 6))
    pf = wrapper.prefill(input_ids)
    assert pf["report"]["logits_metrics"]["allclose"] is True


def test_masked_decode_step_allclose() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    input_ids = torch.randint(0, 32, (1, 6))
    pf = wrapper.prefill(input_ids)
    next_t = pf["logits_recovered"][:, -1, :].argmax(dim=-1)
    ds = wrapper.decode_step(
        next_t, pf["kv_cache"], position=6,
        plain_layer_caches=pf["plain_layer_caches"],
    )
    assert ds["report"]["logits_metrics"]["allclose"] is True
    assert ds["report"]["logits_metrics"]["top1_match_rate"] == 1.0


def test_masked_greedy_generate_matches_plain() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    input_ids = torch.randint(0, 32, (1, 6))
    g = wrapper.greedy_generate(input_ids, max_new_tokens=3)
    assert g["report"]["sequence_exact_match"] is True
    assert g["report"]["token_match_rate"] == 1.0


# ---------------------------------------------------------------------------
# Metadata + boundary mask state
# ---------------------------------------------------------------------------


def test_masked_report_metadata_flags_boundary_masked() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    input_ids = torch.randint(0, 32, (1, 6))
    _, report = wrapper.full_forward(input_ids)
    assert report["inter_block_mask_mode"] == "masked_boundary_experimental"
    assert report["inter_block_plain_recovered"] is False
    assert report["boundary_mask_status"] == "masked"
    assert report["final_mask_status"] == "masked_until_lm_head"


def test_plain_boundary_report_metadata_flags_plain() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
    )
    input_ids = torch.randint(0, 32, (1, 6))
    _, report = wrapper.full_forward(input_ids)
    assert report["inter_block_mask_mode"] == "plain_boundary"
    assert report["inter_block_plain_recovered"] is True
    assert report["boundary_mask_status"] == "plain"


def test_online_extra_matmul_count_remains_zero() -> None:
    w = _weights()
    wrapper = ObfuscatedModernDecoderModelWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
        inter_block_mask_mode="masked_boundary_experimental",
    )
    input_ids = torch.randint(0, 32, (1, 6))
    _, report = wrapper.full_forward(input_ids)
    assert report["online_extra_matmul_count"] == 0
