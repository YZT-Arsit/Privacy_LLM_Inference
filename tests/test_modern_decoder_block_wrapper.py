"""Stage 6.4b — block-level obfuscated wrapper tests for modern decoders.

All tests run against a synthetic LLaMA-shape block (no network, no HF).
"""

from __future__ import annotations

import pytest
import torch

from pllo.hf_wrappers.modern_decoder_block_wrapper import (
    ModernDecoderBlockWeights,
    ObfuscatedModernDecoderBlockWrapper,
    plain_block_forward,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_weights() -> ModernDecoderBlockWeights:
    return ModernDecoderBlockWeights.from_synthetic(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        seed=2026,
    )


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 8, 64)


def _allclose_default(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.allclose(a, b, atol=1e-4, rtol=1e-4))


# ---------------------------------------------------------------------------
# Plain reference
# ---------------------------------------------------------------------------


def test_plain_block_forward_shape(synthetic_weights, x) -> None:
    out = plain_block_forward(x, synthetic_weights)
    assert out["y"].shape == x.shape
    assert out["h_mid"].shape == x.shape
    assert out["attn_out"].shape == x.shape
    assert out["mlp_out"].shape == x.shape


# ---------------------------------------------------------------------------
# Compatible islands allclose (the four combinations specified)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bundle", ["fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"])
@pytest.mark.parametrize("use_pad", [False, True])
def test_compatible_islands_allclose(synthetic_weights, x, bundle, use_pad) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
        mitigation_bundle=bundle,
    )
    y_rec, report = wrapper.forward(x)
    plain = plain_block_forward(x, synthetic_weights)["y"]
    assert _allclose_default(y_rec, plain), (
        f"y_rec vs plain mismatch: max_err={report['max_abs_error']}"
        f" rel_l2={report['relative_l2_error']}"
    )
    assert report["allclose"] is True
    assert report["online_extra_matmul_count"] == 0
    assert report["mitigation_bundle"] == bundle
    assert report["use_pad"] is use_pad


def test_online_extra_matmul_count_is_zero_everywhere(synthetic_weights, x) -> None:
    for bundle in ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"):
        for use_pad in (False, True):
            wrapper = ObfuscatedModernDecoderBlockWrapper(
                synthetic_weights,
                use_pad=use_pad,
                nonlinear_mode="compatible_islands",
                mitigation_bundle=bundle,
            )
            _, report = wrapper.forward(x)
            assert report["online_extra_matmul_count"] == 0


# ---------------------------------------------------------------------------
# Mitigation bundle metadata
# ---------------------------------------------------------------------------


def test_full_bundle_metadata_when_use_pad_true(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        use_pad=True,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    _, report = wrapper.forward(x)
    assert report["dense_sandwich_enabled"] is True
    assert report["boundary_pad_enabled"] is True
    assert report["default_on_candidate_under_stage_5_4"] is True
    assert report["fresh_permutation_enabled"] is True
    assert report["activation_input_form"] == "ZP"


def test_full_bundle_without_pad_not_default_on_candidate(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        use_pad=False,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    _, report = wrapper.forward(x)
    assert report["dense_sandwich_enabled"] is True
    assert report["boundary_pad_enabled"] is False
    assert report["default_on_candidate_under_stage_5_4"] is False


def test_fresh_perm_only_default_bundle_metadata(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights, nonlinear_mode="compatible_islands"
    )
    _, report = wrapper.forward(x)
    assert report["mitigation_bundle"] == "fresh_perm_only"
    assert report["dense_sandwich_enabled"] is False
    assert report["default_on_candidate_under_stage_5_4"] is False


# ---------------------------------------------------------------------------
# Sub-path handling fields populated
# ---------------------------------------------------------------------------


def test_handling_status_fields_populated(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    _, report = wrapper.forward(x)
    assert report["rmsnorm_status"] == (
        "orthogonal_island_with_gamma_folded_into_qkv"
    )
    assert report["rope_attention_status"] == "rope_post_mask_only"
    assert report["gqa_status"] == "per_kv_head_mask_with_repeat_kv"
    assert report["swiglu_status"] == "compatible_island_paired_permutation"
    assert "residual_alignment_status" in report


def test_intermediate_metrics_attn_and_mlp(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    _, report = wrapper.forward(x)
    inter = report["intermediate_metrics"]
    assert inter["attn_branch"]["allclose"] is True
    assert inter["mlp_branch"]["allclose"] is True
    assert inter["h_mid"]["allclose"] is True


# ---------------------------------------------------------------------------
# GQA / MQA shape handling
# ---------------------------------------------------------------------------


def test_mqa_shape_allclose() -> None:
    w = ModernDecoderBlockWeights.from_synthetic(
        hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=1, head_dim=8,
    )
    x = torch.randn(2, 6, 32)
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        w, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad", use_pad=True,
    )
    _, report = wrapper.forward(x)
    assert report["allclose"] is True


def test_mha_shape_allclose() -> None:
    w = ModernDecoderBlockWeights.from_synthetic(
        hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
    )
    x = torch.randn(2, 6, 32)
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        w, nonlinear_mode="compatible_islands",
    )
    _, report = wrapper.forward(x)
    assert report["allclose"] is True


# ---------------------------------------------------------------------------
# Default mode and bundle remain trusted / fresh_perm_only
# ---------------------------------------------------------------------------


def test_default_nonlinear_mode_is_trusted(synthetic_weights) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(synthetic_weights)
    assert wrapper.nonlinear_mode == "trusted"


def test_default_mitigation_bundle_is_fresh_perm_only(synthetic_weights) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(synthetic_weights)
    assert wrapper.mitigation_bundle == "fresh_perm_only"


def test_trusted_mode_unaffected_by_bundle(synthetic_weights, x) -> None:
    a = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights, nonlinear_mode="trusted",
        mitigation_bundle="fresh_perm_only",
    ).forward(x)[0]
    b = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights, nonlinear_mode="trusted",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    ).forward(x)[0]
    assert torch.equal(a, b)


def test_invalid_mitigation_bundle_raises(synthetic_weights) -> None:
    with pytest.raises(ValueError):
        ObfuscatedModernDecoderBlockWrapper(
            synthetic_weights, mitigation_bundle="not_a_bundle"
        )


def test_invalid_nonlinear_mode_raises(synthetic_weights) -> None:
    with pytest.raises(ValueError):
        ObfuscatedModernDecoderBlockWrapper(
            synthetic_weights, nonlinear_mode="raw"
        )


# ---------------------------------------------------------------------------
# Caveats / honesty
# ---------------------------------------------------------------------------


def test_report_includes_honesty_caveats(synthetic_weights, x) -> None:
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    _, report = wrapper.forward(x)
    caveats = " ".join(report["caveats"]).lower()
    assert "block-level" in caveats
    assert "not full" not in caveats or "not a full" in caveats
    assert "no generation" in caveats
    assert "not a real tee" in caveats
    assert "not formal security" in caveats


# ---------------------------------------------------------------------------
# Real tiny LLaMA — skip if HF hub / transformers unavailable
# ---------------------------------------------------------------------------


def test_real_tiny_llama_block_allclose() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny-random LLaMA unavailable: {exc}")
    from pllo.model_zoo.modern_decoder_spec import inspect_modern_decoder_block

    spec = inspect_modern_decoder_block(
        model, model_id="hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    block = model.model.layers[0]
    weights = ModernDecoderBlockWeights.from_hf_block(block, spec)
    torch.manual_seed(123)
    x = torch.randn(2, 6, spec.hidden_size, dtype=torch.float32)
    for bundle in ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"):
        for use_pad in (False, True):
            wrapper = ObfuscatedModernDecoderBlockWrapper(
                weights,
                nonlinear_mode="compatible_islands",
                mitigation_bundle=bundle,
                use_pad=use_pad,
            )
            y_rec, report = wrapper.forward(x)
            assert report["allclose"] is True, (
                f"bundle={bundle} use_pad={use_pad}"
                f" max_err={report['max_abs_error']}"
            )
            assert report["online_extra_matmul_count"] == 0
