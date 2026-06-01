"""Stage 6.4c — model-level wrapper tests (synthetic model, no network)."""

from __future__ import annotations

import json
import re

import pytest
import torch

from pllo.hf_wrappers.modern_decoder_model_wrapper import (
    ModernDecoderModelWeights,
    ObfuscatedModernDecoderModelWrapper,
    plain_model_forward,
)


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


@pytest.fixture
def synthetic_weights() -> ModernDecoderModelWeights:
    torch.manual_seed(0)
    return ModernDecoderModelWeights.from_synthetic(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_layers=2, seed=0,
    )


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 32, (1, 6))


def _allclose(a, b, atol=1e-4, rtol=1e-4):
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))


# ---------------------------------------------------------------------------
# Full forward — both bundles, both use_pad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bundle", [
    "fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad",
])
@pytest.mark.parametrize("use_pad", [False, True])
def test_full_forward_allclose(synthetic_weights, input_ids, bundle, use_pad) -> None:
    torch.manual_seed(7)
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle=bundle,
        use_pad=use_pad,
    )
    logits, report = wrapper.full_forward(input_ids)
    plain = plain_model_forward(input_ids, synthetic_weights)
    assert _allclose(logits, plain), report["logits_metrics"]
    assert report["logits_metrics"]["allclose"] is True
    assert report["logits_metrics"]["top1_match_rate"] == 1.0
    assert report["online_extra_matmul_count"] == 0


# ---------------------------------------------------------------------------
# Trusted mode default behaviour
# ---------------------------------------------------------------------------


def test_default_mode_is_trusted(synthetic_weights) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(synthetic_weights)
    assert wrapper.nonlinear_mode == "trusted"
    assert wrapper.mitigation_bundle == "fresh_perm_only"


def test_trusted_mode_full_forward_exact(synthetic_weights, input_ids) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights, nonlinear_mode="trusted",
    )
    logits, report = wrapper.full_forward(input_ids)
    plain = plain_model_forward(input_ids, synthetic_weights)
    assert torch.equal(logits, plain)
    assert report["final_norm_status"] == "trusted_shortcut"
    assert report["lm_head_status"] == "trusted_shortcut"


def test_compatible_islands_must_be_explicit(synthetic_weights, input_ids) -> None:
    """Default config must NOT activate compatible_islands."""
    wrapper = ObfuscatedModernDecoderModelWrapper(synthetic_weights)
    _, report = wrapper.full_forward(input_ids)
    assert report["nonlinear_mode"] == "trusted"
    assert report["lm_head_status"] == "trusted_shortcut"


# ---------------------------------------------------------------------------
# LM head recovery
# ---------------------------------------------------------------------------


def test_compatible_mode_lm_head_recovery_allclose(synthetic_weights, input_ids) -> None:
    torch.manual_seed(11)
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    logits, report = wrapper.full_forward(input_ids)
    assert report["lm_head_status"] == "single_dense_mask_pair_with_vocab_mask"
    plain = plain_model_forward(input_ids, synthetic_weights)
    assert _allclose(logits, plain)


def test_final_norm_status_metadata_present(synthetic_weights, input_ids) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights, nonlinear_mode="compatible_islands",
    )
    _, report = wrapper.full_forward(input_ids)
    assert report["final_norm_status"] in {
        "trusted_shortcut", "trusted_final_rmsnorm",
    }


# ---------------------------------------------------------------------------
# Mitigation metadata
# ---------------------------------------------------------------------------


def test_full_bundle_metadata_correct(synthetic_weights, input_ids) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    _, report = wrapper.full_forward(input_ids)
    meta = report["mitigation_bundle_metadata"]
    assert meta["dense_sandwich_enabled"] is True
    assert meta["boundary_pad_enabled"] is True
    assert meta["default_on_candidate_under_stage_5_4"] is True
    assert meta["online_extra_matmul_count"] == 0


# ---------------------------------------------------------------------------
# Trace hook available but off by default
# ---------------------------------------------------------------------------


def test_trace_hook_disabled_by_default(synthetic_weights) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(synthetic_weights)
    assert wrapper.collect_traces is False


def test_trace_hook_enabled_when_requested(synthetic_weights) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights, collect_traces=True,
    )
    assert wrapper.collect_traces is True


# ---------------------------------------------------------------------------
# No secret tensor in report
# ---------------------------------------------------------------------------


def test_report_has_no_secret_tensor(synthetic_weights, input_ids) -> None:
    wrapper = ObfuscatedModernDecoderModelWrapper(
        synthetic_weights,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    _, report = wrapper.full_forward(input_ids)
    blob = json.dumps(report)
    assert "tensor(" not in blob
    assert _LONG_NUMBER_ARRAY.search(blob) is None


# ---------------------------------------------------------------------------
# Invalid bundle / mode
# ---------------------------------------------------------------------------


def test_invalid_mitigation_bundle_raises(synthetic_weights) -> None:
    with pytest.raises(ValueError):
        ObfuscatedModernDecoderModelWrapper(
            synthetic_weights, mitigation_bundle="not_a_bundle",
        )


def test_invalid_nonlinear_mode_raises(synthetic_weights) -> None:
    with pytest.raises(ValueError):
        ObfuscatedModernDecoderModelWrapper(
            synthetic_weights, nonlinear_mode="raw",
        )


# ---------------------------------------------------------------------------
# Real tiny LLaMA — opt-in, skip if unavailable
# ---------------------------------------------------------------------------


def test_real_tiny_llama_full_forward_allclose() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny-random LLaMA unavailable: {exc}")
    from pllo.model_zoo.modern_decoder_spec import inspect_modern_decoder_block

    spec = inspect_modern_decoder_block(
        model, model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
    )
    weights = ModernDecoderModelWeights.from_hf_model(model, spec=spec, max_layers=2)
    input_ids = torch.randint(0, weights.vocab_size, (1, 4))
    torch.manual_seed(2026)
    wrapper = ObfuscatedModernDecoderModelWrapper(
        weights, nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
        use_pad=True,
    )
    logits, report = wrapper.full_forward(input_ids)
    assert report["logits_metrics"]["allclose"] is True
    assert report["logits_metrics"]["top1_match_rate"] == 1.0
