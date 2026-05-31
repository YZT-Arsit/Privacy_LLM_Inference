"""Stage 5.3c — BERT FFN compatible-island probe tests."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from pllo.experiments.encoder_ffn_island_probe import (
    EncoderFFNIslandProbeConfig,
    run_encoder_ffn_island_probe,
)
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    VALID_NONLINEAR_MODES,
)


def _run(use_pad: bool, nonlinear_mode: str = "compatible_islands", seed: int = 1):
    cfg = EncoderFFNIslandProbeConfig(
        batch_size=2,
        seq_len=8,
        use_pad=use_pad,
        nonlinear_mode=nonlinear_mode,
        seed=seed,
    )
    result = run_encoder_ffn_island_probe(cfg)
    if result.get("status") == "skipped":
        pytest.skip(f"BERT FFN probe skipped: {result.get('reason')}")
    assert result["status"] == "loaded", result
    return result


# ---------------------------------------------------------------------------
# Mode acceptance
# ---------------------------------------------------------------------------


def test_bert_probe_accepts_nonlinear_modes() -> None:
    for mode in VALID_NONLINEAR_MODES:
        cfg = EncoderFFNIslandProbeConfig(
            batch_size=1, seq_len=4, use_pad=False, nonlinear_mode=mode
        )
        result = run_encoder_ffn_island_probe(cfg)
        if result.get("status") == "skipped":
            pytest.skip(f"BERT unavailable: {result.get('reason')}")
        assert result["status"] == "loaded"
        assert result["nonlinear_mode"] == mode


def test_bert_probe_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        EncoderFFNIslandProbeConfig(nonlinear_mode="not_a_mode")
        # normalize is called inside run_*; trigger it explicitly:
        cfg = EncoderFFNIslandProbeConfig(nonlinear_mode="not_a_mode")
        run_encoder_ffn_island_probe(cfg)


def test_bert_probe_default_mode_is_trusted() -> None:
    cfg = EncoderFFNIslandProbeConfig(batch_size=1, seq_len=4, use_pad=False)
    result = run_encoder_ffn_island_probe(cfg)
    if result.get("status") == "skipped":
        pytest.skip(f"BERT unavailable: {result.get('reason')}")
    assert result["nonlinear_mode"] == DEFAULT_NONLINEAR_MODE == "trusted"
    assert result["nonlinear_mode_active"] is False


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_compatible_bert_ffn_island_use_pad_false_allclose() -> None:
    result = _run(use_pad=False)
    assert result["nonlinear_mode_active"] is True
    metrics = result["ffn_metrics"]
    assert metrics["allclose"] is True, metrics
    tilde = result["tilde_invariant_metrics"]
    assert tilde["allclose"] is True


def test_compatible_bert_ffn_island_use_pad_true_allclose() -> None:
    result = _run(use_pad=True)
    assert result["nonlinear_mode_active"] is True
    metrics = result["ffn_metrics"]
    assert metrics["allclose"] is True, metrics


# ---------------------------------------------------------------------------
# Audit invariants
# ---------------------------------------------------------------------------


def test_bert_permutation_dim_equals_intermediate_size() -> None:
    result = _run(use_pad=False)
    assert result["permutation_dim"] == result["intermediate_size"]
    # Hidden size for BERT-tiny is 128, intermediate 512 — strict inequality.
    assert result["permutation_dim"] != result["hidden_size"]


def test_bert_online_extra_matmul_count_zero() -> None:
    result = _run(use_pad=True)
    assert result["online_extra_matmul_count"] == 0


def test_bert_pad_placement_linear_boundary_only_when_padded() -> None:
    result = _run(use_pad=True)
    assert result["pad_placement"] == "linear_boundary_only"


def test_bert_pad_placement_na_when_unpadded() -> None:
    result = _run(use_pad=False)
    assert result["pad_placement"] == "n/a"


def test_bert_layernorm_remains_trusted() -> None:
    for use_pad in (False, True):
        result = _run(use_pad=use_pad)
        assert result["layernorm_remains_trusted"] is True


def test_bert_mlm_pooler_classifier_not_integrated() -> None:
    result = _run(use_pad=False)
    assert result["mlm_head_not_modified"] is True
    assert result["pooler_not_modified"] is True
    assert result["classifier_not_modified"] is True


def test_bert_security_caveats_present() -> None:
    result = _run(use_pad=True)
    assert result["security_profile"] == "proxy-evaluated, not formal"
    text = " ".join(result["security_caveats"])
    assert "Compatible mask families are weaker" in text
    assert "Fresh permutation" in text
    assert "not a real TEE measurement" in text


def test_bert_activation_type_detected() -> None:
    result = _run(use_pad=False)
    assert result["activation_type"] in {"gelu", "relu", "silu"}
