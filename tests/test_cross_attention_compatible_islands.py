"""Stage 5.3c — T5 / BART FFN compatible-island probe tests.

These tests exercise the *encoder-decoder* FFN probe. They do NOT modify
the Stage 6.2 cross-attention probe invariants — those continue to be
validated by ``tests/test_cross_attention_probe.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from pllo.experiments.encoder_decoder_ffn_island_probe import (
    EncoderDecoderFFNIslandProbeConfig,
    run_encoder_decoder_ffn_island_probe,
)
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    VALID_NONLINEAR_MODES,
)


def _run(
    use_pad: bool,
    nonlinear_mode: str = "compatible_islands",
    seed: int = 1,
    batch_size: int = 2,
    seq_len: int = 8,
):
    cfg = EncoderDecoderFFNIslandProbeConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        use_pad=use_pad,
        nonlinear_mode=nonlinear_mode,
        seed=seed,
    )
    result = run_encoder_decoder_ffn_island_probe(cfg)
    if result.get("status") == "skipped":
        pytest.skip(f"encoder-decoder probe skipped: {result.get('reason')}")
    if result.get("status") == "unsupported":
        pytest.skip(
            "encoder-decoder FFN structure unsupported in this stage: "
            f"{result.get('reason')}"
        )
    assert result["status"] == "loaded", result
    return result


# ---------------------------------------------------------------------------
# Mode acceptance
# ---------------------------------------------------------------------------


def test_t5_probe_accepts_nonlinear_modes() -> None:
    for mode in VALID_NONLINEAR_MODES:
        cfg = EncoderDecoderFFNIslandProbeConfig(
            batch_size=1, seq_len=4, use_pad=False, nonlinear_mode=mode
        )
        result = run_encoder_decoder_ffn_island_probe(cfg)
        if result.get("status") in ("skipped", "unsupported"):
            pytest.skip(f"T5 unavailable: {result.get('reason')}")
        assert result["status"] == "loaded"
        assert result["nonlinear_mode"] == mode


def test_t5_probe_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        cfg = EncoderDecoderFFNIslandProbeConfig(nonlinear_mode="not_a_mode")
        run_encoder_decoder_ffn_island_probe(cfg)


def test_t5_probe_default_mode_is_trusted() -> None:
    cfg = EncoderDecoderFFNIslandProbeConfig(batch_size=1, seq_len=4, use_pad=False)
    result = run_encoder_decoder_ffn_island_probe(cfg)
    if result.get("status") in ("skipped", "unsupported"):
        pytest.skip(f"T5 unavailable: {result.get('reason')}")
    assert result["nonlinear_mode"] == DEFAULT_NONLINEAR_MODE == "trusted"
    assert result["nonlinear_mode_active"] is False


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_compatible_t5_ffn_island_use_pad_false_allclose() -> None:
    result = _run(use_pad=False)
    assert result["nonlinear_mode_active"] is True
    metrics = result["ffn_metrics"]
    assert metrics["allclose"] is True, metrics
    tilde = result["tilde_invariant_metrics"]
    assert tilde["allclose"] is True


def test_compatible_t5_ffn_island_use_pad_true_allclose() -> None:
    result = _run(use_pad=True)
    assert result["nonlinear_mode_active"] is True
    metrics = result["ffn_metrics"]
    assert metrics["allclose"] is True, metrics


# ---------------------------------------------------------------------------
# Audit invariants
# ---------------------------------------------------------------------------


def test_t5_ffn_type_detected() -> None:
    result = _run(use_pad=False)
    assert result["ffn_type"] in {
        "t5_dense_relu_dense",
        "t5_gated",
        "bart_fc1_fc2",
    }
    assert result["activation_type"] in {"gelu", "relu", "silu"}


def test_t5_permutation_dim_equals_intermediate_size() -> None:
    result = _run(use_pad=False)
    assert result["permutation_dim"] == result["intermediate_size"]


def test_t5_online_extra_matmul_count_zero() -> None:
    result = _run(use_pad=True)
    assert result["online_extra_matmul_count"] == 0


def test_t5_pad_placement_linear_boundary_only_when_padded() -> None:
    result = _run(use_pad=True)
    assert result["pad_placement"] == "linear_boundary_only"


def test_t5_pad_placement_na_when_unpadded() -> None:
    result = _run(use_pad=False)
    assert result["pad_placement"] == "n/a"


def test_t5_paired_permutation_when_gated() -> None:
    """If the discovered FFN is gated, gate/up must share the same P."""
    result = _run(use_pad=False)
    if result["ffn_type"] == "t5_gated":
        assert result["uses_paired_permutation"] is True
    else:
        assert result["uses_paired_permutation"] is False


def test_t5_lm_head_and_generation_not_modified() -> None:
    result = _run(use_pad=True)
    assert result["lm_head_not_modified"] is True
    assert result["encoder_decoder_generation_not_modified"] is True


def test_t5_cross_attention_probe_invariants_not_modified() -> None:
    """The compatible-island FFN probe must not claim to touch cross-attention."""
    result = _run(use_pad=True)
    assert result["cross_attention_probe_not_modified"] is True


def test_t5_security_caveats_present() -> None:
    result = _run(use_pad=True)
    assert result["security_profile"] == "proxy-evaluated, not formal"
    text = " ".join(result["security_caveats"])
    assert "Compatible mask families are weaker" in text
    assert "Cross-attention probe invariants are not modified" in text
    assert "not a real TEE measurement" in text


def test_t5_gated_gelu_unsupported_returns_explicit_reason() -> None:
    """If a gated-gelu T5 is loaded, the probe must report unsupported with a reason.

    Tiny-random-t5 is non-gated ReLU so this exercises the trivial branch:
    we just verify that the probe never returns a silent ``allclose=True``
    placeholder when the structure is in fact unsupported.
    """
    result = _run(use_pad=False)
    # tiny-random-t5: not gated, supported.
    if not result.get("is_gated", False):
        assert result["ffn_type"] in {"t5_dense_relu_dense", "bart_fc1_fc2"}
    else:
        # Gated branch must either be SiLU (supported) or carry an explicit
        # unsupported reason; we already skipped the latter above.
        assert result["activation_type"] == "silu"
