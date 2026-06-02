"""Stage 7.4 — tests for the stronger LoRA dummy strategies primitive."""

from __future__ import annotations

import json

import pytest
import torch

from pllo.ops.lora_dummy_strategies import (
    StrongDummyConfig,
    VALID_STRONG_DUMMY_STRATEGIES,
    apply_dummy_correction,
    create_stronger_rank_padded_lora_adapters,
    dummy_correction_norm,
    validate_strong_dummy_config,
    visible_strong_dummy_fingerprint,
)


D_IN, D_OUT = 16, 12
TRUE_RANK = 2
PADDED_RANK = 8


def _adapter_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(D_IN, TRUE_RANK, dtype=torch.float64)
    b = torch.randn(TRUE_RANK, D_OUT, dtype=torch.float64)
    return a, b


# ---------------------------------------------------------------------------
# 1. all valid dummy strategies accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(VALID_STRONG_DUMMY_STRATEGIES))
def test_all_strategies_accepted(strategy: str) -> None:
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy=strategy, dtype="float64",
    )
    validate_strong_dummy_config(cfg)


# ---------------------------------------------------------------------------
# 2. invalid dummy strategy raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_strategy_rejected() -> None:
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy="nope",
    )
    with pytest.raises(ValueError):
        validate_strong_dummy_config(cfg)


def test_negative_true_rank_rejected() -> None:
    cfg = StrongDummyConfig(true_rank=-1, padded_rank=4)
    with pytest.raises(ValueError):
        validate_strong_dummy_config(cfg)


def test_padded_less_than_true_rejected() -> None:
    cfg = StrongDummyConfig(true_rank=4, padded_rank=2)
    with pytest.raises(ValueError):
        validate_strong_dummy_config(cfg)


def test_negative_noise_scale_rejected() -> None:
    cfg = StrongDummyConfig(
        true_rank=2, padded_rank=4,
        dummy_strategy="paired_cancellation_dummy", noise_scale=-1e-3,
    )
    with pytest.raises(ValueError):
        validate_strong_dummy_config(cfg)


# ---------------------------------------------------------------------------
# 3 - 8. each strategy preserves A_pad B_pad = A B (+ correction if any)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(VALID_STRONG_DUMMY_STRATEGIES))
def test_each_strategy_preserves_or_corrects(strategy: str) -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy=strategy, noise_scale=1e-3, dtype="float64",
    )
    gen = torch.Generator(device="cpu").manual_seed(42)
    pack = create_stronger_rank_padded_lora_adapters(
        a, b, cfg, generator=gen,
    )
    a_pad, b_pad, correction = pack["a_pad"], pack["b_pad"], pack["correction"]
    lhs = a_pad @ b_pad
    rhs = a @ b
    if correction is not None:
        rhs = rhs + correction
    assert (lhs - rhs).abs().max().item() < 1e-12, strategy


# ---------------------------------------------------------------------------
# 9. visible_rank_from_shape == padded_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(VALID_STRONG_DUMMY_STRATEGIES))
def test_visible_rank_equals_padded(strategy: str) -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy=strategy, dtype="float64",
    )
    pack = create_stronger_rank_padded_lora_adapters(a, b, cfg)
    fp = visible_strong_dummy_fingerprint(
        pack["a_pad"], pack["b_pad"], dummy_strategy=strategy,
    )
    assert fp["visible_rank_from_a_shape"] == PADDED_RANK
    assert fp["visible_rank_from_b_shape"] == PADDED_RANK


# ---------------------------------------------------------------------------
# 10. true_rank_hidden_from_shape interpretation
# ---------------------------------------------------------------------------


def test_true_rank_hidden_from_shape_via_fingerprint() -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy="spectrum_matched_dummy", dtype="float64",
    )
    pack = create_stronger_rank_padded_lora_adapters(a, b, cfg)
    # padded_rank > true_rank → true_rank cannot be read off shape.
    assert pack["a_pad"].shape[1] == PADDED_RANK
    assert pack["b_pad"].shape[0] == PADDED_RANK


# ---------------------------------------------------------------------------
# 11. no raw tensors in metadata fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(VALID_STRONG_DUMMY_STRATEGIES))
def test_metadata_is_json_safe(strategy: str) -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy=strategy, dtype="float64",
    )
    pack = create_stronger_rank_padded_lora_adapters(a, b, cfg)
    encoded = json.dumps(pack["metadata"], default=str)
    assert "tensor(" not in encoded


# ---------------------------------------------------------------------------
# Misc — correction term subtraction matches the cancellation residual.
# ---------------------------------------------------------------------------


def test_noise_injected_correction_makes_recovery_exact() -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy="noise_injected_cancellation_dummy",
        noise_scale=1e-3, dtype="float64",
    )
    gen = torch.Generator(device="cpu").manual_seed(7)
    pack = create_stronger_rank_padded_lora_adapters(
        a, b, cfg, generator=gen,
    )
    a_pad, b_pad, correction = pack["a_pad"], pack["b_pad"], pack["correction"]
    assert correction is not None
    # X @ (A_pad B_pad) = X @ (A B) + X @ correction
    x = torch.randn(4, D_IN, dtype=torch.float64)
    y_with_dummy = x @ a_pad @ b_pad
    y_plain = x @ a @ b
    y_recovered = apply_dummy_correction(
        y_with_dummy, x, correction, true_rank=TRUE_RANK, alpha=TRUE_RANK,
    )  # alpha == true_rank → scale = 1
    assert (y_recovered - y_plain).abs().max().item() < 1e-13


def test_dummy_correction_norm_zero_for_cancellation() -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=PADDED_RANK,
        dummy_strategy="paired_cancellation_dummy", dtype="float64",
    )
    pack = create_stronger_rank_padded_lora_adapters(a, b, cfg)
    assert dummy_correction_norm(
        pack["a_pad"], pack["b_pad"], TRUE_RANK,
    ) < 1e-12


def test_padded_equals_true_rank_is_no_padding() -> None:
    a, b = _adapter_pair()
    cfg = StrongDummyConfig(
        true_rank=TRUE_RANK, padded_rank=TRUE_RANK,
        dummy_strategy="paired_cancellation_dummy", dtype="float64",
    )
    pack = create_stronger_rank_padded_lora_adapters(a, b, cfg)
    assert pack["dummy_strategy_effective"] == "no_padding"
    assert pack["correction"] is None
