"""Stage 7.2 — tests for the rank-padded LoRA primitive."""

from __future__ import annotations

import pytest
import torch

from pllo.ops.lora import (
    LoRAConfig,
    MaskedLoRAForwardConfig,
    init_lora_adapters,
    plain_lora_linear_forward,
)
from pllo.ops.lora_backward import plain_lora_backward_reference
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    VALID_DUMMY_STRATEGIES,
    create_rank_padded_lora_adapters,
    dummy_contribution_norm,
    extract_real_gradients,
    plain_rank_padded_lora_backward_reference,
    plain_rank_padded_lora_forward,
    run_masked_rank_padded_lora_backward,
    run_masked_rank_padded_lora_linear,
    validate_rank_padding_config,
    visible_shape_fingerprint,
)


def _setup(
    *,
    seed: int = 0,
    d_in: int = 12,
    d_out: int = 8,
    true_rank: int = 2,
    padded_rank: int = 8,
    alpha: float = 2.0,
    with_bias: bool = True,
):
    torch.manual_seed(seed)
    cfg = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=true_rank, alpha=alpha, dtype="float64",
    )
    a, b = init_lora_adapters(cfg)
    b = b + 0.1 * torch.randn(true_rank, d_out, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    x = torch.randn(5, d_in, dtype=torch.float64)
    bias = torch.randn(d_out, dtype=torch.float64) if with_bias else None
    return x, w, a, b, bias, cfg


# ---------------------------------------------------------------------------
# 1. validate_rank_padding_config accepts valid config
# ---------------------------------------------------------------------------


def test_validate_rank_padding_config_accepts_valid() -> None:
    cfg = RankPaddingConfig(true_rank=4, padded_rank=8)
    validate_rank_padding_config(cfg)  # should not raise


# ---------------------------------------------------------------------------
# 2. invalid padded_rank < true_rank raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_padded_rank_raises() -> None:
    with pytest.raises(ValueError):
        validate_rank_padding_config(
            RankPaddingConfig(true_rank=4, padded_rank=2)
        )
    with pytest.raises(ValueError):
        validate_rank_padding_config(
            RankPaddingConfig(true_rank=0, padded_rank=8)
        )
    with pytest.raises(ValueError):
        validate_rank_padding_config(
            RankPaddingConfig(true_rank=4, padded_rank=8, dummy_strategy="bogus")
        )


# ---------------------------------------------------------------------------
# 3. create_rank_padded_lora_adapters shapes correct
# ---------------------------------------------------------------------------


def test_create_rank_padded_adapters_shapes() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=3, padded_rank=7)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=3, padded_rank=7),
    )
    assert tuple(pack["a_pad"].shape) == (cfg.d_in, 7)
    assert tuple(pack["b_pad"].shape) == (7, cfg.d_out)
    assert pack["true_rank"] == 3
    assert pack["padded_rank"] == 7
    assert pack["real_slice"] == slice(0, 3)
    assert pack["dummy_slice"] == slice(3, 7)


# ---------------------------------------------------------------------------
# 4. A_pad B_pad equals A B for zero_dummy
# ---------------------------------------------------------------------------


def test_a_pad_b_pad_equals_a_b_zero_dummy() -> None:
    x, w, a, b, _, cfg = _setup(true_rank=2, padded_rank=6)
    pack = create_rank_padded_lora_adapters(
        a, b,
        RankPaddingConfig(true_rank=2, padded_rank=6, dummy_strategy="zero_dummy"),
    )
    assert torch.allclose(
        pack["a_pad"] @ pack["b_pad"], a @ b, atol=1e-12, rtol=1e-12,
    )
    assert dummy_contribution_norm(pack["a_pad"], pack["b_pad"], 2) < 1e-12


# ---------------------------------------------------------------------------
# 5. A_pad B_pad equals A B for paired_cancellation_dummy
# ---------------------------------------------------------------------------


def test_a_pad_b_pad_equals_a_b_paired_cancellation() -> None:
    x, w, a, b, _, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b,
        RankPaddingConfig(
            true_rank=2, padded_rank=8,
            dummy_strategy="paired_cancellation_dummy",
        ),
    )
    assert torch.allclose(
        pack["a_pad"] @ pack["b_pad"], a @ b, atol=1e-12, rtol=1e-12,
    )
    assert dummy_contribution_norm(pack["a_pad"], pack["b_pad"], 2) < 1e-12


def test_paired_cancellation_odd_pad_has_zero_tail() -> None:
    x, w, a, b, _, cfg = _setup(true_rank=2, padded_rank=7)
    pack = create_rank_padded_lora_adapters(
        a, b,
        RankPaddingConfig(
            true_rank=2, padded_rank=7,
            dummy_strategy="paired_cancellation_dummy",
        ),
    )
    # 5 dummies total → 2 pairs + 1 zero tail.
    assert pack["metadata"]["dummy_strategy_effective"] == (
        "paired_cancellation_dummy_with_zero_tail"
    )
    # Last row of B_pad is zero.
    assert torch.allclose(pack["b_pad"][-1, :], torch.zeros_like(pack["b_pad"][-1, :]))


# ---------------------------------------------------------------------------
# 6. plain_rank_padded_forward equals plain rank-r forward
# ---------------------------------------------------------------------------


def test_plain_rank_padded_forward_matches_plain_rank_r() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_padded = plain_rank_padded_lora_forward(
        x, w, pack["a_pad"], pack["b_pad"], true_rank=2,
        bias=bias, alpha=cfg.alpha,
    )
    assert torch.allclose(y_plain, y_padded, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. masked rank-padded forward no_pad allclose
# ---------------------------------------------------------------------------


def test_masked_rank_padded_forward_no_pad_allclose() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_masked, _ = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], bias,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 8. masked rank-padded forward use_pad allclose
# ---------------------------------------------------------------------------


def test_masked_rank_padded_forward_use_pad_allclose() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_masked, _ = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], bias,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 9. fresh U_pad changes visible tensors but output unchanged
# ---------------------------------------------------------------------------


def test_fresh_u_pad_changes_visible_but_not_output() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=False, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype="float64",
    )
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y1, state1 = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], bias,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    y2, state2 = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], bias,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    assert not torch.allclose(state1.u, state2.u, atol=1e-4)
    assert torch.allclose(y_plain, y1, atol=1e-9)
    assert torch.allclose(y_plain, y2, atol=1e-9)


# ---------------------------------------------------------------------------
# 10. scale uses alpha / true_rank, not alpha / padded_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
def test_scale_uses_alpha_over_true_rank(alpha: float) -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=16, alpha=alpha)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=16),
    )
    # Manually compute with alpha / true_rank
    expected = x @ w + (alpha / 2) * (x @ a) @ b + bias
    got = plain_rank_padded_lora_forward(
        x, w, pack["a_pad"], pack["b_pad"], true_rank=2,
        bias=bias, alpha=alpha,
    )
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)
    # If scale erroneously used alpha / padded_rank=16, it would be much smaller.
    wrong = x @ w + (alpha / 16) * (x @ a) @ b + bias
    assert not torch.allclose(got, wrong, atol=1e-4)


# ---------------------------------------------------------------------------
# Backward: real slices match plain grad_a / grad_b
# ---------------------------------------------------------------------------


def test_backward_real_slices_match_plain_rank_r() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    G = torch.randn(x.shape[0], cfg.d_out, dtype=torch.float64)
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    _, state = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], bias,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    ref = plain_lora_backward_reference(x, w, a, b, G, alpha=cfg.alpha)
    got = run_masked_rank_padded_lora_backward(
        x, w, pack["a_pad"], pack["b_pad"], G,
        true_rank=2, padded_rank=8, alpha=cfg.alpha,
        state=state, recover_grad_x=True,
    )
    real = extract_real_gradients(got["grad_a_pad"], got["grad_b_pad"], 2)
    assert torch.allclose(real["grad_a_real"], ref["grad_a"], atol=1e-9)
    assert torch.allclose(real["grad_b_real"], ref["grad_b"], atol=1e-9)
    assert torch.allclose(got["grad_x"], ref["grad_x"], atol=1e-9)


# ---------------------------------------------------------------------------
# Plain padded backward reference equals plain rank-r grad in real slice
# ---------------------------------------------------------------------------


def test_plain_padded_backward_reference_matches_plain() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=2, padded_rank=8)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=2, padded_rank=8),
    )
    G = torch.randn(x.shape[0], cfg.d_out, dtype=torch.float64)
    ref = plain_lora_backward_reference(x, w, a, b, G, alpha=cfg.alpha)
    padded_ref = plain_rank_padded_lora_backward_reference(
        x, w, pack["a_pad"], pack["b_pad"], 2, G, alpha=cfg.alpha,
    )
    assert torch.allclose(padded_ref["grad_a_pad"][:, :2], ref["grad_a"], atol=1e-12)
    assert torch.allclose(padded_ref["grad_b_pad"][:2, :], ref["grad_b"], atol=1e-12)


# ---------------------------------------------------------------------------
# bias=None path
# ---------------------------------------------------------------------------


def test_bias_none_path_rank_padded() -> None:
    x, w, a, b, _, cfg = _setup(with_bias=False, true_rank=3, padded_rank=10)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=3, padded_rank=10),
    )
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, None, alpha=cfg.alpha)
    y_masked, _ = run_masked_rank_padded_lora_linear(
        x, w, pack["a_pad"], pack["b_pad"], None,
        true_rank=3, padded_rank=10, alpha=cfg.alpha,
        state=None, forward_config=fcfg,
    )
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# visible_shape_fingerprint exposes padded_rank only
# ---------------------------------------------------------------------------


def test_visible_shape_fingerprint_shows_padded_rank_only() -> None:
    a_tilde_pad = torch.empty(12, 8, dtype=torch.float64)
    b_tilde_pad = torch.empty(8, 16, dtype=torch.float64)
    fp = visible_shape_fingerprint(a_tilde_pad, b_tilde_pad)
    assert fp["visible_rank_from_a_shape"] == 8
    assert fp["visible_rank_from_b_shape"] == 8


# ---------------------------------------------------------------------------
# Edge case: padded_rank == true_rank ⇒ no padding, no dummy slice
# ---------------------------------------------------------------------------


def test_padded_rank_equals_true_rank_degenerate() -> None:
    x, w, a, b, bias, cfg = _setup(true_rank=4, padded_rank=4)
    pack = create_rank_padded_lora_adapters(
        a, b, RankPaddingConfig(true_rank=4, padded_rank=4),
    )
    assert pack["metadata"]["dummy_size"] == 0
    assert pack["metadata"]["dummy_strategy_effective"] == "no_padding"
    assert torch.allclose(pack["a_pad"], a)
    assert torch.allclose(pack["b_pad"], b)
