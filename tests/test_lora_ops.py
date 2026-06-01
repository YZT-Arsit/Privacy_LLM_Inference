"""Stage 7.0 — tests for the new LoRA primitive."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    init_lora_adapters,
    lora_state_fingerprint,
    make_lora_pad_compensation,
    masked_lora_linear_forward,
    obfuscate_lora_input,
    plain_lora_linear_forward,
    recover_masked_output,
    run_masked_lora_linear,
    transform_linear_weight_lora,
    transform_lora_adapter,
)


def _setup(
    *,
    seed: int = 0,
    d_in: int = 12,
    d_out: int = 16,
    rank: int = 3,
    seq_len: int = 5,
    alpha: float = 2.0,
    with_bias: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
    LoRAConfig,
]:
    torch.manual_seed(seed)
    cfg = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=rank, alpha=alpha,
        use_bias=with_bias, dtype="float64",
    )
    x = torch.randn(seq_len, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    a, b = init_lora_adapters(cfg)
    # Make B non-zero so the LoRA branch matters.
    b = b + 0.1 * torch.randn(rank, d_out, dtype=torch.float64)
    bias = torch.randn(d_out, dtype=torch.float64) if with_bias else None
    return x, w, a, b, bias, cfg


# ---------------------------------------------------------------------------
# 1. plain_lora_linear_forward shape correct
# ---------------------------------------------------------------------------


def test_plain_lora_linear_forward_shape() -> None:
    x, w, a, b, bias, cfg = _setup()
    y = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    assert tuple(y.shape) == (x.shape[0], w.shape[1])


# ---------------------------------------------------------------------------
# 2. transform_lora_adapter shape correct
# ---------------------------------------------------------------------------


def test_transform_lora_adapter_shape() -> None:
    x, w, a, b, bias, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    a_tilde, b_tilde = transform_lora_adapter(
        a, b, state.n_in_inv, state.n_out, state.u, state.u_inv,
        alpha=cfg.alpha,
    )
    assert tuple(a_tilde.shape) == tuple(a.shape)
    assert tuple(b_tilde.shape) == tuple(b.shape)


# ---------------------------------------------------------------------------
# 3. masked_lora_forward no_pad allclose
# ---------------------------------------------------------------------------


def test_masked_lora_no_pad_allclose() -> None:
    x, w, a, b, bias, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_masked, _ = run_masked_lora_linear(x, w, a, b, bias, cfg, fcfg)
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 4. masked_lora_forward use_pad allclose
# ---------------------------------------------------------------------------


def test_masked_lora_with_pad_allclose() -> None:
    x, w, a, b, bias, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_masked, _ = run_masked_lora_linear(x, w, a, b, bias, cfg, fcfg)
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 5. fresh U changes A_tilde/B_tilde but recovered output unchanged
# ---------------------------------------------------------------------------


def test_fresh_u_changes_adapter_but_not_recovered_output() -> None:
    x, w, a, b, bias, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(
        use_pad=True, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype="float64",
    )
    state1 = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    state2 = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    a_tilde_1, b_tilde_1 = transform_lora_adapter(
        a, b, state1.n_in_inv, state1.n_out, state1.u, state1.u_inv,
        alpha=cfg.alpha,
    )
    a_tilde_2, b_tilde_2 = transform_lora_adapter(
        a, b, state2.n_in_inv, state2.n_out, state2.u, state2.u_inv,
        alpha=cfg.alpha,
    )
    # Adapters should be different.
    assert not torch.allclose(a_tilde_1, a_tilde_2, atol=1e-4)
    assert not torch.allclose(b_tilde_1, b_tilde_2, atol=1e-4)

    # But recovered output should equal plain in both cases.
    y_plain = plain_lora_linear_forward(x, w, a, b, bias, alpha=cfg.alpha)
    y_masked_1, _ = run_masked_lora_linear(x, w, a, b, bias, cfg, fcfg)
    y_masked_2, _ = run_masked_lora_linear(x, w, a, b, bias, cfg, fcfg)
    assert torch.allclose(y_plain, y_masked_1, atol=1e-9)
    assert torch.allclose(y_plain, y_masked_2, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. bias=None path works
# ---------------------------------------------------------------------------


def test_lora_no_bias_path() -> None:
    x, w, a, b, _, cfg = _setup(with_bias=False)
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    y_plain = plain_lora_linear_forward(x, w, a, b, None, alpha=cfg.alpha)
    y_masked, _ = run_masked_lora_linear(x, w, a, b, None, cfg, fcfg)
    assert torch.allclose(y_plain, y_masked, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 7. rank scaling alpha/r correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("rank", [2, 4, 8])
def test_alpha_over_rank_scaling(alpha: float, rank: int) -> None:
    x, w, a, b, bias, cfg = _setup(rank=rank, alpha=alpha)
    # Plain reference computed two ways to verify the formula.
    expected = x @ w + (alpha / rank) * (x @ a) @ b
    if bias is not None:
        expected = expected + bias
    y = plain_lora_linear_forward(x, w, a, b, bias, alpha=alpha)
    assert torch.allclose(y, expected, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 8. JSON summary does not expose full adapter tensors
# ---------------------------------------------------------------------------


def test_lora_state_fingerprint_does_not_expose_raw_tensors() -> None:
    _, _, _, _, _, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=5)
    fp = lora_state_fingerprint(state)
    text = json.dumps(fp, default=str)
    # No raw tensor literals.
    assert "tensor(" not in text
    # Only fingerprints (norm + shape) leak.
    for key in ("n_in", "n_out", "u"):
        assert isinstance(fp[key], dict)
        assert set(fp[key].keys()) == {"shape", "frobenius_norm_digest"}
    assert isinstance(fp["pad_present"], bool)
    assert fp["rank"] == cfg.rank


# ---------------------------------------------------------------------------
# Extra: shape validators raise on misuse
# ---------------------------------------------------------------------------


def test_plain_lora_rejects_bad_rank() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.float64)
    w = torch.randn(4, 5, dtype=torch.float64)
    a = torch.randn(4, 2, dtype=torch.float64)
    b = torch.randn(3, 5, dtype=torch.float64)  # WRONG rank
    with pytest.raises(ValueError):
        plain_lora_linear_forward(x, w, a, b)


def test_masked_lora_rejects_bad_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.float64)
    w_tilde = torch.randn(4, 5, dtype=torch.float64)
    a_tilde = torch.randn(4, 2, dtype=torch.float64)
    b_tilde = torch.randn(3, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        masked_lora_linear_forward(x, w_tilde, a_tilde, b_tilde)
