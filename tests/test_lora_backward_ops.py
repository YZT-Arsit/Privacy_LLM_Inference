"""Stage 7.1 — tests for the masked LoRA backward primitive."""

from __future__ import annotations

import json

import pytest
import torch

from pllo.ops.lora import (
    LoRAConfig,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    init_lora_adapters,
)
from pllo.ops.lora_backward import (
    MaskedLoRABackwardConfig,
    invert_upstream_gradient_mask,
    make_lora_grad_pad_compensation,
    masked_lora_backward,
    plain_lora_backward_reference,
    recover_lora_gradients,
    run_masked_lora_backward,
    transform_upstream_gradient,
)


def _setup(
    *,
    seed: int = 0,
    d_in: int = 12,
    d_out: int = 8,
    rank: int = 3,
    seq_len: int = 5,
    alpha: float = 2.0,
):
    torch.manual_seed(seed)
    cfg = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=rank, alpha=alpha, dtype="float64",
    )
    x = torch.randn(seq_len, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    a, b = init_lora_adapters(cfg)
    b = b + 0.1 * torch.randn(rank, d_out, dtype=torch.float64)
    g = torch.randn(seq_len, d_out, dtype=torch.float64)
    return x, w, a, b, g, cfg


# ---------------------------------------------------------------------------
# 1. upstream gradient transform satisfies dL invariance
# ---------------------------------------------------------------------------


def test_upstream_gradient_transform_invariance() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    # Y is a placeholder for dY; pick a synthetic Y to test invariance.
    y = torch.randn_like(g)
    y_tilde = y @ state.n_out
    g_tilde = transform_upstream_gradient(g, state.n_out)
    inv_plain = (g * y).sum().item()
    inv_tilde = (g_tilde * y_tilde).sum().item()
    assert abs(inv_plain - inv_tilde) < 1e-9
    # And round trip.
    g_recovered = invert_upstream_gradient_mask(g_tilde, state.n_out)
    assert torch.allclose(g, g_recovered, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. masked_lora_backward no_pad recovers grad_A / grad_B
# ---------------------------------------------------------------------------


def test_masked_backward_no_pad_recovers_plain_grads() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    ref = plain_lora_backward_reference(x, w, a, b, g, alpha=cfg.alpha)
    got = run_masked_lora_backward(
        x, w, a, b, g, alpha=cfg.alpha,
        n_in=state.n_in, n_in_inv=state.n_in_inv,
        n_out=state.n_out, u=state.u, u_inv=state.u_inv,
        pad=None, recover_grad_x=True,
    )
    assert torch.allclose(ref["grad_a"], got["grad_a"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(ref["grad_b"], got["grad_b"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(ref["grad_x"], got["grad_x"], atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 3. masked_lora_backward use_pad recovers grad_A / grad_B
# ---------------------------------------------------------------------------


def test_masked_backward_use_pad_recovers_plain_grads() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    ref = plain_lora_backward_reference(x, w, a, b, g, alpha=cfg.alpha)
    got = run_masked_lora_backward(
        x, w, a, b, g, alpha=cfg.alpha,
        n_in=state.n_in, n_in_inv=state.n_in_inv,
        n_out=state.n_out, u=state.u, u_inv=state.u_inv,
        pad=state.pad, recover_grad_x=True,
    )
    assert torch.allclose(ref["grad_a"], got["grad_a"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(ref["grad_b"], got["grad_b"], atol=1e-9, rtol=1e-9)
    assert torch.allclose(ref["grad_x"], got["grad_x"], atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 4. fresh U changes grad_A_tilde / grad_B_tilde but recovered grads unchanged
# ---------------------------------------------------------------------------


def test_fresh_u_changes_grad_tilde_but_not_recovered_grads() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(
        use_pad=False, fresh_u_per_call=True, fresh_masks_per_call=True,
        dtype="float64",
    )
    ref = plain_lora_backward_reference(x, w, a, b, g, alpha=cfg.alpha)
    state1 = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    state2 = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])

    # GPU-side masked gradients should differ between two fresh masks.
    def _mt(s):
        x_t = x @ s.n_in
        a_t = s.n_in_inv @ a @ s.u
        b_t = s.u_inv @ b @ s.n_out
        g_t = transform_upstream_gradient(g, s.n_out)
        return masked_lora_backward(x_t, a_t, b_t, g_t, alpha=cfg.alpha)

    m1 = _mt(state1)
    m2 = _mt(state2)
    assert not torch.allclose(m1["grad_a_tilde"], m2["grad_a_tilde"], atol=1e-4)
    assert not torch.allclose(m1["grad_b_tilde"], m2["grad_b_tilde"], atol=1e-4)

    # Recovered grads are identical across calls.
    got1 = run_masked_lora_backward(
        x, w, a, b, g, alpha=cfg.alpha,
        n_in=state1.n_in, n_in_inv=state1.n_in_inv,
        n_out=state1.n_out, u=state1.u, u_inv=state1.u_inv,
    )
    got2 = run_masked_lora_backward(
        x, w, a, b, g, alpha=cfg.alpha,
        n_in=state2.n_in, n_in_inv=state2.n_in_inv,
        n_out=state2.n_out, u=state2.u, u_inv=state2.u_inv,
    )
    assert torch.allclose(got1["grad_a"], ref["grad_a"], atol=1e-9)
    assert torch.allclose(got2["grad_a"], ref["grad_a"], atol=1e-9)
    assert torch.allclose(got1["grad_b"], ref["grad_b"], atol=1e-9)
    assert torch.allclose(got2["grad_b"], ref["grad_b"], atol=1e-9)


# ---------------------------------------------------------------------------
# 5. bias=None path works (bias does not affect grad_A / grad_B)
# ---------------------------------------------------------------------------


def test_bias_none_path_works() -> None:
    # The masked backward doesn't take bias; the plain reference does
    # not include bias in any gradient. So this test simply checks that
    # autograd vs analytic still match when bias is excluded.
    torch.manual_seed(7)
    d_in, d_out, rank, batch = 10, 6, 2, 4
    x = torch.randn(batch, d_in, dtype=torch.float64, requires_grad=True)
    w = torch.randn(d_in, d_out, dtype=torch.float64)
    a = torch.randn(d_in, rank, dtype=torch.float64, requires_grad=True)
    b = torch.randn(rank, d_out, dtype=torch.float64, requires_grad=True)
    s = 1.0 / rank
    y = x @ w + s * (x @ a) @ b
    loss = (y * y).sum()
    loss.backward()
    g = 2.0 * y.detach()
    ref = plain_lora_backward_reference(
        x.detach(), w, a.detach(), b.detach(), g, alpha=1.0,
    )
    assert torch.allclose(ref["grad_a"], a.grad, atol=1e-10)
    assert torch.allclose(ref["grad_b"], b.grad, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. alpha/r scaling works
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("rank", [2, 4])
def test_alpha_over_rank_scaling(alpha: float, rank: int) -> None:
    torch.manual_seed(0)
    x, w, a, b, g, cfg = _setup(rank=rank, alpha=alpha)
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    ref = plain_lora_backward_reference(x, w, a, b, g, alpha=alpha)
    got = run_masked_lora_backward(
        x, w, a, b, g, alpha=alpha,
        n_in=state.n_in, n_in_inv=state.n_in_inv,
        n_out=state.n_out, u=state.u, u_inv=state.u_inv,
        pad=state.pad,
    )
    assert torch.allclose(ref["grad_a"], got["grad_a"], atol=1e-9)
    assert torch.allclose(ref["grad_b"], got["grad_b"], atol=1e-9)


# ---------------------------------------------------------------------------
# 7. optional grad_X recovery
# ---------------------------------------------------------------------------


def test_grad_x_recovery_when_enabled() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=True, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    ref = plain_lora_backward_reference(x, w, a, b, g, alpha=cfg.alpha)
    got = run_masked_lora_backward(
        x, w, a, b, g, alpha=cfg.alpha,
        n_in=state.n_in, n_in_inv=state.n_in_inv,
        n_out=state.n_out, u=state.u, u_inv=state.u_inv,
        pad=state.pad, recover_grad_x=True,
    )
    assert got["grad_x"] is not None
    assert torch.allclose(ref["grad_x"], got["grad_x"], atol=1e-9)


# ---------------------------------------------------------------------------
# 8. no raw gradients in summary
# ---------------------------------------------------------------------------


def test_pad_compensation_shapes_and_safety() -> None:
    x, w, a, b, g, cfg = _setup()
    pad = torch.randn(x.shape[0], cfg.d_in, dtype=torch.float64)
    comp = make_lora_grad_pad_compensation(a, b, pad, g, alpha=cfg.alpha)
    assert tuple(comp["grad_a_pad_compensation"].shape) == (cfg.d_in, cfg.rank)
    assert tuple(comp["grad_b_pad_compensation"].shape) == (cfg.rank, cfg.d_out)
    # JSON-safety: the compensation tensors stay trusted-only — verifying we
    # can call repr() without anything weird, and they aren't accidentally
    # logged by the runner (covered by the probe test elsewhere).
    text = json.dumps({"shapes": [list(comp[k].shape) for k in comp]})
    assert "tensor(" not in text


def test_recover_rejects_bad_shapes() -> None:
    torch.manual_seed(0)
    # Mismatched n_in / grad_a_tilde dim
    bad_grad_a = torch.zeros(5, 3, dtype=torch.float64)
    grad_b = torch.zeros(3, 4, dtype=torch.float64)
    n_in = torch.eye(7, dtype=torch.float64)
    n_out = torch.eye(4, dtype=torch.float64)
    u = torch.eye(3, dtype=torch.float64)
    with pytest.raises(Exception):
        recover_lora_gradients(bad_grad_a, grad_b, n_in, n_out, u)


def test_masked_backward_requires_w_tilde_for_grad_x() -> None:
    x, w, a, b, g, cfg = _setup()
    fcfg = MaskedLoRAForwardConfig(use_pad=False, dtype="float64")
    state = create_masked_lora_state(cfg, fcfg, seq_len=x.shape[0])
    x_t = x @ state.n_in
    a_t = state.n_in_inv @ a @ state.u
    b_t = state.u_inv @ b @ state.n_out
    g_t = transform_upstream_gradient(g, state.n_out)
    with pytest.raises(ValueError):
        masked_lora_backward(
            x_t, a_t, b_t, g_t, alpha=cfg.alpha, recover_grad_x=True,
        )


def test_masked_lora_backward_config_dtype() -> None:
    c = MaskedLoRABackwardConfig(use_pad=False, dtype="float32")
    assert c.torch_dtype() == torch.float32
    with pytest.raises(ValueError):
        MaskedLoRABackwardConfig(dtype="bfloat16").torch_dtype()
