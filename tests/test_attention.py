"""Tests for tiny Transformer attention helpers."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.ops.attention import (
    causal_self_attention_obfuscated,
    causal_self_attention_plain,
    generate_qk_mask_pair,
    qk_mask_constraint_error,
)
from pllo.utils.seed import set_seed


def _weights(hidden: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(hidden, hidden, dtype=dtype) * 0.02 for _ in range(4))


def test_attention_output_shape() -> None:
    set_seed(1001)
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    out = causal_self_attention_plain(x, *_weights(16), num_heads=4)
    assert out.shape == x.shape


def test_causal_mask_prevents_future_token_influence() -> None:
    set_seed(1002)
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    w_q, w_k, w_v, w_o = _weights(16)
    out = causal_self_attention_plain(x, w_q, w_k, w_v, w_o, num_heads=4)
    x_changed = x.clone()
    x_changed[:, 1:, :] = torch.randn_like(x_changed[:, 1:, :]) * 10.0
    out_changed = causal_self_attention_plain(x_changed, w_q, w_k, w_v, w_o, num_heads=4)
    assert torch.allclose(out[:, 0, :], out_changed[:, 0, :], atol=1e-10, rtol=1e-10)


def test_obfuscated_attention_recovered_output_matches_plain() -> None:
    set_seed(1003)
    x = torch.randn(2, 8, 16, dtype=torch.float64)
    w_q, w_k, w_v, w_o = _weights(16)
    tee = SimulatedTEE()
    executor = UntrustedGPUExecutor()
    output_state = tee.create_linear_mask_state(x.reshape(-1, 16), 16, use_pad=False)
    output_state.n_out = output_state.n_in
    output_state.n_out_inv = output_state.n_in_inv
    out_tilde = causal_self_attention_obfuscated(
        x,
        w_q,
        w_k,
        w_v,
        w_o,
        num_heads=4,
        output_state=output_state,
        tee=tee,
        executor=executor,
        use_pad=True,
    )
    recovered = tee.recover_output(out_tilde.reshape(-1, 16), output_state).reshape_as(x)
    plain = causal_self_attention_plain(x, w_q, w_k, w_v, w_o, num_heads=4)
    assert torch.allclose(plain, recovered, atol=1e-8, rtol=1e-6)


def test_qk_mask_constraint_holds_per_head() -> None:
    set_seed(1004)
    n_q, _, n_k, _ = generate_qk_mask_pair(num_heads=4, d_head=4, dtype=torch.float64, device="cpu")
    assert qk_mask_constraint_error(n_q, n_k, num_heads=4) < 1e-10
