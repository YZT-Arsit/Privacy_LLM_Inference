"""Tests for obfuscated LM head recovery."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.ops.lm_head import lm_head_obfuscated, lm_head_plain
from pllo.utils.seed import set_seed


def test_lm_head_logits_recovery() -> None:
    set_seed(1201)
    hidden = torch.randn(2, 8, 16, dtype=torch.float64)
    weight = torch.randn(16, 32, dtype=torch.float64) * 0.02
    bias = torch.randn(32, dtype=torch.float64) * 0.02
    tee = SimulatedTEE()
    vocab_state = tee.create_linear_mask_state(hidden.reshape(-1, 16), 32, use_pad=False)
    logits = lm_head_obfuscated(hidden, weight, bias, vocab_state, tee, UntrustedGPUExecutor(), use_pad=True)
    assert torch.allclose(lm_head_plain(hidden, weight, bias), logits, atol=1e-8, rtol=1e-6)


def test_lm_head_vocab_mask_recovery() -> None:
    set_seed(1202)
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    weight = torch.randn(8, 16, dtype=torch.float64) * 0.02
    tee = SimulatedTEE()
    vocab_state = tee.create_linear_mask_state(hidden.reshape(-1, 8), 16, use_pad=False)
    logits = lm_head_obfuscated(hidden, weight, None, vocab_state, tee, UntrustedGPUExecutor())
    assert logits.shape == (1, 4, 16)
    assert torch.allclose(lm_head_plain(hidden, weight), logits, atol=1e-8, rtol=1e-6)
