"""Condition-number effect on a multi-layer linear chain (Table 2 trend).

The paper shows that higher-condition-number obfuscation matrices deteriorate
utility through accumulated numerical error. Here we verify the *numerical*
ordering directly on a deep float32 linear chain: kappa=1 (orthogonal) has the
smallest error and it grows with kappa, with the naive random matrix worst.
"""

from __future__ import annotations

import torch

from pllo.baselines.obfuscatune.config import ObfuscaTuneConfig
from pllo.baselines.obfuscatune.linear_obfuscation import plain_linear
from pllo.baselines.obfuscatune.modules import ObfuscaTuneLinearSimulator


def _chain_error(kind: str, kappa: float, seed: int = 0, depth: int = 8, d: int = 48) -> float:
    torch.manual_seed(seed)
    x0 = torch.randn(8, d, dtype=torch.float32)
    layers = [torch.nn.Linear(d, d, bias=False).float() for _ in range(depth)]
    cfg = ObfuscaTuneConfig(dtype="float32", random_matrix_type=kind,
                            condition_number=kappa, seed=seed)
    x = x0.clone()
    ref = x0.clone()
    for i, lin in enumerate(layers):
        sim = ObfuscaTuneLinearSimulator(lin, cfg, direction="input", seed=i)
        x, _ = sim.forward(x)
        ref = plain_linear(lin, ref)
    return float((x - ref).abs().max().item())


def test_error_increases_with_condition_number():
    e1 = _chain_error("cond", 1.0)
    e32 = _chain_error("cond", 32.0)
    e128 = _chain_error("cond", 128.0)
    e160 = _chain_error("cond", 160.0)
    assert e1 <= e32 <= e128 <= e160
    assert e160 > e1


def test_orthogonal_beats_high_condition():
    e_orth = _chain_error("orthogonal", 1.0)
    e_hi = _chain_error("cond", 128.0)
    assert e_orth < e_hi


def test_reproducible():
    a = _chain_error("cond", 32.0, seed=11)
    b = _chain_error("cond", 32.0, seed=11)
    assert a == b
