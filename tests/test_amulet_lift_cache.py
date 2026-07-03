"""The Amulet selector-lift factor cache is a pure efficiency change.

The lift factors (``valid``, ``R``) depend only on (h, lift_k, seed) with a fixed
seed, so caching them across calls/layers must be BIT-IDENTICAL to regenerating
them per call (the historical path). These tests pin exact equality (``torch.equal``,
not ``allclose``) against a faithful reimplementation of the old per-call code,
plus cache reuse + cross-call determinism.

CPU torch only. Run:
    python -m pytest tests/test_amulet_lift_cache.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _reference_selector_lift(x, act, lift_k, seed):
    """Faithful copy of the OLD per-call selector lift (regenerate every call)."""
    import torch
    lead = x.shape[:-1]
    h = x.shape[-1]
    U = x.reshape(-1, h)
    m = U.shape[0]
    gen = torch.Generator().manual_seed(seed)
    valid = torch.randint(0, lift_k, (h,), generator=gen).to(U.device)
    R = (torch.rand(h, lift_k, generator=gen) + 0.5).to(
        device=U.device, dtype=U.dtype)
    R[torch.arange(h, device=U.device), valid] = 1.0
    lift = U.unsqueeze(-1) * R.unsqueeze(0)
    Af = act(lift)
    idx = valid.view(1, h, 1).expand(m, h, 1)
    return Af.gather(2, idx).squeeze(-1).reshape(*lead, h)


def test_cached_silu_bit_identical_to_per_call_reference() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=3, seed=2035)
    torch.manual_seed(0)
    x = torch.randn(4, 11, 9, dtype=torch.float32)
    got = b.silu(x).output
    ref = _reference_selector_lift(x, F.silu, 3, 2035)
    assert torch.equal(got, ref), "cached SiLU lift diverged from per-call path"


def test_cached_gelu_bit_identical_to_per_call_reference() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=4, seed=7)
    torch.manual_seed(1)
    x = torch.randn(6, 13, dtype=torch.float64)
    got = b.gelu(x).output
    ref = _reference_selector_lift(x, F.gelu, 4, 7)
    assert torch.equal(got, ref)


def test_repeated_calls_are_deterministic_and_reuse_factors() -> None:
    torch = pytest.importorskip("torch")
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=2, seed=2035)
    torch.manual_seed(3)
    # different m (batch*seq) each call, same feature width h -> one cache entry
    outs = []
    for m in (1, 8, 1, 4, 1):
        x = torch.randn(m, 16, dtype=torch.float32)
        outs.append((x, b.silu(x).output))
    # exactly one cached (h, device, dtype) entry, generated once
    assert len(b._lift_factors_cache) == 1
    key = next(iter(b._lift_factors_cache))
    assert key[0] == 16
    valid0, R0 = b._lift_factors_cache[key]
    # calling again returns the SAME cached tensor objects (no regeneration)
    valid1, R1 = b._lift_factors(16, R0.device, R0.dtype)
    assert valid1 is valid0 and R1 is R0
    # and every recorded output is reproducible from the cached factors
    for x, out in outs:
        assert torch.equal(b.silu(x).output, out)


def test_distinct_width_gets_distinct_cache_entry() -> None:
    torch = pytest.importorskip("torch")
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=2, seed=2035)
    torch.manual_seed(4)
    b.silu(torch.randn(2, 8))
    b.silu(torch.randn(2, 16))
    assert {k[0] for k in b._lift_factors_cache} == {8, 16}


def test_runner_trusted_shortcut_still_exact_with_cache() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    ts = make_folded_nonlinear_runner("trusted_shortcut", lift_k=4)
    x = torch.randn(2, 7, 16, dtype=torch.float32)
    # two calls through the runner: both exact vs plain SiLU, and identical
    y1, y2 = ts.silu(x), ts.silu(x)
    assert torch.allclose(y1, F.silu(x), atol=1e-6)
    assert torch.equal(y1, y2)
    ev = ts.execution_evidence()
    assert ev["amulet_lift_executed"] is True
    assert ev["lifted_nonlinear_ops_count"] > 0
