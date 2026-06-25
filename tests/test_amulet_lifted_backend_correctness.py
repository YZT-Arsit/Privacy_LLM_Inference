"""Correctness of the Amulet-style lifted nonlinear backend (design B).

The selector lift must be functionally EXACT: the lifted activation, squeezed
back through the folded selector, equals the ordinary activation. Covers the op
backend (GELU/SiLU lift), the SwiGLU lifted island, the FoldedNonlinearRunner
dispatch, and that the ``current`` backend still runs in the trusted boundary.

CPU torch only (no CUDA / model / H800). Run:
    python -m pytest tests/test_amulet_lifted_backend_correctness.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_amulet_gelu_lift_executes() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=4, seed=1)
    x = torch.randn(4, 17, dtype=torch.float64)
    r = b.gelu(x)
    # exact vs the ordinary GELU (selector squeeze picks the valid column)
    assert torch.allclose(r.output, F.gelu(x), atol=1e-12)
    # genuinely lifted onto the untrusted accelerator (no boundary crossing)
    assert r.extra["location"] == "untrusted_accelerator"
    assert int(r.extra["lift_k"]) == 4
    assert r.trusted_calls == 0
    assert r.gpu_bytes > 0


def test_amulet_silu_lift_executes() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.nonlinear.amulet_backend import AmuletMigratedNonlinearBackend
    b = AmuletMigratedNonlinearBackend(lift_k=3, seed=2)
    x = torch.randn(5, 9, dtype=torch.float64)
    r = b.silu(x)
    assert torch.allclose(r.output, F.silu(x), atol=1e-12)
    assert r.trusted_calls == 0 and r.gpu_bytes > 0


def test_swiglu_selector_lifted_island_exact() -> None:
    torch = pytest.importorskip("torch")
    from pllo.ops.amulet_lifted_islands import (
        run_swiglu_selector_lifted_mlp_island)
    torch.manual_seed(0)
    d, h, dout = 6, 8, 5
    x = torch.randn(4, d, dtype=torch.float64)
    n_in = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    n_in_inv = torch.linalg.inv(n_in)
    n_out = torch.linalg.qr(torch.randn(dout, dout, dtype=torch.float64))[0]
    w_up = torch.randn(d, h, dtype=torch.float64)
    w_gate = torch.randn(d, h, dtype=torch.float64)
    w_down = torch.randn(h, dout, dtype=torch.float64)
    b_up = torch.randn(h, dtype=torch.float64)
    b_gate = torch.randn(h, dtype=torch.float64)
    b_down = torch.randn(dout, dtype=torch.float64)
    out = run_swiglu_selector_lifted_mlp_island(
        x, w_up, b_up, w_gate, b_gate, w_down, b_down, n_in, n_in_inv, n_out,
        k=4, seed=7)
    # the lifted masked output equals the plain output @ n_out (exact migration)
    assert torch.allclose(out["y_tilde"], out["expected_y_tilde"], atol=1e-9)


def test_runner_silu_lift_matches_reference() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    x = torch.randn(2, 7, 16, dtype=torch.float32)
    ts = make_folded_nonlinear_runner("trusted_shortcut", lift_k=4)
    assert torch.allclose(ts.silu(x), F.silu(x), atol=1e-6)
    # rmsnorm + softmax migrations are numerically equivalent too
    assert torch.allclose(ts.rmsnorm_core(x, 1e-6),
                          x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6),
                          atol=1e-6)
    sc = torch.randn(2, 3, 5, 5)
    assert torch.allclose(ts.softmax(sc, -1), torch.softmax(sc, -1), atol=1e-6)


def test_current_backend_still_trusted() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    x = torch.randn(3, 12, dtype=torch.float32)
    cur = make_folded_nonlinear_runner("current")
    assert torch.allclose(cur.silu(x), F.silu(x))
    ev = cur.execution_evidence()
    assert ev["amulet_lift_executed"] is False
    assert ev["lifted_nonlinear_ops_count"] == 0
    assert ev["trusted_nonlinear_ops_count"] >= 1
    assert ev["nonlinear_execution_status"] == "executed_trusted_boundary_inline"
