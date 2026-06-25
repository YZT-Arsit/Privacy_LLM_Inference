"""trusted_shortcut must never silently fall back to the current trusted path.

If design B is selected, the MLP activation must actually dispatch through the
Amulet lift (recorded), and a genuinely unsupported op must fail LOUDLY with a
structured error rather than quietly running the current trusted path under a
trusted_shortcut tag.

CPU torch only. Run:
    python -m pytest tests/test_trusted_shortcut_no_silent_fallback.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_trusted_shortcut_silu_dispatches_to_lift_not_current() -> None:
    torch = pytest.importorskip("torch")
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    ts = make_folded_nonlinear_runner("trusted_shortcut", lift_k=2)
    ts.silu(torch.randn(2, 6, 8))
    ev = ts.acc
    # the activation was LIFTED (design B), not counted as a trusted island
    assert ev.amulet_lift_executed is True
    assert ev.lifted_nonlinear_ops_count == 1
    assert ev.trusted_nonlinear_ops_count == 0


def test_unsupported_op_fails_loudly() -> None:
    pytest.importorskip("torch")
    from pllo.deployment.folded_nonlinear import (
        UnsupportedNonlinearOp, make_folded_nonlinear_runner)
    ts = make_folded_nonlinear_runner("trusted_shortcut")
    with pytest.raises(UnsupportedNonlinearOp):
        ts.fail_unsupported("some_exotic_activation")
    assert "some_exotic_activation" in ts.acc.unsupported_ops


def test_current_and_trusted_shortcut_are_distinguishable() -> None:
    pytest.importorskip("torch")
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    cur = make_folded_nonlinear_runner("current")
    ts = make_folded_nonlinear_runner("trusted_shortcut")
    assert cur.op_backend == "current"
    assert ts.op_backend == "amulet_migrated"
    assert cur._amulet is None and ts._amulet is not None
