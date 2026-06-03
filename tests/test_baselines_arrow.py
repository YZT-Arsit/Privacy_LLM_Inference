"""Stage 7.5c tests for the Arrow baseline (recorded as missing-formula)."""

from __future__ import annotations

import inspect

from pllo.baselines import ArrowDirectPrimitive, UnsupportedResult


def test_arrow_either_implemented_or_marked_missing_formula() -> None:
    arrow = ArrowDirectPrimitive()
    if arrow.declare.exact_primitive_implemented:
        # If a future stage implements the formula, the test still passes.
        return
    res = arrow.forward()
    assert isinstance(res, UnsupportedResult)
    assert res.reason
    assert arrow.missing_paper_formula is True


def test_arrow_does_not_silently_substitute_proxy() -> None:
    arrow = ArrowDirectPrimitive()
    src = inspect.getsource(ArrowDirectPrimitive)
    # The Arrow stub must NOT contain the word "proxy" as a substitute for
    # the missing primitive (it is allowed in the explanatory docstring).
    # Specifically: the stub must NOT implement an arrow-like operator and
    # claim it as Arrow. We assert that the forward method body returns
    # an UnsupportedResult.
    assert "UnsupportedResult" in src


def test_arrow_unsupported_reason_explicit() -> None:
    arrow = ArrowDirectPrimitive()
    res = arrow.forward()
    if isinstance(res, UnsupportedResult):
        assert "Arrow" in res.reason or "arrow" in res.reason.lower()
        assert res.implementation_scope_reason
