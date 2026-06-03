"""Stage 7.5c tests for the Slalom delegated linear baseline."""

from __future__ import annotations

import torch

from pllo.baselines import SlalomConfig, SlalomDelegatedLinear, UnsupportedResult


def test_delegated_linear_correctness() -> None:
    sl = SlalomDelegatedLinear(SlalomConfig(dtype="float64"))
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(4, 8, dtype=torch.float64, generator=gen)
    w = torch.randn(8, 16, dtype=torch.float64, generator=gen)
    res = sl.delegated_linear(x, w, generator=gen)
    assert res["allclose"] is True
    assert res["max_abs_error"] < 1e-9


def test_freivalds_verification_passes_on_correct_recovery() -> None:
    sl = SlalomDelegatedLinear(SlalomConfig(dtype="float64", num_freivalds_rounds=4))
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(4, 8, dtype=torch.float64, generator=gen)
    w = torch.randn(8, 16, dtype=torch.float64, generator=gen)
    res = sl.delegated_linear(x, w, generator=gen)
    assert res["verification_passed"] is True


def test_declaration_marks_exact_primitive_and_no_full_system() -> None:
    sl = SlalomDelegatedLinear()
    assert sl.declare.exact_primitive_implemented is True
    assert sl.declare.full_system_reproduced is False
    assert sl.declare.supports_static_forward is True


def test_generation_unsupported_with_explicit_reason() -> None:
    sl = SlalomDelegatedLinear()
    res = sl.decode_step()
    assert isinstance(res, UnsupportedResult)
    assert "Slalom" in res.paper_scope_reason or "feed-forward" in res.paper_scope_reason


def test_train_step_unsupported_with_reason() -> None:
    sl = SlalomDelegatedLinear()
    res = sl.train_step()
    assert isinstance(res, UnsupportedResult)
    assert "inference-only" in res.reason or "inference-only" in res.paper_scope_reason
