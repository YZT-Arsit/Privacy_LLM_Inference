"""Stage 7.5c tests for the Amulet static-PHQ baseline + KV counterexample."""

from __future__ import annotations

import torch

from pllo.baselines import (
    AmuletConfig,
    AmuletStaticPHQ,
    UnsupportedResult,
    ours_right_mask_kv_append,
)


def test_phq_linear_correctness() -> None:
    amu = AmuletStaticPHQ(AmuletConfig(dtype="float64"))
    gen = torch.Generator().manual_seed(2026)
    h = torch.randn(4, 8, dtype=torch.float64, generator=gen)
    w = torch.randn(8, 16, dtype=torch.float64, generator=gen)
    res = amu.static_linear_forward(h, w, generator=gen)
    assert res["allclose"] is True
    assert res["max_abs_error"] < 1e-9


def test_kv_append_counterexample_present() -> None:
    amu = AmuletStaticPHQ(AmuletConfig(dtype="float64"))
    gen = torch.Generator().manual_seed(2026)
    ce = amu.kv_append_counterexample(seq_len_old=2, seq_len_new=1, d=8, generator=gen)
    assert ce["counterexample_present"] is True
    assert ce["max_gap"] > 1e-3
    assert ce["block_compatible_max_gap"] < 1e-9
    assert ce["kv_append_supported"] is False
    assert "block-diag" in ce["mathematical_reason"].lower() or "block-compatible" in ce["block_compatible_condition"].lower()


def test_decoder_generation_unsupported_under_fresh_left_mask() -> None:
    amu = AmuletStaticPHQ()
    res = amu.decode_step()
    assert isinstance(res, UnsupportedResult)
    assert "block-compatible" in res.mathematical_reason.lower() or "concatenation" in res.mathematical_reason.lower()


def test_ours_right_mask_kv_append_passes() -> None:
    gen = torch.Generator().manual_seed(2026)
    res = ours_right_mask_kv_append(seq_len_old=2, seq_len_new=1, d=8, generator=gen)
    assert res["ours_append_supported"] is True
    assert res["max_abs_error"] < 1e-9


def test_declaration_marks_full_system_not_reproduced() -> None:
    amu = AmuletStaticPHQ()
    assert amu.declare.full_system_reproduced is False
    assert amu.declare.supports_decoder_generation is False
