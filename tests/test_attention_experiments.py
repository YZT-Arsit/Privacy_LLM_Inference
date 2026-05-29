"""Tests for Stage 5.0 attention probe."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.experiments import AttentionProbeConfig, run_attention_probe


def _try_run(config: AttentionProbeConfig):
    try:
        return run_attention_probe(config)
    except Exception as exc:  # noqa: BLE001 — runtime model-load failure should skip, not fail.
        msg = str(exc).lower()
        if "huggingface" in msg or "not found" in msg or "connection" in msg or "tiny-gpt2" in msg:
            pytest.skip(f"sshleifer/tiny-gpt2 unavailable: {exc}")
        raise


REQUIRED_TOP_LEVEL_KEYS = {
    "config",
    "full_attention",
    "prefill_attention",
    "decode_attention",
    "mask_structure",
    "pad_report",
}


def test_attention_probe_returns_complete_fields() -> None:
    result = _try_run(
        AttentionProbeConfig(batch_size=1, seq_len=4, decode_steps=1, use_pad=True)
    )
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(result.keys())
    full = result["full_attention"]
    for key in ("output_metrics", "score_metrics", "prob_metrics", "v_aggr_metrics", "qk_constraint_error", "allclose"):
        assert key in full
    prefill = result["prefill_attention"]
    for key in ("output_metrics", "cache_key_metrics", "cache_value_metrics", "cache_invariant_allclose"):
        assert key in prefill
    decode = result["decode_attention"]
    for key in ("per_step", "decode_output_max_abs_error_max", "cache_append_invariant_allclose"):
        assert key in decode
    mask = result["mask_structure"]
    assert mask["right_multiply_mask"] is True
    assert mask["fused_c_attn_block_diagonal"] is True
    assert "N_Q N_K^T" in mask["qk_constraint"]


@pytest.mark.parametrize("use_pad", [True, False])
def test_attention_probe_invariants_hold(use_pad: bool) -> None:
    result = _try_run(
        AttentionProbeConfig(
            batch_size=2, seq_len=8, decode_steps=2, use_pad=use_pad
        )
    )
    full = result["full_attention"]
    prefill = result["prefill_attention"]
    decode = result["decode_attention"]

    assert full["allclose"] is True, f"full attention invariants failed: {full}"
    assert full["score_metrics"]["allclose"] is True
    assert full["prob_metrics"]["allclose"] is True
    assert full["v_aggr_metrics"]["allclose"] is True
    assert full["output_metrics"]["allclose"] is True
    assert full["qk_constraint_error"] < 1e-5, (
        f"N_Q N_K^T constraint error {full['qk_constraint_error']} >= 1e-5"
    )

    assert prefill["cache_invariant_allclose"] is True
    assert prefill["cache_key_metrics"]["max_abs_error"] < 1e-4
    assert prefill["cache_value_metrics"]["max_abs_error"] < 1e-4

    assert decode["cache_append_invariant_allclose"] is True
    assert len(decode["per_step"]) == 2
    for step in decode["per_step"]:
        assert step["output_metrics"]["allclose"] is True
        assert step["new_key_metrics"]["allclose"] is True
        assert step["new_value_metrics"]["allclose"] is True


def test_attention_probe_pad_report_records_use_pad_true() -> None:
    result = _try_run(
        AttentionProbeConfig(batch_size=1, seq_len=4, decode_steps=1, use_pad=True)
    )
    pad = result["pad_report"]
    assert pad["use_pad"] is True
    assert pad["attn_c_attn_pad"] is True
    assert pad["attn_c_proj_pad"] is True
    assert pad["compensation_formula"] == "C_T = T W N_out"


def test_attention_probe_pad_report_records_use_pad_false() -> None:
    result = _try_run(
        AttentionProbeConfig(batch_size=1, seq_len=4, decode_steps=1, use_pad=False)
    )
    pad = result["pad_report"]
    assert pad["use_pad"] is False
    assert pad["attn_c_attn_pad"] is False
    assert pad["attn_c_proj_pad"] is False


def test_attention_probe_decode_steps_zero_is_supported() -> None:
    result = _try_run(
        AttentionProbeConfig(batch_size=1, seq_len=4, decode_steps=0, use_pad=True)
    )
    decode = result["decode_attention"]
    assert decode["per_step"] == []
    assert decode["decode_output_max_abs_error_max"] is None
    # With zero decode steps the cumulative cache is just the prefill cache,
    # so the append invariant collapses to the prefill cache invariant.
    assert decode["cache_append_invariant_allclose"] is True
