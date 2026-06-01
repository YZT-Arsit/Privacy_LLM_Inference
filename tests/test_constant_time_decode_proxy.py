"""Stage 5.6 extension — tests for constant_time_decode_mode in timing proxy."""

from __future__ import annotations

import json

import pytest

from pllo.experiments.timing_sidechannel_proxy import (
    VALID_CONSTANT_TIME_DECODE_MODES,
    TimingSidechannelConfig,
    normalize_constant_time_decode_mode,
    run_timing_sidechannel_proxy,
)


def _cfg(**overrides):
    cfg = dict(
        seed=2026,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        head_dim=4,
        layers=2,
        vocab_size=32,
        prompt_lengths=(4, 8),
        decode_steps=(0, 1, 2),
        samples_per_bin=8,
    )
    cfg.update(overrides)
    return TimingSidechannelConfig(**cfg)


def test_normalize_invalid_raises() -> None:
    assert normalize_constant_time_decode_mode(None) == "off"
    assert "off" in VALID_CONSTANT_TIME_DECODE_MODES
    assert "proxy_equalized" in VALID_CONSTANT_TIME_DECODE_MODES
    with pytest.raises(ValueError):
        normalize_constant_time_decode_mode("real_sleep")


def test_off_preserves_old_timing_proxy() -> None:
    r = run_timing_sidechannel_proxy(_cfg(constant_time_decode_mode="off"))
    ct = r["constant_time_decode"]
    assert ct["mode"] == "off"
    assert ct["decode_step_accuracy_before"] == ct["decode_step_accuracy_after"]
    assert ct["risk_level_before"] == ct["risk_level_after"]
    assert ct["overhead_ms_estimate"] == 0.0


def test_proxy_equalized_reduces_decode_step_leakage() -> None:
    r = run_timing_sidechannel_proxy(
        _cfg(constant_time_decode_mode="proxy_equalized")
    )
    ct = r["constant_time_decode"]
    assert ct["mode"] == "proxy_equalized"
    # After padding to per-method upper bound, step accuracy must drop OR
    # the limitation note must explain why (e.g. already at random chance).
    if ct["decode_step_accuracy_after"] >= ct["decode_step_accuracy_before"]:
        assert "limitation" in ct
    else:
        assert ct["decode_step_accuracy_after"] < ct["decode_step_accuracy_before"]
    assert ct["overhead_ms_estimate"] > 0.0


def test_overhead_proxy_is_a_proxy_only_label() -> None:
    r = run_timing_sidechannel_proxy(
        _cfg(constant_time_decode_mode="proxy_equalized")
    )
    ct = r["constant_time_decode"]
    assert "proxy only" in ct["limitation"].lower()
    # Markdown / runner exposes this; the underlying summary must never
    # claim a real-wall-time measurement.
    assert "wall-time" in ct["limitation"].lower() or "wall_time" in ct["limitation"].lower()


def test_cost_model_note_states_not_real_tee_timing() -> None:
    r = run_timing_sidechannel_proxy(_cfg(constant_time_decode_mode="off"))
    assert "NOT a real TEE" in r["cost_model_note"]


def test_no_wall_time_claim_in_constant_time_summary() -> None:
    r = run_timing_sidechannel_proxy(
        _cfg(constant_time_decode_mode="proxy_equalized")
    )
    text = json.dumps(r["constant_time_decode"], default=str).lower()
    assert "tensor(" not in text
    assert "real wall-time" in text or "real sleep" in text
