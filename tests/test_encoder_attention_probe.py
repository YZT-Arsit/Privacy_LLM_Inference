"""Tests for the Stage 6.1 encoder-only attention probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from pllo.experiments import (
    EncoderAttentionProbeConfig,
    run_encoder_attention_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_encoder_attention_experiments.py"


def _run_or_skip(config: EncoderAttentionProbeConfig) -> dict:
    result = run_encoder_attention_probe(config)
    if result["model_loading"]["status"] != "loaded":
        pytest.skip(
            f"tiny-bert unavailable: {result['model_loading'].get('reason')}"
        )
    return result


# ---------------------------------------------------------------------------
# Loading + structural discovery
# ---------------------------------------------------------------------------


def test_probe_loads_first_layer_qkv_and_output_dense() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=1, seq_len=4, use_pad=True)
    )
    loading = result["model_loading"]
    assert loading["model_class"].startswith("Bert")
    assert loading["num_attention_heads"] > 0
    assert loading["hidden_size"] > 0
    assert loading["head_dim"] * loading["num_attention_heads"] == loading["hidden_size"]


# ---------------------------------------------------------------------------
# N_Q N_K^T = I (per head)
# ---------------------------------------------------------------------------


def test_qk_constraint_holds() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=1, seq_len=4, use_pad=False)
    )
    qkv = result["qkv_invariants"]
    assert qkv["qk_constraint_error"] < 1e-5, qkv["qk_constraint_error"]
    assert qkv["qkv_allclose"] is True


def test_q_k_v_match_expected_mask_application() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=2, seq_len=8, use_pad=True)
    )
    qkv = result["qkv_invariants"]
    assert qkv["q_metrics"]["allclose"] is True
    assert qkv["k_metrics"]["allclose"] is True
    assert qkv["v_metrics"]["allclose"] is True
    assert qkv["q_metrics"]["max_abs_error"] < 1e-4
    assert qkv["k_metrics"]["max_abs_error"] < 1e-4
    assert qkv["v_metrics"]["max_abs_error"] < 1e-4


# ---------------------------------------------------------------------------
# Score / softmax / V aggregation / output invariants, both mask kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_attention_invariants_under_both_mask_kinds(use_pad: bool) -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=2, seq_len=8, use_pad=use_pad)
    )
    for mask_kind in ("all_ones", "padding"):
        payload = result["results_per_mask"][mask_kind]
        assert payload["score_metrics"]["allclose"] is True, (
            f"score invariant failed for use_pad={use_pad} mask={mask_kind}: "
            f"{payload['score_metrics']}"
        )
        assert payload["prob_metrics"]["allclose"] is True
        assert payload["v_aggr_metrics"]["allclose"] is True
        assert payload["output_metrics"]["allclose"] is True
        assert payload["output_metrics"]["max_abs_error"] < 1e-4
        assert payload["allclose"] is True


# ---------------------------------------------------------------------------
# Pad compensation: all four projections (Q/K/V/O)
# ---------------------------------------------------------------------------


def test_use_pad_true_propagates_pad_to_all_four_projections() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=2, seq_len=8, use_pad=True)
    )
    for mask_kind in ("all_ones", "padding"):
        pad = result["pad_report"]["per_mask"][mask_kind]
        assert pad["q_pad"] is True
        assert pad["k_pad"] is True
        assert pad["v_pad"] is True
        assert pad["o_pad"] is True


def test_use_pad_false_records_no_pad_anywhere() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=2, seq_len=8, use_pad=False)
    )
    for mask_kind in ("all_ones", "padding"):
        pad = result["pad_report"]["per_mask"][mask_kind]
        assert pad == {
            "q_pad": False,
            "k_pad": False,
            "v_pad": False,
            "o_pad": False,
        }


# ---------------------------------------------------------------------------
# Mask structure metadata
# ---------------------------------------------------------------------------


def test_mask_structure_records_bidirectional_no_cache() -> None:
    result = _run_or_skip(
        EncoderAttentionProbeConfig(batch_size=1, seq_len=4, use_pad=True)
    )
    mask = result["mask_structure"]
    assert mask["attention_kind"] == "bidirectional_self_attention"
    assert mask["right_multiply_mask"] is True
    assert mask["qk_constraint"] == "N_Q N_K^T = I"
    assert mask["cache_type"] == "none"


# ---------------------------------------------------------------------------
# Skip-on-unavailable: bogus model id ⇒ status="skipped", not a crash
# ---------------------------------------------------------------------------


def test_probe_returns_skipped_on_unresolvable_model_id() -> None:
    result = run_encoder_attention_probe(
        EncoderAttentionProbeConfig(
            model_id="does/not/exist-stage61-bogus",
            batch_size=1,
            seq_len=4,
            use_pad=True,
        )
    )
    assert result["model_loading"]["status"] == "skipped"
    assert result["results_per_mask"] == {}


# ---------------------------------------------------------------------------
# End-to-end script smoke
# ---------------------------------------------------------------------------


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    for filename in (
        "encoder_attention_experiments.json",
        "encoder_attention_experiments.csv",
        "encoder_attention_experiments.md",
    ):
        assert (tmp_path / filename).exists(), filename

    payload = json.loads(
        (tmp_path / "encoder_attention_experiments.json").read_text(encoding="utf-8")
    )
    assert "results" in payload
    # Sweep size = 2 batch × 3 seq × 2 pad = 12 cells.
    assert len(payload["results"]) == 12

    md = (tmp_path / "encoder_attention_experiments.md").read_text(encoding="utf-8")
    assert "Encoder-only Attention Probe" in md
    assert "Limitations" in md
    assert "Next stage plan" in md
    assert "encoder-decoder cross-attention" in md.lower()
    assert "cells=" in result.stdout


def test_script_handles_unresolvable_model_id_gracefully(tmp_path) -> None:
    """A bogus override flips every cell to skipped without crashing the script."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model-id",
            "does/not/exist-stage61-bogus",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(
        (tmp_path / "encoder_attention_experiments.json").read_text(encoding="utf-8")
    )
    assert all(
        r["model_loading"]["status"] == "skipped" for r in payload["results"]
    )
    assert "skipped=12" in result.stdout
