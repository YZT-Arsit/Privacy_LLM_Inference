"""Tests for the Stage 6.2 encoder-decoder cross-attention probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from pllo.experiments import (
    CrossAttentionProbeConfig,
    EncoderMemoryCache,
    run_cross_attention_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_cross_attention_experiments.py"


def _run_or_skip(config: CrossAttentionProbeConfig) -> dict:
    result = run_cross_attention_probe(config)
    if result["model_loading"]["status"] != "loaded":
        pytest.skip(
            f"tiny encoder-decoder model unavailable: "
            f"{result['model_loading'].get('reason')}"
        )
    return result


# ---------------------------------------------------------------------------
# Loading + structural discovery
# ---------------------------------------------------------------------------


def test_probe_loads_first_decoder_cross_attention() -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=1, dec_seq_len=4, enc_seq_len=8, use_pad=True
        )
    )
    loading = result["model_loading"]
    assert loading["family"] in {"t5", "bart"}
    assert loading["num_attention_heads"] > 0
    assert loading["hidden_size"] > 0
    assert loading["head_dim"] > 0
    assert loading["inner_dim"] == loading["num_attention_heads"] * loading["head_dim"]
    # T5 has no biases on cross-attention; BART has biases. Either is fine.
    assert set(loading["bias_present"].keys()) == {"q", "k", "v", "o"}


def test_projection_helper_supports_bias_none() -> None:
    """T5 attention has ``bias=None`` on every projection — the probe must not crash."""
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=1, dec_seq_len=4, enc_seq_len=4, use_pad=False
        )
    )
    loading = result["model_loading"]
    if loading["family"] == "t5":
        for key in ("q", "k", "v", "o"):
            assert loading["bias_present"][key] is False, (
                f"T5 cross-attention should have no bias on {key}, "
                f"got {loading['bias_present']}"
            )
    else:
        # BART: every projection has a bias. Just confirm the probe ran.
        assert any(loading["bias_present"].values())
    # And the QKV invariants still pass under bias=None.
    qkv = result["qkv_invariants"]
    assert qkv["qkv_allclose"] is True


# ---------------------------------------------------------------------------
# N_Q_dec N_K_enc^T = I (per head)
# ---------------------------------------------------------------------------


def test_qk_constraint_holds() -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=1, dec_seq_len=4, enc_seq_len=8, use_pad=False
        )
    )
    qkv = result["qkv_invariants"]
    assert qkv["qk_constraint_error"] < 1e-5, qkv["qk_constraint_error"]
    assert qkv["qkv_allclose"] is True


# ---------------------------------------------------------------------------
# Score / softmax / V aggregation / output invariants, both mask kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_attention_invariants_under_both_encoder_mask_kinds(use_pad: bool) -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=2, dec_seq_len=4, enc_seq_len=8, use_pad=use_pad
        )
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
        CrossAttentionProbeConfig(
            batch_size=2, dec_seq_len=4, enc_seq_len=8, use_pad=True
        )
    )
    for mask_kind in ("all_ones", "padding"):
        pad = result["pad_report"]["per_mask"][mask_kind]
        assert pad["q_pad"] is True
        assert pad["k_pad"] is True
        assert pad["v_pad"] is True
        assert pad["o_pad"] is True


def test_use_pad_false_records_no_pad_anywhere() -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=2, dec_seq_len=4, enc_seq_len=8, use_pad=False
        )
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
# EncoderMemoryCache invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_encoder_memory_cache_invariants_hold(use_pad: bool) -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=2, dec_seq_len=4, enc_seq_len=8, use_pad=use_pad
        )
    )
    cache = result["encoder_memory_cache"]
    assert cache["key_metrics"]["allclose"] is True
    assert cache["value_metrics"]["allclose"] is True
    assert cache["key_metrics"]["max_abs_error"] < 1e-4
    assert cache["value_metrics"]["max_abs_error"] < 1e-4
    assert cache["allclose"] is True
    assert cache["encoder_seq_len"] == 8
    assert cache["batch_size"] == 2


def test_encoder_memory_cache_dataclass_shape() -> None:
    """``EncoderMemoryCache`` is the probe-level structure exported for downstream stages."""
    import torch

    cache = EncoderMemoryCache(
        key_tilde=torch.zeros(1, 1, 1, 1),
        value_tilde=torch.zeros(1, 1, 1, 1),
        key_plain=torch.zeros(1, 1, 1, 1),
        value_plain=torch.zeros(1, 1, 1, 1),
        n_k=torch.eye(1),
        n_v=torch.eye(1),
        encoder_seq_len=1,
        batch_size=1,
    )
    assert cache.encoder_seq_len == 1
    assert cache.batch_size == 1


# ---------------------------------------------------------------------------
# Mask structure metadata
# ---------------------------------------------------------------------------


def test_mask_structure_records_cross_attention_topology() -> None:
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=1, dec_seq_len=4, enc_seq_len=8, use_pad=True
        )
    )
    mask = result["mask_structure"]
    assert mask["attention_kind"] == "encoder_decoder_cross_attention"
    assert mask["right_multiply_mask"] is True
    assert mask["qk_constraint"] == "N_Q_dec N_K_enc^T = I"
    assert mask["decoder_query_input_mask_independent_from_encoder_kv_input_mask"] is True
    assert mask["cache_type"] == "encoder_memory_cache (probe-level)"


# ---------------------------------------------------------------------------
# dec_seq_len=1 corresponds to a single decode step
# ---------------------------------------------------------------------------


def test_single_decoder_query_step() -> None:
    """``dec_seq_len=1`` exercises the one-step decode shape without hitting full generation."""
    result = _run_or_skip(
        CrossAttentionProbeConfig(
            batch_size=2, dec_seq_len=1, enc_seq_len=8, use_pad=True
        )
    )
    payload = result["results_per_mask"]["padding"]
    assert payload["allclose"] is True


# ---------------------------------------------------------------------------
# Skip-on-unavailable: bogus model id ⇒ status="skipped", not a crash
# ---------------------------------------------------------------------------


def test_probe_returns_skipped_on_unresolvable_model_id() -> None:
    result = run_cross_attention_probe(
        CrossAttentionProbeConfig(
            model_id="does/not/exist-stage62-bogus",
            batch_size=1,
            dec_seq_len=4,
            enc_seq_len=4,
            use_pad=True,
        )
    )
    assert result["model_loading"]["status"] == "skipped"
    assert result["results_per_mask"] == {}
    assert result["encoder_memory_cache"]["per_mask"] == {}


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
        "cross_attention_experiments.json",
        "cross_attention_experiments.csv",
        "cross_attention_experiments.md",
    ):
        assert (tmp_path / filename).exists(), filename

    payload = json.loads(
        (tmp_path / "cross_attention_experiments.json").read_text(encoding="utf-8")
    )
    assert "results" in payload
    # Sweep size = 2 batch × 2 dec × 3 enc × 2 pad = 24 cells.
    assert len(payload["results"]) == 24

    md = (tmp_path / "cross_attention_experiments.md").read_text(encoding="utf-8")
    assert "Encoder-decoder Cross-attention Probe" in md
    assert "Limitations" in md
    assert "Next stage plan" in md
    assert "Encoder memory cache invariants" in md
    assert "cells=" in result.stdout
    assert "cache_allclose=" in result.stdout


def test_script_handles_unresolvable_model_id_gracefully(tmp_path) -> None:
    """A bogus override flips every cell to skipped without crashing the script."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model-id",
            "does/not/exist-stage62-bogus",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(
        (tmp_path / "cross_attention_experiments.json").read_text(encoding="utf-8")
    )
    assert all(
        r["model_loading"]["status"] == "skipped" for r in payload["results"]
    )
    assert "skipped=24" in result.stdout
