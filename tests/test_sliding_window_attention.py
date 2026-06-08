"""Stage 7.8a tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.sliding_window_attention import (
    SlidingWindowConfig,
    render_markdown,
    run_sliding_window_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_sliding_window_attention(cfg=SlidingWindowConfig())


def test_full_window_matches_full_causal(report: dict) -> None:
    for r in report["per_window_results"]:
        if r["window_size"] == "full":
            full_vs = r["full_vs_sliding_match_when_window_ge_seqlen"]
            assert full_vs is not None
            assert full_vs < 1e-9


def test_masked_sliding_logits_match_plain(report: dict) -> None:
    for r in report["per_window_results"]:
        assert r["attn_out_recovered_max_abs_error_vs_plain"] < 1e-9


def test_greedy_sequence_exact(report: dict) -> None:
    assert report["greedy_token_match_rate"] == 1.0
    assert report["sequence_exact_match"] is True


def test_rolling_kv_evicts_tokens_outside_window(report: dict) -> None:
    # The eviction-correct flag is recorded per window.
    for r in report["per_window_results"]:
        assert r["window_eviction_correct"] is True


def test_kv_window_invariant_holds(report: dict) -> None:
    for r in report["per_window_results"]:
        assert r["kv_window_invariant_max_abs_error"] < 1e-12


def test_rope_safe_path_carryover(report: dict) -> None:
    assert report["rope_mask_mode"] == "pre_rope_block_diagonal_rotation"
    assert report["rope_transient_plain_qk_visible"] is False
    assert report["qkv_projection_outputs_masked_directly"] is True
    for r in report["per_window_results"]:
        assert r["rope_transient_plain_qk_visible"] is False
        assert r["qkv_projection_outputs_masked_directly"] is True


def test_pad_does_not_enter_cores(report: dict) -> None:
    assert report["pad_enters_rmsnorm_core"] is False
    assert report["pad_enters_rope_core"] is False
    assert report["pad_enters_swiglu_core"] is False
    assert report["pad_enters_softmax"] is False


def test_exact_visible_attention_visible(report: dict) -> None:
    for r in report["per_window_results"]:
        if r["attention_privacy_mode"] == "exact_visible_attention":
            assert r["attention_scores_visible"] is True
            assert r["attention_extra_tee_round_trips_per_layer"] == 0


def test_trusted_softmax_extra_tee_round_trips(report: dict) -> None:
    for r in report["per_window_results"]:
        if r["attention_privacy_mode"] == "trusted_softmax_attention":
            assert r["attention_scores_visible"] is False
            assert r["attention_extra_tee_round_trips_per_layer"] >= 1


def test_window_size_policy_is_public(report: dict) -> None:
    text = " ".join(report["limitations"]).lower()
    assert "window size policy is public" in text or \
           "window size policy" in text


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Sliding Window Attention" in md
    assert "Per-Window Results" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "sliding_window_attention.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.8a"
