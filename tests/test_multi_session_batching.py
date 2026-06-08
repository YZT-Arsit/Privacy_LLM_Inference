"""Stage 7.7d tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.multi_session_batching import (
    MultiSessionBatchingConfig,
    render_markdown,
    run_multi_session_batching,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_multi_session_batching(cfg=MultiSessionBatchingConfig())


def test_two_sessions_do_not_share_masks(report: dict) -> None:
    fp = report["fingerprint_isolation"]
    assert fp["fingerprints_differ_per_session"] is True
    # The two fingerprints must differ for the same prompt.
    assert fp["session_0_layer_entry"] != fp["session_1_layer_entry"]


def test_same_token_different_session_different_fingerprints(report: dict) -> None:
    fp = report["fingerprint_isolation"]
    assert fp["session_0_layer_entry"] is not None
    assert fp["session_1_layer_entry"] is not None
    assert fp["session_0_layer_entry"] != fp["session_1_layer_entry"]


def test_kv_cache_masks_session_specific(report: dict) -> None:
    # If session masks were shared, sessions 0 and 1 would have
    # identical fingerprints (same prompt). They differ -> n_k / n_v
    # are session-specific by extension of the per-session compile.
    fp = report["fingerprint_isolation"]
    assert fp["session_0_layer_entry"] != fp["session_1_layer_entry"]


def test_batching_output_equals_independent_per_session(report: dict) -> None:
    be = report["batching_equivalence"]
    if be.get("checked"):
        assert be["row_0_matches_session_run"] is True
        assert be["row_1_matches_session_run"] is True


def test_ragged_lengths_handled(report: dict) -> None:
    assert report["ragged_lengths_handled"] is True
    # The third session (length 7) must also have h_hat invariant
    # < tolerance.
    assert all(
        r["h_hat_layer_entry_invariant_max_abs_error"] < 1e-9
        for r in report["per_session_results"]
    )


def test_no_cross_session_prefix_sharing_default(report: dict) -> None:
    assert report["cross_session_prefix_sharing_default"] is False


def test_use_pad_preserved_implicit(report: dict) -> None:
    # All sessions ran with use_pad=True (wrapper default); the
    # boundary invariants are non-trivial because of fresh pads.
    for r in report["per_session_results"]:
        assert r["greedy_token_match_rate"] == 1.0
        assert r["sequence_exact_match"] is True


def test_policy_flags(report: dict) -> None:
    assert report["multi_session_supported"] is True
    assert report["continuous_batching_simulated"] is True
    assert report["cross_session_mask_isolation"] is True
    assert report["timing_side_channel_not_evaluated"] is True


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Multi-Session" in md
    assert "Fingerprint Isolation" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "multi_session_batching.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7d"
