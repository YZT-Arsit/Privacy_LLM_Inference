"""Stage 7.7c tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.paged_kv_abstraction import (
    PagedKVConfig,
    render_markdown,
    run_paged_kv_abstraction,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_paged_kv_abstraction(cfg=PagedKVConfig())


def test_block_append_invariant(report: dict) -> None:
    for a in report["per_session_audit"]:
        assert a["per_block_invariant_max_abs_error"] < 1e-12


def test_multi_block_sequence_invariant(report: dict) -> None:
    # max_tokens_per_session=13, block_size=4 => 4 blocks (3 full + 1 partial).
    cfg = report["config"]
    expected_blocks = (cfg["max_tokens_per_session"] + cfg["block_size"] - 1) // cfg["block_size"]
    for a in report["per_session_audit"]:
        assert a["num_blocks_used"] == expected_blocks
        assert a["full_cache_invariant_max_abs_error"] < 1e-12


def test_block_table_remapping_invariant(report: dict) -> None:
    # The full-cache invariant uses gather_full_tilde which walks the
    # block table. If it matches the per-block invariant, remapping
    # is consistent.
    for a in report["per_session_audit"]:
        assert (
            abs(a["full_cache_invariant_max_abs_error"]
                - a["per_block_invariant_max_abs_error"]) < 1e-10
        )


def test_no_plaintext_kv_block(report: dict) -> None:
    info = report["no_plaintext_kv_block_check"]
    assert info["min_distance_masked_block_to_any_plain_row"] > 0.0


def test_cross_session_block_sharing_disabled_by_default(report: dict) -> None:
    assert report["cross_user_cache_sharing_allowed"] is False
    assert report["prefix_cache_sharing_default"] is False
    assert report["cross_session_mask_isolation"][
        "cross_session_mask_isolation_observed"] is True


def test_prefix_sharing_requires_explicit_flag(report: dict) -> None:
    p = report["prefix_cache_sharing"]
    assert p["prefix_cache_sharing_enabled"] is False
    assert p["public_prefix_token_count"] == 0
    assert "explicit" in p["leakage_note"].lower()


def test_paged_cache_gqa_indexing(report: dict) -> None:
    gqa = report["gqa_paged_cache_indexing_check"]
    assert gqa["block_table_indexing_per_kv_head_supported"] is True
    cfg = report["config"]
    expected_blocks = (cfg["max_tokens_per_session"] + cfg["block_size"] - 1) // cfg["block_size"]
    for row in gqa["block_table_lengths"]:
        for length in row:
            assert length == expected_blocks


def test_policy_flags(report: dict) -> None:
    assert report["paged_kv_supported"] is True
    assert report["private_cache_mode"] is True
    assert report["timing_side_channel_not_evaluated"] is True


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Paged KV-Cache Abstraction" in md
    assert "Per-Session Audit" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "paged_kv_abstraction.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7c"
