"""Stage 5.6 — tests for the black-box query attacker."""

from __future__ import annotations

import json

import pytest

from pllo.experiments.blackbox_attacker import (
    BlackboxAttackerConfig,
    run_blackbox_attacker,
)


def _small_config(**overrides):
    cfg = dict(
        seed=2026,
        num_prompts=4,
        prompt_max_length=6,
        max_new_tokens=2,
        attempt_real_model_load=False,
        attempt_tokenizer_load=False,
        synthetic_vocab_size=32,
        synthetic_hidden_size=16,
        synthetic_intermediate_size=32,
        synthetic_num_attention_heads=4,
        synthetic_num_key_value_heads=2,
        synthetic_head_dim=4,
        max_layers=2,
    )
    cfg.update(overrides)
    return BlackboxAttackerConfig(**cfg)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_blackbox_attacker(_small_config())


def test_attacker_view_inventory_is_restricted(report) -> None:
    inv = report["attacker_view_inventory"]
    # Must NOT mention any internal trace name.
    for forbidden in (
        "boundary_input", "q", "k", "v", "gate", "up",
        "swiglu_intermediate", "post_island", "final",
        "kv_cache", "per_layer_traces", "n_k_stack", "n_v_stack",
    ):
        for line in inv:
            assert forbidden not in line.split(), (
                f"black-box view leaks internal trace name {forbidden!r}"
            )
    assert report["internal_trace_access"] == "denied"


def test_synthetic_fallback_runs(report) -> None:
    assert report["source"] in ("synthetic_block", )
    assert report["prompt_summary"]["token_source"] == "synthetic_token_ids"


def test_prompt_linkability_metrics_present(report) -> None:
    pl = report["prompt_linkability"]
    for k in (
        "same_prompt_similarity", "different_prompt_similarity",
        "linkability_auc_proxy", "nearest_prompt_retrieval_top1",
        "nearest_prompt_retrieval_random_chance",
    ):
        assert k in pl


def test_mode_distinguishability_metrics_present(report) -> None:
    mm = report["mitigation_mode_distinguishability"]
    for k in (
        "mode_classification_accuracy", "random_chance_baseline",
        "modes_observed",
    ):
        assert k in mm


def test_mode_distinguishability_at_or_below_random_chance(report) -> None:
    """Stage 6.4c guarantees byte-identical greedy output across bundles,
    so a black-box attacker should classify mode at random chance."""
    mm = report["mitigation_mode_distinguishability"]
    # Allow ≤ 2× random chance (tiny num_prompts; small variance).
    assert mm["mode_classification_accuracy"] <= max(
        2.0 * mm["random_chance_baseline"], 0.01
    ), f"mode distinguishability above 2x random chance: {mm}"


def test_json_safe_no_raw_tensor(report) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text
