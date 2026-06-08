"""Stage 7.7f tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.paper_cost_model import (
    MODES,
    PaperCostModelConfig,
    render_markdown,
    run_paper_cost_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_paper_cost_model(cfg=PaperCostModelConfig())


def test_all_modes_present(report: dict) -> None:
    assert set(report["modes_evaluated"]) == set(MODES)
    for m in MODES:
        assert m in report["tiny_config_counts"]
        assert m in report["real_config_estimates"]


def test_baseline_zero_round_trip(report: dict) -> None:
    r = report["real_config_estimates"]["baseline_plain"]
    assert r["round_trips_per_decode_step"] == 0
    assert r["intermediate_tee_reentry"] is False


def test_low_interaction_sequence_one_round_trip(report: dict) -> None:
    r = report["real_config_estimates"][
        "low_interaction_sequence_norm_exact_visible_attention"
    ]
    assert r["round_trips_per_decode_step"] == 1
    assert r["intermediate_tee_reentry"] is False


def test_trusted_softmax_round_trips_scale_with_layers(report: dict) -> None:
    L = report["config_real"]["L"]
    r = report["real_config_estimates"]["trusted_softmax_attention"]
    assert r["round_trips_per_decode_step"] == 1 + L
    assert r["intermediate_tee_reentry"] is True


def test_token_norm_more_expensive_than_sequence(report: dict) -> None:
    seq = report["real_config_estimates"][
        "low_interaction_sequence_norm_exact_visible_attention"
    ]
    tok = report["real_config_estimates"][
        "low_interaction_token_norm_exact_visible_attention"
    ]
    assert tok["accelerator_compute_ops"] >= seq["accelerator_compute_ops"]


def test_lora_overhead_added(report: dict) -> None:
    base = report["real_config_estimates"][
        "low_interaction_sequence_norm_exact_visible_attention"
    ]
    lora = report["real_config_estimates"]["lora_enabled"]
    assert lora["accelerator_compute_ops"] > base["accelerator_compute_ops"]


def test_lm_head_permutation_more_scalable(report: dict) -> None:
    dense = report["real_config_estimates"][
        "low_interaction_sequence_norm_exact_visible_attention"
    ]
    perm = report["real_config_estimates"]["scalable_lm_head_permutation"]
    block = report["real_config_estimates"]["scalable_lm_head_block"]
    assert perm["lm_head_mask_overhead_bytes"] \
        < dense["lm_head_mask_overhead_bytes"]
    assert block["lm_head_mask_overhead_bytes"] \
        < dense["lm_head_mask_overhead_bytes"]


def test_no_real_wall_clock_claim(report: dict) -> None:
    assert report["real_gpu_wall_clock_measured"] is False
    assert report["real_tee_wall_clock_measured"] is False


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    for m in MODES:
        assert m in md
    assert "Symbolic Formulas" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "paper_cost_model.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7f"
