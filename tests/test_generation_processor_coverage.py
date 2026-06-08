"""Stage 7.8c tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.generation_processor_coverage import (
    GenerationProcessorCoverageConfig,
    render_markdown,
    run_generation_processor_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_generation_processor_coverage(
        cfg=GenerationProcessorCoverageConfig()
    )


def test_recovered_logits_equal_plain(report: dict) -> None:
    assert report["logit_recovery_max_abs_error"] < 1e-9


def test_greedy_exact(report: dict) -> None:
    g = report["per_processor_detail"]["greedy"]
    assert g["argmax_match_rate"] == 1.0
    assert g["discrete_equal"] is True


def test_top_k_candidate_set_exact(report: dict) -> None:
    t = report["per_processor_detail"]["top_k"]
    # Distribution match must be very tight; -inf entries collapse to 0.
    assert t["max_abs_error_distribution"] < 1e-9
    assert t["argmax_match_rate"] == 1.0


def test_top_p_candidate_set_exact(report: dict) -> None:
    t = report["per_processor_detail"]["top_p"]
    assert t["max_abs_error_distribution"] < 1e-9
    assert t["argmax_match_rate"] == 1.0


def test_temperature_distribution_exact(report: dict) -> None:
    t = report["per_processor_detail"]["temperature"]
    assert t["max_abs_error_distribution"] < 1e-9


def test_repetition_penalty_uses_trusted_history(report: dict) -> None:
    r = report["per_processor_detail"]["repetition_penalty"]
    assert r["status"] == "tested"
    assert r["history_used_inside_trusted_side"] is True
    assert r["argmax_match_rate"] == 1.0


def test_bad_words_and_forced_token_hidden_from_accel(report: dict) -> None:
    bw = report["per_processor_detail"]["bad_words"]
    ft = report["per_processor_detail"]["forced_token"]
    assert bw["accelerator_sees_bad_word_list"] is False
    assert bw["bad_words_masked_to_neg_inf"] is True
    assert ft["forced_id_visible_to_accelerator"] is False


def test_temperature_sampling_reproducible_under_trusted_seed(report: dict) -> None:
    s = report["per_processor_detail"]["temperature_sampling_reproducible"]
    assert s["reproducible_under_same_trusted_seed"] is True


def test_beam_search_and_grammar_marked_audit_only(report: dict) -> None:
    assert report["processors"]["beam_search"] == "audit_only"
    assert report["processors"]["grammar_constrained"] == "audit_only"
    assert report["per_processor_detail"]["beam_search"]["theorem_applies"] is True
    assert report["per_processor_detail"]["grammar_constrained"]["theorem_applies"] is True


def test_processors_run_in_trusted_side_flags(report: dict) -> None:
    assert report["processors_run_inside_trusted_side"] is True
    assert report["accelerator_sees_processed_logits"] is False
    assert report["accelerator_sees_sampling_candidates"] is False


def test_output_length_side_channel_not_hidden_unless_padded(report: dict) -> None:
    text = " ".join(report["limitations"]).lower()
    assert "output length" in text
    assert ("not implemented here" in text or "not padded" in text
            or "leak" in text)


def test_unsafe_wording_avoidance(report: dict) -> None:
    text = " ".join(report["unsafe_wording_to_avoid"]).lower()
    assert "output length hidden" in text
    assert "stop timing side channel evaluated" in text


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Generation Processor Coverage" in md
    for k in ("greedy", "temperature", "top_k", "top_p",
              "repetition_penalty", "stop_token", "bad_words",
              "forced_token", "beam_search", "grammar_constrained"):
        assert k in md, k


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "generation_processor_coverage.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.8c"
