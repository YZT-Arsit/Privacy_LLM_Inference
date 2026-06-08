"""Stage 7.7e tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.integrity_spotcheck import (
    IntegritySpotCheckConfig,
    render_markdown,
    run_integrity_spotcheck,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_integrity_spotcheck(cfg=IntegritySpotCheckConfig())


def test_no_corruption_no_false_alarm(report: dict) -> None:
    for mode in report["modes_evaluated"]:
        for row in report["per_mode"][mode]["clean_curves"]:
            assert row["false_positive_rate"] == 0.0


def test_corrupted_q_proj_detected(report: dict) -> None:
    info = report["per_mode"]["spot_check_linear_projection"]
    # At the highest checked_fraction, detection rate must be > 0.
    last = info["corruption_present_curves"][-1]
    assert last["empirical_detection_rate"] > 0.0


def test_corrupted_kv_cache_append_detected(report: dict) -> None:
    info = report["per_mode"]["spot_check_kv_cache_append"]
    last = info["corruption_present_curves"][-1]
    assert last["empirical_detection_rate"] > 0.0


def test_corrupted_lm_head_slice_detected(report: dict) -> None:
    info = report["per_mode"]["spot_check_lm_head_slice"]
    last = info["corruption_present_curves"][-1]
    assert last["empirical_detection_rate"] > 0.0


def test_detection_probability_increases_with_checked_fraction(report: dict) -> None:
    # Sanity flag (with finite-sample tolerance).
    assert report["sanity"][
        "linear_projection_detection_increases_with_checked_fraction"
    ] is True
    # Spot-check: checked_fraction=0 -> detection_rate=0.
    for mode in ("spot_check_linear_projection",
                 "spot_check_kv_cache_append",
                 "spot_check_lm_head_slice"):
        first = report["per_mode"][mode]["corruption_present_curves"][0]
        assert first["checked_fraction"] == 0.0
        assert first["empirical_detection_rate"] == 0.0


def test_report_says_not_full_malicious_security(report: dict) -> None:
    assert report["full_verifiable_computation"] is False
    assert report["malicious_accelerator_privacy_not_addressed"] is True
    assert report["active_adversary_integrity_supported"] == (
        "probabilistic spot-check only"
    )
    text = " ".join(report["limitations"])
    assert "NOT a verifiable computation" in text or \
           "not full malicious" in text.lower() or \
           "NOT a verifiable" in text


def test_no_check_mode_undetected(report: dict) -> None:
    assert report["sanity"]["no_check_corruption_undetected"] is True


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Integrity Spot-Check" in md
    for mode in report["modes_evaluated"]:
        assert f"`{mode}`" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "integrity_spotcheck.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.7e"
