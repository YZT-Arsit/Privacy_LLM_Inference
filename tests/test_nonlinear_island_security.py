"""Tests for the Stage 5.2 nonlinear island security proxy."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments import (
    NonlinearIslandSecurityConfig,
    run_nonlinear_island_security_experiments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_nonlinear_island_security.py"


@pytest.fixture(scope="module")
def security_payload():
    cfg = NonlinearIslandSecurityConfig(
        num_sessions=8,
        num_samples_per_session=32,
        num_trials=16,
        hidden_size=32,
        seed=2025,
    )
    return run_nonlinear_island_security_experiments(cfg)


def test_fixed_permutation_more_recoverable_than_fresh(security_payload) -> None:
    """Fixed permutation enables cross-session signature alignment."""
    p = security_payload["permutation_recovery_proxy"]["per_strategy"]
    assert (
        p["fixed_permutation"]["permutation_recovery_top1"]
        > p["fresh_permutation_per_session"]["permutation_recovery_top1"]
    )
    # Dense sandwich should be the lowest of the four.
    sandwich = p["dense_sandwich_reference"]["permutation_recovery_top1"]
    assert sandwich <= p["fresh_permutation_per_session"]["permutation_recovery_top1"]


def test_fixed_perm_no_pad_linkability_higher_than_fresh_with_pad(
    security_payload,
) -> None:
    l = security_payload["island_linkability_proxy"]["per_strategy"]
    assert (
        l["fixed_perm_no_pad"]["mean_pairwise_cosine"]
        > l["fresh_perm_with_linear_boundary_pad"]["mean_pairwise_cosine"]
    )
    # fixed_perm_no_pad ⇒ identical visible tensor ⇒ cosine ≈ 1.
    assert l["fixed_perm_no_pad"]["mean_pairwise_cosine"] > 0.99


def test_security_accounting_includes_coordinate_multiset_note(
    security_payload,
) -> None:
    by_family = {
        e["mask_family"]: e
        for e in security_payload["mask_family_accounting"]["table"]
    }
    perm_note = by_family["permutation"]["leakage_note"].lower()
    assert "coordinate-value multiset" in perm_note
    paired = by_family["paired_permutation"]["leakage_note"].lower()
    assert "paired" in paired
    # Dense mask family is present.
    assert "dense_invertible" in by_family


def test_global_limitations_recorded(security_payload) -> None:
    lims = security_payload["global_limitations"]
    text = " ".join(lims).lower()
    assert "proxy attacks" in text
    assert "fresh permutation" in text
    assert "coordinate-value multiset" in text
    assert "real tee" in text


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--num-sessions",
            "6",
            "--num-samples-per-session",
            "32",
            "--num-trials",
            "12",
            "--hidden-size",
            "32",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    for filename in (
        "nonlinear_island_security.json",
        "nonlinear_island_security.csv",
        "nonlinear_island_security.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "nonlinear_island_security.md").read_text(encoding="utf-8")
    for section in (
        "Permutation Recovery Proxy",
        "Island Linkability Proxy",
        "Mask Family Security Accounting",
        "Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert section in md, f"missing section: {section}"
    # spec-required limitations phrases:
    assert "proxy attacks" in md.lower()
    assert "coordinate-value multiset" in md.lower()
    assert "real tee" in md.lower()

    payload = json.loads(
        (tmp_path / "nonlinear_island_security.json").read_text(encoding="utf-8")
    )
    assert "permutation_recovery_proxy" in payload
    assert "island_linkability_proxy" in payload
    assert "mask_family_accounting" in payload
    assert "perm_recovery" in result.stdout
