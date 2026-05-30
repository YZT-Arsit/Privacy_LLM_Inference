"""Tests for the Stage 5.1 norm probes + experiment script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pllo.experiments import (
    RMSNormOrthogonalProbeConfig,
    TrustedNormProbeConfig,
    run_rmsnorm_orthogonal_probe,
    run_trusted_norm_probe,
)
from pllo.experiments.norm_probe import _generate_orthogonal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_norm_experiments.py"


# ---------------------------------------------------------------------------
# Orthogonal mask basics
# ---------------------------------------------------------------------------


def test_qr_generated_mask_is_orthogonal() -> None:
    N = _generate_orthogonal(64, torch.float32, torch.device("cpu"))
    eye = torch.eye(64, dtype=N.dtype)
    err = float((N.T @ N - eye).abs().max().item())
    assert err < 1e-5


# ---------------------------------------------------------------------------
# RMSNorm probe results
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def orthogonal_probe() -> dict:
    cfg = RMSNormOrthogonalProbeConfig(
        batch_size=2,
        seq_len=8,
        hidden_size=64,
        num_trials=12,
        seed=2024,
    )
    return run_rmsnorm_orthogonal_probe(cfg)


def test_rms_preservation_under_orthogonal_mask(orthogonal_probe) -> None:
    assert orthogonal_probe["rms_preservation_error"] < 1e-4


def test_normalized_state_commutes_without_gamma(orthogonal_probe) -> None:
    assert orthogonal_probe["allclose_without_gamma"] is True
    assert orthogonal_probe["normalized_state_error"] < 1e-4


def test_scalar_gamma_commutes(orthogonal_probe) -> None:
    assert orthogonal_probe["allclose_with_scalar_gamma"] is True
    assert orthogonal_probe["gamma_commutation_error"]["scalar_gamma_max"] < 1e-4


def test_vector_gamma_does_not_generally_commute(orthogonal_probe) -> None:
    # The whole point of the probe: vector gamma must fail.
    assert orthogonal_probe["allclose_with_vector_gamma"] is False
    assert (
        orthogonal_probe["gamma_commutation_error"]["vector_gamma_max"] > 1e-3
    ), "vector_gamma should produce a clearly observable commutation gap"


def test_orthogonal_probe_note_mentions_vector_gamma(orthogonal_probe) -> None:
    note = orthogonal_probe["note"].lower()
    assert "vector gamma" in note
    assert "scalar gamma" in note


# ---------------------------------------------------------------------------
# Trusted norm probe cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("use_pad", [True, False])
def test_trusted_norm_probe_cell_runs(norm_type: str, use_pad: bool) -> None:
    result = run_trusted_norm_probe(
        TrustedNormProbeConfig(
            norm_type=norm_type,
            batch_size=2,
            seq_len=8,
            hidden_size=64,
            use_pad=use_pad,
        )
    )
    assert result["metrics"]["allclose"] is True
    assert result["y_tilde_invariant_metrics"]["allclose"] is True
    assert result["reference_metrics"]["allclose"] is True
    assert result["pad_present"]["pad_in"] is use_pad
    assert result["pad_present"]["pad_out"] is use_pad


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
        "norm_experiments.json",
        "norm_experiments.csv",
        "norm_experiments.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "norm_experiments.md").read_text(encoding="utf-8")
    # spec-required strings
    assert "General right masks do not commute with LayerNorm" in md
    assert "Vector gamma breaks simple right-mask commutation" in md
    # canonical section headings
    for section in (
        "Experiment scope",
        "Trusted norm primitive correctness",
        "Restricted RMSNorm orthogonal-mask feasibility",
        "Gamma commutation analysis",
        "Limitations",
        "Next stage plan",
    ):
        assert section in md, f"missing section: {section}"

    payload = json.loads(
        (tmp_path / "norm_experiments.json").read_text(encoding="utf-8")
    )
    assert "trusted_norm" in payload
    assert "rmsnorm_orthogonal" in payload
    # 32 trusted cells (2 norm × 2 batch × 2 seq × 2 hidden × 2 pad)
    assert len(payload["trusted_norm"]) == 32
    assert "trusted_cells=32" in result.stdout


def test_script_markdown_lists_limitations(tmp_path) -> None:
    """The Limitations section must enumerate the eight required bullets."""
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    md = (tmp_path / "norm_experiments.md").read_text(encoding="utf-8")
    required_phrases = [
        "TrustedNormPrimitive still runs norm in the trusted side",
        "does not eliminate trusted compute",
        "General right masks do not commute with LayerNorm",
        "norm-preserving restrictions",
        "Vector gamma breaks simple right-mask commutation",
        "does not implement GELU",
        "does not implement real TEE",
        "does not claim formal security",
    ]
    for phrase in required_phrases:
        assert phrase in md, f"limitations section missing: {phrase!r}"
