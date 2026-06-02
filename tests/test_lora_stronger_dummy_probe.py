"""Stage 7.4 — tests for the stronger-dummy training correctness probe."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_stronger_dummy_probe import (
    StrongerDummyProbeConfig,
    VALID_OPTIMIZERS,
    normalize_optimizer,
    run_lora_stronger_dummy_probe,
    stronger_dummy_probe_csv_rows,
)
from pllo.ops.lora_dummy_strategies import VALID_STRONG_DUMMY_STRATEGIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_stronger_dummy_experiments.py"


def _cfg(**overrides) -> StrongerDummyProbeConfig:
    base = dict(
        seed=2026, batch_size=2, d_in=16, d_out=8,
        true_rank=2, padded_rank=8, num_steps=2,
        lr=1e-2, optimizer="sgd",
        use_pad=True, fresh_u_per_step=True,
        dummy_strategies=tuple(VALID_STRONG_DUMMY_STRATEGIES),
        noise_scale=1e-3,
        dtype="float64",
    )
    base.update(overrides)
    return StrongerDummyProbeConfig(**base)


# ---------------------------------------------------------------------------
# 1. small config runs
# ---------------------------------------------------------------------------


def test_run_lora_stronger_dummy_probe_runs() -> None:
    report = run_lora_stronger_dummy_probe(_cfg())
    assert report["lora_stronger_dummy_status"] == "implemented"
    assert (
        report["lora_spectral_rank_hardening_status"] == "proxy-evaluated"
    )
    assert len(report["per_strategy"]) == len(VALID_STRONG_DUMMY_STRATEGIES)


# ---------------------------------------------------------------------------
# 2. forward allclose for supported strategies
# ---------------------------------------------------------------------------


def test_forward_allclose_per_strategy() -> None:
    report = run_lora_stronger_dummy_probe(_cfg())
    for entry in report["per_strategy"]:
        assert entry["allclose"] is True, entry["dummy_strategy"]
        assert entry["max_forward_err"] < 1e-9, entry["dummy_strategy"]


# ---------------------------------------------------------------------------
# 3. backward grad real slices allclose
# ---------------------------------------------------------------------------


def test_grad_real_slices_allclose_per_strategy() -> None:
    report = run_lora_stronger_dummy_probe(_cfg())
    for entry in report["per_strategy"]:
        assert entry["max_grad_a_real_err"] < 1e-7, entry["dummy_strategy"]
        assert entry["max_grad_b_real_err"] < 1e-7, entry["dummy_strategy"]


# ---------------------------------------------------------------------------
# 4. SGD update allclose
# ---------------------------------------------------------------------------


def test_sgd_update_allclose() -> None:
    report = run_lora_stronger_dummy_probe(_cfg(optimizer="sgd"))
    for entry in report["per_strategy"]:
        assert entry["max_update_a_err"] < 1e-7, entry["dummy_strategy"]
        assert entry["max_update_b_err"] < 1e-7, entry["dummy_strategy"]


# ---------------------------------------------------------------------------
# 5. AdamW update allclose
# ---------------------------------------------------------------------------


def test_adamw_update_allclose() -> None:
    report = run_lora_stronger_dummy_probe(
        _cfg(optimizer="adamw", lr=1e-3)
    )
    for entry in report["per_strategy"]:
        assert entry["max_update_a_err"] < 1e-7, entry["dummy_strategy"]
        assert entry["max_update_b_err"] < 1e-7, entry["dummy_strategy"]


# ---------------------------------------------------------------------------
# 6. dummy_update_applied=False
# ---------------------------------------------------------------------------


def test_dummy_update_not_applied() -> None:
    report = run_lora_stronger_dummy_probe(_cfg())
    for entry in report["per_strategy"]:
        assert entry["optimizer_handling"]["dummy_update_applied"] is False


# ---------------------------------------------------------------------------
# 7. optimizer_state_contains_dummy=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt", ["sgd", "adamw"])
def test_optimizer_state_no_dummy(opt: str) -> None:
    report = run_lora_stronger_dummy_probe(_cfg(optimizer=opt, lr=1e-3))
    for entry in report["per_strategy"]:
        oh = entry["optimizer_handling"]
        assert oh["optimizer_state_contains_dummy"] is False
        assert oh["trainable_adapter_shape_a"][1] == 2  # true_rank
        assert oh["trainable_adapter_shape_b"][0] == 2


# ---------------------------------------------------------------------------
# 8. correction_norm recorded when needed
# ---------------------------------------------------------------------------


def test_correction_norm_recorded_for_noise_injected() -> None:
    report = run_lora_stronger_dummy_probe(_cfg(noise_scale=1e-2))
    by_strategy = {
        e["dummy_strategy"]: e for e in report["per_strategy"]
    }
    # Cancellation strategies must keep correction_norm == 0.
    for strategy in (
        "paired_cancellation_dummy",
        "gaussian_matched_dummy",
        "spectrum_matched_dummy",
        "orthogonalized_cancellation_dummy",
        "mixed_dummy_ensemble",
        "zero_dummy",
    ):
        assert by_strategy[strategy]["max_correction_norm"] < 1e-12, strategy
    # Noise-injected strategy must have non-zero correction_norm.
    assert by_strategy["noise_injected_cancellation_dummy"][
        "max_correction_norm"
    ] > 0.0


# ---------------------------------------------------------------------------
# 9. Markdown generated (runner script)
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--d-in", "16", "--d-out", "8",
        "--true-rank", "2", "--padded-rank", "8",
        "--batch-size", "2", "--num-steps", "2",
        "--noise-scale", "1e-3",
        "--dtype", "float64",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_path = tmp_path / "lora_stronger_dummy_experiments.json"
    csv_path = tmp_path / "lora_stronger_dummy_experiments.csv"
    md_path = tmp_path / "lora_stronger_dummy_experiments.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    md_text = md_path.read_text().lower()
    for needle in (
        "stronger dummy",
        "experiment scope",
        "stronger dummy strategy design",
        "forward correctness",
        "backward correctness",
        "optimizer handling",
        "dummy contribution and correction",
        "comparison with stage 7.2 / 7.3",
        "limitations",
        "next stage plan",
    ):
        assert needle in md_text, f"missing markdown section: {needle!r}"


# ---------------------------------------------------------------------------
# 10. Markdown includes "spectral hardening does not imply cryptographic hiding"
# ---------------------------------------------------------------------------


def test_markdown_states_no_cryptographic_hiding(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--d-in", "16", "--d-out", "8",
        "--true-rank", "2", "--padded-rank", "8",
        "--batch-size", "2", "--num-steps", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    md_text = (
        tmp_path / "lora_stronger_dummy_experiments.md"
    ).read_text().lower()
    assert "spectral hardening does not imply cryptographic hiding" in md_text
    assert "no real tee training" in md_text
    assert "padded rank remains visible" in md_text


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_normalize_optimizer() -> None:
    assert normalize_optimizer(None) == "sgd"
    assert normalize_optimizer("sgd") == "sgd"
    assert normalize_optimizer("adamw") == "adamw"
    with pytest.raises(ValueError):
        normalize_optimizer("rmsprop")


def test_csv_rows_no_tensor_text() -> None:
    report = run_lora_stronger_dummy_probe(_cfg())
    rows = stronger_dummy_probe_csv_rows(report)
    for row in rows:
        assert "tensor(" not in str(row["value"])


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--d-in", "16", "--d-out", "8",
        "--true-rank", "2", "--padded-rank", "8",
        "--batch-size", "2", "--num-steps", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_text = (
        tmp_path / "lora_stronger_dummy_experiments.json"
    ).read_text()
    md_text = (
        tmp_path / "lora_stronger_dummy_experiments.md"
    ).read_text()
    csv_text = (
        tmp_path / "lora_stronger_dummy_experiments.csv"
    ).read_text()
    for text in (json_text, md_text, csv_text):
        assert "tensor(" not in text
