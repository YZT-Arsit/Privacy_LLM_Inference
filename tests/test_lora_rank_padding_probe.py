"""Stage 7.2 — tests for the LoRA rank-padded training probe."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_rank_padding_probe import (
    LoRARankPaddingProbeConfig,
    VALID_OPTIMIZERS,
    normalize_optimizer,
    run_lora_rank_padding_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_rank_padding_experiments.py"


def _cfg(**overrides) -> LoRARankPaddingProbeConfig:
    base = dict(
        seed=2026, batch_size=4, d_in=16, d_out=8,
        true_rank=2, padded_rank=8, num_steps=3,
        lr=1e-2, optimizer="sgd",
        use_pad=True, fresh_u_per_step=True,
        dummy_strategy="paired_cancellation_dummy",
        dtype="float64",
    )
    base.update(overrides)
    return LoRARankPaddingProbeConfig(**base)


# ---------------------------------------------------------------------------
# 1. run_lora_rank_padding_probe small config runs
# ---------------------------------------------------------------------------


def test_run_lora_rank_padding_probe_runs() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    assert report["lora_rank_padding_status"] == "implemented"
    assert report["lora_hidden_rank_status"] == "padded-rank-prototype"


# ---------------------------------------------------------------------------
# 2. forward allclose
# ---------------------------------------------------------------------------


def test_forward_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    rp = report["rank_padding_correctness"]
    assert rp["allclose"] is True
    assert rp["max_loss_diff"] < 1e-9
    assert rp["max_dummy_contribution_norm"] < 1e-9


# ---------------------------------------------------------------------------
# 3. backward grad real slices allclose
# ---------------------------------------------------------------------------


def test_grad_real_slices_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    rp = report["rank_padding_correctness"]
    assert rp["max_grad_a_real_err"] < 1e-9
    assert rp["max_grad_b_real_err"] < 1e-9


# ---------------------------------------------------------------------------
# 4. SGD update allclose
# ---------------------------------------------------------------------------


def test_sgd_update_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg(optimizer="sgd"))
    rp = report["rank_padding_correctness"]
    assert rp["allclose"] is True
    assert rp["final_adapter_a_update_err"] < 1e-9
    assert rp["final_adapter_b_update_err"] < 1e-9


# ---------------------------------------------------------------------------
# 5. AdamW update allclose
# ---------------------------------------------------------------------------


def test_adamw_update_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg(optimizer="adamw", lr=1e-3))
    rp = report["rank_padding_correctness"]
    assert rp["allclose"] is True
    assert rp["final_adapter_a_update_err"] < 1e-9
    assert rp["final_adapter_b_update_err"] < 1e-9


# ---------------------------------------------------------------------------
# 6. dummy_update_applied=False
# ---------------------------------------------------------------------------


def test_dummy_update_not_applied() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    assert report["optimizer_handling"]["dummy_update_applied"] is False


# ---------------------------------------------------------------------------
# 7. optimizer_state_contains_dummy=False
# ---------------------------------------------------------------------------


def test_optimizer_state_does_not_contain_dummy() -> None:
    for optim in ("sgd", "adamw"):
        report = run_lora_rank_padding_probe(_cfg(optimizer=optim, lr=1e-3))
        oh = report["optimizer_handling"]
        assert oh["optimizer_state_contains_dummy"] is False
        # Trainable adapters are sized to true_rank, never padded_rank.
        assert oh["trainable_adapter_shape_a"][-1] == _cfg().true_rank
        assert oh["trainable_adapter_shape_b"][0] == _cfg().true_rank


# ---------------------------------------------------------------------------
# 8. visible_rank_from_shape == padded_rank
# ---------------------------------------------------------------------------


def test_visible_rank_from_shape_equals_padded_rank() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    sh = report["shape_level_rank_hiding"]
    assert sh["visible_rank_from_a_shape"] == _cfg().padded_rank
    assert sh["visible_rank_from_b_shape"] == _cfg().padded_rank


# ---------------------------------------------------------------------------
# 9. true_rank_hidden_from_shape=True
# ---------------------------------------------------------------------------


def test_true_rank_hidden_from_shape() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    assert report["shape_level_rank_hiding"]["true_rank_hidden_from_shape"] is True
    assert report["lora_true_rank_hidden_from_shape"] is True
    assert report["lora_padded_rank_visible"] is True


# ---------------------------------------------------------------------------
# 10. Markdown generated and includes "padded rank remains visible"
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-steps", "3",
            "--d-in", "16", "--d-out", "8",
            "--true-rank", "2", "--padded-rank", "8",
            "--batch-size", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_rank_padding_experiments.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Rank Padding Formula",
        "Dummy Rank Strategy",
        "Forward Correctness",
        "Backward Correctness",
        "Optimizer Handling",
        "Shape-Level Rank Hiding",
        "Limitations",
        "Next Stage Plan",
    ):
        assert phrase in md, f"missing markdown section: {phrase!r}"
    md_lower = md.lower()
    assert "padded rank r_pad remains visible" in md_lower
    assert "no formal" in md_lower  # 'no formal / cryptographic / semantic security'
    assert "no real tee training" in md_lower
    assert "rank padding is not implemented" not in md_lower  # this is for the OLDER stage
    json_text = (tmp_path / "lora_rank_padding_experiments.json").read_text(
        encoding="utf-8"
    )
    assert "tensor(" not in json_text
    with (tmp_path / "lora_rank_padding_experiments.csv").open(
        encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "summary" in sections
    assert "per_step" in sections
    assert "shape_level_rank_hiding" in sections


# ---------------------------------------------------------------------------
# Visibility / leakage contract
# ---------------------------------------------------------------------------


def test_normalize_optimizer_rejects_unknown() -> None:
    assert normalize_optimizer(None) == "sgd"
    assert "sgd" in VALID_OPTIMIZERS
    assert "adamw" in VALID_OPTIMIZERS
    with pytest.raises(ValueError):
        normalize_optimizer("adamax")


def test_no_raw_tensors_in_json() -> None:
    report = run_lora_rank_padding_probe(_cfg())
    text = json.dumps(report, default=str)
    assert "tensor(" not in text


def test_use_pad_path_remains_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg(use_pad=True))
    assert report["rank_padding_correctness"]["allclose"] is True


def test_no_pad_path_remains_allclose() -> None:
    report = run_lora_rank_padding_probe(_cfg(use_pad=False))
    assert report["rank_padding_correctness"]["allclose"] is True


def test_zero_dummy_strategy_correctness_only() -> None:
    """zero_dummy must remain mathematically correct (the security
    side-effects are evaluated separately by the security proxy).
    """
    report = run_lora_rank_padding_probe(_cfg(dummy_strategy="zero_dummy"))
    rp = report["rank_padding_correctness"]
    assert rp["allclose"] is True
    assert rp["max_dummy_contribution_norm"] < 1e-12
