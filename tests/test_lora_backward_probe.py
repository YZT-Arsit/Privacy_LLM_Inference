"""Stage 7.1 — tests for the LoRA masked-backward probe."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_backward_probe import (
    LoRABackwardProbeConfig,
    VALID_OPTIMIZERS,
    normalize_optimizer,
    run_lora_backward_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_backward_experiments.py"


def _cfg(**overrides) -> LoRABackwardProbeConfig:
    base = dict(
        seed=2026, batch_size=4, d_in=16, d_out=8, rank=3,
        num_steps=3, lr=1e-2, optimizer="sgd",
        use_pad=True, fresh_u_per_step=True, fresh_masks_per_step=True,
        dtype="float64",
    )
    base.update(overrides)
    return LoRABackwardProbeConfig(**base)


# ---------------------------------------------------------------------------
# 1. run_lora_backward_probe small config runs
# ---------------------------------------------------------------------------


def test_run_lora_backward_probe_runs() -> None:
    report = run_lora_backward_probe(_cfg())
    assert report["lora_backward_status"] == "masked_backward_prototype"
    assert report["masked_backward_correctness"]["num_steps"] == _cfg().num_steps


# ---------------------------------------------------------------------------
# 2. SGD masked backward update allclose
# ---------------------------------------------------------------------------


def test_sgd_masked_backward_update_allclose() -> None:
    report = run_lora_backward_probe(_cfg(optimizer="sgd"))
    mb = report["masked_backward_correctness"]
    assert mb["allclose"] is True
    assert mb["masked_backward_allclose"] is True
    assert mb["final_adapter_a_update_err"] < 1e-9
    assert mb["final_adapter_b_update_err"] < 1e-9


# ---------------------------------------------------------------------------
# 3. AdamW masked backward update allclose
# ---------------------------------------------------------------------------


def test_adamw_masked_backward_update_allclose() -> None:
    report = run_lora_backward_probe(_cfg(optimizer="adamw", lr=1e-3))
    mb = report["masked_backward_correctness"]
    assert mb["allclose"] is True
    assert mb["final_adapter_a_update_err"] < 1e-9
    assert mb["final_adapter_b_update_err"] < 1e-9


# ---------------------------------------------------------------------------
# 4. use_pad=True works
# ---------------------------------------------------------------------------


def test_use_pad_path_remains_allclose() -> None:
    report = run_lora_backward_probe(_cfg(use_pad=True))
    assert report["pad_compensation"]["use_pad"] is True
    assert report["masked_backward_correctness"]["allclose"] is True


def test_no_pad_path_remains_allclose() -> None:
    report = run_lora_backward_probe(_cfg(use_pad=False))
    assert report["pad_compensation"]["use_pad"] is False
    assert report["masked_backward_correctness"]["allclose"] is True


# ---------------------------------------------------------------------------
# 5. fresh_u_per_step=True works
# ---------------------------------------------------------------------------


def test_fresh_u_per_step_path_remains_allclose() -> None:
    report = run_lora_backward_probe(_cfg(fresh_u_per_step=True))
    assert report["config"]["fresh_u_per_step"] is True
    assert report["masked_backward_correctness"]["allclose"] is True


# ---------------------------------------------------------------------------
# 6. grad_A / grad_B errors below tolerance
# ---------------------------------------------------------------------------


def test_grad_errors_below_tolerance() -> None:
    report = run_lora_backward_probe(_cfg())
    mb = report["masked_backward_correctness"]
    assert mb["max_grad_a_err"] < 1e-9
    assert mb["max_grad_b_err"] < 1e-9
    assert mb["max_upstream_gradient_invariance_err"] < 1e-9


# ---------------------------------------------------------------------------
# 7. Markdown generated
# ---------------------------------------------------------------------------


def test_runner_script_generates_json_csv_md(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-steps", "3",
            "--d-in", "16", "--d-out", "8", "--rank", "3",
            "--batch-size", "4",
            "--recover-grad-x",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_backward_experiments.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Masked LoRA Backward Formula",
        "Upstream Gradient Masking",
        "Grad-A / Grad-B Recovery",
        "Training-Step Correctness",
        "Optimizer Handling",
        "Limitations",
        "Next Stage Plan",
    ):
        assert phrase in md, f"missing markdown section: {phrase!r}"
    json_text = (tmp_path / "lora_backward_experiments.json").read_text(encoding="utf-8")
    assert "tensor(" not in json_text
    with (tmp_path / "lora_backward_experiments.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "summary" in sections
    assert "per_step" in sections
    assert "autograd_cross_check" in sections


# ---------------------------------------------------------------------------
# 8. Markdown includes loss computation remains trusted
# ---------------------------------------------------------------------------


def test_markdown_includes_loss_and_optimizer_trusted(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-steps", "2",
            "--d-in", "12", "--d-out", "8", "--rank", "2",
            "--batch-size", "3",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_backward_experiments.md").read_text(encoding="utf-8")
    md_lower = md.lower()
    assert "loss computation remains trusted" in md_lower
    assert "optimizer remains trusted" in md_lower
    assert "trusted_loss" in md_lower
    assert "trusted_optimizer" in md_lower
    assert "masked_backward_prototype" in md_lower
    assert "rank padding is not implemented" in md_lower


# ---------------------------------------------------------------------------
# 9. Visibility / leakage contract
# ---------------------------------------------------------------------------


def test_gpu_does_not_see_raw_adapter_gradients_or_private_data() -> None:
    report = run_lora_backward_probe(_cfg())
    gh = report["gradient_handling"]
    vis = gh["gpu_visibility"]
    assert vis["raw_a"] is False
    assert vis["raw_b"] is False
    assert vis["raw_x"] is False
    assert vis["raw_grad_a"] is False
    assert vis["raw_grad_b"] is False
    assert vis["raw_upstream_gradient_g"] is False
    assert vis["optimizer_state"] is False
    assert vis["private_target_y"] is False
    # Masked variants ARE visible by design.
    assert vis["a_tilde"] is True
    assert vis["b_tilde"] is True
    assert vis["grad_a_tilde"] is True
    assert vis["grad_b_tilde"] is True
    assert vis["grad_y_tilde"] is True


def test_no_raw_tensors_in_json() -> None:
    report = run_lora_backward_probe(_cfg())
    text = json.dumps(report, default=str)
    assert "tensor(" not in text


def test_autograd_vs_analytic_cross_check_is_accurate() -> None:
    report = run_lora_backward_probe(_cfg())
    cross = report["autograd_vs_analytic_step0"]
    assert cross["grad_a"] < 1e-9
    assert cross["grad_b"] < 1e-9
    assert cross["grad_x"] < 1e-9


def test_normalize_optimizer_rejects_unknown() -> None:
    assert normalize_optimizer(None) == "sgd"
    assert "sgd" in VALID_OPTIMIZERS
    assert "adamw" in VALID_OPTIMIZERS
    with pytest.raises(ValueError):
        normalize_optimizer("adamax")
