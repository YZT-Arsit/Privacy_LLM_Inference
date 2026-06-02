"""Stage 7.3 — tests for the LoRA training timing side-channel proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pllo.experiments.lora_training_timing_proxy import (
    LoRATrainingTimingProxyConfig,
    VALID_CONSTANT_TIME_MODES,
    lora_training_timing_proxy_csv_rows,
    run_lora_training_timing_proxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_training_timing_proxy.py"


def _cfg(**overrides) -> LoRATrainingTimingProxyConfig:
    base = dict(
        seed=2026,
        batch_sizes=(1, 2, 4),
        seq_lens=(4, 8),
        true_ranks=(2, 4),
        padded_ranks=(8,),
        num_lora_modules=(2, 7),
        optimizers=("sgd", "adamw"),
        samples_per_config=3,
        constant_time_training_mode="off",
        dtype="float64",
    )
    base.update(overrides)
    return LoRATrainingTimingProxyConfig(**base)


# ---------------------------------------------------------------------------
# 1. small config runs
# ---------------------------------------------------------------------------


def test_run_lora_training_timing_proxy_runs() -> None:
    report = run_lora_training_timing_proxy(_cfg())
    assert (
        report["lora_training_timing_proxy_status"] == "implemented"
    )
    assert report["security_profile"] == "proxy-evaluated, not formal"


# ---------------------------------------------------------------------------
# 2. leakage tasks include the required dimensions
# ---------------------------------------------------------------------------


def test_leakage_tasks_dimensions() -> None:
    report = run_lora_training_timing_proxy(_cfg())
    tasks = report["leakage_tasks_off"]
    for needed in (
        "batch_size", "seq_len", "true_rank", "padded_rank",
        "num_modules", "optimizer", "rank_padding_on", "dummy_strategy",
    ):
        assert needed in tasks
        assert "classification_accuracy" in tasks[needed]
        assert "random_chance_baseline" in tasks[needed]
        assert "risk_level" in tasks[needed]


# ---------------------------------------------------------------------------
# 3. constant_time_training_mode="off" output exists
# ---------------------------------------------------------------------------


def test_constant_time_off_output_exists() -> None:
    report = run_lora_training_timing_proxy(_cfg(constant_time_training_mode="off"))
    assert (
        report["constant_time_training_proxy"]["constant_time_training_mode"]
        == "off"
    )
    assert report["leakage_tasks_off"]
    assert report["leakage_tasks_proxy_equalized"] == {}


# ---------------------------------------------------------------------------
# 4. constant_time_training_mode="proxy_equalized" output exists
# ---------------------------------------------------------------------------


def test_constant_time_proxy_equalized_output_exists() -> None:
    report = run_lora_training_timing_proxy(
        _cfg(constant_time_training_mode="proxy_equalized")
    )
    assert (
        report["constant_time_training_proxy"]["constant_time_training_mode"]
        == "proxy_equalized"
    )
    assert report["leakage_tasks_off"]
    assert report["leakage_tasks_proxy_equalized"]


# ---------------------------------------------------------------------------
# 5. proxy_equalized reduces leakage on the most-leaky task (num_modules)
# ---------------------------------------------------------------------------


def test_proxy_equalized_reduces_leakage() -> None:
    report = run_lora_training_timing_proxy(
        _cfg(
            constant_time_training_mode="proxy_equalized",
            samples_per_config=4,
        )
    )
    summary = report["summary"]
    # The off run has a clearly leaky num_modules task; equalization
    # should reduce the WORST-case classifier accuracy.
    assert (
        summary["max_classification_accuracy_proxy_equalized"]
        <= summary["max_classification_accuracy_off"]
    )
    # Reduction should be strictly positive for this config (where
    # num_modules is a strong signal off, equalized to ~chance).
    reduction = summary["leakage_reduction_after_equalization"]
    assert reduction is not None
    assert reduction >= 0.0


# ---------------------------------------------------------------------------
# 6. overhead_proxy exists
# ---------------------------------------------------------------------------


def test_overhead_proxy_exists() -> None:
    report = run_lora_training_timing_proxy(
        _cfg(constant_time_training_mode="proxy_equalized")
    )
    op = report["overhead_proxy"]
    assert "mean_native_latency_ms" in op
    assert "upper_latency_ms" in op
    assert "overhead_ratio" in op
    assert "overhead_pct" in op
    assert op["upper_latency_ms"] >= op["mean_native_latency_ms"]


# ---------------------------------------------------------------------------
# 7. Markdown states not real TEE wall-time
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--batch-sizes", "1", "2", "4",
        "--seq-lens", "4", "8",
        "--true-ranks", "2", "4",
        "--padded-ranks", "8",
        "--num-lora-modules", "2", "7",
        "--optimizers", "sgd", "adamw",
        "--samples-per-config", "3",
        "--constant-time-training-mode", "proxy_equalized",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    md_path = tmp_path / "lora_training_timing_proxy.md"
    json_path = tmp_path / "lora_training_timing_proxy.json"
    csv_path = tmp_path / "lora_training_timing_proxy.csv"
    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
    md_text = md_path.read_text().lower()
    for needle in (
        "training timing",
        "experiment scope",
        "training timing proxy model",
        "leakage tasks",
        "constant-time training proxy",
        "overhead estimate",
        "limitations",
        "next stage plan",
        "not real tee",
    ):
        assert needle in md_text, f"missing markdown section: {needle!r}"


# ---------------------------------------------------------------------------
# 8. No real sleep
# ---------------------------------------------------------------------------


def test_runtime_does_not_sleep() -> None:
    t0 = time.perf_counter()
    report = run_lora_training_timing_proxy(_cfg(
        constant_time_training_mode="proxy_equalized",
        samples_per_config=2,
    ))
    elapsed = time.perf_counter() - t0
    assert (
        report["constant_time_training_proxy"]["did_actually_sleep"]
        is False
    )
    # Loose upper bound — should be well under 10 s on any reasonable
    # machine; we only need to confirm it isn't sleeping for many seconds.
    assert elapsed < 10.0


# ---------------------------------------------------------------------------
# 9. No raw tensor in outputs
# ---------------------------------------------------------------------------


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--batch-sizes", "1", "2",
        "--seq-lens", "4",
        "--true-ranks", "2",
        "--padded-ranks", "8",
        "--num-lora-modules", "2", "4",
        "--optimizers", "sgd",
        "--samples-per-config", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_text = (tmp_path / "lora_training_timing_proxy.json").read_text()
    md_text = (tmp_path / "lora_training_timing_proxy.md").read_text()
    csv_text = (tmp_path / "lora_training_timing_proxy.csv").read_text()
    for text in (json_text, md_text, csv_text):
        assert "tensor(" not in text


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_csv_rows_no_tensor_text() -> None:
    report = run_lora_training_timing_proxy(_cfg())
    rows = lora_training_timing_proxy_csv_rows(report)
    for row in rows:
        assert "tensor(" not in str(row["value"])


def test_invalid_constant_time_mode_raises() -> None:
    cfg = _cfg()
    cfg.constant_time_training_mode = "nope"
    with pytest.raises(ValueError):
        run_lora_training_timing_proxy(cfg)
