"""Stage 7.5 — tests for measured runtime evaluation (local emulation)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pllo.experiments.measured_runtime_evaluation import (
    MeasuredRuntimeEvaluationConfig,
    run_measured_runtime_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_measured_runtime_evaluation.py"


def _cfg(output_dir: Path, **overrides) -> MeasuredRuntimeEvaluationConfig:
    base = dict(
        output_dir=str(output_dir),
        num_warmup=1,
        num_repeats=2,
        linear_d_in=16, linear_d_out=8, linear_batch=4,
        lora_rank=2, lora_padded_rank=4,
        multilayer_num_layers=2,
        multilayer_hidden_size=8,
        multilayer_intermediate_size=12,
        multilayer_seq_len=4,
        multilayer_batch=2,
        multilayer_num_steps=1,
        dtype="float64",
    )
    base.update(overrides)
    return MeasuredRuntimeEvaluationConfig(**base)


# ---------------------------------------------------------------------------
# 1. small config runs
# ---------------------------------------------------------------------------


def test_runtime_evaluator_runs(tmp_path: Path) -> None:
    report = run_measured_runtime_evaluation(_cfg(tmp_path))
    assert (
        report["measured_runtime_evaluation_status"] == "implemented"
    )
    assert report["is_real_tee_wall_time"] is False


# ---------------------------------------------------------------------------
# 2. json/csv/md/tex generated
# ---------------------------------------------------------------------------


def test_artifacts_generated(tmp_path: Path) -> None:
    run_measured_runtime_evaluation(_cfg(tmp_path))
    assert (tmp_path / "json" / "measured_runtime.json").exists()
    assert (tmp_path / "csv" / "measured_runtime.csv").exists()
    assert (tmp_path / "markdown" / "measured_runtime.md").exists()
    assert (tmp_path / "latex" / "measured_runtime.tex").exists()


# ---------------------------------------------------------------------------
# 3. mean / median / std fields present
# ---------------------------------------------------------------------------


def test_required_fields_present(tmp_path: Path) -> None:
    report = run_measured_runtime_evaluation(_cfg(tmp_path))
    for r in report["rows"]:
        for k in (
            "component", "variant", "num_warmup", "num_repeats",
            "device", "dtype", "wall_time_source",
            "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms",
            "skipped_with_reason",
        ):
            assert k in r


# ---------------------------------------------------------------------------
# 4. Markdown explicitly says NOT real TEE wall-time
# ---------------------------------------------------------------------------


def test_md_says_not_real_tee(tmp_path: Path) -> None:
    run_measured_runtime_evaluation(_cfg(tmp_path))
    text = (tmp_path / "markdown" / "measured_runtime.md").read_text()
    low = text.lower()
    assert "not real tee" in low or "local emulation" in low
    assert "no real sleep" in low or "real sleep" in low


# ---------------------------------------------------------------------------
# 5. skipped benchmarks have a reason
# ---------------------------------------------------------------------------


def test_skipped_benchmarks_carry_reason(tmp_path: Path) -> None:
    # modern_decoder_model_wrapper is opt-in; should be skipped by default.
    report = run_measured_runtime_evaluation(_cfg(tmp_path))
    has_skip = any(
        r.get("skipped_with_reason") for r in report["rows"]
    )
    assert has_skip
    for r in report["rows"]:
        if r.get("mean_ms") is None:
            assert r.get("skipped_with_reason")


# ---------------------------------------------------------------------------
# 6. Does not call sleep — quick smoke that runtime is finite & fast
# ---------------------------------------------------------------------------


def test_runtime_does_not_sleep(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    run_measured_runtime_evaluation(_cfg(tmp_path))
    elapsed = time.perf_counter() - t0
    # Loose upper bound; the synthetic tile should complete well within 30s.
    assert elapsed < 30.0


# ---------------------------------------------------------------------------
# 7. Does not depend on network (smoke: HF wrapper benchmark is opt-in)
# ---------------------------------------------------------------------------


def test_no_network_dependency(tmp_path: Path) -> None:
    report = run_measured_runtime_evaluation(_cfg(tmp_path))
    # The runner finishes without making network requests because the
    # modern-decoder benchmark is opt-in and we did not enable it.
    assert any(
        r["component"] == "modern_decoder_model_wrapper"
        for r in report["rows"]
    )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_runner_script_executes(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--num-warmup", "1", "--num-repeats", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "json" / "measured_runtime.json").exists()


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    run_measured_runtime_evaluation(_cfg(tmp_path))
    for sub in ("json", "csv", "markdown", "latex"):
        for path in (tmp_path / sub).glob("*"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "tensor(" not in text, path
