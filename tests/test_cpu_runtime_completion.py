"""Stage 7.5b tests for cpu_runtime_completion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pllo.experiments.cpu_runtime_completion import (
    CPURuntimeCompletionConfig,
    run_cpu_runtime_completion,
)


def _cfg(tmp: Path, **overrides) -> CPURuntimeCompletionConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        num_warmup=1, num_repeats=2,
        batch_sizes=(1,),
        seq_lens=(4,),
        hidden_sizes=(16,),
        true_rank=2, padded_rank=4,
    )
    base.update(overrides)
    return CPURuntimeCompletionConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_cpu_runtime_completion(_cfg(tmp_path))
    assert report["cpu_runtime_completion_status"] == "implemented"
    assert report["wall_time_source"] == "measured_local_emulation"
    assert report["is_real_tee_wall_time"] is False
    assert report["is_gpu_throughput"] is False


def test_required_components_present(tmp_path: Path) -> None:
    report = run_cpu_runtime_completion(_cfg(tmp_path))
    names = {r["component"] for r in report["rows"]}
    for required in (
        "lora_forward",
        "lora_backward",
        "modern_decoder_full_forward",
        "linear_masked_forward",
        "nonlinear_island_forward",
    ):
        assert required in names, f"missing {required}"


def test_stat_fields_present(tmp_path: Path) -> None:
    report = run_cpu_runtime_completion(_cfg(tmp_path))
    for row in report["rows"]:
        for key in ("mean_ms", "median_ms", "std_ms", "min_ms", "max_ms"):
            assert key in row, f"missing {key}"


def test_md_disclaimer(tmp_path: Path) -> None:
    run_cpu_runtime_completion(_cfg(tmp_path))
    md = (tmp_path / "cpu_runtime_completion.md").read_text(encoding="utf-8").lower()
    assert "not real tee" in md or "not a real tee" in md
    assert "not gpu throughput" in md or "not a gpu" in md or "gpu throughput" in md


def test_no_sleep_in_implementation() -> None:
    # The implementation file must not call ``time.sleep`` (the docstring is
    # allowed to mention the word -- look for the call form only).
    src = Path(__file__).resolve().parents[1] / "src" / "pllo" / "experiments" / "cpu_runtime_completion.py"
    text = src.read_text(encoding="utf-8")
    assert "time.sleep(" not in text


def test_outputs_emitted(tmp_path: Path) -> None:
    run_cpu_runtime_completion(_cfg(tmp_path))
    for name in (
        "cpu_runtime_completion.json",
        "cpu_runtime_completion.csv",
        "cpu_runtime_completion.md",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_no_raw_tensors(tmp_path: Path) -> None:
    run_cpu_runtime_completion(_cfg(tmp_path))
    blob = (tmp_path / "cpu_runtime_completion.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
