"""Stage 7.5b tests for paper_stability_study."""

from __future__ import annotations

from pathlib import Path

from pllo.experiments.paper_stability_study import (
    PaperStabilityStudyConfig,
    run_paper_stability_study,
)


def _cfg(tmp: Path, **overrides) -> PaperStabilityStudyConfig:
    base = dict(
        output_dir=str(tmp),
        seeds=(2021, 2022),
        batch_sizes=(1, 2),
        seq_lens=(4,),
        hidden_sizes=(16,),
        true_ranks=(2,),
        padded_ranks=(8,),
    )
    base.update(overrides)
    return PaperStabilityStudyConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_paper_stability_study(_cfg(tmp_path))
    assert report["paper_stability_study_status"] == "implemented"


def test_multi_seed_rows(tmp_path: Path) -> None:
    report = run_paper_stability_study(_cfg(tmp_path))
    seeds = {r["seed"] for r in report["trial_rows"]}
    assert seeds == {2021, 2022}


def test_summary_fields_present(tmp_path: Path) -> None:
    report = run_paper_stability_study(_cfg(tmp_path))
    for s in report["summary_rows"]:
        for key in (
            "allclose_rate",
            "max_error_p95",
            "max_error_max",
            "runtime_mean",
            "runtime_std",
            "failure_count",
        ):
            assert key in s, f"missing {key}"


def test_outputs_emitted(tmp_path: Path) -> None:
    run_paper_stability_study(_cfg(tmp_path))
    for name in (
        "paper_stability_study.json",
        "paper_stability_study.csv",
        "paper_stability_study.md",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_no_raw_tensors(tmp_path: Path) -> None:
    run_paper_stability_study(_cfg(tmp_path))
    blob = (tmp_path / "paper_stability_study.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
