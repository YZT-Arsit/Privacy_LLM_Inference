"""Stage 7.5b tests for paper_ablation_study."""

from __future__ import annotations

from pathlib import Path

from pllo.experiments.paper_ablation_study import (
    PaperAblationStudyConfig,
    run_paper_ablation_study,
)


def _cfg(tmp: Path, **overrides) -> PaperAblationStudyConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        batch_size=2, seq_len=4, hidden_size=8,
        num_trials=2, true_rank=2, padded_rank=4,
    )
    base.update(overrides)
    return PaperAblationStudyConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_paper_ablation_study(_cfg(tmp_path))
    assert report["paper_ablation_study_status"] == "implemented"
    assert report["is_real_tee_wall_time"] is False
    assert report["is_gpu_throughput"] is False


def test_required_components_present(tmp_path: Path) -> None:
    report = run_paper_ablation_study(_cfg(tmp_path))
    components = {r["component"] for r in report["rows"]}
    for required in (
        "boundary_pad",
        "permutation_freshness",
        "dense_sandwich",
        "inter_block_boundary",
        "rank_padding",
    ):
        assert required in components, f"missing {required}"


def test_each_row_has_correctness_and_risk(tmp_path: Path) -> None:
    report = run_paper_ablation_study(_cfg(tmp_path))
    for row in report["rows"]:
        assert "correctness_preserved" in row
        assert "risk_level" in row
        assert row["risk_level"] in {"low", "medium", "needs_more_evaluation", "high"}


def test_md_lists_critical_categories(tmp_path: Path) -> None:
    run_paper_ablation_study(_cfg(tmp_path))
    md = (tmp_path / "paper_ablation_study.md").read_text(encoding="utf-8")
    assert "correctness-critical" in md.lower() or "correctness critical" in md.lower()
    assert "experimental" in md.lower()
    assert "metadata" in md.lower() or "timing" in md.lower()


def test_outputs_emitted(tmp_path: Path) -> None:
    run_paper_ablation_study(_cfg(tmp_path))
    for name in (
        "paper_ablation_study.json",
        "paper_ablation_study.csv",
        "paper_ablation_study.md",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_no_raw_tensors(tmp_path: Path) -> None:
    run_paper_ablation_study(_cfg(tmp_path))
    blob = (tmp_path / "paper_ablation_study.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
