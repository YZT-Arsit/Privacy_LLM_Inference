"""Stage 7.5b tests for paper_baseline_comparison."""

from __future__ import annotations

from pathlib import Path

from pllo.experiments.paper_baseline_comparison import (
    PaperBaselineComparisonConfig,
    run_paper_baseline_comparison,
)


def _cfg(tmp: Path, **overrides) -> PaperBaselineComparisonConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        batch_size=2, seq_len=4, hidden_size=8,
        true_rank=2, padded_rank=4, num_repeats=2,
    )
    base.update(overrides)
    return PaperBaselineComparisonConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_paper_baseline_comparison(_cfg(tmp_path))
    assert report["paper_baseline_comparison_status"] == "implemented"
    assert report["wall_time_source"] == "measured_local_emulation"
    assert report["is_real_tee_wall_time"] is False
    assert report["is_gpu_throughput"] is False


def test_required_variants_present(tmp_path: Path) -> None:
    report = run_paper_baseline_comparison(_cfg(tmp_path))
    names = {r["variant"] for r in report["rows"]}
    for required in (
        "plain_cpu",
        "trusted_nonlinear_partition",
        "fresh_perm_only",
        "full_mitigation_bundle",
    ):
        assert required in names, f"missing {required}"


def test_each_variant_has_correctness_and_runtime(tmp_path: Path) -> None:
    report = run_paper_baseline_comparison(_cfg(tmp_path))
    for row in report["rows"]:
        assert "correctness_error" in row
        assert "local_runtime_ms" in row
        assert isinstance(row["local_runtime_ms"], float)


def test_risk_level_is_proxy_derived(tmp_path: Path) -> None:
    report = run_paper_baseline_comparison(_cfg(tmp_path))
    assert "proxy-derived" in report["risk_level_derivation"]
    for row in report["rows"]:
        assert "proxy_risk_level" in row
        assert row["proxy_risk_level"] in {"low", "medium", "needs_more_evaluation", "high"}


def test_outputs_emitted(tmp_path: Path) -> None:
    run_paper_baseline_comparison(_cfg(tmp_path))
    for name in (
        "paper_baseline_comparison.json",
        "paper_baseline_comparison.csv",
        "paper_baseline_comparison.md",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_md_says_proxy_derived(tmp_path: Path) -> None:
    run_paper_baseline_comparison(_cfg(tmp_path))
    md = (tmp_path / "paper_baseline_comparison.md").read_text(encoding="utf-8")
    assert "proxy-derived" in md.lower() or "proxy derived" in md.lower()
    assert "not real tee" in md.lower() or "not a real tee" in md.lower()


def test_no_raw_tensors(tmp_path: Path) -> None:
    run_paper_baseline_comparison(_cfg(tmp_path))
    blob = (tmp_path / "paper_baseline_comparison.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
