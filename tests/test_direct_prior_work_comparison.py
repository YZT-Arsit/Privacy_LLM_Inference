"""Stage 7.5c tests for the direct prior-work comparison runner."""

from __future__ import annotations

import json
from pathlib import Path

from pllo.experiments.direct_prior_work_comparison import (
    DirectPriorWorkComparisonConfig,
    run_direct_prior_work_comparison,
)


def _cfg(tmp: Path, **overrides) -> DirectPriorWorkComparisonConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        batch_size=2, seq_len=4, hidden_size=8,
        true_rank=2, padded_rank=4, num_repeats=2,
    )
    base.update(overrides)
    return DirectPriorWorkComparisonConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    assert report["direct_prior_work_comparison_status"] == "implemented"
    assert report["is_real_tee_wall_time"] is False
    assert report["is_gpu_throughput"] is False


def test_comparison_includes_required_protocols(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    names = {r["protocol_name"] for r in report["rows"]}
    for required in (
        "ours_right_mask_full_bundle",
        "slalom_delegated_linear",
        "amulet_static_left_right_mask",
        "darknight_blinding_primitive",
        "arrow_direct_primitive_or_unavailable",
        "cryptonets_polynomial_skeleton",
        "gazelle_cost_model_only",
        "delphi_cost_model_only",
        "secureml_cost_model_only",
        "minionn_cost_model_only",
    ):
        assert required in names, f"missing protocol {required}"


def test_every_row_has_implementation_and_reproduction_flags(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    for row in report["rows"]:
        for key in (
            "exact_primitive_implemented",
            "full_system_reproduced",
            "arithmetic_skeleton_only",
            "cost_model_only",
            "runtime_directly_comparable",
        ):
            assert key in row, f"row missing field {key}"


def test_unsupported_rows_have_mathematical_reason(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    for row in report["rows"]:
        if (
            row["protocol_name"].startswith("ours_")
            or row["protocol_name"] == "slalom_delegated_linear"
            or row["protocol_name"] == "darknight_blinding_primitive"
            or row["protocol_name"] == "amulet_static_left_right_mask"
            or row["protocol_name"] == "cryptonets_polynomial_skeleton"
        ):
            # These have a primitive implemented; mathematical_reason may be brief.
            continue
        assert row["mathematical_reason"], row["protocol_name"]


def test_no_row_falsely_claims_full_reproduction(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    for row in report["rows"]:
        if row["protocol_name"].startswith("ours_"):
            continue
        assert row["full_system_reproduced"] is False, row["protocol_name"]


def test_outputs_written(tmp_path: Path) -> None:
    run_direct_prior_work_comparison(_cfg(tmp_path))
    for name in (
        "direct_prior_work_comparison.json",
        "direct_prior_work_comparison.csv",
        "direct_prior_work_comparison.md",
    ):
        assert (tmp_path / name).exists()


def test_amulet_kv_append_counterexample_in_kv_result(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    amu = [r for r in report["rows"] if r["protocol_name"] == "amulet_static_left_right_mask"][0]
    assert "counterexample" in amu["kv_append_result"].lower()


def test_arrow_marked_missing_formula(tmp_path: Path) -> None:
    report = run_direct_prior_work_comparison(_cfg(tmp_path))
    arr = [r for r in report["rows"] if r["protocol_name"] == "arrow_direct_primitive_or_unavailable"][0]
    assert arr["exact_primitive_implemented"] is False
    assert arr["runtime_directly_comparable"] is False
