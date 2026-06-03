"""Stage 7.5b tests for paper_toy_tasks."""

from __future__ import annotations

import json
from pathlib import Path

from pllo.experiments.paper_toy_tasks import (
    PaperToyTaskConfig,
    run_paper_toy_tasks,
)


def _cfg(tmp: Path, **overrides) -> PaperToyTaskConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        num_samples=16,
        seq_len=4,
        vocab_size=16,
        hidden_size=8,
        num_classes=2,
        num_layers=1,
        true_rank=2,
        padded_rank=4,
        batch_size=4,
        num_train_steps=2,
        lr=1e-2,
    )
    base.update(overrides)
    return PaperToyTaskConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_paper_toy_tasks(_cfg(tmp_path))
    assert report["paper_toy_tasks_status"] == "implemented"
    assert report["wall_time_source"] == "measured_local_emulation"
    assert report["is_real_tee_wall_time"] is False
    assert report["is_gpu_throughput"] is False


def test_three_task_rows(tmp_path: Path) -> None:
    report = run_paper_toy_tasks(_cfg(tmp_path))
    names = [r["task_name"] for r in report["rows"]]
    assert names == [
        "token_parity_classification",
        "first_last_token_relation",
        "next_token_toy_lm",
    ]


def test_required_fields_present(tmp_path: Path) -> None:
    report = run_paper_toy_tasks(_cfg(tmp_path))
    for row in report["rows"]:
        for key in (
            "train_loss_plain", "train_loss_masked", "loss_diff",
            "accuracy_plain", "accuracy_masked", "accuracy_diff",
            "logits_max_abs_error", "token_match_rate", "allclose",
        ):
            assert key in row, f"missing {key}"


def test_outputs_emitted(tmp_path: Path) -> None:
    run_paper_toy_tasks(_cfg(tmp_path))
    for name in ("paper_toy_tasks.json", "paper_toy_tasks.csv", "paper_toy_tasks.md"):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_md_disclaimer(tmp_path: Path) -> None:
    run_paper_toy_tasks(_cfg(tmp_path))
    md = (tmp_path / "paper_toy_tasks.md").read_text(encoding="utf-8")
    assert "CPU" in md or "local" in md.lower()
    assert "not a real" in md.lower() or "not real tee" in md.lower() or "not a real qwen" in md.lower()


def test_no_raw_tensors_or_input_ids(tmp_path: Path) -> None:
    run_paper_toy_tasks(_cfg(tmp_path))
    blob = (tmp_path / "paper_toy_tasks.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
    # We do not export input_ids -- only labels are derived from them.
    assert '"input_ids"' not in blob
    assert '"private_data"' not in blob


def test_loss_diff_close_to_zero(tmp_path: Path) -> None:
    report = run_paper_toy_tasks(_cfg(tmp_path))
    for row in report["rows"]:
        # The masked path implements Theorem 7 exactly; round-off only.
        assert row["loss_diff"] < 1e-6, row
        assert row["token_match_rate"] > 0.95, row


def test_accuracy_diff_close_to_zero(tmp_path: Path) -> None:
    report = run_paper_toy_tasks(_cfg(tmp_path))
    for row in report["rows"]:
        assert row["accuracy_diff"] < 1e-6
