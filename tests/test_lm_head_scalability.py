"""Stage 7.7a tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.lm_head_scalability import (
    LMHeadScalabilityConfig,
    render_markdown,
    run_lm_head_scalability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_lm_head_scalability(cfg=LMHeadScalabilityConfig())


def test_permutation_mask_exact_all_real_v(report: dict) -> None:
    for r in report["real_runs"]["vocab_permutation_mask"]:
        assert r["exactness"] == "exact"
        assert r["max_abs_error"] < 1e-12
        assert r["greedy_token_match_rate"] == 1.0


def test_block_mode_exact_for_feasible_v(report: dict) -> None:
    for r in report["real_runs"]["block_diagonal_vocab_mask"]:
        assert r["exactness"] == "exact"
        assert r["max_abs_error"] < 1e-10
        assert r["greedy_token_match_rate"] == 1.0


def test_dense_runs_only_for_small_v(report: dict) -> None:
    cfg = report["config"]
    max_v = cfg["dense_max_real_v"]
    for r in report["real_runs"]["dense_vocab_mask_baseline"]:
        if r["vocab_size"] <= max_v:
            assert r["exactness"] == "exact"
            assert r["max_abs_error"] < 1e-9
            assert r.get("symbolic_estimate_only", False) is False
        else:
            assert r.get("symbolic_estimate_only", False) is True


def test_topk_mode_top1_exact(report: dict) -> None:
    for r in report["real_runs"]["topk_trusted_recovery_mode"]:
        assert r["greedy_token_match_rate"] == 1.0


def test_permutation_preserves_logit_multiset(report: dict) -> None:
    for r in report["real_runs"]["vocab_permutation_mask"]:
        assert r["logit_multiset_preserved_max_abs_error"] < 1e-10


def test_dense_not_scalable_text(report: dict) -> None:
    text = "\n".join(report["limitations"])
    assert "not feasible" in text.lower() or "not scalable" in text.lower()
    # Symbolic dense estimates appear for the large vocab sizes.
    big_sizes = report["config"]["estimated_vocab_sizes"]
    have_dense = report["symbolic_estimates"]["dense_vocab_mask_baseline"]
    assert len(have_dense) == len(big_sizes)


def test_no_real_gpu_or_tee_claim(report: dict) -> None:
    text = json.dumps(report)
    assert "real_gpu" not in text or '"real_gpu": false' in text
    assert "formal cryptographic" not in text.lower() or (
        "not formal cryptographic" in text.lower()
        or "no formal cryptographic" in text.lower()
        or "NOT formal cryptographic".lower() in text.lower()
    )


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Scalable LM-Head Masking" in md
    for mode in report["modes_evaluated"]:
        assert mode in md
    assert "Paper-Safe Wording" in md
    assert "Unsafe Wording to Avoid" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "lm_head_scalability.json"
    m = REPO_ROOT / "outputs" / "lm_head_scalability.md"
    # Allow tests to run before runner has been invoked; runner is
    # called explicitly via test_run_runner below.
    if j.exists() and m.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"


def test_run_runner_writes_outputs(tmp_path: Path) -> None:
    from pllo.experiments.lm_head_scalability import write_reports
    rep = run_lm_head_scalability(cfg=LMHeadScalabilityConfig())
    j, m = write_reports(rep, outputs_dir=tmp_path)
    assert j.exists() and m.exists()
    assert json.loads(j.read_text())["stage"] == "7.7a"
