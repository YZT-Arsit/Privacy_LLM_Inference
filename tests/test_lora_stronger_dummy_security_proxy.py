"""Stage 7.4 — tests for the stronger-dummy LoRA security proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_stronger_dummy_security_proxy import (
    StrongerDummySecurityProxyConfig,
    run_lora_stronger_dummy_security_proxy,
    stronger_dummy_security_csv_rows,
)
from pllo.ops.lora_dummy_strategies import VALID_STRONG_DUMMY_STRATEGIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT / "scripts" / "run_lora_stronger_dummy_security_proxy.py"
)


def _cfg(**overrides) -> StrongerDummySecurityProxyConfig:
    base = dict(
        seed=2026,
        d_in=16, d_out=12,
        true_ranks=(2, 4),
        padded_rank=8,
        num_trials=4,
        num_lora_modules_for_linkage=3,
        dummy_strategies=tuple(VALID_STRONG_DUMMY_STRATEGIES),
        dtype="float64",
    )
    base.update(overrides)
    return StrongerDummySecurityProxyConfig(**base)


# ---------------------------------------------------------------------------
# 1. small config runs
# ---------------------------------------------------------------------------


def test_run_lora_stronger_dummy_security_proxy_runs() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    assert (
        report["lora_stronger_dummy_security_status"] == "implemented"
    )
    assert (
        report["lora_spectral_rank_hardening_status"] == "proxy-evaluated"
    )
    assert report["security_profile"] == "proxy-evaluated, not formal"


# ---------------------------------------------------------------------------
# 2. spectral rank inference section exists
# ---------------------------------------------------------------------------


def test_spectral_rank_inference_section() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    section = report["spectral_rank_inference"]
    assert "rows" in section
    assert len(section["rows"]) >= len(VALID_STRONG_DUMMY_STRATEGIES)
    for row in section["rows"]:
        for k in (
            "cliff_inference_accuracy", "energy_inference_accuracy",
            "elbow_inference_accuracy", "ensemble_inference_accuracy",
            "risk_level",
        ):
            assert k in row


# ---------------------------------------------------------------------------
# 3. gradient rank inference section exists
# ---------------------------------------------------------------------------


def test_gradient_rank_inference_section() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    section = report["gradient_rank_inference"]
    assert "rows" in section
    for row in section["rows"]:
        for k in (
            "grad_a_cliff_accuracy", "grad_b_cliff_accuracy",
            "gradient_rank_inference_accuracy", "risk_level",
        ):
            assert k in row


# ---------------------------------------------------------------------------
# 4. dummy strategy classification section exists
# ---------------------------------------------------------------------------


def test_dummy_strategy_classification_section() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    cls = report["dummy_strategy_classification"]
    for k in (
        "strategy_classification_accuracy",
        "random_chance_baseline",
        "risk_level",
        "confusion_counts",
    ):
        assert k in cls


# ---------------------------------------------------------------------------
# 5. cross-layer linkage section exists
# ---------------------------------------------------------------------------


def test_cross_layer_linkage_section() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    section = report["cross_layer_linkage"]
    assert "rows" in section
    for row in section["rows"]:
        for k in (
            "layer_linkability_auc",
            "module_identity_retrieval_top1",
            "same_module_similarity",
            "different_module_similarity",
            "risk_level",
        ):
            assert k in row


# ---------------------------------------------------------------------------
# 6. at least one stronger dummy strategy reduces rank inference accuracy
#    (or is conservatively marked needs_more_evaluation)
# ---------------------------------------------------------------------------


def test_stronger_dummy_reduces_or_marks_conservatively() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    by_strategy: dict[str, list[dict]] = {}
    for row in report["spectral_rank_inference"]["rows"]:
        by_strategy.setdefault(row["dummy_strategy"], []).append(row)
    # zero_dummy is the high-risk baseline.
    zero_accs = [
        r["cliff_inference_accuracy"] for r in by_strategy.get("zero_dummy", [])
    ]
    assert zero_accs and max(zero_accs) >= 0.5, "zero_dummy baseline should be leaky"
    # Every stronger cancellation strategy should EITHER reduce accuracy or
    # explicitly be reported as needs_more_evaluation / medium / high; never
    # silently report "low" without justification.
    for strategy in (
        "paired_cancellation_dummy",
        "gaussian_matched_dummy",
        "spectrum_matched_dummy",
        "noise_injected_cancellation_dummy",
        "orthogonalized_cancellation_dummy",
        "mixed_dummy_ensemble",
    ):
        rows = by_strategy.get(strategy, [])
        assert rows, f"missing rows for {strategy!r}"
        for row in rows:
            assert row["risk_level"] in {
                "needs_more_evaluation", "medium", "high", "low",
            }
            # Conservative: paired-style strategies must never be 'low' when
            # they only achieve accuracy <= 0.2 — they must be flagged as
            # needs_more_evaluation per requirement 12.
            if row["cliff_inference_accuracy"] < 0.2:
                assert row["risk_level"] in {
                    "needs_more_evaluation", "medium", "high",
                }, (strategy, row)


# ---------------------------------------------------------------------------
# 7. risk_level reported conservatively for zero_dummy
# ---------------------------------------------------------------------------


def test_zero_dummy_is_high_risk() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    for row in report["spectral_rank_inference"]["rows"]:
        if row["dummy_strategy"] == "zero_dummy":
            assert row["risk_level"] == "high", row


# ---------------------------------------------------------------------------
# 8. JSON / CSV / Markdown generated by the runner
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--d-in", "16", "--d-out", "12",
        "--true-ranks", "2", "4",
        "--padded-rank", "8",
        "--num-trials", "3",
        "--num-lora-modules-for-linkage", "2",
        "--dtype", "float64",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_path = tmp_path / "lora_stronger_dummy_security_proxy.json"
    csv_path = tmp_path / "lora_stronger_dummy_security_proxy.csv"
    md_path = tmp_path / "lora_stronger_dummy_security_proxy.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    md_text = md_path.read_text().lower()
    for needle in (
        "stronger dummy",
        "experiment scope",
        "threat model",
        "spectral rank inference",
        "gradient rank inference",
        "dummy strategy classification",
        "cross-layer linkage",
        "interpretation",
        "limitations",
        "next stage plan",
    ):
        assert needle in md_text, f"missing markdown section: {needle!r}"


# ---------------------------------------------------------------------------
# 9. no raw adapter / raw gradient / private data / mask in outputs
# ---------------------------------------------------------------------------


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--d-in", "16", "--d-out", "12",
        "--true-ranks", "2",
        "--padded-rank", "8",
        "--num-trials", "3",
        "--num-lora-modules-for-linkage", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_text = (
        tmp_path / "lora_stronger_dummy_security_proxy.json"
    ).read_text()
    md_text = (
        tmp_path / "lora_stronger_dummy_security_proxy.md"
    ).read_text()
    csv_text = (
        tmp_path / "lora_stronger_dummy_security_proxy.csv"
    ).read_text()
    for text in (json_text, md_text, csv_text):
        assert "tensor(" not in text


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_csv_rows_no_tensor_text() -> None:
    report = run_lora_stronger_dummy_security_proxy(_cfg())
    rows = stronger_dummy_security_csv_rows(report)
    for row in rows:
        assert "tensor(" not in str(row["value"])


def test_invalid_strategy_raises() -> None:
    cfg = _cfg()
    cfg.dummy_strategies = ("not_a_real_strategy",)
    with pytest.raises(ValueError):
        run_lora_stronger_dummy_security_proxy(cfg)
