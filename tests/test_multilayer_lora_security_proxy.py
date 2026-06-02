"""Stage 7.3 — tests for the multi-layer LoRA cross-layer security proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.multilayer_lora_security_proxy import (
    MultiLayerLoRASecurityProxyConfig,
    VALID_LINKAGE_STRATEGIES,
    multilayer_security_csv_rows,
    run_multilayer_lora_security_proxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_multilayer_lora_security_proxy.py"


def _cfg(**overrides) -> MultiLayerLoRASecurityProxyConfig:
    base = dict(
        seed=2026,
        num_layers=2,
        hidden_size=16,
        intermediate_size=24,
        true_ranks=(2, 4),
        padded_rank=8,
        num_trials=4,
        membership_trials_per_sample=2,
        membership_num_steps=1,
        dummy_strategy="paired_cancellation_dummy",
        dtype="float64",
    )
    base.update(overrides)
    return MultiLayerLoRASecurityProxyConfig(**base)


# ---------------------------------------------------------------------------
# 1. small config runs
# ---------------------------------------------------------------------------


def test_run_multilayer_lora_security_proxy_runs() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    assert (
        report["lora_multilayer_security_proxy_status"] == "implemented"
    )
    assert report["security_profile"] == "proxy-evaluated, not formal"


# ---------------------------------------------------------------------------
# 2. cross-layer adapter linkage section exists
# ---------------------------------------------------------------------------


def test_cross_layer_adapter_linkage_section() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    section = report["cross_layer_adapter_linkage"]
    assert "rows" in section
    strategies = [r["strategy"] for r in section["rows"]]
    assert set(strategies) == set(VALID_LINKAGE_STRATEGIES)
    for row in section["rows"]:
        assert "layer_linkability_auc" in row
        assert "module_identity_retrieval_top1" in row
        assert "same_module_similarity" in row
        assert "different_module_similarity" in row
        assert "risk_level" in row


# ---------------------------------------------------------------------------
# 3. heterogeneous rank section exists
# ---------------------------------------------------------------------------


def test_heterogeneous_rank_section() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    section = report["heterogeneous_true_rank_with_shared_padded_rank"]
    assert "rows" in section
    assert len(section["rows"]) > 0
    for row in section["rows"]:
        assert "true_rank" in row
        assert "padded_rank" in row
        assert "visible_rank_from_shape" in row
        assert "true_rank_shape_hidden_rate" in row
        assert "spectral_rank_inference_accuracy" in row
        assert "gradient_rank_inference_accuracy" in row


# ---------------------------------------------------------------------------
# 4. true_rank shape hidden rate == 1.0 when padded_rank shared
# ---------------------------------------------------------------------------


def test_true_rank_shape_hidden_rate_is_one() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    assert report["interpretation"]["true_rank_shape_hidden_rate"] == 1.0
    for row in report["heterogeneous_true_rank_with_shared_padded_rank"]["rows"]:
        assert row["true_rank_shape_hidden_rate"] == 1.0
        assert row["visible_rank_from_shape"] == 8


# ---------------------------------------------------------------------------
# 5. spectral risk reported conservatively (never "low" under paired)
# ---------------------------------------------------------------------------


def test_spectral_risk_conservative_under_paired() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    for row in report["heterogeneous_true_rank_with_shared_padded_rank"]["rows"]:
        assert row["risk_level"] in {
            "needs_more_evaluation", "medium", "high",
        }


def test_zero_dummy_risk_is_high() -> None:
    report = run_multilayer_lora_security_proxy(
        _cfg(dummy_strategy="zero_dummy")
    )
    for row in report["heterogeneous_true_rank_with_shared_padded_rank"]["rows"]:
        assert row["risk_level"] == "high"


# ---------------------------------------------------------------------------
# 6. membership linkability section exists
# ---------------------------------------------------------------------------


def test_membership_linkability_section() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    section = report["multi_step_membership_linkability"]
    assert "rows" in section
    assert "aggregate" in section
    for row in section["rows"]:
        assert "membership_auc_proxy" in row
        assert "linkability_rank" in row
        assert "risk_level" in row


# ---------------------------------------------------------------------------
# 7. JSON / CSV / MD generated
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--num-layers", "2",
        "--hidden-size", "16",
        "--intermediate-size", "24",
        "--true-ranks", "2", "4",
        "--padded-rank", "8",
        "--num-trials", "3",
        "--membership-trials-per-sample", "2",
        "--membership-num-steps", "1",
        "--dtype", "float64",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_path = tmp_path / "multilayer_lora_security_proxy.json"
    csv_path = tmp_path / "multilayer_lora_security_proxy.csv"
    md_path = tmp_path / "multilayer_lora_security_proxy.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    md_text = md_path.read_text().lower()
    for needle in (
        "multi-layer lora",
        "experiment scope",
        "threat model",
        "cross-layer adapter linkage",
        "heterogeneous true rank",
        "multi-step membership",
        "interpretation",
        "limitations",
        "next stage plan",
    ):
        assert needle in md_text, f"missing markdown section: {needle!r}"


# ---------------------------------------------------------------------------
# 8. No raw adapter / gradient / mask in outputs
# ---------------------------------------------------------------------------


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--num-layers", "2",
        "--hidden-size", "16",
        "--intermediate-size", "24",
        "--true-ranks", "2", "4",
        "--padded-rank", "8",
        "--num-trials", "3",
        "--membership-trials-per-sample", "2",
        "--membership-num-steps", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_text = (
        tmp_path / "multilayer_lora_security_proxy.json"
    ).read_text()
    md_text = (
        tmp_path / "multilayer_lora_security_proxy.md"
    ).read_text()
    csv_text = (
        tmp_path / "multilayer_lora_security_proxy.csv"
    ).read_text()
    for text in (json_text, md_text, csv_text):
        assert "tensor(" not in text


# ---------------------------------------------------------------------------
# 9. Misc
# ---------------------------------------------------------------------------


def test_csv_rows_no_tensor_text() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    rows = multilayer_security_csv_rows(report)
    for row in rows:
        assert "tensor(" not in str(row["value"])


def test_membership_aggregate_present() -> None:
    report = run_multilayer_lora_security_proxy(_cfg())
    agg = report["multi_step_membership_linkability"]["aggregate"]
    assert "mean_membership_auc_proxy" in agg
    assert "adapter_update_linkability" in agg


def test_invalid_dummy_strategy_raises() -> None:
    cfg = _cfg()
    cfg.dummy_strategy = "nope"
    with pytest.raises(ValueError):
        run_multilayer_lora_security_proxy(cfg)
