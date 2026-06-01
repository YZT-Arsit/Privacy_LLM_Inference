"""Stage 5.6 — tests for the stronger_attackers orchestrator + runner."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.stronger_attackers import (
    StrongerAttackersConfig,
    run_stronger_attackers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_stronger_attackers.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _small_cfg(**overrides):
    cfg = dict(
        seed=2026,
        num_prompts=4,
        prompt_max_length=6,
        max_new_tokens=2,
        synthetic_vocab_size=32,
        synthetic_hidden_size=16,
        synthetic_intermediate_size=32,
        synthetic_num_attention_heads=4,
        synthetic_num_key_value_heads=2,
        synthetic_head_dim=4,
        max_layers=2,
    )
    cfg.update(overrides)
    return StrongerAttackersConfig(**cfg)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_stronger_attackers(_small_cfg())


def test_returns_four_main_sections(report) -> None:
    for k in (
        "blackbox_attacker", "timing_sidechannel_proxy",
        "inter_block_masking_gap", "overall_risk_summary",
    ):
        assert k in report


def test_envelope_integrity_risk_present(report) -> None:
    o = report["overall_risk_summary"]
    assert "envelope_integrity_risk_level" in o
    assert "structural_leakage_risk_level" in o
    assert "overall_risk_level" in o


def test_recommendation_provides_promotion_eligibility(report) -> None:
    rec = report["recommendation"]
    assert rec["security_profile_detail_with_stronger_attackers"] == (
        "adaptive-blackbox-and-timing-proxy-evaluated, not formal"
    )
    assert "promotion_eligibility_note" in rec


def test_limitations_contain_required_phrases(report) -> None:
    text = " ".join(report["limitations"]).lower()
    assert "stronger proxy attacks, not formal security proofs" in text
    assert "timing results are model-based proxies" in text
    assert "inter-block masking is experimental" in text


def test_json_safe_no_internal_tensor(report) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text


# ---------------------------------------------------------------------------
# Script end-to-end
# ---------------------------------------------------------------------------


def test_script_generates_json_csv_markdown(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-prompts", "4",
            "--prompt-max-length", "6",
            "--max-new-tokens", "2",
            "--synthetic-vocab-size", "32",
            "--synthetic-hidden-size", "16",
            "--synthetic-intermediate-size", "32",
            "--synthetic-num-query-heads", "4",
            "--synthetic-num-kv-heads", "2",
            "--synthetic-head-dim", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_path = tmp_path / "stronger_attackers.json"
    csv_path = tmp_path / "stronger_attackers.csv"
    md_path = tmp_path / "stronger_attackers.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Threat Model",
        "Black-Box Query Attacker",
        "Timing / Boundary-Call Side-Channel Proxy",
        "Inter-Block Residual Masking Gap",
        "Single-Transition Masking Probe",
        "Comparison with Stage 5.4 / 5.5 / 5.5b",
        "Overall Risk Summary",
        "Recommendation",
        "Limitations",
        "Next Stage Plan",
        "stronger proxy attacks, not formal security proofs",
        "timing results are model-based proxies",
        "inter-block residual masking gap",
        "not formal security",
        "not a real TEE measurement",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert {
        "model_loading", "tokenizer_loading",
        "blackbox", "timing", "inter_block",
        "overall_risk_summary", "recommendation",
    } <= sections


def test_committed_outputs_sanity_when_present() -> None:
    p = OUTPUT_DIR / "stronger_attackers.json"
    if not p.exists():
        pytest.skip("outputs/stronger_attackers.json not present")
    payload = json.loads(p.read_text(encoding="utf-8"))
    for k in (
        "blackbox_attacker", "timing_sidechannel_proxy",
        "inter_block_masking_gap", "overall_risk_summary",
        "recommendation", "limitations",
    ):
        assert k in payload


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


def test_script_output_has_no_long_numeric_arrays(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-prompts", "3",
            "--prompt-max-length", "6",
            "--max-new-tokens", "2",
            "--synthetic-vocab-size", "32",
            "--synthetic-hidden-size", "16",
            "--synthetic-intermediate-size", "32",
            "--synthetic-num-query-heads", "4",
            "--synthetic-num-kv-heads", "2",
            "--synthetic-head-dim", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    text = (tmp_path / "stronger_attackers.json").read_text(encoding="utf-8")
    assert _LONG_NUMBER_ARRAY.search(text) is None
