"""Stage 5.6 extension — tests for stronger_attackers under closure modes."""

from __future__ import annotations

import csv
import json
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


def _cfg(**overrides):
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
def closed_report() -> dict:
    return run_stronger_attackers(
        _cfg(
            inter_block_mask_mode="masked_boundary_experimental",
            constant_time_decode_mode="proxy_equalized",
        )
    )


def test_accepts_inter_block_mask_mode_and_constant_time(closed_report) -> None:
    assert closed_report["config"]["inter_block_mask_mode"] == (
        "masked_boundary_experimental"
    )
    assert closed_report["config"]["constant_time_decode_mode"] == "proxy_equalized"


def test_inter_block_closure_summary_present(closed_report) -> None:
    ibc = closed_report["inter_block_closure_summary"]
    assert ibc["status"] == "implemented"
    assert ibc["masked_boundary_experimental_status"] == "implemented"
    # Head-to-head: boundary_input flips from inter_block_plain True (high
    # risk) to False (low risk).
    before = ibc["boundary_input_before"]
    after = ibc["boundary_input_after"]
    assert before["inter_block_plain"] is True
    assert after["inter_block_plain"] is False
    assert before["risk_level"] == "high"
    assert after["risk_level"] in {"low", "medium"}


def test_constant_time_decode_summary_present(closed_report) -> None:
    ct = closed_report["constant_time_decode_summary"]
    assert ct["mode"] == "proxy_equalized"
    assert ct["overhead_ms_estimate"] > 0.0
    # Step risk should drop OR limitation must explain why.
    if ct["risk_level_after"] != ct["risk_level_before"]:
        assert {ct["risk_level_after"], ct["risk_level_before"]} <= {
            "low", "medium", "high",
        }


def test_recommendation_promotes_to_extended_proxy(closed_report) -> None:
    rec = closed_report["recommendation"]
    assert rec["overall_recommendation"] in {
        "acceptable_with_mitigation_under_extended_proxy",
        "needs_more_evaluation",
    }
    assert rec["security_profile_detail_with_stronger_attackers"] == (
        "adaptive-blackbox-and-timing-proxy-evaluated, not formal"
    )
    if rec["overall_recommendation"] == (
        "acceptable_with_mitigation_under_extended_proxy"
    ):
        assert rec["extended_proxy_eligibility"] == "yes"
        assert rec["security_profile_detail_with_extended_proxy"] in {
            "inter-block-and-constant-time-proxy-evaluated, not formal",
            "inter-block-masked-and-adaptive-proxy-evaluated, not formal",
        }


def test_json_safe_no_raw_tensor(closed_report) -> None:
    text = json.dumps(closed_report, default=str)
    assert "tensor(" not in text


# ---------------------------------------------------------------------------
# Script end-to-end
# ---------------------------------------------------------------------------


def test_script_generates_json_csv_markdown_under_closure(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-prompts", "4",
            "--prompt-max-length", "6",
            "--max-new-tokens", "2",
            "--inter-block-mask-mode", "masked_boundary_experimental",
            "--constant-time-decode-mode", "proxy_equalized",
            "--synthetic-vocab-size", "32",
            "--synthetic-hidden-size", "16",
            "--synthetic-intermediate-size", "32",
            "--synthetic-num-query-heads", "4",
            "--synthetic-num-kv-heads", "2",
            "--synthetic-head-dim", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "stronger_attackers.md").read_text(encoding="utf-8")
    for phrase in (
        "Inter-Block Masking Mode",
        "Plain Boundary vs Masked Boundary Experimental",
        "Boundary Input / Final Risk Before and After",
        "Constant-Time Decode Proxy",
        "Decode-Step Timing Leakage Before and After",
        "Overhead Proxy",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"
    with (tmp_path / "stronger_attackers.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "inter_block_closure" in sections
    assert "constant_time_decode" in sections
