"""Stage 5.5b — tests for the real-token-prompted attacker + runner script."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.real_token_activation_attacker import (
    INTER_BLOCK_PLAIN_TENSORS,
    RealTokenActivationAttackConfig,
    run_real_token_activation_attacks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_real_token_activation_attacks.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _small_config(**overrides):
    cfg = dict(
        seed=2026,
        attempt_real_model_load=False,
        attempt_tokenizer_load=False,
        allow_synthetic_fallback=True,
        num_prompts=4,
        prompt_max_length=6,
        max_layers=2,
        max_new_tokens=2,
        attacker_steps=10,
        attacker_lr=1e-2,
        mlp_hidden_size=32,
        mlp_batch_size=16,
        ridge_lambda=1e-3,
        train_fraction=0.7,
        mitigation_bundles=("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"),
        use_pad=True,
        nonlinear_mode="compatible_islands",
        synthetic_vocab_size=64,
        synthetic_hidden_size=16,
        synthetic_intermediate_size=32,
        synthetic_num_attention_heads=4,
        synthetic_num_key_value_heads=2,
        synthetic_head_dim=4,
    )
    cfg.update(overrides)
    return RealTokenActivationAttackConfig(**cfg)


@pytest.fixture(scope="module")
def attacker_report() -> dict:
    return run_real_token_activation_attacks(_small_config())


def test_run_returns_expected_top_level_keys(attacker_report) -> None:
    for key in (
        "config", "model_loading", "tokenizer_loading", "source",
        "prompt_summary", "block_spec_summary", "metadata",
        "generation_summary", "trace_summary",
        "target_tensor_results", "bundle_comparison",
        "attacker_summary", "recommendation",
        "comparison_with_stage_5_5", "limitations",
    ):
        assert key in attacker_report, f"missing top-level key {key!r}"


def test_both_bundles_present(attacker_report) -> None:
    tt = attacker_report["target_tensor_results"]
    assert "fresh_perm_only" in tt
    assert "fresh_perm_plus_sandwich_plus_pad" in tt


def test_coverage_includes_key_tensors(attacker_report) -> None:
    full = attacker_report["target_tensor_results"][
        "fresh_perm_plus_sandwich_plus_pad"
    ]
    prefill = full.get("prefill", {})
    must_cover = {"gate", "up", "swiglu_intermediate", "q", "k", "v"}
    assert must_cover <= set(prefill.keys()), (
        f"missing tensors in prefill: {must_cover - set(prefill.keys())}"
    )


def test_inter_block_plain_tensors_flagged(attacker_report) -> None:
    full = attacker_report["target_tensor_results"][
        "fresh_perm_plus_sandwich_plus_pad"
    ]["prefill"]
    for name in INTER_BLOCK_PLAIN_TENSORS:
        if name in full:
            assert full[name]["inter_block_plain"] is True


def test_masked_only_recommendation_low(attacker_report) -> None:
    rec = attacker_report["recommendation"]
    assert (
        rec["default_on_recommendation_full_bundle_masked_only"]
        == "acceptable_with_mitigation_under_real_token_proxy"
    )


def test_limitations_phrases_present(attacker_report) -> None:
    text = " ".join(attacker_report["limitations"]).lower()
    assert "real-token-prompted adaptive proxy attacks, not formal security proofs" in text
    assert "synthetic token fallback" in text
    assert "dense sandwiching reduces tested recovery but does not imply semantic security" in text


def test_json_has_no_raw_tensor(attacker_report) -> None:
    text = json.dumps(
        {
            "trace_summary": attacker_report["trace_summary"],
            "metadata": attacker_report["metadata"],
            "recommendation": attacker_report["recommendation"],
            "limitations": attacker_report["limitations"],
        },
        default=str,
    )
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
            "--attacker-steps", "10",
            "--synthetic-vocab-size", "64",
            "--synthetic-hidden-size", "16",
            "--synthetic-intermediate-size", "32",
            "--synthetic-num-query-heads", "4",
            "--synthetic-num-kv-heads", "2",
            "--synthetic-head-dim", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_path = tmp_path / "real_token_activation_attacks.json"
    csv_path = tmp_path / "real_token_activation_attacks.csv"
    md_path = tmp_path / "real_token_activation_attacks.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Model and Tokenizer Loading Status",
        "Prompt Set Summary",
        "Trace Collection Summary",
        "Prefill Real-Token Activation Attacks",
        "Decode-Step Real-Token Activation Attacks",
        "Linear Inverter Results",
        "Small MLP Inverter Results",
        "Permutation Recovery Results",
        "Linkability Results",
        "Bundle Comparison",
        "Comparison with Stage 5.5 Random-Hidden Real-Activation Attacker",
        "Recommendation",
        "Limitations",
        "Next Stage Plan",
        "real-token-prompted adaptive proxy attacks, not formal security proofs",
        "synthetic token fallback",
        "Dense sandwiching reduces tested recovery but does not imply semantic security",
        "not formal security",
        "not a real TEE measurement",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"

    # CSV uses long format with at least these sections.
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert {
        "model_loading", "tokenizer_loading", "prompt_summary",
        "linear_inverter", "mlp_inverter", "linkability",
        "permutation_recovery", "decision", "bundle_comparison",
        "recommendation",
    } <= sections


def test_committed_outputs_sanity_when_present() -> None:
    """If `outputs/real_token_activation_attacks.json` was committed, sanity-check it."""
    p = OUTPUT_DIR / "real_token_activation_attacks.json"
    if not p.exists():
        pytest.skip("outputs/real_token_activation_attacks.json not present")
    text = p.read_text(encoding="utf-8")
    assert "tensor(" not in text
    payload = json.loads(text)
    assert "target_tensor_results" in payload
    assert "recommendation" in payload


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
            "--attacker-steps", "10",
            "--synthetic-vocab-size", "64",
            "--synthetic-hidden-size", "16",
            "--synthetic-intermediate-size", "32",
            "--synthetic-num-query-heads", "4",
            "--synthetic-num-kv-heads", "2",
            "--synthetic-head-dim", "4",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_text = (tmp_path / "real_token_activation_attacks.json").read_text(
        encoding="utf-8"
    )
    assert _LONG_NUMBER_ARRAY.search(json_text) is None
