"""Stage 6.4b — modern decoder block-level smoke script tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_modern_decoder_block_smoke.py"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "modern_decoder_block_wrapper_smoke.json"
OUTPUT_MD = PROJECT_ROOT / "outputs" / "modern_decoder_block_wrapper_smoke.md"


# ---------------------------------------------------------------------------
# Smoke script emits JSON + Markdown using synthetic fallback (no network)
# ---------------------------------------------------------------------------


def test_smoke_emits_json_and_markdown(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--hidden-size", "32",
            "--intermediate-size", "64",
            "--num-query-heads", "4",
            "--num-kv-heads", "2",
            "--head-dim", "8",
            "--seq-len", "6",
            "--batch-size", "2",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_path = tmp_path / "modern_decoder_block_wrapper_smoke.json"
    md_path = tmp_path / "modern_decoder_block_wrapper_smoke.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source"] == "synthetic_fallback"
    assert payload["summary"]["all_runs_allclose"] is True
    assert payload["summary"]["online_extra_matmul_count"] == 0
    assert payload["summary"]["implemented_block_level"] is True
    assert payload["summary"]["full_runtime_integrated"] is False
    assert set(payload["summary"]["mitigation_bundles_evaluated"]) == {
        "fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad",
    }
    # Default config: 2 bundles × 2 use_pad = 4 runs.
    assert len(payload["per_run"]) == 4


def test_smoke_synthetic_fallback_does_not_hit_network(tmp_path) -> None:
    """If attempt_real_model_load is OFF, candidates_tried is empty."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    payload = json.loads(
        (tmp_path / "modern_decoder_block_wrapper_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["model_loading"]["load_status"] == "synthetic_only"
    assert payload["model_loading"]["candidates_tried"] == []


# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------


def test_smoke_markdown_contains_required_sections(tmp_path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "modern_decoder_block_wrapper_smoke.md").read_text(
        encoding="utf-8"
    )
    for header in (
        "Experiment Scope",
        "Model Loading Status",
        "Modern Decoder Block Spec",
        "Plain Reference vs HF Block Status",
        "RMSNorm Handling",
        "RoPE-Aware Attention Handling",
        "GQA / MQA Handling",
        "SwiGLU Compatible Island Handling",
        "Mitigation Bundle Results",
        "Limitations",
        "Next Stage Plan",
    ):
        assert header in md, f"missing markdown header: {header!r}"
    md_lower = md.lower()
    assert "block-level integration" in md_lower
    assert "not a full" in md_lower or "not full" in md_lower
    assert "not a real tee" in md_lower
    assert "not formal security" in md_lower
    assert "fresh_perm_plus_sandwich_plus_pad" in md
    assert "fresh_perm_only" in md


# ---------------------------------------------------------------------------
# Committed outputs (best-effort)
# ---------------------------------------------------------------------------


def test_committed_smoke_outputs_present_or_skipped() -> None:
    if not OUTPUT_JSON.exists():
        pytest.skip(
            "outputs/modern_decoder_block_wrapper_smoke.json missing — run "
            "`python scripts/run_modern_decoder_block_smoke.py` first."
        )
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert payload["summary"]["all_runs_allclose"] is True
    assert payload["summary"]["online_extra_matmul_count"] == 0


def test_committed_smoke_markdown_present_or_skipped() -> None:
    if not OUTPUT_MD.exists():
        pytest.skip("outputs/modern_decoder_block_wrapper_smoke.md missing")
    md = OUTPUT_MD.read_text(encoding="utf-8")
    assert "Mitigation Bundle Results" in md
