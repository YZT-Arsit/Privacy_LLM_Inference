"""Stage 6.4c — smoke script tests (synthetic, no network)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_modern_decoder_model_smoke.py"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "modern_decoder_model_wrapper_smoke.json"
OUTPUT_MD = PROJECT_ROOT / "outputs" / "modern_decoder_model_wrapper_smoke.md"

_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


def test_smoke_writes_json_md(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--hidden-size", "32",
            "--intermediate-size", "64",
            "--num-query-heads", "4",
            "--num-kv-heads", "2",
            "--head-dim", "8",
            "--vocab-size", "32",
            "--prompt-length", "5",
            "--max-new-tokens", "3",
            "--max-layers", "2",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_path = tmp_path / "modern_decoder_model_wrapper_smoke.json"
    md_path = tmp_path / "modern_decoder_model_wrapper_smoke.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["all_full_forward_allclose"] is True
    assert summary["all_prefill_allclose"] is True
    assert summary["all_decode_top1_match"] is True
    assert summary["all_generation_exact_match"] is True
    assert summary["implemented_model_level"] is True
    assert summary["full_runtime_integrated"] is False
    assert summary["modern_decoder_generation_status"] == "greedy_generation_implemented"
    assert summary["modern_decoder_kv_cache_status"] == "implemented"
    # 2 bundles × 2 use_pad = 4 runs by default.
    assert len(payload["per_run"]) == 4


def test_smoke_synthetic_fallback_no_network(tmp_path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    payload = json.loads(
        (tmp_path / "modern_decoder_model_wrapper_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["model_loading"]["load_status"] == "synthetic_only"
    assert payload["model_loading"]["candidates_tried"] == []


def test_smoke_markdown_has_required_sections(tmp_path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "modern_decoder_model_wrapper_smoke.md").read_text(
        encoding="utf-8"
    )
    for header in (
        "Experiment Scope",
        "Model Loading Status",
        "Model-Level Wrapper Configuration",
        "Full Forward Correctness",
        "Prefill / Decode-Step Correctness",
        "Greedy Generation Correctness",
        "KV Cache Invariants",
        "RoPE / GQA Handling",
        "Mitigation Bundle Results",
        "Trace Hook Status",
        "Limitations",
        "Next Stage Plan",
    ):
        assert header in md, f"missing header: {header!r}"
    md_lower = md.lower()
    assert "model-level wrapper" in md_lower
    assert "greedy generation" in md_lower
    assert "not a real tee" in md_lower
    assert "not formal security" in md_lower
    assert "fresh_perm_plus_sandwich_plus_pad" in md


def test_smoke_outputs_no_secret_tensor(tmp_path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    for name in (
        "modern_decoder_model_wrapper_smoke.json",
        "modern_decoder_model_wrapper_smoke.md",
    ):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "tensor(" not in text
        assert _LONG_NUMBER_ARRAY.search(text) is None, name


def test_committed_outputs_present_or_skipped() -> None:
    if not OUTPUT_JSON.exists():
        pytest.skip(
            "outputs/modern_decoder_model_wrapper_smoke.json missing — run "
            "`python scripts/run_modern_decoder_model_smoke.py` first."
        )
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert payload["summary"]["all_full_forward_allclose"] is True
    assert payload["summary"]["all_generation_exact_match"] is True
