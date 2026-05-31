"""Stage 5.3e — BERT / T5 / modern decoder dense-sandwich probe tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROSS_ARCH_JSON = PROJECT_ROOT / "outputs" / "cross_architecture_summary.json"
CROSS_ARCH_MD = PROJECT_ROOT / "outputs" / "cross_architecture_summary.md"
SCRIPT = PROJECT_ROOT / "scripts" / "run_cross_architecture_compatible_island_smoke.py"


# ---------------------------------------------------------------------------
# BERT FFN probe with full bundle
# ---------------------------------------------------------------------------


def test_bert_ffn_full_bundle_allclose() -> None:
    pytest.importorskip("transformers")
    from pllo.experiments.encoder_ffn_island_probe import (
        EncoderFFNIslandProbeConfig,
        run_encoder_ffn_island_probe,
    )
    for use_pad in (False, True):
        r = run_encoder_ffn_island_probe(
            EncoderFFNIslandProbeConfig(
                batch_size=2, seq_len=8, use_pad=use_pad,
                nonlinear_mode="compatible_islands",
                mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
                seed=1,
            )
        )
        if r.get("status") == "skipped":
            pytest.skip(f"BERT unavailable: {r.get('reason')}")
        assert r["status"] == "loaded"
        assert r["ffn_metrics"]["allclose"] is True
        assert r["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"
        assert r["dense_sandwich_enabled"] is True
        assert r["boundary_pad_enabled"] is use_pad
        assert r["online_extra_matmul_count"] == 0
        assert r["default_on_candidate_under_stage_5_4"] is use_pad


# ---------------------------------------------------------------------------
# T5 FFN probe with full bundle
# ---------------------------------------------------------------------------


def test_t5_ffn_full_bundle_allclose_or_explicit_unsupported() -> None:
    pytest.importorskip("transformers")
    from pllo.experiments.encoder_decoder_ffn_island_probe import (
        EncoderDecoderFFNIslandProbeConfig,
        run_encoder_decoder_ffn_island_probe,
    )
    for use_pad in (False, True):
        r = run_encoder_decoder_ffn_island_probe(
            EncoderDecoderFFNIslandProbeConfig(
                batch_size=2, seq_len=8, use_pad=use_pad,
                nonlinear_mode="compatible_islands",
                mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
                seed=1,
            )
        )
        if r.get("status") in ("skipped", "unsupported"):
            assert r.get("reason"), "skip/unsupported must carry explicit reason"
            continue
        assert r["status"] == "loaded"
        assert r["ffn_metrics"]["allclose"] is True
        assert r["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"
        assert r["dense_sandwich_enabled"] is True
        assert r["boundary_pad_enabled"] is use_pad
        assert r["online_extra_matmul_count"] == 0


# ---------------------------------------------------------------------------
# Modern decoder SwiGLU probe with full bundle
# ---------------------------------------------------------------------------


def test_modern_decoder_swiglu_full_bundle_allclose() -> None:
    from pllo.experiments.modern_decoder_probe import (
        ModernDecoderProbeConfig,
        run_modern_decoder_probe,
    )
    r = run_modern_decoder_probe(
        ModernDecoderProbeConfig(
            batch_size=2, seq_len=8,
            hidden_size=64, intermediate_size=128,
            num_query_heads=4, num_kv_heads=2, head_dim=16,
            mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
            attempt_real_model_load=False,
        )
    )
    swi = r["swiglu_probe"]
    assert swi["allclose"] is True
    assert swi["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"
    for use_pad_key, entry in swi["per_use_pad"].items():
        assert entry["dense_sandwich_enabled"] is True
        assert entry["online_extra_matmul_count"] == 0
        assert entry["activation_input_form"] == "ZP"
    g = r["global_summary"]
    assert g["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"


# ---------------------------------------------------------------------------
# Cross-architecture smoke includes both bundles when --both-bundles
# ---------------------------------------------------------------------------


def test_cross_architecture_smoke_both_bundles(tmp_path) -> None:
    pytest.importorskip("transformers")
    # Source GPT-2 smoke does not need to exist (script handles "missing").
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--seed",
            "2026",
            "--both-bundles",
            "--gpt2-smoke-json",
            str(tmp_path / "missing.json"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    js = json.loads(
        (tmp_path / "cross_architecture_compatible_island_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(js["mitigation_bundles_evaluated"]) == {
        "fresh_perm_only",
        "fresh_perm_plus_sandwich_plus_pad",
    }
    runs = js["mitigation_bundle_runs"]
    for bundle, bundles in runs.items():
        assert bundle in {"fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"}
        # BERT probe present.
        bert = bundles["encoder_only"]
        # T5 probe present.
        t5 = bundles["encoder_decoder"]
        for r in bert.get("runs", []) + t5.get("runs", []):
            assert r.get("mitigation_bundle") == bundle


# ---------------------------------------------------------------------------
# Cross-architecture summary surface
# ---------------------------------------------------------------------------


def test_cross_architecture_summary_records_mitigation_bundle_support() -> None:
    if not CROSS_ARCH_JSON.exists():
        pytest.skip(
            "outputs/cross_architecture_summary.json missing — run "
            "`python scripts/run_cross_architecture_summary.py` first."
        )
    payload = json.loads(CROSS_ARCH_JSON.read_text(encoding="utf-8"))
    integ = payload["compatible_island_integration_status"]
    assert integ.get("mitigation_bundle_selectable") is True
    assert integ.get("default_mitigation_bundle") == "fresh_perm_only"
    assert (
        integ.get("recommended_default_on_bundle")
        == "fresh_perm_plus_sandwich_plus_pad"
    )
    support = integ.get("mitigation_bundle_support") or []
    assert len(support) >= 3
    archs = {row["architecture"] for row in support}
    # At least the three baseline architecture rows are present.
    assert {"decoder_only", "encoder_only", "encoder_decoder"} <= archs
    for row in support:
        assert row["fresh_perm_only"] == "supported"
        assert row["fresh_perm_plus_sandwich_plus_pad"] == "supported"
        assert row["dense_sandwich_enabled"] is True
        assert row["online_extra_matmul_count"] == 0


def test_cross_architecture_summary_markdown_has_mitigation_bundle_table() -> None:
    if not CROSS_ARCH_MD.exists():
        pytest.skip("outputs/cross_architecture_summary.md missing")
    md = CROSS_ARCH_MD.read_text(encoding="utf-8")
    assert "Mitigation Bundle Support" in md
    assert "fresh_perm_plus_sandwich_plus_pad" in md
    assert "default_mitigation_bundle" in md
    assert "recommended_default_on_bundle" in md
    assert "adaptive-proxy-mitigated, not formal" in md
    assert "not a real TEE measurement" in md
