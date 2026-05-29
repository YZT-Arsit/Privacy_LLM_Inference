"""Tests for the Stage 6.0 architecture coverage script + inspector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from pllo.architectures import (
    ArchitectureType,
    inspect_architecture,
    load_for_architecture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_architecture_coverage.py"


def _try_load(key: str):
    try:
        return load_for_architecture(key)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{key} unavailable: {type(exc).__name__}: {str(exc)[:80]}")


# ---------------------------------------------------------------------------
# Inspector classification on each architecture family
# ---------------------------------------------------------------------------


def test_inspector_classifies_gpt2_as_decoder_only() -> None:
    model_id, model = _try_load("decoder_only")
    spec = inspect_architecture(model, model_id=model_id)
    assert spec.architecture_type is ArchitectureType.DECODER_ONLY
    assert spec.has_decoder is True
    assert spec.has_encoder is False
    assert spec.has_cross_attention is False
    assert spec.has_causal_self_attention is True
    assert spec.has_bidirectional_self_attention is False
    assert spec.supports_past_key_values is True
    assert spec.has_lm_head is True
    assert spec.num_layers is not None and spec.num_layers > 0
    assert spec.num_heads is not None and spec.num_heads > 0


def test_inspector_classifies_bert_as_encoder_only() -> None:
    model_id, model = _try_load("encoder_only")
    spec = inspect_architecture(model, model_id=model_id)
    assert spec.architecture_type is ArchitectureType.ENCODER_ONLY
    assert spec.has_encoder is True
    assert spec.has_decoder is False
    assert spec.has_cross_attention is False
    assert spec.has_bidirectional_self_attention is True
    assert spec.has_causal_self_attention is False
    # BertForMaskedLM exposes the MLM head; AutoModel-only checkpoints may not.
    assert spec.has_mlm_head or "Masked" in spec.model_class or not spec.has_lm_head


def test_inspector_classifies_t5_or_bart_as_encoder_decoder() -> None:
    model_id, model = _try_load("encoder_decoder")
    spec = inspect_architecture(model, model_id=model_id)
    assert spec.architecture_type is ArchitectureType.ENCODER_DECODER
    assert spec.has_encoder is True
    assert spec.has_decoder is True
    assert spec.has_cross_attention is True
    assert spec.has_causal_self_attention is True  # decoder has causal masking
    assert spec.has_bidirectional_self_attention is True  # encoder has bidirectional
    # Both T5 and BART support past_key_values for decoder.
    assert spec.supports_past_key_values is True


# ---------------------------------------------------------------------------
# spec_to_dict serialisation
# ---------------------------------------------------------------------------


def test_spec_to_dict_serialises_enum_as_string() -> None:
    from pllo.architectures import spec_to_dict

    model_id, model = _try_load("decoder_only")
    spec = inspect_architecture(model, model_id=model_id)
    payload = spec_to_dict(spec)
    assert payload["architecture_type"] == "decoder_only"
    # Must be JSON-serialisable.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Script end-to-end (skipped-models tolerated)
# ---------------------------------------------------------------------------


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    """End-to-end smoke: the script writes JSON/CSV/Markdown even if some
    models are unavailable. Run with explicit overrides so failures map to
    `skipped` rows rather than crashing the pytest worker."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    for filename in (
        "architecture_coverage.json",
        "architecture_coverage.csv",
        "architecture_coverage.md",
    ):
        assert (tmp_path / filename).exists(), filename
    payload = json.loads(
        (tmp_path / "architecture_coverage.json").read_text(encoding="utf-8")
    )
    # Every architecture key appears, whether loaded or skipped.
    keys = {entry["architecture_key"] for entry in payload["coverage"]}
    assert keys == {"decoder_only", "encoder_only", "encoder_decoder"}
    # Three attention kinds present.
    taxonomy_names = {k["name"] for k in payload["attention_taxonomy"]}
    assert taxonomy_names == {
        "causal_self_attention",
        "bidirectional_self_attention",
        "cross_attention",
    }
    md = (tmp_path / "architecture_coverage.md").read_text(encoding="utf-8")
    assert "Attention taxonomy" in md
    assert "Required invariants" in md
    assert "Q_tilde K_tilde^T = Q K^T" in md
    assert "Q_dec_tilde K_enc_tilde^T" in md
    assert "summary=" in result.stdout


def test_script_handles_unresolvable_model_id_gracefully(tmp_path) -> None:
    """A bogus override must produce a `skipped` row instead of crashing."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--encoder-model-id",
            "does/not/exist-stage6-bogus",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(
        (tmp_path / "architecture_coverage.json").read_text(encoding="utf-8")
    )
    encoder_entry = next(
        e for e in payload["coverage"] if e["architecture_key"] == "encoder_only"
    )
    assert encoder_entry["status"] == "skipped"
    assert "does/not/exist-stage6-bogus" in (encoder_entry.get("model_id") or "")
