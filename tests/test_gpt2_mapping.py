"""Tests for GPT-2 linear mapping reports."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from pllo.model_zoo import ExternalModelConfig, get_model_loader
from pllo.model_zoo.gpt2_mapping import build_gpt2_linear_mapping_report


def _load_tiny_gpt2_model():
    config = ExternalModelConfig(
        source="huggingface",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        dtype="float32",
    )
    try:
        _, model = get_model_loader("hf").load(config)
        return model
    except Exception as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable in this environment: {exc}")


def test_build_gpt2_linear_mapping_report_contains_layer_paths() -> None:
    model = _load_tiny_gpt2_model()
    report = build_gpt2_linear_mapping_report(model)
    assert report["num_layers"] == len(model.transformer.h)
    first = report["layers"][0]
    assert first["c_attn"]["path"] == "transformer.h.0.attn.c_attn"
    assert first["c_proj"]["path"] == "transformer.h.0.attn.c_proj"
    assert first["mlp_c_fc"]["path"] == "transformer.h.0.mlp.c_fc"
    assert first["mlp_c_proj"]["path"] == "transformer.h.0.mlp.c_proj"


def test_build_gpt2_linear_mapping_report_tied_embedding_and_limitations() -> None:
    model = _load_tiny_gpt2_model()
    report = build_gpt2_linear_mapping_report(model)
    assert report["tied_embedding"] is True
    assert report["lm_head"]["is_nn_linear"] is True
    assert report["token_embedding"]["path"] == "transformer.wte"
    assert len(report["mapping_limitations"]) >= 4
    assert any("Fused c_attn" in item for item in report["mapping_limitations"])
