"""Optional HuggingFace GPT-2 loader tests."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.model_zoo import (
    ExternalModelConfig,
    get_gpt2_module_spec,
    get_model_loader,
    inspect_model_modules,
)


def _load_tiny_gpt2():
    config = ExternalModelConfig(
        source="huggingface",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        dtype="float32",
    )
    loader = get_model_loader("hf")
    try:
        return loader.load(config)
    except Exception as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable in this environment: {exc}")


def test_hf_tiny_gpt2_loads_tokenizer_and_model() -> None:
    tokenizer, model = _load_tiny_gpt2()
    assert tokenizer is not None
    assert model is not None
    assert model.training is False


def test_hf_tiny_gpt2_plain_forward_logits_shape() -> None:
    tokenizer, model = _load_tiny_gpt2()
    encoded = tokenizer("Hello, my name is", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    assert outputs.logits.ndim == 3
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == encoded["input_ids"].shape[1]


def test_hf_tiny_gpt2_inspection_and_spec() -> None:
    _, model = _load_tiny_gpt2()
    inspection = inspect_model_modules(model)
    spec = get_gpt2_module_spec(model)
    assert "module_type_counts" in inspection
    assert inspection["recognizes_hf_conv1d"] is True
    assert spec["embedding_paths"]["token_embedding"] == "transformer.wte"
    assert spec["attention_projection_paths"][0]["qkv_fused"].endswith("attn.c_attn")
    assert spec["lm_head_path"] == "lm_head"


def test_hf_tiny_gpt2_generation_smoke() -> None:
    tokenizer, model = _load_tiny_gpt2()
    encoded = tokenizer("Hello, my name is", return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    assert generated.shape[0] == 1
    assert generated.shape[1] == encoded["input_ids"].shape[1] + 2
