"""Stage 5.3e — GPT-2 wrapper + dense-sandwich mitigation bundle integration tests."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import (
    sequence_exact_match,
    token_match_rate,
)
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper, ObfuscatedGPT2ModelWrapper
from pllo.ops.mitigation_bundles import DEFAULT_MITIGATION_BUNDLE
from pllo.model_zoo import ExternalModelConfig, get_model_loader


def _load_model(dtype: str = "float32"):
    config = ExternalModelConfig(
        source="huggingface",
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
        dtype=dtype,
    )
    try:
        _, model = get_model_loader("hf").load(config)
        return model
    except Exception as exc:
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable: {exc}")


# ---------------------------------------------------------------------------
# Acceptance of the bundle argument
# ---------------------------------------------------------------------------


def test_model_wrapper_accepts_mitigation_bundle() -> None:
    model = _load_model()
    for bundle in ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"):
        wrapper = ObfuscatedGPT2ModelWrapper(
            model,
            dtype=torch.float32,
            use_pad=True,
            nonlinear_mode="compatible_islands",
            mitigation_bundle=bundle,
        )
        assert wrapper.mitigation_bundle == bundle


def test_block_wrapper_accepts_mitigation_bundle() -> None:
    model = _load_model()
    for bundle in ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"):
        wrapper = ObfuscatedGPT2BlockWrapper(
            model.transformer.h[0],
            model.config,
            dtype=torch.float32,
            use_pad=True,
            nonlinear_mode="compatible_islands",
            mitigation_bundle=bundle,
        )
        assert wrapper.mitigation_bundle == bundle


def test_invalid_bundle_raises() -> None:
    model = _load_model()
    with pytest.raises(ValueError):
        ObfuscatedGPT2ModelWrapper(
            model, dtype=torch.float32, mitigation_bundle="not_a_bundle"
        )


def test_default_mitigation_bundle_is_fresh_perm_only() -> None:
    model = _load_model()
    w = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32)
    assert w.mitigation_bundle == DEFAULT_MITIGATION_BUNDLE == "fresh_perm_only"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_trusted_mode_unaffected_by_mitigation_bundle() -> None:
    """``nonlinear_mode='trusted'`` outputs must NOT depend on the bundle arg."""
    torch.manual_seed(0)
    model = _load_model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))

    torch.manual_seed(42)
    a = ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, use_pad=False, nonlinear_mode="trusted",
        mitigation_bundle="fresh_perm_only",
    ).forward(prompt)
    torch.manual_seed(42)
    b = ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, use_pad=False, nonlinear_mode="trusted",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    ).forward(prompt)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Compatible islands + full bundle correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [False, True])
def test_compatible_islands_full_bundle_forward_allclose(use_pad: bool) -> None:
    torch.manual_seed(7)
    model = _load_model()
    prompt = torch.randint(0, model.config.vocab_size, (2, 6))
    plain = model(prompt).logits
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    with torch.no_grad():
        rec = wrapper.forward(prompt)
    m = compute_correctness_metrics(plain, rec, atol=1e-4, rtol=1e-4)
    assert m["allclose"] is True


@pytest.mark.parametrize("use_pad", [False, True])
def test_compatible_islands_full_bundle_greedy_token_match(use_pad: bool) -> None:
    torch.manual_seed(11)
    model = _load_model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    # Hand-written plain greedy.
    with torch.no_grad():
        out = model(prompt, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(dim=-1)
        past = out.past_key_values
        toks = [nxt]
        for _ in range(2):
            step = model(nxt.unsqueeze(-1), past_key_values=past, use_cache=True)
            past = step.past_key_values
            nxt = step.logits[:, -1, :].argmax(dim=-1)
            toks.append(nxt)
    plain_gen = torch.cat([prompt, torch.stack(toks, dim=1)], dim=1)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    with torch.no_grad():
        obf_gen, _ = wrapper.generate_greedy(prompt, max_new_tokens=3)
    new = slice(4, 7)
    assert torch.equal(plain_gen, obf_gen)
    assert token_match_rate(plain_gen[:, new], obf_gen[:, new]) == 1.0
    assert sequence_exact_match(plain_gen[:, new], obf_gen[:, new]) == 1.0


# ---------------------------------------------------------------------------
# island_summary metadata
# ---------------------------------------------------------------------------


def test_island_summary_records_full_bundle() -> None:
    torch.manual_seed(13)
    model = _load_model()
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        wrapper.forward(prompt)
    s = wrapper.island_summary
    assert s["mitigation_bundle"] == "fresh_perm_plus_sandwich_plus_pad"
    assert s["dense_sandwich_enabled"] is True
    assert s["fresh_permutation_enabled"] is True
    assert s["boundary_pad_enabled"] is True
    assert s["default_on_candidate_under_stage_5_4"] is True
    assert s["online_extra_matmul_count"] == 0


def test_island_summary_records_default_bundle() -> None:
    torch.manual_seed(17)
    model = _load_model()
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        wrapper.forward(prompt)
    s = wrapper.island_summary
    assert s["mitigation_bundle"] == "fresh_perm_only"
    assert s["dense_sandwich_enabled"] is False
    assert s["default_on_candidate_under_stage_5_4"] is False


def test_block_wrapper_default_island_report_is_fresh_perm_only() -> None:
    model = _load_model()
    w = ObfuscatedGPT2BlockWrapper(
        model.transformer.h[0], model.config, dtype=torch.float32
    )
    r = w.island_report
    assert r["mitigation_bundle"] == "fresh_perm_only"
    assert r["dense_sandwich_enabled"] is False
