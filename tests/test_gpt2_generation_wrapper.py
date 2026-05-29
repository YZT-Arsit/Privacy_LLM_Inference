"""Tests for Stage 4.9 GPT-2 greedy generation correctness."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import (
    sequence_exact_match,
    token_match_rate,
    top1_match_rate,
)
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper, gpt2_cache_invariant_metrics
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
        pytest.skip(f"sshleifer/tiny-gpt2 unavailable in this environment: {exc}")


def _plain_greedy(model, input_ids: torch.Tensor, max_new_tokens: int):
    """Hand-written HF greedy loop (avoids model.generate())."""
    step_logits: list[torch.Tensor] = []
    with torch.no_grad():
        prefill = model(input_ids, use_cache=True)
        last_logits = prefill.logits[:, -1:, :]
        step_logits.append(last_logits)
        next_token = last_logits[:, -1, :].argmax(dim=-1)
        tokens = [next_token]
        past = prefill.past_key_values
        for _ in range(max_new_tokens - 1):
            step = model(next_token.unsqueeze(-1), past_key_values=past, use_cache=True)
            past = step.past_key_values
            step_logits.append(step.logits)
            next_token = step.logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_token)
    generated = torch.cat([input_ids, torch.stack(tokens, dim=1)], dim=1)
    return generated, step_logits


# ---------------------------------------------------------------------------
# HF model integrity guards
# ---------------------------------------------------------------------------


def test_gpt2_generation_hf_c_attn_not_replaced() -> None:
    model = _load_model()
    before = type(model.transformer.h[0].attn.c_attn).__name__
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    _ = wrapper.generate_greedy(prompt, max_new_tokens=2)
    assert type(model.transformer.h[0].attn.c_attn).__name__ == before == "Conv1D"


def test_gpt2_generation_tied_embedding_status_not_broken() -> None:
    """Wrapper must not change lm_head.weight tying status (tied or untied)."""
    model = _load_model()
    was_tied = model.lm_head.weight is model.transformer.wte.weight
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    _ = wrapper.generate_greedy(prompt, max_new_tokens=2)
    is_tied_after = model.lm_head.weight is model.transformer.wte.weight
    assert was_tied == is_tied_after


# ---------------------------------------------------------------------------
# Token-level generation correctness across batch / prompt / max_new_tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "prompt_len", "max_new_tokens"),
    [(1, 4, 1), (1, 4, 3), (2, 8, 4)],
)
@pytest.mark.parametrize("use_pad", [True, False])
def test_gpt2_generation_token_match(
    batch_size: int,
    prompt_len: int,
    max_new_tokens: int,
    use_pad: bool,
) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len))

    plain_generated, _ = _plain_greedy(model, prompt, max_new_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=use_pad)
    with torch.no_grad():
        obf_generated, _trace = wrapper.generate_greedy(prompt, max_new_tokens)

    assert obf_generated.shape == (batch_size, prompt_len + max_new_tokens)
    assert torch.equal(plain_generated, obf_generated), (
        f"token mismatch: plain={plain_generated.tolist()}, obf={obf_generated.tolist()}"
    )
    new_slice = slice(prompt_len, prompt_len + max_new_tokens)
    assert token_match_rate(plain_generated[:, new_slice], obf_generated[:, new_slice]) == 1.0
    assert sequence_exact_match(plain_generated[:, new_slice], obf_generated[:, new_slice]) == 1.0
    assert top1_match_rate(plain_generated, obf_generated) == 1.0


# ---------------------------------------------------------------------------
# Per-step logits correctness during generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_gpt2_generation_per_step_logits_allclose(use_pad: bool) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (2, 8))
    max_new_tokens = 4

    plain_generated, plain_step_logits = _plain_greedy(model, prompt, max_new_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=use_pad)
    with torch.no_grad():
        obf_generated, trace = wrapper.generate_greedy(prompt, max_new_tokens)

    assert torch.equal(plain_generated, obf_generated)
    assert len(plain_step_logits) == len(trace["step_logits"]) == max_new_tokens
    for step, (plain_step, obf_step) in enumerate(
        zip(plain_step_logits, trace["step_logits"])
    ):
        m = compute_correctness_metrics(plain_step, obf_step, atol=1e-4, rtol=1e-4)
        assert m["allclose"] is True, (
            f"step {step}: allclose=False, max_abs_error={m['max_abs_error']}"
        )
        assert m["max_abs_error"] < 1e-4
        assert top1_match_rate(plain_step, obf_step) == 1.0


# ---------------------------------------------------------------------------
# Cache invariant after generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pad", [True, False])
def test_gpt2_generation_cache_invariant_after_generate(use_pad: bool) -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (2, 8))
    max_new_tokens = 4
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=use_pad)
    with torch.no_grad():
        obf_generated, trace = wrapper.generate_greedy(prompt, max_new_tokens)

    cache = trace["final_cache"]
    metrics = gpt2_cache_invariant_metrics(cache, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True, f"cache invariant failed: {metrics}"
    assert metrics["max_key_error"] < 1e-4
    assert metrics["max_value_error"] < 1e-4
    # Generation extends the cache by (max_new_tokens - 1) decode steps; the
    # final new token is selected from prefill output and never enters the cache.
    expected_seq = prompt.shape[1] + (max_new_tokens - 1)
    assert cache.seq_len == expected_seq
    assert obf_generated.shape == (2, prompt.shape[1] + max_new_tokens)


# ---------------------------------------------------------------------------
# Boundary: max_new_tokens=1 must not call decode_step
# ---------------------------------------------------------------------------


def test_gpt2_generation_max_new_tokens_one_skips_decode() -> None:
    model = _load_model("float32")
    vocab_size = model.config.vocab_size
    prompt = torch.randint(0, vocab_size, (1, 4))
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=True)
    with torch.no_grad():
        generated, trace = wrapper.generate_greedy(prompt, max_new_tokens=1)
    assert generated.shape == (1, 5)
    assert trace["cache_seq_len"] == prompt.shape[1]
    assert len(trace["step_logits"]) == 1


def test_gpt2_generation_invalid_max_new_tokens() -> None:
    model = _load_model("float32")
    wrapper = ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, use_pad=False)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        wrapper.generate_greedy(prompt, max_new_tokens=0)
