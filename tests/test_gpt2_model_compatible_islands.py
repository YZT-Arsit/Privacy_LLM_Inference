"""Stage 5.3b — GPT-2 model-level compatible nonlinear island integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import (
    sequence_exact_match,
    token_match_rate,
    top1_match_rate,
)
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper
from pllo.hf_wrappers.nonlinear_modes import DEFAULT_NONLINEAR_MODE
from pllo.model_zoo import ExternalModelConfig, get_model_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_PROFILE_JSON = PROJECT_ROOT / "outputs" / "workload_profile.json"


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


def _plain_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits


def _plain_greedy(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    with torch.no_grad():
        prefill = model(input_ids, use_cache=True)
        next_token = prefill.logits[:, -1, :].argmax(dim=-1)
        past = prefill.past_key_values
        tokens = [next_token]
        for _ in range(max_new_tokens - 1):
            step = model(next_token.unsqueeze(-1), past_key_values=past, use_cache=True)
            past = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_token)
    return torch.cat([input_ids, torch.stack(tokens, dim=1)], dim=1)


# ---------------------------------------------------------------------------
# Mode acceptance
# ---------------------------------------------------------------------------


def test_model_wrapper_accepts_nonlinear_mode() -> None:
    model = _load_model("float32")
    ObfuscatedGPT2ModelWrapper(model, dtype=torch.float32, nonlinear_mode="trusted")
    ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, nonlinear_mode="compatible_islands"
    )
    with pytest.raises(ValueError):
        ObfuscatedGPT2ModelWrapper(
            model, dtype=torch.float32, nonlinear_mode="not_a_mode"
        )


def test_default_model_mode_is_trusted() -> None:
    torch.manual_seed(0)
    model = _load_model("float32")
    prompt = torch.randint(0, model.config.vocab_size, (2, 6))

    torch.manual_seed(321)
    wrapper_default = ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, use_pad=False
    )
    with torch.no_grad():
        logits_default = wrapper_default.forward(prompt)

    torch.manual_seed(321)
    wrapper_trusted = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode=DEFAULT_NONLINEAR_MODE,
    )
    with torch.no_grad():
        logits_trusted = wrapper_trusted.forward(prompt)

    assert wrapper_default.nonlinear_mode == "trusted"
    assert torch.equal(logits_default, logits_trusted)


# ---------------------------------------------------------------------------
# Full forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_compatible_model_forward_matches_plain_no_pad(
    batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(41)
    model = _load_model("float32")
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    plain = _plain_logits(model, input_ids)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(input_ids)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert top1_match_rate(plain, recovered) == 1.0


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_compatible_model_forward_matches_plain_with_pad(
    batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(43)
    model = _load_model("float32")
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    plain = _plain_logits(model, input_ids)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(input_ids)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert top1_match_rate(plain, recovered) == 1.0


# ---------------------------------------------------------------------------
# Greedy generation correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "prompt_len", "max_new_tokens"),
    [(1, 4, 3), (2, 8, 4)],
)
def test_compatible_model_greedy_generation_matches_plain_no_pad(
    batch_size: int, prompt_len: int, max_new_tokens: int
) -> None:
    torch.manual_seed(47)
    model = _load_model("float32")
    prompt = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len))
    plain_generated = _plain_greedy(model, prompt, max_new_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        obf_generated, _ = wrapper.generate_greedy(prompt, max_new_tokens)
    assert obf_generated.shape == (batch_size, prompt_len + max_new_tokens)
    new_slice = slice(prompt_len, prompt_len + max_new_tokens)
    assert torch.equal(plain_generated, obf_generated)
    assert (
        token_match_rate(
            plain_generated[:, new_slice], obf_generated[:, new_slice]
        )
        == 1.0
    )
    assert (
        sequence_exact_match(
            plain_generated[:, new_slice], obf_generated[:, new_slice]
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("batch_size", "prompt_len", "max_new_tokens"),
    [(1, 4, 3), (2, 8, 4)],
)
def test_compatible_model_greedy_generation_matches_plain_with_pad(
    batch_size: int, prompt_len: int, max_new_tokens: int
) -> None:
    torch.manual_seed(53)
    model = _load_model("float32")
    prompt = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len))
    plain_generated = _plain_greedy(model, prompt, max_new_tokens)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        obf_generated, _ = wrapper.generate_greedy(prompt, max_new_tokens)
    new_slice = slice(prompt_len, prompt_len + max_new_tokens)
    assert torch.equal(plain_generated, obf_generated)
    assert (
        token_match_rate(
            plain_generated[:, new_slice], obf_generated[:, new_slice]
        )
        == 1.0
    )


# ---------------------------------------------------------------------------
# Island summary
# ---------------------------------------------------------------------------


def test_compatible_model_collects_island_summary() -> None:
    torch.manual_seed(57)
    model = _load_model("float32")
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    prompt = torch.randint(0, model.config.vocab_size, (2, 6))
    with torch.no_grad():
        wrapper.forward(prompt)
    summary = wrapper.island_summary
    num_blocks = len(model.transformer.h)
    assert summary["nonlinear_mode"] == "compatible_islands"
    assert summary["num_blocks"] == num_blocks
    assert summary["blocks_with_compatible_islands"] == num_blocks
    # One forward = one fresh permutation draw per block.
    assert summary["total_mlp_island_permutation_draws"] >= num_blocks
    assert summary["online_extra_matmul_count"] == 0
    assert summary["layernorm_remains_trusted"] is True
    assert summary["lm_head_not_modified"] is True
    assert summary["generation_path_not_modified"] is True
    assert summary["pad_placement"] == "linear_boundary_only"
    assert summary["wrapper_integration_scope"] == "gpt2_model_level"
    assert summary["security_profile"] == "proxy-evaluated, not formal"


def test_compatible_model_no_pad_reports_linear_boundary_inactive() -> None:
    torch.manual_seed(59)
    model = _load_model("float32")
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        wrapper.forward(prompt)
    summary = wrapper.island_summary
    assert summary["pad_placement"] == "n/a"
    assert summary["online_extra_matmul_count"] == 0


def test_trusted_model_summary_inactive() -> None:
    torch.manual_seed(61)
    model = _load_model("float32")
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="trusted",
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        wrapper.forward(prompt)
    summary = wrapper.island_summary
    assert summary["nonlinear_mode"] == "trusted"
    assert summary["blocks_with_compatible_islands"] == 0
    assert summary["total_mlp_island_permutation_draws"] == 0
    assert summary["security_profile"] == "n/a"
    assert summary["security_caveats"] == []


# ---------------------------------------------------------------------------
# Co-existence and module-integrity guards
# ---------------------------------------------------------------------------


def test_trusted_and_compatible_modes_both_available() -> None:
    torch.manual_seed(67)
    model = _load_model("float32")
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    plain = _plain_logits(model, prompt)

    wrapper_trusted = ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, use_pad=False, nonlinear_mode="trusted"
    )
    wrapper_islands = ObfuscatedGPT2ModelWrapper(
        model, dtype=torch.float32, use_pad=False, nonlinear_mode="compatible_islands"
    )
    with torch.no_grad():
        rec_trusted = wrapper_trusted.forward(prompt)
        rec_islands = wrapper_islands.forward(prompt)
    assert compute_correctness_metrics(plain, rec_trusted, atol=1e-4, rtol=1e-4)[
        "allclose"
    ]
    assert compute_correctness_metrics(plain, rec_islands, atol=1e-4, rtol=1e-4)[
        "allclose"
    ]
    assert wrapper_trusted.island_summary["nonlinear_mode"] == "trusted"
    assert wrapper_islands.island_summary["nonlinear_mode"] == "compatible_islands"


def test_model_level_wrapper_does_not_replace_hf_modules() -> None:
    model = _load_model("float32")
    before_attn = type(model.transformer.h[0].attn.c_attn).__name__
    before_mlp_fc = type(model.transformer.h[0].mlp.c_fc).__name__
    before_mlp_proj = type(model.transformer.h[0].mlp.c_proj).__name__
    before_ln1 = type(model.transformer.h[0].ln_1).__name__
    before_ln_f = type(model.transformer.ln_f).__name__
    before_lm_head = type(model.lm_head).__name__

    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        wrapper.forward(prompt)
        wrapper.generate_greedy(prompt, max_new_tokens=2)

    assert type(model.transformer.h[0].attn.c_attn).__name__ == before_attn
    assert type(model.transformer.h[0].mlp.c_fc).__name__ == before_mlp_fc
    assert type(model.transformer.h[0].mlp.c_proj).__name__ == before_mlp_proj
    assert type(model.transformer.h[0].ln_1).__name__ == before_ln1
    assert type(model.transformer.ln_f).__name__ == before_ln_f
    assert type(model.lm_head).__name__ == before_lm_head


# ---------------------------------------------------------------------------
# Workload profiler integration status
# ---------------------------------------------------------------------------


def test_workload_profile_records_gpt2_model_level_integration() -> None:
    if not WORKLOAD_PROFILE_JSON.exists():
        pytest.skip(
            "outputs/workload_profile.json missing — run "
            "`python scripts/run_workload_profile.py` first."
        )
    payload = json.loads(WORKLOAD_PROFILE_JSON.read_text(encoding="utf-8"))
    top = payload.get("wrapper_integration_status", {}).get(
        "ours_compatible_nonlinear_islands"
    )
    assert top is not None
    assert top["gpt2_single_block"] == "implemented"
    assert top["gpt2_model_level"] == "implemented"
    # Stage 5.3c moved BERT/T5 from "not_yet" → "implemented_probe_level".
    assert top["bert"] in ("implemented_probe_level", "not_yet")
    assert top["t5"] in ("implemented_probe_level", "not_yet")
    method = payload["methods"]["ours_compatible_nonlinear_islands"]
    # Method record mirrors top-level status.
    method_status = method.get("wrapper_integration_status")
    assert method_status is not None
    assert method_status["gpt2_model_level"] == "implemented"
    assert method.get("partial_implementation") is True
    assert method["implemented"] is False  # Probe-level only, not full runtime.
    assert method["wall_time_source"] == "projected_from_op_counts"
    # measured_integration_scope was "gpt2_model_level" in 5.3b and is
    # "cross_architecture_probe_level" in 5.3c. Either is acceptable.
    assert method.get("measured_integration_scope") in (
        "gpt2_model_level",
        "cross_architecture_probe_level",
    )
