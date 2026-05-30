"""Stage 5.3a — feature-flagged GPT-2 single-block compatible-island integration."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from pllo.evaluation import compute_correctness_metrics
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    VALID_NONLINEAR_MODES,
)
from pllo.model_zoo import ExternalModelConfig, get_model_loader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _first_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _plain_block(block, hidden_states: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _first_hidden(block(hidden_states, use_cache=False))


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------


def test_valid_modes_includes_trusted_and_compatible() -> None:
    assert "trusted" in VALID_NONLINEAR_MODES
    assert "compatible_islands" in VALID_NONLINEAR_MODES
    assert DEFAULT_NONLINEAR_MODE == "trusted"


def test_invalid_mode_rejected() -> None:
    model = _load_model("float32")
    block = model.transformer.h[0]
    with pytest.raises(ValueError):
        ObfuscatedGPT2BlockWrapper(
            block,
            model.config,
            dtype=torch.float32,
            use_pad=False,
            nonlinear_mode="not_a_mode",
        )


# ---------------------------------------------------------------------------
# Default behaviour preservation
# ---------------------------------------------------------------------------


def test_default_mode_matches_existing_behavior() -> None:
    """Not passing nonlinear_mode must be byte-for-byte identical to trusted."""
    torch.manual_seed(0)
    model = _load_model("float32")
    block = model.transformer.h[0]
    hidden_states = torch.randn(2, 6, model.config.n_embd, dtype=torch.float32)

    torch.manual_seed(123)
    wrapper_default = ObfuscatedGPT2BlockWrapper(
        block, model.config, dtype=torch.float32, use_pad=False
    )
    with torch.no_grad():
        recovered_default = wrapper_default.forward(hidden_states)

    torch.manual_seed(123)
    wrapper_trusted = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="trusted",
    )
    with torch.no_grad():
        recovered_trusted = wrapper_trusted.forward(hidden_states)

    assert wrapper_default.nonlinear_mode == "trusted"
    assert torch.equal(recovered_default, recovered_trusted)


def test_trusted_mode_island_report_inactive() -> None:
    torch.manual_seed(0)
    model = _load_model("float32")
    block = model.transformer.h[0]
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="trusted",
    )
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        wrapper.forward(hidden_states)
    rep = wrapper.island_report
    assert rep["nonlinear_mode"] == "trusted"
    assert rep["mlp_gelu_island_active"] is False
    assert rep["mlp_island_permutation_dim"] is None
    assert rep["security_profile"] == "n/a"


# ---------------------------------------------------------------------------
# Compatible-island correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_compatible_island_single_block_matches_plain_no_pad(
    batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(7)
    model = _load_model("float32")
    block = model.transformer.h[0]
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.n_embd, dtype=torch.float32
    )

    plain = _plain_block(block, hidden_states)
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert metrics["max_abs_error"] < 1e-4


@pytest.mark.parametrize(("batch_size", "seq_len"), [(1, 4), (2, 8)])
def test_compatible_island_single_block_matches_plain_with_pad(
    batch_size: int, seq_len: int
) -> None:
    torch.manual_seed(11)
    model = _load_model("float32")
    block = model.transformer.h[0]
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.n_embd, dtype=torch.float32
    )

    plain = _plain_block(block, hidden_states)
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=True,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True


def test_compatible_island_float64_matches_plain() -> None:
    torch.manual_seed(13)
    model = _load_model("float64")
    block = model.transformer.h[0]
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float64)
    plain = _plain_block(block, hidden_states)
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float64,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(hidden_states)
    metrics = compute_correctness_metrics(plain, recovered, atol=1e-8, rtol=1e-6)
    assert metrics["allclose"] is True


# ---------------------------------------------------------------------------
# Audit-report invariants
# ---------------------------------------------------------------------------


def _run_island(model, use_pad: bool) -> ObfuscatedGPT2BlockWrapper:
    torch.manual_seed(17)
    block = model.transformer.h[0]
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
    )
    hidden_states = torch.randn(2, 6, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        wrapper.forward(hidden_states)
    return wrapper


def test_compatible_island_mlp_intermediate_permutation_dimension() -> None:
    model = _load_model("float32")
    wrapper = _run_island(model, use_pad=False)
    rep = wrapper.island_report
    inter = wrapper.block.mlp.c_fc.weight.shape[1]
    # GPT-2 Conv1D weight: [d_in, d_out]; c_fc out dim is intermediate_size.
    assert rep["mlp_island_permutation_dim"] == inter
    assert rep["mlp_island_intermediate_size"] == inter
    # Crucially, must NOT equal hidden_size (only when inter != hidden).
    assert rep["mlp_island_permutation_dim"] != model.config.n_embd


def test_compatible_island_online_extra_matmul_zero() -> None:
    model = _load_model("float32")
    wrapper = _run_island(model, use_pad=True)
    assert wrapper.island_report["online_extra_matmul_count"] == 0


def test_compatible_island_does_not_push_pad_through_gelu() -> None:
    model = _load_model("float32")
    wrapper = _run_island(model, use_pad=True)
    rep = wrapper.island_report
    assert rep["mlp_gelu_island_active"] is True
    assert rep["mlp_island_pad_placement"] == "linear_boundary_only"


def test_compatible_island_uses_fresh_permutation_per_call() -> None:
    model = _load_model("float32")
    block = model.transformer.h[0]
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    torch.manual_seed(19)
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        wrapper.forward(hidden_states)
    first_draws = int(wrapper.island_report["mlp_island_permutation_draws"])
    with torch.no_grad():
        wrapper.forward(hidden_states)
    # Each forward resets the report and draws exactly one fresh permutation.
    assert first_draws == 1
    assert int(wrapper.island_report["mlp_island_permutation_draws"]) == 1
    assert wrapper.island_report["mlp_island_uses_fresh_permutation"] is True


def test_compatible_island_reports_security_caveats() -> None:
    model = _load_model("float32")
    wrapper = _run_island(model, use_pad=True)
    rep = wrapper.island_report
    assert rep["security_profile"] == "proxy-evaluated, not formal"
    text = " ".join(rep["security_caveats"])
    assert "Compatible mask families are weaker" in text
    assert "Fresh permutation" in text and "dense sandwich" in text.lower()


# ---------------------------------------------------------------------------
# Co-existence
# ---------------------------------------------------------------------------


def test_trusted_and_compatible_modes_both_available() -> None:
    torch.manual_seed(23)
    model = _load_model("float32")
    block = model.transformer.h[0]
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    plain = _plain_block(block, hidden_states)

    wrapper_trusted = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="trusted",
    )
    wrapper_islands = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered_trusted = wrapper_trusted.forward(hidden_states)
        recovered_islands = wrapper_islands.forward(hidden_states)
    assert compute_correctness_metrics(plain, recovered_trusted, atol=1e-4, rtol=1e-4)["allclose"]
    assert compute_correctness_metrics(plain, recovered_islands, atol=1e-4, rtol=1e-4)["allclose"]


# ---------------------------------------------------------------------------
# Generation / KV-cache path is untouched
# ---------------------------------------------------------------------------


def test_compatible_island_block_does_not_replace_hf_modules() -> None:
    model = _load_model("float32")
    block = model.transformer.h[0]
    before_attn = block.attn.c_attn.__class__.__name__
    before_mlp = block.mlp.c_fc.__class__.__name__
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        nonlinear_mode="compatible_islands",
    )
    torch.manual_seed(29)
    hidden_states = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    with torch.no_grad():
        wrapper.forward(hidden_states)
    assert block.attn.c_attn.__class__.__name__ == before_attn
    assert block.mlp.c_fc.__class__.__name__ == before_mlp
    # Generation path uses the model wrapper, which we explicitly did not
    # alter; the block wrapper exposes prefill / decode_step and they
    # continue to work under compatible_islands (covered by the next test).


def test_compatible_island_prefill_and_decode_recover_plain() -> None:
    """Block-level prefill + decode_step must still recover plain semantics."""
    torch.manual_seed(31)
    model = _load_model("float32")
    block = model.transformer.h[0]

    prompt = torch.randn(1, 4, model.config.n_embd, dtype=torch.float32)
    new_step = torch.randn(1, 1, model.config.n_embd, dtype=torch.float32)
    plain_concat = torch.cat([prompt, new_step], dim=1)
    plain_full = _plain_block(block, plain_concat)

    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model.config,
        dtype=torch.float32,
        use_pad=False,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        prefill_hidden, cache = wrapper.prefill(prompt)
        decode_hidden, _ = wrapper.decode_step(new_step, cache)
    recovered = torch.cat([prefill_hidden, decode_hidden], dim=1)
    metrics = compute_correctness_metrics(plain_full, recovered, atol=1e-4, rtol=1e-4)
    assert metrics["allclose"] is True
    assert wrapper.island_report["mlp_gelu_island_active"] is True
