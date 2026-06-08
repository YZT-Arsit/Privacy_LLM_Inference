"""Stage 7.6e -- tests for the padded modern-decoder full-generation
correctness wrapper.

All tests are CPU-only, float64, deterministic under fixed seeds, and
run without any network access or HuggingFace download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pllo.experiments.modern_decoder_generation_correctness import (
    GenerationCorrectnessConfig,
    KV_CACHE_TOLERANCE,
    LOGITS_TOLERANCE,
    run_modern_decoder_generation_correctness,
    write_reports,
)
from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
    apply_rope,
)
from pllo.wrappers.padded_modern_decoder_generation_wrapper import (
    PaddedMaskedGenerationDiagnostics,
    PaddedMaskedTinyModernDecoderWrapper,
    apply_padded_linear,
    sample_invertible_mask,
    sample_pad_like,
    tensor_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_model() -> TinyModernDecoderForCausalLM:
    torch.manual_seed(2026)
    cfg = TinyModernDecoderConfig()
    m = TinyModernDecoderForCausalLM(cfg)
    m.init_random_weights(torch.Generator(device="cpu").manual_seed(2026))
    return m


@pytest.fixture(scope="module")
def prompt_ids(tiny_model: TinyModernDecoderForCausalLM) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(2027)
    return torch.randint(
        0, tiny_model.cfg.vocab_size, (2, 5), generator=g
    )


def _wrap(
    model: TinyModernDecoderForCausalLM, **kwargs
) -> PaddedMaskedTinyModernDecoderWrapper:
    return PaddedMaskedTinyModernDecoderWrapper(model, **kwargs)


# ---------------------------------------------------------------------------
# 1. Plain reference shapes
# ---------------------------------------------------------------------------


def test_tiny_modern_decoder_plain_generation_shapes(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    cfg = tiny_model.cfg
    logits, past = tiny_model.forward(prompt_ids)
    assert logits.shape == (prompt_ids.shape[0], prompt_ids.shape[1], cfg.vocab_size)
    assert len(past) == cfg.num_layers
    for k, v in past:
        assert k.shape == (
            prompt_ids.shape[0],
            cfg.num_kv_heads,
            prompt_ids.shape[1],
            cfg.head_dim,
        )
        assert v.shape == k.shape
    toks = tiny_model.greedy_generate(prompt_ids, 4)
    assert toks.shape == (prompt_ids.shape[0], prompt_ids.shape[1] + 4)
    # Plain forward + greedy must be deterministic across two calls.
    again = tiny_model.greedy_generate(prompt_ids, 4)
    assert torch.equal(toks, again)


# ---------------------------------------------------------------------------
# 2. Padded linear helper -- correctness round-trip
# ---------------------------------------------------------------------------


def test_padded_linear_compensation_correctness():
    torch.manual_seed(0)
    d_in, d_out, b, s = 8, 6, 2, 4
    x = torch.randn(b, s, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64) * 0.1
    bias = torch.randn(d_out, dtype=torch.float64) * 0.01

    g = torch.Generator(device="cpu").manual_seed(1)
    n_in, n_in_inv = sample_invertible_mask(d_in, generator=g)
    n_out, n_out_inv = sample_invertible_mask(d_out, generator=g)
    pad = sample_pad_like(x, generator=g)

    pack = apply_padded_linear(
        x, w, bias,
        n_in=n_in, n_in_inv=n_in_inv,
        n_out=n_out, n_out_inv=n_out_inv,
        pad=pad,
    )
    plain_y = x @ w + bias
    # Y_tilde ≈ Y @ N_out
    expected_y_tilde = plain_y @ n_out
    torch.testing.assert_close(pack["y_tilde"], expected_y_tilde, atol=1e-10, rtol=1e-10)
    # Y_recovered ≈ Y
    torch.testing.assert_close(pack["y_recovered"], plain_y, atol=1e-10, rtol=1e-10)


def test_same_input_fresh_pad_produces_different_masked_x_tilde():
    """Smaller-scale helper sanity check: same X with fresh pads / masks
    must produce different ``X_tilde`` bytes."""
    torch.manual_seed(2)
    d_in, d_out, b, s = 8, 6, 2, 4
    x = torch.randn(b, s, d_in, dtype=torch.float64)
    w = torch.randn(d_in, d_out, dtype=torch.float64) * 0.1

    g1 = torch.Generator(device="cpu").manual_seed(11)
    n_in1, n_in_inv1 = sample_invertible_mask(d_in, generator=g1)
    n_out1, n_out_inv1 = sample_invertible_mask(d_out, generator=g1)
    pad1 = sample_pad_like(x, generator=g1)
    p1 = apply_padded_linear(
        x, w, None,
        n_in=n_in1, n_in_inv=n_in_inv1,
        n_out=n_out1, n_out_inv=n_out_inv1, pad=pad1,
    )

    g2 = torch.Generator(device="cpu").manual_seed(22)
    n_in2, n_in_inv2 = sample_invertible_mask(d_in, generator=g2)
    n_out2, n_out_inv2 = sample_invertible_mask(d_out, generator=g2)
    pad2 = sample_pad_like(x, generator=g2)
    p2 = apply_padded_linear(
        x, w, None,
        n_in=n_in2, n_in_inv=n_in_inv2,
        n_out=n_out2, n_out_inv=n_out_inv2, pad=pad2,
    )

    assert tensor_fingerprint(p1["x_tilde"]) != tensor_fingerprint(p2["x_tilde"])
    torch.testing.assert_close(p1["y_recovered"], p2["y_recovered"], atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3-5. Pad never enters nonlinear cores
# ---------------------------------------------------------------------------


def _diagnostics_for_main_run(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
) -> PaddedMaskedGenerationDiagnostics:
    wrapper = _wrap(tiny_model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(2028)
    _, diag = wrapper.padded_masked_generate(
        prompt_ids, 4, generator=g, diagnostics=PaddedMaskedGenerationDiagnostics()
    )
    return diag


def test_pad_does_not_enter_rmsnorm_core(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    assert diag.pad_entered_rmsnorm_core is False
    assert diag.rmsnorm_mode == "trusted_fallback_with_repad"


def test_pad_does_not_enter_rope_core(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    assert diag.pad_entered_rope_core is False
    assert diag.rope_mode == "post_rope_masking"


def test_pad_does_not_enter_swiglu_core(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    assert diag.pad_entered_swiglu_core is False
    assert diag.swiglu_mode == "paired_permutation_with_boundary_pad"


# ---------------------------------------------------------------------------
# 6. SwiGLU shared permutation + boundary pad correctness
# ---------------------------------------------------------------------------


def test_swiglu_shared_permutation_with_pad_correctness(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    """The wrapper's SwiGLU paired-permutation arithmetic must recover
    plain Y up to float64 round-off; diagnostics record the worst error."""
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    assert diag.swiglu_paired_permutation_max_error < 1e-10


# ---------------------------------------------------------------------------
# 7. RoPE -- post-mask invariant used, pre-mask commutation NOT used
# ---------------------------------------------------------------------------


def test_rope_post_mask_invariant_used_not_pre_mask_commutation(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    assert diag.rope_mode == "post_rope_masking"
    # The wrapper sets the QK constraint error from the per-head mask
    # pair; this must be near zero (orthogonal masks).
    assert diag.qk_constraint_max_error < 1e-10

    # Cross-check: generic dense pre-RoPE mask does NOT commute. We
    # construct a small example and verify the failure is large -- this
    # is the *reason* the wrapper uses post-RoPE masking.
    torch.manual_seed(7)
    head_dim = 16
    q = torch.randn(1, 1, 4, head_dim, dtype=torch.float64)
    n_dense, _ = sample_invertible_mask(head_dim)
    positions = torch.arange(4)
    lhs = apply_rope(q @ n_dense, positions, 10000.0)
    rhs = apply_rope(q, positions, 10000.0) @ n_dense
    pre_mask_commutation_gap = (lhs - rhs).abs().max().item()
    assert pre_mask_commutation_gap > 1e-3, (
        "Generic dense mask must NOT commute with RoPE; if this passes "
        "trivially the test is meaningless."
    )


# ---------------------------------------------------------------------------
# 8. GQA mask indexing
# ---------------------------------------------------------------------------


def test_gqa_mask_indexing_head_dim_not_hidden_size(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    cfg = tiny_model.cfg
    assert cfg.num_query_heads == 4
    assert cfg.num_kv_heads == 2
    diag = _diagnostics_for_main_run(tiny_model, prompt_ids)
    # The wrapper's QK constraint is computed per Q head with mask
    # dimension = head_dim. If the wrapper had mistakenly used a hidden-
    # size mask, the QK constraint error would be wildly large.
    assert diag.qk_constraint_max_error < 1e-10


# ---------------------------------------------------------------------------
# 9. Padded prefill logits allclose to plain
# ---------------------------------------------------------------------------


def test_padded_prefill_logits_allclose(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    wrapper = _wrap(tiny_model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(2028)
    recovered, _, _, _ = wrapper.padded_masked_forward(
        prompt_ids, generator=g
    )
    plain_logits, _ = tiny_model.forward(prompt_ids)
    torch.testing.assert_close(recovered, plain_logits, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. Padded decode-step logits allclose to plain
# ---------------------------------------------------------------------------


def test_padded_decode_step_logits_allclose(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    wrapper = _wrap(tiny_model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(2029)
    # Padded prefill + 3 decode steps, comparing recovered logits at
    # every step to plain forward.
    recovered_prefill, past_tilde, diag, session = wrapper.padded_masked_forward(
        prompt_ids, generator=g
    )
    plain_logits, plain_past = tiny_model.forward(prompt_ids)
    torch.testing.assert_close(
        recovered_prefill, plain_logits, atol=1e-10, rtol=1e-10
    )

    next_token = plain_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    plain_step_past = list(plain_past)
    for _ in range(3):
        plain_logits, plain_step_past = tiny_model.forward(
            next_token, past_key_values=plain_step_past
        )
        recovered_step, past_tilde, diag, _ = wrapper.padded_masked_forward(
            next_token,
            past_key_values_tilde=past_tilde,
            session_masks=session,
            generator=g,
            diagnostics=diag,
        )
        torch.testing.assert_close(
            recovered_step, plain_logits, atol=1e-10, rtol=1e-10
        )
        next_token = recovered_step[:, -1, :].argmax(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 11. Padded KV cache append invariant
# ---------------------------------------------------------------------------


def test_padded_kv_cache_append_invariant(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    wrapper = _wrap(tiny_model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(2030)
    diag = PaddedMaskedGenerationDiagnostics()
    _, _ = wrapper.padded_masked_generate(
        prompt_ids, 4, generator=g, diagnostics=diag
    )
    assert diag.kv_cache_invariant_max_abs_error < KV_CACHE_TOLERANCE
    assert diag.kv_cache_contains_plaintext is False
    assert diag.kv_cache_pad_compensated_before_append is True
    assert diag.kv_cache_mask_fixed_within_session is True


# ---------------------------------------------------------------------------
# 12. Greedy generation exact match
# ---------------------------------------------------------------------------


def test_padded_greedy_generation_exact_match(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    wrapper = _wrap(tiny_model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(2031)
    masked_tokens, diag = wrapper.padded_masked_generate(
        prompt_ids, 4, generator=g
    )
    plain_tokens = tiny_model.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, masked_tokens)
    assert diag.prefill_logits_max_abs_error < LOGITS_TOLERANCE
    assert diag.decode_step_logits_max_abs_error_max < LOGITS_TOLERANCE


# ---------------------------------------------------------------------------
# 13. Same input + fresh pad/mask -> different masked fingerprints
# ---------------------------------------------------------------------------


def test_same_input_fresh_pad_produces_different_masked_fingerprints(
    tiny_model: TinyModernDecoderForCausalLM, prompt_ids: torch.Tensor
):
    wrapper = _wrap(tiny_model, use_pad=True)
    g1 = torch.Generator(device="cpu").manual_seed(101)
    tokens_a, diag_a = wrapper.padded_masked_generate(
        prompt_ids, 4, generator=g1
    )
    g2 = torch.Generator(device="cpu").manual_seed(202)
    tokens_b, diag_b = wrapper.padded_masked_generate(
        prompt_ids, 4, generator=g2
    )
    # Recovered token sequences are equal.
    assert torch.equal(tokens_a, tokens_b)
    # Every masked-boundary fingerprint must differ across the two runs.
    for key in ("prefill_x_tilde", "prefill_kv_cache", "prefill_logits_tilde"):
        assert diag_a.masked_boundary_fingerprints[key] != \
            diag_b.masked_boundary_fingerprints[key], (
                f"fingerprint key {key} unexpectedly matched across two "
                "fresh-pad/fresh-mask runs"
            )


# ---------------------------------------------------------------------------
# 14. Report marks use_pad=True as main mode
# ---------------------------------------------------------------------------


def test_report_marks_use_pad_as_main_mode(tmp_path: Path):
    cfg = GenerationCorrectnessConfig()
    report = run_modern_decoder_generation_correctness(cfg=cfg)
    assert report["status"] == "ok"
    assert report["main_mode"] == "padded_masked_execution"
    assert report["use_pad"] is True
    assert report["fresh_pad"] is True
    assert report["fresh_mask"] is True
    assert report["pad_policy"]["pad_at_linear_boundaries"] is True
    assert report["pad_policy"]["pad_enters_rmsnorm_core"] is False
    assert report["pad_policy"]["pad_enters_rope_core"] is False
    assert report["pad_policy"]["pad_enters_swiglu_core"] is False
    assert report["pad_policy"]["pad_enters_softmax"] is False
    assert report["correctness"]["greedy_token_match_rate"] == 1.0
    assert report["correctness"]["sequence_exact_match"] is True

    json_path, md_path = write_reports(report, outputs_dir=tmp_path)
    assert json_path.is_file()
    assert md_path.is_file()


# ---------------------------------------------------------------------------
# 15. Attention map limitation is reported
# ---------------------------------------------------------------------------


def test_attention_map_limitation_is_reported(tmp_path: Path):
    cfg = GenerationCorrectnessConfig()
    report = run_modern_decoder_generation_correctness(cfg=cfg)
    limitations = " ".join(report["limitations"])
    assert "attention" in limitations.lower()
    assert (
        "scores" in limitations.lower()
        or "probabilities" in limitations.lower()
        or "softmax" in limitations.lower()
    )
    unsafe = report["unsafe_wording_to_avoid"]
    assert any("Attention maps" in s for s in unsafe)
    # Limitations also flag that this is not a formal-security claim.
    assert any("formal" in s.lower() or "cryptographic" in s.lower() for s in unsafe)


# ---------------------------------------------------------------------------
# 16. Public outputs/ artifact is produced by the runner script
# ---------------------------------------------------------------------------


def test_public_outputs_artifact_present():
    """Smoke test that the runner has produced both report files on disk
    after the last manual run (or after the test below regenerates them)."""
    json_path = REPO_ROOT / "outputs" / "modern_decoder_generation_correctness.json"
    md_path = REPO_ROOT / "outputs" / "modern_decoder_generation_correctness.md"
    assert json_path.is_file(), "run_modern_decoder_generation_correctness.py has not been executed"
    assert md_path.is_file(), "Markdown report missing"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["main_mode"] == "padded_masked_execution"
    assert payload["use_pad"] is True
    assert payload["correctness"]["sequence_exact_match"] is True
