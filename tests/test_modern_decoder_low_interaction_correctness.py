"""Stage 7.6f -- tests for the low-interaction operator-compatible
modern-decoder generation wrapper and its norm-leakage audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pllo.experiments.modern_decoder_low_interaction_correctness import (
    INVARIANT_TOLERANCE,
    KV_CACHE_TOLERANCE,
    LOGITS_TOLERANCE,
    LowInteractionConfig,
    run_low_interaction_correctness,
    write_reports,
)
from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
)
from pllo.wrappers.low_interaction_modern_decoder_generation_wrapper import (
    LowInteractionDiagnostics,
    LowInteractionTinyModernDecoderWrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_model_1layer() -> TinyModernDecoderForCausalLM:
    torch.manual_seed(2026)
    cfg = TinyModernDecoderConfig(num_layers=1)
    m = TinyModernDecoderForCausalLM(cfg)
    m.init_random_weights(torch.Generator(device="cpu").manual_seed(2026))
    return m


@pytest.fixture(scope="module")
def tiny_model_2layer() -> TinyModernDecoderForCausalLM:
    torch.manual_seed(2026)
    cfg = TinyModernDecoderConfig(num_layers=2)
    m = TinyModernDecoderForCausalLM(cfg)
    m.init_random_weights(torch.Generator(device="cpu").manual_seed(2026))
    return m


@pytest.fixture(scope="module")
def prompt_ids(tiny_model_1layer: TinyModernDecoderForCausalLM) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(2027)
    return torch.randint(
        0, tiny_model_1layer.cfg.vocab_size, (2, 5), generator=g
    )


def _wrap(
    model: TinyModernDecoderForCausalLM, **kwargs
) -> LowInteractionTinyModernDecoderWrapper:
    return LowInteractionTinyModernDecoderWrapper(model, **kwargs)


def _generate(
    model: TinyModernDecoderForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    seed: int = 2028,
) -> tuple[torch.Tensor, LowInteractionDiagnostics]:
    wrapper = _wrap(model, use_pad=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return wrapper.low_interaction_generate(
        prompt_ids, 4, generator=g, diagnostics=LowInteractionDiagnostics()
    )


# ---------------------------------------------------------------------------
# 1. H_hat = H Q invariant holds at every layer boundary
# ---------------------------------------------------------------------------


def test_main_layer_invariant_holds_at_every_boundary(
    tiny_model_1layer, prompt_ids
):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.main_layer_invariant == "H_hat_l = H_l Q_l"
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 2. RMSNorm runs in operator_compatible_orthogonal mode (no fallback)
# ---------------------------------------------------------------------------


def test_rmsnorm_runs_in_operator_compatible_orthogonal_mode(
    tiny_model_1layer, prompt_ids
):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.rmsnorm_mode == "operator_compatible_orthogonal"
    assert diag.trusted_fallback_used_in_main_path is False
    assert diag.rmsnorm_core_orthogonal_commutation_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 3. No intermediate TEE re-entry; exactly one boundary round-trip per step
# ---------------------------------------------------------------------------


def test_no_intermediate_tee_reentry(tiny_model_1layer, prompt_ids):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.intermediate_tee_reentry is False
    assert diag.online_boundary_round_trips_per_decode_step == 1


# ---------------------------------------------------------------------------
# 4. Pad is mandatory and used at every Linear boundary (default mode)
# ---------------------------------------------------------------------------


def test_use_pad_default_true_at_every_linear_boundary(
    tiny_model_1layer, prompt_ids
):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.use_pad is True
    assert diag.fresh_pad_used_at_linear_boundaries is True
    assert diag.pad_at_linear_boundaries is True
    assert diag.pad_enters_rmsnorm_core is False
    assert diag.pad_enters_swiglu_core is False
    assert diag.pad_enters_rope_core is False
    assert diag.pad_enters_softmax is False


# ---------------------------------------------------------------------------
# 5. Transition trick A = Q^{-1} M, C = T M reproduces (X - T) M correctly
# ---------------------------------------------------------------------------


def test_transition_trick_round_trip(tiny_model_1layer, prompt_ids):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.transition_trick_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 6. SwiGLU paired-permutation invariant + boundary pad
# ---------------------------------------------------------------------------


def test_swiglu_paired_permutation_with_boundary_pad(
    tiny_model_1layer, prompt_ids
):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.swiglu_mode == "paired_permutation_with_boundary_pad"
    assert diag.swiglu_paired_permutation_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 7. RoPE blocker: post-RoPE masking used, no pre-RoPE generic commutation
# ---------------------------------------------------------------------------


def test_rope_blocker_post_rope_masking_used(tiny_model_1layer, prompt_ids):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert "post_rope_masking" in diag.rope_mode
    assert diag.rope_blocker_transient_plain_qk_on_accelerator is True
    assert diag.qk_constraint_max_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 8. KV cache append invariant + masked-only storage
# ---------------------------------------------------------------------------


def test_kv_cache_append_invariant(tiny_model_1layer, prompt_ids):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.kv_cache_invariant_max_abs_error < KV_CACHE_TOLERANCE


# ---------------------------------------------------------------------------
# 9. Recovered prefill / decode logits match plain reference
# ---------------------------------------------------------------------------


def test_recovered_logits_match_plain(tiny_model_1layer, prompt_ids):
    _, diag = _generate(tiny_model_1layer, prompt_ids)
    assert diag.prefill_logits_max_abs_error < LOGITS_TOLERANCE
    assert diag.decode_step_logits_max_abs_error_max < LOGITS_TOLERANCE
    assert diag.lm_head_recovery_max_abs_error < LOGITS_TOLERANCE


# ---------------------------------------------------------------------------
# 10. Greedy generation produces token-for-token equal output
# ---------------------------------------------------------------------------


def test_greedy_generation_exact_match(tiny_model_1layer, prompt_ids):
    tokens, _ = _generate(tiny_model_1layer, prompt_ids)
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, tokens)


# ---------------------------------------------------------------------------
# 11. Two-layer model also satisfies the no-reentry invariant
# ---------------------------------------------------------------------------


def test_two_layer_invariant_holds(tiny_model_2layer, prompt_ids):
    _, diag = _generate(tiny_model_2layer, prompt_ids)
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE
    assert diag.lm_head_recovery_max_abs_error < LOGITS_TOLERANCE
    tokens, _ = _generate(tiny_model_2layer, prompt_ids)
    plain_tokens = tiny_model_2layer.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, tokens)


# ---------------------------------------------------------------------------
# 12. Same input + fresh Q produces same tokens but different fingerprints
# ---------------------------------------------------------------------------


def test_same_input_fresh_masks_different_fingerprints(
    tiny_model_1layer, prompt_ids
):
    tokens_a, diag_a = _generate(tiny_model_1layer, prompt_ids, seed=1234)
    tokens_b, diag_b = _generate(tiny_model_1layer, prompt_ids, seed=5678)
    assert torch.equal(tokens_a, tokens_b)
    fp_a = diag_a.masked_boundary_fingerprints
    fp_b = diag_b.masked_boundary_fingerprints
    assert fp_a["prefill_layer_entry_h_hat"] != fp_b["prefill_layer_entry_h_hat"]
    assert (
        fp_a["prefill_lm_head_logits_tilde"]
        != fp_b["prefill_lm_head_logits_tilde"]
    )


# ---------------------------------------------------------------------------
# 13. Norm leakage audit: row norms and Gram matrix are preserved
# ---------------------------------------------------------------------------


def test_norm_leakage_audit_row_norms_and_gram_preserved():
    report = run_low_interaction_correctness(cfg=LowInteractionConfig())
    leak = report["norm_leakage_audit"]
    # Row L2 norms are mathematically invariant under orthogonal Q;
    # the audit number should be at float64 round-off.
    assert leak["row_norm_error"] < 1e-12
    assert leak["gram_matrix_error"] < 1e-10
    # Same prompt + fresh Q1, Q2: the Gram matrix is unchanged (both
    # equal the plain Gram). The audit number must therefore also be
    # near zero. This is the *leakage* surface.
    assert leak["same_prompt_fresh_Q_gram_linkability"] < 1e-10
    # Different prompts should have a substantial Gram distance.
    assert leak["different_prompt_gram_distance"] > 1.0


# ---------------------------------------------------------------------------
# 14. Report carries all Stage 7.6f mandated fields
# ---------------------------------------------------------------------------


def test_report_carries_mandated_main_fields(tmp_path: Path):
    report = run_low_interaction_correctness(cfg=LowInteractionConfig())
    assert report["status"] == "ok"
    assert report["main_layer_invariant"] == "H_hat_l = H_l Q_l"
    assert report["rmsnorm_mode"] == "operator_compatible_orthogonal"
    assert report["trusted_fallback_used_in_main_path"] is False
    assert report["intermediate_tee_reentry"] is False
    assert report["online_boundary_round_trips_per_decode_step"] == 1
    assert report["use_pad"] is True
    assert report["fresh_pad_used_at_linear_boundaries"] is True
    assert report["pad_policy"]["pad_enters_rmsnorm_core"] is False
    assert report["pad_policy"]["pad_enters_swiglu_core"] is False
    assert report["correctness"]["greedy_token_match_rate"] == 1.0
    assert report["correctness"]["sequence_exact_match"] is True
    # RoPE blocker honestly reported.
    assert report["rope_blocker"]["blocker_present"] is True
    assert report["rope_blocker"]["transient_plain_qk_on_accelerator"] is True
    assert report["rope_blocker"]["tee_reentry_inside_rope_block"] is False

    json_path, md_path = write_reports(report, outputs_dir=tmp_path)
    assert json_path.is_file()
    assert md_path.is_file()


# ---------------------------------------------------------------------------
# 15. Report names the norm leakage surface in limitations
# ---------------------------------------------------------------------------


def test_norm_leakage_listed_in_limitations():
    report = run_low_interaction_correctness(cfg=LowInteractionConfig())
    limitations = " ".join(report["limitations"]).lower()
    assert "gram" in limitations or "norm" in limitations
    assert "leakage" in limitations
    assert "not formal" in limitations or "formal" in limitations
    unsafe = report["unsafe_wording_to_avoid"]
    assert any("gram" in s.lower() or "row norm" in s.lower() for s in unsafe)


# ---------------------------------------------------------------------------
# 16. Public outputs/ artifact present after the runner has been executed
# ---------------------------------------------------------------------------


def test_public_outputs_artifact_present():
    json_path = (
        REPO_ROOT
        / "outputs"
        / "modern_decoder_low_interaction_correctness.json"
    )
    md_path = (
        REPO_ROOT / "outputs" / "modern_decoder_low_interaction_correctness.md"
    )
    assert json_path.is_file(), (
        "run_modern_decoder_low_interaction_correctness.py must be executed"
    )
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["main_layer_invariant"] == "H_hat_l = H_l Q_l"
    assert payload["trusted_fallback_used_in_main_path"] is False
    assert payload["intermediate_tee_reentry"] is False
    assert payload["online_boundary_round_trips_per_decode_step"] == 1
    assert payload["use_pad"] is True
