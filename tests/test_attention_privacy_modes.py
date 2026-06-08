"""Stage 7.6i tests -- attention-privacy modes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from pllo.experiments.attention_privacy_modes import (
    AttentionPrivacyModesConfig,
    LOGITS_TOLERANCE,
    render_markdown,
    run_attention_privacy_modes,
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
def stage_7_6i_report() -> dict:
    return run_attention_privacy_modes(cfg=AttentionPrivacyModesConfig())


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


def test_exact_visible_attention_preserves_stage_7_6h(
    stage_7_6i_report: dict,
) -> None:
    m = stage_7_6i_report["per_mode_results"]["exact_visible_attention"]
    d = m["diagnostics"]
    assert m["greedy_token_match_rate"] == 1.0
    assert m["sequence_exact_match"] is True
    # Stage 7.6h carry-over.
    assert d["use_pad"] is True
    assert d["rope_mask_mode"] == "pre_rope_block_diagonal_rotation"
    assert d["rope_transient_plain_qk_visible"] is False
    assert d["qkv_projection_outputs_masked_directly"] is True
    assert d["intermediate_tee_reentry"] is False
    assert d["online_boundary_round_trips_per_decode_step"] == 1
    assert d["trusted_fallback_used_in_main_path"] is False
    assert d["pad_enters_rmsnorm_core"] is False
    assert d["pad_enters_rope_core"] is False
    assert d["pad_enters_swiglu_core"] is False
    assert d["pad_enters_softmax"] is False
    # New attention-privacy fields for the baseline.
    assert d["attention_privacy_mode"] == "exact_visible_attention"
    assert d["attention_scores_visible"] is True
    assert d["attention_probs_visible"] is True
    assert d["attention_exact"] is True
    assert d["lm_head_recovery_max_abs_error"] < LOGITS_TOLERANCE


def test_trusted_softmax_attention_hides_scores_from_transcript(
    stage_7_6i_report: dict,
) -> None:
    d = (
        stage_7_6i_report["per_mode_results"]["trusted_softmax_attention"]
        ["diagnostics"]
    )
    assert d["attention_privacy_mode"] == "trusted_softmax_attention"
    assert d["attention_scores_visible"] is False
    assert d["attention_probs_visible"] is False
    assert d["attention_score_persistent_transcript_visible"] is False
    assert d["attention_prob_persistent_transcript_visible"] is False
    assert d["attention_entropy_visible"] is False
    assert d["attention_top1_index_visible"] is False
    assert d["attention_topk_indices_visible"] is False
    assert d["attention_relative_margin_visible"] is False
    assert d["attention_map_fingerprint_available"] is False
    assert d["trusted_softmax_used"] is True
    assert d["attention_map_hidden_from_accelerator_transcript"] is True


def test_trusted_softmax_attention_exact_logits_match(
    stage_7_6i_report: dict,
) -> None:
    m = stage_7_6i_report["per_mode_results"]["trusted_softmax_attention"]
    d = m["diagnostics"]
    assert m["greedy_token_match_rate"] == 1.0
    assert m["sequence_exact_match"] is True
    assert d["attention_exact"] is True
    assert d["lm_head_recovery_max_abs_error"] < LOGITS_TOLERANCE


def test_trusted_softmax_attention_has_extra_tee_round_trips(
    stage_7_6i_report: dict,
) -> None:
    d = (
        stage_7_6i_report["per_mode_results"]["trusted_softmax_attention"]
        ["diagnostics"]
    )
    cfg = stage_7_6i_report["config"]
    assert d["online_boundary_round_trips_per_decode_step"] > 1
    # one entry/exit + one per layer attention round-trip
    assert d["online_boundary_round_trips_per_decode_step"] == 1 + cfg["num_layers"]
    assert d["intermediate_tee_reentry"] is True
    assert d["attention_extra_tee_round_trips_per_layer"] >= 1


def test_row_constant_score_shift_softmax_exact_but_not_private(
    stage_7_6i_report: dict,
) -> None:
    d = (
        stage_7_6i_report["per_mode_results"]["score_blinding_experimental"]
        ["diagnostics"]
    )
    # Row-constant shift -> softmax exact.
    assert d["row_constant_shift_used"] is True
    assert d["row_constant_blinding_softmax_max_abs_error"] < 1e-12
    # ... but does NOT hide ranking / relative margins / topology.
    assert d["hides_relative_attention"] is False
    assert d["attention_privacy_gain"].startswith("none")
    # Scores and probs are still on the accelerator transcript.
    assert d["attention_scores_visible"] is True
    assert d["attention_probs_visible"] is True


def test_nonconstant_score_blinding_breaks_softmax_exactness_if_implemented(
    stage_7_6i_report: dict,
) -> None:
    d = (
        stage_7_6i_report["per_mode_results"]["score_blinding_experimental"]
        ["diagnostics"]
    )
    # The wrapper records a side-channel diagnostic showing that
    # a non-row-constant random R additive shift breaks softmax.
    assert d["nonconstant_blinding_softmax_max_abs_error"] > 1e-3


# ---------------------------------------------------------------------------
# Audit, report, and integrity
# ---------------------------------------------------------------------------


def test_attention_leakage_audit_fields_exist(
    stage_7_6i_report: dict,
) -> None:
    audit = stage_7_6i_report["attention_leakage_audit"]
    expected_fields = {
        "attention_score_persistent_transcript_visible",
        "attention_prob_persistent_transcript_visible",
        "attention_entropy_visible",
        "attention_top1_index_visible",
        "attention_topk_indices_visible",
        "attention_relative_margin_visible",
        "attention_map_fingerprint_available",
        "attention_score_max_abs_error_vs_plain",
        "attention_prob_max_abs_error_vs_plain",
        "attention_top1_match_rate",
    }
    for mode in ("exact_visible_attention", "trusted_softmax_attention",
                 "score_blinding_experimental"):
        assert set(audit[mode].keys()) >= expected_fields, mode


def test_report_comparison_table_contains_tradeoff(
    stage_7_6i_report: dict,
) -> None:
    comp = stage_7_6i_report["comparison"]
    e = comp["exact_visible_attention"]
    t = comp["trusted_softmax_attention"]
    s = comp["score_blinding_experimental"]
    # exact_visible: exact + one round-trip, NOT hidden.
    assert e["exact"] is True
    assert e["one_round_trip"] is True
    assert e["attention_hidden"] is False
    # trusted_softmax: exact, NOT one round-trip, hidden.
    assert t["exact"] is True
    assert t["one_round_trip"] is False
    assert t["attention_hidden"] is True
    # score_blinding: exact + one round-trip, NOT hidden.
    assert s["exact"] is True
    assert s["one_round_trip"] is True
    assert s["attention_hidden"] is False


def test_no_false_claim_attention_hidden_in_exact_visible_mode(
    stage_7_6i_report: dict,
) -> None:
    d = (
        stage_7_6i_report["per_mode_results"]["exact_visible_attention"]
        ["diagnostics"]
    )
    # The exact baseline MUST report that the attention map is visible.
    assert d["attention_map_hidden_from_accelerator_transcript"] is False
    assert d["attention_scores_visible"] is True
    assert d["attention_probs_visible"] is True
    assert d["hides_relative_attention"] is False
    # Unsafe wording list must contain the canonical false claim.
    unsafe = stage_7_6i_report["unsafe_wording_to_avoid"]
    assert any("exact low-interaction mode hides attention maps" in s.lower()
               for s in unsafe)


def test_use_pad_and_rope_safe_invariants_preserved(
    stage_7_6i_report: dict,
) -> None:
    for mode in ("exact_visible_attention", "trusted_softmax_attention",
                 "score_blinding_experimental"):
        d = stage_7_6i_report["per_mode_results"][mode]["diagnostics"]
        assert d["use_pad"] is True
        assert d["rope_mask_mode"] == "pre_rope_block_diagonal_rotation"
        assert d["rope_transient_plain_qk_visible"] is False
        assert d["rope_transient_plain_v_visible"] is False
        assert d["qkv_projection_outputs_masked_directly"] is True
        assert d["pad_enters_rmsnorm_core"] is False
        assert d["pad_enters_rope_core"] is False
        assert d["pad_enters_swiglu_core"] is False
        assert d["pad_enters_softmax"] is False


def test_required_acceptance_fields_present(
    stage_7_6i_report: dict,
) -> None:
    assert stage_7_6i_report["attention_privacy_modes_completed"] is True
    e = stage_7_6i_report["per_mode_results"]["exact_visible_attention"]
    t = stage_7_6i_report["per_mode_results"]["trusted_softmax_attention"]
    assert e["diagnostics"]["attention_exact"] is True
    assert e["diagnostics"]["online_boundary_round_trips_per_decode_step"] == 1
    assert e["diagnostics"]["attention_scores_visible"] is True
    assert t["diagnostics"]["attention_exact"] is True
    assert t["diagnostics"]["online_boundary_round_trips_per_decode_step"] > 1
    assert t["diagnostics"]["attention_scores_visible"] is False


def test_public_outputs_artifact_present() -> None:
    json_path = REPO_ROOT / "outputs" / "attention_privacy_modes.json"
    md_path = REPO_ROOT / "outputs" / "attention_privacy_modes.md"
    assert json_path.exists(), f"missing {json_path}"
    assert md_path.exists(), f"missing {md_path}"
    obj = json.loads(json_path.read_text())
    assert obj["status"] == "ok"
    assert obj["stage"] == "7.6i"
    text = md_path.read_text()
    assert "Attention-Map Protection" in text
    assert "Summary Comparison" in text
    assert "Attention Privacy: Exactness vs Hiding Tension" in text
    assert "Attention Leakage Audit" in text
    assert "Paper-Safe Wording" in text


def test_render_markdown_smoke(stage_7_6i_report: dict) -> None:
    md = render_markdown(stage_7_6i_report)
    for mode in ("exact_visible_attention", "trusted_softmax_attention",
                 "score_blinding_experimental"):
        assert mode in md


# ---------------------------------------------------------------------------
# Wrapper-level mathematical contracts
# ---------------------------------------------------------------------------


def _build_model() -> TinyModernDecoderForCausalLM:
    cfg = TinyModernDecoderConfig(num_layers=1)
    cfg.validate()
    model = TinyModernDecoderForCausalLM(cfg)
    model.init_random_weights(torch.Generator(device="cpu").manual_seed(2026))
    return model


def _build_prompt(model: TinyModernDecoderForCausalLM) -> torch.Tensor:
    return torch.randint(
        0, model.cfg.vocab_size, (2, 6),
        generator=torch.Generator(device="cpu").manual_seed(2027),
    )


def test_trusted_softmax_attention_requires_pre_rope_block_diag_rotation() -> None:
    model = _build_model()
    with pytest.raises(ValueError, match="pre_rope_block_diagonal_rotation"):
        LowInteractionTinyModernDecoderWrapper(
            model,
            rope_mask_mode="post_rope_masking",
            attention_privacy_mode="trusted_softmax_attention",
        )


def test_unknown_attention_privacy_mode_raises() -> None:
    model = _build_model()
    with pytest.raises(ValueError, match="attention_privacy_mode"):
        LowInteractionTinyModernDecoderWrapper(
            model,
            rope_mask_mode="pre_rope_block_diagonal_rotation",
            attention_privacy_mode="cryptographic_secret_sauce",
        )


def test_softmax_row_constant_invariance_mathematical() -> None:
    """softmax(S + c_i * 1_row) == softmax(S) at machine precision."""
    g = torch.Generator(device="cpu").manual_seed(31415)
    s = torch.randn(2, 4, 6, 6, dtype=torch.float64, generator=g)
    c = torch.randn(2, 4, 6, 1, dtype=torch.float64, generator=g)
    p_base = torch.softmax(s, dim=-1)
    p_shift = torch.softmax(s + c, dim=-1)
    err = float((p_base - p_shift).abs().max().item())
    assert err < 1e-12


def test_softmax_nonconstant_shift_breaks_exactness_mathematical() -> None:
    """softmax(S + R) != softmax(S) for arbitrary non-row-constant R."""
    g = torch.Generator(device="cpu").manual_seed(27182)
    s = torch.randn(2, 4, 6, 6, dtype=torch.float64, generator=g)
    r = torch.randn(2, 4, 6, 6, dtype=torch.float64, generator=g)
    p_base = torch.softmax(s, dim=-1)
    p_r = torch.softmax(s + r, dim=-1)
    err = float((p_base - p_r).abs().max().item())
    assert err > 1e-3
