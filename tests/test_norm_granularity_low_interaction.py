"""Stage 7.6h -- tests for the RMSNorm-compatible mask granularity
(sequence / chunk / token) on top of the rope-safe low-interaction
path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pllo.experiments.norm_granularity_low_interaction import (
    INVARIANT_TOLERANCE,
    KV_CACHE_TOLERANCE,
    LOGITS_TOLERANCE,
    NormGranularityConfig,
    run_norm_granularity_low_interaction,
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
def prompt_ids(tiny_model_1layer: TinyModernDecoderForCausalLM) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(2027)
    return torch.randint(
        0, tiny_model_1layer.cfg.vocab_size, (2, 6), generator=g
    )


def _wrap(
    model: TinyModernDecoderForCausalLM,
    *,
    granularity: str,
    chunk_size: int = 1,
) -> LowInteractionTinyModernDecoderWrapper:
    return LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=True,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
        norm_mask_granularity=granularity,
        norm_chunk_size=chunk_size,
    )


def _generate(
    wrapper: LowInteractionTinyModernDecoderWrapper,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 3,
    seed: int = 2028,
) -> tuple[torch.Tensor, LowInteractionDiagnostics]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    diag = LowInteractionDiagnostics()
    return wrapper.low_interaction_generate(
        prompt_ids, max_new_tokens, generator=g, diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# 1. Sequence mode unchanged (Stage 7.6g baseline)
# ---------------------------------------------------------------------------


def test_sequence_mode_matches_baseline(tiny_model_1layer, prompt_ids):
    wrapper = _wrap(tiny_model_1layer, granularity="sequence")
    tokens, diag = _generate(wrapper, prompt_ids)
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 3)
    assert torch.equal(plain_tokens, tokens)
    assert diag.norm_mask_granularity == "sequence"
    assert diag.norm_q_is_per_row is False
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 2. Chunk mode greedy match
# ---------------------------------------------------------------------------


def test_chunk_mode_greedy_exact_match(tiny_model_1layer, prompt_ids):
    wrapper = _wrap(tiny_model_1layer, granularity="chunk", chunk_size=2)
    tokens, diag = _generate(wrapper, prompt_ids)
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 3)
    assert torch.equal(plain_tokens, tokens)
    assert diag.norm_mask_granularity == "chunk"
    assert diag.norm_chunk_size == 2
    assert diag.norm_q_is_per_row is True
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 3. Token mode greedy match
# ---------------------------------------------------------------------------


def test_token_mode_greedy_exact_match(tiny_model_1layer, prompt_ids):
    wrapper = _wrap(tiny_model_1layer, granularity="token")
    tokens, diag = _generate(wrapper, prompt_ids)
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 3)
    assert torch.equal(plain_tokens, tokens)
    assert diag.norm_mask_granularity == "token"
    assert diag.norm_q_is_per_row is True


# ---------------------------------------------------------------------------
# 4. Row norms preserved in all three modes (RMSNorm correctness)
# ---------------------------------------------------------------------------


def test_row_norms_preserved_in_all_modes(tiny_model_1layer, prompt_ids):
    """For exact RMSNorm correctness ``Q_i`` must be orthogonal, so per-
    row L2 norms are preserved in every granularity. This is captured by
    ``rmsnorm_core_orthogonal_commutation_max_abs_error`` being near 0."""
    for granularity, chunk in (("sequence", 1), ("chunk", 2), ("token", 1)):
        wrapper = _wrap(
            tiny_model_1layer, granularity=granularity, chunk_size=chunk
        )
        _, diag = _generate(wrapper, prompt_ids)
        assert (
            diag.rmsnorm_core_orthogonal_commutation_max_abs_error
            < INVARIANT_TOLERANCE
        ), f"granularity={granularity}"


# ---------------------------------------------------------------------------
# 5. All modes preserve every Stage 7.6g invariant
# ---------------------------------------------------------------------------


def test_all_modes_preserve_stage_7_6g_invariants(tiny_model_1layer, prompt_ids):
    for granularity, chunk in (("sequence", 1), ("chunk", 2), ("token", 1)):
        wrapper = _wrap(
            tiny_model_1layer, granularity=granularity, chunk_size=chunk
        )
        _, diag = _generate(wrapper, prompt_ids)
        assert diag.use_pad is True
        assert diag.fresh_pad_used_at_linear_boundaries is True
        assert diag.rope_mask_mode == "pre_rope_block_diagonal_rotation"
        assert diag.rope_transient_plain_qk_visible is False
        assert diag.rope_transient_plain_v_visible is False
        assert diag.qkv_projection_outputs_masked_directly is True
        assert diag.intermediate_tee_reentry is False
        assert diag.online_boundary_round_trips_per_decode_step == 1
        assert diag.trusted_fallback_used_in_main_path is False
        assert diag.pad_enters_rmsnorm_core is False
        assert diag.pad_enters_swiglu_core is False
        assert diag.pad_enters_rope_core is False
        assert diag.pad_enters_softmax is False
        assert diag.kv_cache_invariant_max_abs_error < KV_CACHE_TOLERANCE
        assert diag.qk_score_invariant_max_abs_error < INVARIANT_TOLERANCE
        assert diag.prefill_logits_max_abs_error < LOGITS_TOLERANCE
        assert diag.decode_step_logits_max_abs_error_max < LOGITS_TOLERANCE


# ---------------------------------------------------------------------------
# 6. Sequence mode: full Gram preserved (Stage 7.6g leak)
# ---------------------------------------------------------------------------


def test_sequence_mode_full_gram_preserved():
    report = run_norm_granularity_low_interaction(cfg=NormGranularityConfig())
    leak = report["norm_and_gram_leakage_audit"]["sequence"]
    assert leak["row_norm_error"] < 1e-12
    assert leak["full_gram_error"] < 1e-10  # exactly preserved
    assert leak["same_prompt_fresh_Q_gram_distance"] < 1e-10


# ---------------------------------------------------------------------------
# 7. Token mode: row norm preserved, full Gram off-diagonal disrupted
# ---------------------------------------------------------------------------


def test_token_mode_row_norm_preserved_offdiag_disrupted():
    report = run_norm_granularity_low_interaction(cfg=NormGranularityConfig())
    leak = report["norm_and_gram_leakage_audit"]["token"]
    assert leak["row_norm_error"] < 1e-12
    # Off-diagonal Gram is now scrambled because Q_i != Q_j.
    assert leak["off_diagonal_gram_error"] > 1.0
    assert leak["full_gram_error"] > 1.0
    assert leak["same_prompt_fresh_Q_gram_distance"] > 1.0


# ---------------------------------------------------------------------------
# 8. Chunk mode: within-chunk Gram preserved, cross-chunk disrupted
# ---------------------------------------------------------------------------


def test_chunk_mode_within_chunk_preserved_cross_chunk_disrupted():
    report = run_norm_granularity_low_interaction(cfg=NormGranularityConfig())
    leak = report["norm_and_gram_leakage_audit"]["chunk"]
    assert leak["row_norm_error"] < 1e-12
    # Within-chunk Gram preserved (same Q within each chunk).
    assert leak["within_chunk_gram_error"] < 1e-10
    # Cross-chunk Gram is disrupted.
    assert leak["cross_chunk_gram_error"] > 1.0
    assert leak["off_diagonal_gram_error"] > 1.0


# ---------------------------------------------------------------------------
# 9. Same input fresh masks: same tokens, different fingerprints (all modes)
# ---------------------------------------------------------------------------


def test_same_input_fresh_masks_different_fingerprints(tiny_model_1layer, prompt_ids):
    for granularity, chunk in (("sequence", 1), ("chunk", 2), ("token", 1)):
        wrapper = _wrap(
            tiny_model_1layer, granularity=granularity, chunk_size=chunk
        )
        tokens_a, diag_a = _generate(wrapper, prompt_ids, seed=1234)
        tokens_b, diag_b = _generate(wrapper, prompt_ids, seed=5678)
        assert torch.equal(tokens_a, tokens_b), f"tokens differ in {granularity}"
        fp_a = diag_a.masked_boundary_fingerprints
        fp_b = diag_b.masked_boundary_fingerprints
        assert (
            fp_a["prefill_layer_entry_h_hat"]
            != fp_b["prefill_layer_entry_h_hat"]
        ), f"layer-entry fingerprint identical for {granularity}"
        assert (
            fp_a["prefill_lm_head_logits_tilde"]
            != fp_b["prefill_lm_head_logits_tilde"]
        ), f"logits fingerprint identical for {granularity}"


# ---------------------------------------------------------------------------
# 10. Report carries mandated fields
# ---------------------------------------------------------------------------


def test_report_carries_required_fields(tmp_path: Path):
    report = run_norm_granularity_low_interaction(cfg=NormGranularityConfig())
    assert report["status"] == "ok"
    assert report["stage"] == "7.6h"
    assert set(report["modes_evaluated"]) == {"sequence", "chunk", "token"}
    inh = report["stage_7_6g_inherited"]
    assert inh["use_pad"] is True
    assert inh["rope_mask_mode"] == "pre_rope_block_diagonal_rotation"
    assert inh["rope_transient_plain_qk_visible"] is False
    assert inh["qkv_projection_outputs_masked_directly"] is True
    assert inh["intermediate_tee_reentry"] is False
    assert inh["online_boundary_round_trips_per_decode_step"] == 1
    assert inh["trusted_fallback_used_in_main_path"] is False

    for mode in ("sequence", "chunk", "token"):
        result = report["per_mode_results"][mode]
        assert result["greedy_token_match_rate"] == 1.0
        assert result["sequence_exact_match"] is True
        d = result["diagnostics"]
        assert d["norm_mask_granularity"] == mode
        assert d["use_pad"] is True
        assert d["intermediate_tee_reentry"] is False
        assert d["online_boundary_round_trips_per_decode_step"] == 1

    # Leakage audit must include all the required metrics.
    leak = report["norm_and_gram_leakage_audit"]
    for mode in ("sequence", "chunk", "token"):
        m = leak[mode]
        for key in (
            "row_norm_error", "full_gram_error", "off_diagonal_gram_error",
            "within_chunk_gram_error", "cross_chunk_gram_error",
            "same_prompt_fresh_Q_gram_distance",
            "same_prompt_fresh_Q_offdiag_distance",
        ):
            assert key in m

    # Limitations + paper-safe wording present.
    assert any("not formal" in s.lower() or "formal" in s.lower()
               for s in report["limitations"])
    assert "security-efficiency knob" in report["paper_safe_wording"]
    assert any("token-wise masking hides row norms" in s.lower()
               for s in report["unsafe_wording_to_avoid"])

    json_path, md_path = write_reports(report, outputs_dir=tmp_path)
    assert json_path.is_file()
    assert md_path.is_file()


# ---------------------------------------------------------------------------
# 11. Public artifacts present on disk
# ---------------------------------------------------------------------------


def test_public_outputs_artifact_present():
    json_path = REPO_ROOT / "outputs" / "norm_granularity_low_interaction.json"
    md_path = REPO_ROOT / "outputs" / "norm_granularity_low_interaction.md"
    assert json_path.is_file(), (
        "run_norm_granularity_low_interaction.py must be executed"
    )
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["stage"] == "7.6h"
    assert set(payload["modes_evaluated"]) == {"sequence", "chunk", "token"}


# ---------------------------------------------------------------------------
# 12. Decode-step token mode: each new decode step samples one fresh Q
# ---------------------------------------------------------------------------


def test_token_mode_decode_step_sees_per_step_fresh_Q(
    tiny_model_1layer, prompt_ids
):
    """Decode-step granularity matters most when prompt_len * extra rows
    differ; with S_new = 1 per decode step the wrapper should still sample
    one fresh Q per step. This test exercises the path with a 4-token
    decode."""
    wrapper = _wrap(tiny_model_1layer, granularity="token")
    g = torch.Generator(device="cpu").manual_seed(99)
    tokens, diag = wrapper.low_interaction_generate(
        prompt_ids, max_new_tokens=4, generator=g,
        diagnostics=LowInteractionDiagnostics(),
    )
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, tokens)
    assert diag.norm_q_is_per_row is True
    # The h_hat invariant must hold per-row across prefill + every decode.
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE
