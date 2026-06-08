"""Stage 7.6g -- tests for the RoPE-safe low-interaction generation
wrapper and its per-RoPE-pair norm leakage audit.

These tests verify that the *computation path itself* avoids plain
Q / K / V tensors on the accelerator (not just the diagnostic flags).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pllo.experiments.modern_decoder_rope_safe_low_interaction import (
    INVARIANT_TOLERANCE,
    KV_CACHE_TOLERANCE,
    LOGITS_TOLERANCE,
    RopeSafeLowInteractionConfig,
    run_rope_safe_low_interaction_correctness,
    write_reports,
)
from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
    apply_rope,
)
from pllo.wrappers.low_interaction_modern_decoder_generation_wrapper import (
    LowInteractionDiagnostics,
    LowInteractionTinyModernDecoderWrapper,
    generate_rope_plane_rotation_mask,
    verify_rope_commutation,
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


def _wrap_rope_safe(
    model: TinyModernDecoderForCausalLM,
) -> LowInteractionTinyModernDecoderWrapper:
    return LowInteractionTinyModernDecoderWrapper(
        model, use_pad=True,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
    )


def _generate_rope_safe(
    model: TinyModernDecoderForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    seed: int = 2028,
) -> tuple[torch.Tensor, LowInteractionDiagnostics]:
    wrapper = _wrap_rope_safe(model)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return wrapper.low_interaction_generate(
        prompt_ids, 4, generator=g, diagnostics=LowInteractionDiagnostics()
    )


# ---------------------------------------------------------------------------
# 1. RoPE-plane block-rotation mask is orthogonal
# ---------------------------------------------------------------------------


def test_rope_plane_rotation_mask_is_orthogonal():
    head_dim = 16
    g = torch.Generator(device="cpu").manual_seed(31)
    b = generate_rope_plane_rotation_mask(
        head_dim, dtype=torch.float64, device="cpu", generator=g,
    )
    identity = torch.eye(head_dim, dtype=torch.float64)
    torch.testing.assert_close(b @ b.T, identity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b.T @ b, identity, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. RoPE-plane block-rotation mask commutes with repo's apply_rope
# ---------------------------------------------------------------------------


def test_rope_plane_rotation_mask_commutes_with_repo_apply_rope():
    head_dim = 16
    base = 10000.0
    g = torch.Generator(device="cpu").manual_seed(41)
    b = generate_rope_plane_rotation_mask(
        head_dim, dtype=torch.float64, device="cpu", generator=g,
    )
    q = torch.randn(2, 4, 6, head_dim, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(42))
    positions = torch.arange(6)
    err = verify_rope_commutation(q, b, positions, base)
    assert err < 1e-12, f"RoPE commutation must hold to float64 precision; got {err}"
    # Direct equality check too.
    lhs = apply_rope(q @ b, positions, base)
    rhs = apply_rope(q, positions, base) @ b
    torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. qkv projection output IS masked directly (no plain Q/K/V transient)
# ---------------------------------------------------------------------------


def test_qkv_projection_outputs_are_masked_directly(
    tiny_model_1layer, prompt_ids
):
    """Stage 7.6g spec: the q/k projection accelerator-visible output
    is ``Q @ B_Q`` / ``K @ B_K`` directly, NOT plain Q / K. We check
    this by inspecting the precomputed table -- when fed the plain
    RMSNormCore(H) it should produce a tensor that equals
    plain_q @ B_Q_block per head (not plain_q)."""
    wrapper = _wrap_rope_safe(tiny_model_1layer)
    g = torch.Generator(device="cpu").manual_seed(2028)
    session = wrapper.compile_session(generator=g)
    cfg = tiny_model_1layer.cfg
    b, s, hidden = 2, 5, cfg.hidden_size
    tbl = wrapper._compile_layer_step_tables(
        layer_idx=0, h_hat_shape=(b, s, hidden), session=session, generator=g,
    )
    # Synthesise plain x and compute what the accelerator would see.
    plain_x = torch.randn(
        b, s, hidden, dtype=cfg.dtype, device=cfg.device,
        generator=torch.Generator().manual_seed(99),
    )
    q_l = session.q_layer[0]
    # The accelerator state at entry is X Q_l. After RMSNorm core in
    # the wrapper, the state is RMSNormCore(X) Q_l. For this test we
    # bypass RMSNorm by feeding plain_x directly as ``x_hat = X @ Q_l``.
    x_hat = plain_x @ q_l
    qkv_tbl = tbl["qkv"]
    x_pad_qkv = x_hat @ qkv_tbl.a - qkv_tbl.c_t
    q_pre_tilde_flat = x_pad_qkv @ qkv_tbl.w_tilde[0] + qkv_tbl.c_linear[0]

    # Compute what plain_q would be (gamma-folded reference).
    gamma1 = tiny_model_1layer.layers[0].input_norm_weight
    w_q_plain = gamma1.unsqueeze(-1) * tiny_model_1layer.layers[0].attn.q_proj.weight.T
    plain_q_flat = plain_x @ w_q_plain   # NOT what accelerator sees
    plain_q = plain_q_flat.view(
        b, s, cfg.num_query_heads, cfg.head_dim
    ).transpose(1, 2)

    # Accelerator-visible per-head Q: should equal plain_q @ B_Q[h] per head.
    q_pre_tilde = q_pre_tilde_flat.view(
        b, s, cfg.num_query_heads, cfg.head_dim
    ).transpose(1, 2)
    for q_head in range(cfg.num_query_heads):
        b_q = session.n_q[0][q_head]
        expected = plain_q[:, q_head, :, :] @ b_q
        torch.testing.assert_close(
            q_pre_tilde[:, q_head, :, :], expected, atol=1e-10, rtol=1e-10
        )
        # The accelerator output must NOT equal plain_q for any head
        # (with high probability under a random fresh B_Q).
        plain_err = (
            q_pre_tilde[:, q_head, :, :] - plain_q[:, q_head, :, :]
        ).abs().max().item()
        assert plain_err > 1e-6, (
            f"q_pre_tilde unexpectedly equals plain Q on head {q_head} "
            "-- B_Q failed to mask"
        )


# ---------------------------------------------------------------------------
# 4. No transient plain Q/K/V visible in diagnostics
# ---------------------------------------------------------------------------


def test_no_transient_plain_qkv_visible_in_diagnostics(
    tiny_model_1layer, prompt_ids
):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.rope_mask_mode == "pre_rope_block_diagonal_rotation"
    assert diag.rope_mode == "pre_rope_block_diagonal_rotation"
    assert diag.rope_transient_plain_qk_visible is False
    assert diag.rope_transient_plain_v_visible is False
    assert diag.qkv_projection_outputs_masked_directly is True
    assert diag.trusted_rope_recovery_used is False
    assert diag.generic_pre_rope_dense_commutation_used is False
    assert diag.rope_blocker_transient_plain_qk_on_accelerator is False


# ---------------------------------------------------------------------------
# 5. GQA B_Q / B_K constraint
# ---------------------------------------------------------------------------


def test_gqa_b_q_b_k_constraint(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.qk_constraint_max_error < 1e-12


# ---------------------------------------------------------------------------
# 6. RoPE score invariant
# ---------------------------------------------------------------------------


def test_rope_score_invariant(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.qk_score_invariant_max_abs_error < INVARIANT_TOLERANCE
    assert diag.rope_commutation_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 7. KV cache invariant + masked-only storage
# ---------------------------------------------------------------------------


def test_kv_cache_invariant(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.kv_cache_invariant_max_abs_error < KV_CACHE_TOLERANCE


# ---------------------------------------------------------------------------
# 8. Padded greedy generation exact match
# ---------------------------------------------------------------------------


def test_padded_greedy_generation_exact_match(tiny_model_1layer, prompt_ids):
    tokens, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    plain_tokens = tiny_model_1layer.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, tokens)
    assert diag.prefill_logits_max_abs_error < LOGITS_TOLERANCE
    assert diag.decode_step_logits_max_abs_error_max < LOGITS_TOLERANCE


# ---------------------------------------------------------------------------
# 9. online_boundary_round_trips_per_decode_step == 1
# ---------------------------------------------------------------------------


def test_online_boundary_round_trips_eq_one(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.online_boundary_round_trips_per_decode_step == 1


# ---------------------------------------------------------------------------
# 10. intermediate_tee_reentry == false + trusted_fallback_used_in_main_path == false
# ---------------------------------------------------------------------------


def test_no_tee_reentry_no_trusted_fallback(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.intermediate_tee_reentry is False
    assert diag.trusted_fallback_used_in_main_path is False


# ---------------------------------------------------------------------------
# 11. use_pad == True + pad_enters_*_core == False
# ---------------------------------------------------------------------------


def test_use_pad_and_pad_does_not_enter_cores(tiny_model_1layer, prompt_ids):
    _, diag = _generate_rope_safe(tiny_model_1layer, prompt_ids)
    assert diag.use_pad is True
    assert diag.fresh_pad_used_at_linear_boundaries is True
    assert diag.pad_at_linear_boundaries is True
    assert diag.pad_enters_rmsnorm_core is False
    assert diag.pad_enters_rope_core is False
    assert diag.pad_enters_swiglu_core is False
    assert diag.pad_enters_softmax is False


# ---------------------------------------------------------------------------
# 12. Two-layer model also satisfies the RoPE-safe invariants
# ---------------------------------------------------------------------------


def test_two_layer_rope_safe(tiny_model_2layer, prompt_ids):
    tokens, diag = _generate_rope_safe(tiny_model_2layer, prompt_ids)
    plain_tokens = tiny_model_2layer.greedy_generate(prompt_ids, 4)
    assert torch.equal(plain_tokens, tokens)
    assert diag.h_hat_layer_entry_invariant_max_abs_error < INVARIANT_TOLERANCE
    assert diag.rope_commutation_max_abs_error < INVARIANT_TOLERANCE
    assert diag.qk_score_invariant_max_abs_error < INVARIANT_TOLERANCE


# ---------------------------------------------------------------------------
# 13. RoPE-pair norm leakage audit reports the residual leakage surface
# ---------------------------------------------------------------------------


def test_rope_pair_norm_leakage_audit():
    report = run_rope_safe_low_interaction_correctness(
        cfg=RopeSafeLowInteractionConfig()
    )
    leakage = report["rope_pair_norm_leakage_audit"]
    assert leakage["rope_pair_norm_leakage"] is True
    # Per-RoPE-pair 2D norms are mathematically preserved by the
    # block-rotation mask -- error must be at float64 round-off.
    assert leakage["rope_pair_norm_max_abs_error"] < 1e-12
    assert leakage["rope_commutation_max_abs_error_audit"] < 1e-12
    # Explanation must say the per-pair 2D norm is preserved.
    explanation = leakage["explanation"].lower()
    assert "norm" in explanation
    assert "pair" in explanation


# ---------------------------------------------------------------------------
# 14. Report carries all Stage 7.6g mandated fields
# ---------------------------------------------------------------------------


def test_report_carries_mandated_main_fields(tmp_path: Path):
    report = run_rope_safe_low_interaction_correctness(
        cfg=RopeSafeLowInteractionConfig()
    )
    for key, value in (
        ("status", "ok"),
        ("rope_mask_mode", "pre_rope_block_diagonal_rotation"),
        ("rope_transient_plain_qk_visible", False),
        ("rope_transient_plain_v_visible", False),
        ("qkv_projection_outputs_masked_directly", True),
        ("trusted_rope_recovery_used", False),
        ("generic_pre_rope_dense_commutation_used", False),
        ("trusted_fallback_used_in_main_path", False),
        ("intermediate_tee_reentry", False),
        ("online_boundary_round_trips_per_decode_step", 1),
        ("use_pad", True),
        ("main_layer_invariant", "H_hat_l = H_l Q_l"),
        ("rmsnorm_mode", "operator_compatible_orthogonal"),
    ):
        assert report[key] == value, f"{key} != {value!r} (got {report[key]!r})"

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
# 15. Same input + fresh masks -> same tokens, different fingerprints
# ---------------------------------------------------------------------------


def test_same_input_fresh_masks_different_fingerprints(
    tiny_model_1layer, prompt_ids
):
    tokens_a, diag_a = _generate_rope_safe(tiny_model_1layer, prompt_ids, seed=1234)
    tokens_b, diag_b = _generate_rope_safe(tiny_model_1layer, prompt_ids, seed=5678)
    assert torch.equal(tokens_a, tokens_b)
    fp_a = diag_a.masked_boundary_fingerprints
    fp_b = diag_b.masked_boundary_fingerprints
    assert fp_a["prefill_layer_entry_h_hat"] != fp_b["prefill_layer_entry_h_hat"]
    assert (
        fp_a["prefill_lm_head_logits_tilde"]
        != fp_b["prefill_lm_head_logits_tilde"]
    )


# ---------------------------------------------------------------------------
# 16. Public outputs/ artifact is present
# ---------------------------------------------------------------------------


def test_public_outputs_artifact_present():
    json_path = (
        REPO_ROOT / "outputs" / "modern_decoder_rope_safe_low_interaction.json"
    )
    md_path = (
        REPO_ROOT / "outputs" / "modern_decoder_rope_safe_low_interaction.md"
    )
    assert json_path.is_file(), (
        "run_modern_decoder_rope_safe_low_interaction.py must be executed"
    )
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["rope_mask_mode"] == "pre_rope_block_diagonal_rotation"
    assert payload["rope_transient_plain_qk_visible"] is False
    assert payload["rope_transient_plain_v_visible"] is False
    assert payload["qkv_projection_outputs_masked_directly"] is True
    assert payload["use_pad"] is True
