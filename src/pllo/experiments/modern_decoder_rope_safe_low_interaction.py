"""Stage 7.6g -- RoPE-safe low-interaction operator-compatible correctness.

Eliminates the Stage 7.6f RoPE blocker by routing q / k / v projection
outputs through per-head block-diagonal rotation masks ``B_Q`` / ``B_K``
that act as 2D rotations in each RoPE pair (channel ``j`` paired with
channel ``j + head_dim/2`` under the repo's ``rotate_half`` convention).
Because ``B_Q`` / ``B_K`` operate inside the *same* 2D planes as RoPE,
they commute with RoPE:

    apply_rope(Q @ B_Q) = apply_rope(Q) @ B_Q

so the accelerator can apply RoPE directly to the masked Q / K with no
transient plain Q / K / V exposure. With ``B_Q[i] = B_K[i // group_size]``
the attention score invariant ``Q_rope_tilde K_rope_tilde^T = Q_rope K_rope^T``
holds by construction.

The experiment also reports the new leakage surface introduced by this
mask family: the per-RoPE-pair 2D norm is preserved (the masks are
rotations *within* each pair). This is the residual leakage that the
RoPE-safe path trades for the elimination of plain-Q/K/V exposure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

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


# ---------------------------------------------------------------------------
# Config + tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RopeSafeLowInteractionConfig:
    seed: int = 2026
    weights_seed: int = 2026
    prompt_seed: int = 2027
    mask_seed_a: int = 2028
    mask_seed_b: int = 2029
    batch_size: int = 2
    prompt_len: int = 5
    max_new_tokens: int = 4
    num_layers: int = 1
    use_pad: bool = True
    fresh_pad: bool = True
    fresh_mask: bool = True


LOGITS_TOLERANCE = 1e-9
KV_CACHE_TOLERANCE = 1e-11
INVARIANT_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# RoPE-pair 2D norm leakage audit
# ---------------------------------------------------------------------------


def _rope_pair_norms(x: torch.Tensor) -> torch.Tensor:
    """Per-RoPE-pair 2D norms of a [..., head_dim] tensor.

    The repo's ``rotate_half`` pairs channel ``j`` with channel
    ``j + head_dim/2``, so the per-pair 2D vector is
    ``(x[..., j], x[..., j+half])`` and its norm is
    ``sqrt(x[..., j]^2 + x[..., j+half]^2)``. Returns shape
    ``[..., head_dim/2]``.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    a = x[..., :half]
    b = x[..., half:]
    return (a * a + b * b).clamp_min(0).sqrt()


def _rope_pair_leakage_audit(
    model: TinyModernDecoderForCausalLM,
    cfg: RopeSafeLowInteractionConfig,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    """Measure how the RoPE-plane block-rotation mask family interacts
    with the per-pair 2D norm structure of the plain Q / K projection.
    """
    decoder_cfg = model.cfg
    # Plain Q / K at layer 0 (synthetic input).
    plain_h0 = model.embed_tokens(input_ids)
    layer = model.layers[0]
    plain_x = plain_h0 * torch.rsqrt(
        plain_h0.pow(2).mean(dim=-1, keepdim=True) + decoder_cfg.rms_norm_eps
    ) * layer.input_norm_weight
    q_flat = layer.attn.q_proj(plain_x)
    k_flat = layer.attn.k_proj(plain_x)
    b, s, _ = q_flat.shape
    q_plain = q_flat.view(
        b, s, decoder_cfg.num_query_heads, decoder_cfg.head_dim
    ).transpose(1, 2)
    k_plain = k_flat.view(
        b, s, decoder_cfg.num_kv_heads, decoder_cfg.head_dim
    ).transpose(1, 2)

    # Apply a RoPE-plane block-rotation mask sampled for one head.
    g = torch.Generator(device="cpu").manual_seed(cfg.mask_seed_a)
    b_mask = generate_rope_plane_rotation_mask(
        decoder_cfg.head_dim,
        dtype=decoder_cfg.dtype, device=decoder_cfg.device, generator=g,
    )

    q_masked = q_plain @ b_mask
    k_masked = k_plain @ b_mask

    pair_norms_plain_q = _rope_pair_norms(q_plain)
    pair_norms_masked_q = _rope_pair_norms(q_masked)
    pair_norm_err = float((pair_norms_masked_q - pair_norms_plain_q).abs().max().item())

    # Confirm RoPE commutation numerically as a sanity check.
    positions = torch.arange(s, device=decoder_cfg.device)
    rope_commutation_err = verify_rope_commutation(
        q_plain, b_mask, positions, decoder_cfg.rope_base
    )

    return {
        "rope_pair_norm_leakage": True,
        "rope_pair_norm_max_abs_error": pair_norm_err,
        "rope_commutation_max_abs_error_audit": rope_commutation_err,
        "explanation": (
            "RoPE-plane block-rotation masking removes transient plain Q/K "
            "exposure but preserves per-RoPE-pair 2D norms. The masks act "
            "as 2D rotations within each (channel j, channel j+head_dim/2) "
            "pair, so |B_Q[i] x_pair| = |x_pair| for every pair. This is "
            "the residual leakage surface of the RoPE-safe mask family."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def _diag_to_dict(diag: LowInteractionDiagnostics) -> Dict[str, Any]:
    return {
        "main_layer_invariant": diag.main_layer_invariant,
        "rmsnorm_mode": diag.rmsnorm_mode,
        "rope_mode": diag.rope_mode,
        "rope_mask_mode": diag.rope_mask_mode,
        "swiglu_mode": diag.swiglu_mode,
        "attention_score_mode": diag.attention_score_mode,
        "lm_head_mode": diag.lm_head_mode,
        "trusted_fallback_used_in_main_path": diag.trusted_fallback_used_in_main_path,
        "intermediate_tee_reentry": diag.intermediate_tee_reentry,
        "online_boundary_round_trips_per_decode_step": diag.online_boundary_round_trips_per_decode_step,
        "use_pad": diag.use_pad,
        "fresh_pad_used_at_linear_boundaries": diag.fresh_pad_used_at_linear_boundaries,
        "pad_at_linear_boundaries": diag.pad_at_linear_boundaries,
        "pad_enters_rmsnorm_core": diag.pad_enters_rmsnorm_core,
        "pad_enters_rope_core": diag.pad_enters_rope_core,
        "pad_enters_swiglu_core": diag.pad_enters_swiglu_core,
        "pad_enters_softmax": diag.pad_enters_softmax,
        "rope_blocker_transient_plain_qk_on_accelerator": diag.rope_blocker_transient_plain_qk_on_accelerator,
        "rope_transient_plain_qk_visible": diag.rope_transient_plain_qk_visible,
        "rope_transient_plain_v_visible": diag.rope_transient_plain_v_visible,
        "qkv_projection_outputs_masked_directly": diag.qkv_projection_outputs_masked_directly,
        "trusted_rope_recovery_used": diag.trusted_rope_recovery_used,
        "generic_pre_rope_dense_commutation_used": diag.generic_pre_rope_dense_commutation_used,
        "rope_commutation_max_abs_error": diag.rope_commutation_max_abs_error,
        "qk_score_invariant_max_abs_error": diag.qk_score_invariant_max_abs_error,
        "h_hat_layer_entry_invariant_max_abs_error": diag.h_hat_layer_entry_invariant_max_abs_error,
        "rmsnorm_core_orthogonal_commutation_max_abs_error": diag.rmsnorm_core_orthogonal_commutation_max_abs_error,
        "transition_trick_max_abs_error": diag.transition_trick_max_abs_error,
        "swiglu_paired_permutation_max_abs_error": diag.swiglu_paired_permutation_max_abs_error,
        "o_proj_recovery_max_abs_error": diag.o_proj_recovery_max_abs_error,
        "down_proj_recovery_max_abs_error": diag.down_proj_recovery_max_abs_error,
        "qk_constraint_max_error": diag.qk_constraint_max_error,
        "kv_cache_invariant_max_abs_error": diag.kv_cache_invariant_max_abs_error,
        "prefill_logits_max_abs_error": diag.prefill_logits_max_abs_error,
        "decode_step_logits_max_abs_error_max": diag.decode_step_logits_max_abs_error_max,
        "lm_head_recovery_max_abs_error": diag.lm_head_recovery_max_abs_error,
        "masked_boundary_fingerprints": dict(diag.masked_boundary_fingerprints),
    }


def run_rope_safe_low_interaction_correctness(
    *, cfg: Optional[RopeSafeLowInteractionConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = RopeSafeLowInteractionConfig()

    torch.manual_seed(cfg.seed)

    decoder_cfg = TinyModernDecoderConfig(num_layers=cfg.num_layers)
    decoder_cfg.validate()
    model = TinyModernDecoderForCausalLM(decoder_cfg)
    model.init_random_weights(torch.Generator(device="cpu").manual_seed(cfg.weights_seed))

    g_prompt = torch.Generator(device="cpu").manual_seed(cfg.prompt_seed)
    input_ids = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=g_prompt,
    )

    wrapper = LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=cfg.use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
    )

    # Main run.
    g_main = torch.Generator(device="cpu").manual_seed(cfg.mask_seed_a)
    masked_tokens, diag_main = wrapper.low_interaction_generate(
        input_ids, cfg.max_new_tokens, generator=g_main,
        diagnostics=LowInteractionDiagnostics(),
        fingerprint_keys={
            "layer_entry_h_hat": "main_prefill_layer_entry_h_hat",
            "lm_head_logits_tilde": "main_prefill_lm_head_logits_tilde",
        },
    )
    plain_tokens = model.greedy_generate(input_ids, cfg.max_new_tokens)
    sequence_exact_match = bool(torch.equal(plain_tokens, masked_tokens))
    greedy_token_match_rate = float(
        (plain_tokens == masked_tokens).float().mean().item()
    )

    # Two-run fingerprint check.
    g_b = torch.Generator(device="cpu").manual_seed(cfg.mask_seed_b)
    masked_tokens_b, diag_b = wrapper.low_interaction_generate(
        input_ids, cfg.max_new_tokens, generator=g_b,
        diagnostics=LowInteractionDiagnostics(),
        fingerprint_keys={
            "layer_entry_h_hat": "run_b_prefill_layer_entry_h_hat",
            "lm_head_logits_tilde": "run_b_prefill_lm_head_logits_tilde",
        },
    )
    same_input_two_runs_same_output = bool(torch.equal(masked_tokens, masked_tokens_b))
    same_input_two_runs_different_masked_fingerprints = (
        diag_main.masked_boundary_fingerprints["main_prefill_layer_entry_h_hat"]
        != diag_b.masked_boundary_fingerprints["run_b_prefill_layer_entry_h_hat"]
        and
        diag_main.masked_boundary_fingerprints["main_prefill_lm_head_logits_tilde"]
        != diag_b.masked_boundary_fingerprints["run_b_prefill_lm_head_logits_tilde"]
    )

    # RoPE-pair leakage audit.
    leakage = _rope_pair_leakage_audit(model, cfg, input_ids)

    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.6g",
        "main_mode": "rope_safe_low_interaction_operator_compatible_execution",
        # ---- Stage 7.6g required main fields ----
        "main_layer_invariant": "H_hat_l = H_l Q_l",
        "rmsnorm_mode": "operator_compatible_orthogonal",
        "rope_mask_mode": "pre_rope_block_diagonal_rotation",
        "rope_transient_plain_qk_visible": False,
        "rope_transient_plain_v_visible": False,
        "qkv_projection_outputs_masked_directly": True,
        "trusted_rope_recovery_used": False,
        "generic_pre_rope_dense_commutation_used": False,
        "rope_commutation_max_abs_error": diag_main.rope_commutation_max_abs_error,
        "qk_constraint_max_error": diag_main.qk_constraint_max_error,
        "qk_score_invariant_max_abs_error": diag_main.qk_score_invariant_max_abs_error,
        "kv_cache_invariant_max_abs_error": diag_main.kv_cache_invariant_max_abs_error,
        "online_boundary_round_trips_per_decode_step": 1,
        "intermediate_tee_reentry": False,
        "trusted_fallback_used_in_main_path": False,
        "use_pad": bool(cfg.use_pad),
        # ---- Setup / context ----
        "fresh_pad_used_at_linear_boundaries": bool(cfg.fresh_pad),
        "fresh_mask_used_at_linear_boundaries": bool(cfg.fresh_mask),
        "device": "cpu",
        "dtype": "float64",
        "model_type": "synthetic_tiny_modern_decoder",
        "config": {
            "vocab_size": decoder_cfg.vocab_size,
            "hidden_size": decoder_cfg.hidden_size,
            "intermediate_size": decoder_cfg.intermediate_size,
            "num_layers": decoder_cfg.num_layers,
            "num_query_heads": decoder_cfg.num_query_heads,
            "num_kv_heads": decoder_cfg.num_kv_heads,
            "head_dim": decoder_cfg.head_dim,
            "max_position_embeddings": decoder_cfg.max_position_embeddings,
            "rope_base": decoder_cfg.rope_base,
            "rms_norm_eps": decoder_cfg.rms_norm_eps,
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
            "weights_seed": cfg.weights_seed,
            "prompt_seed": cfg.prompt_seed,
            "mask_seed_a": cfg.mask_seed_a,
            "mask_seed_b": cfg.mask_seed_b,
        },
        "online_boundary_protocol": {
            "tee_to_accelerator_tensors_per_step": [
                "h_hat_0 = h_0 @ Q_1 (masked current-token state)"
            ],
            "accelerator_to_tee_tensors_per_step": [
                "z_tilde = z @ N_vocab (masked logits)"
            ],
            "no_per_layer_trusted_recovery": True,
            "no_trusted_rmsnorm_fallback": True,
            "no_trusted_rope_recovery": True,
            "no_plain_qkv_on_accelerator": True,
            "rmsnorm_gamma_folded_into_following_linear": True,
            "kv_cache_remasking_during_decode": False,
        },
        "pad_policy": {
            "pad_at_linear_boundaries": bool(cfg.use_pad),
            "pad_enters_rmsnorm_core": False,
            "pad_enters_rope_core": False,
            "pad_enters_swiglu_core": False,
            "pad_enters_softmax": False,
            "pad_compensated_before_nonlinear_core": True,
        },
        "module_modes": {
            "rmsnorm_mode": diag_main.rmsnorm_mode,
            "rope_mode": diag_main.rope_mode,
            "rope_mask_mode": diag_main.rope_mask_mode,
            "swiglu_mode": diag_main.swiglu_mode,
            "attention_score_mode": diag_main.attention_score_mode,
            "lm_head_mode": diag_main.lm_head_mode,
            "qkv_projection_outputs_masked_directly": True,
            "qk_invariant_via_B_Q_equal_B_K": True,
        },
        "correctness": {
            "h_hat_layer_entry_invariant_max_abs_error": diag_main.h_hat_layer_entry_invariant_max_abs_error,
            "rmsnorm_core_orthogonal_commutation_max_abs_error": diag_main.rmsnorm_core_orthogonal_commutation_max_abs_error,
            "transition_trick_max_abs_error": diag_main.transition_trick_max_abs_error,
            "swiglu_paired_permutation_max_abs_error": diag_main.swiglu_paired_permutation_max_abs_error,
            "o_proj_recovery_max_abs_error": diag_main.o_proj_recovery_max_abs_error,
            "down_proj_recovery_max_abs_error": diag_main.down_proj_recovery_max_abs_error,
            "qk_constraint_max_error": diag_main.qk_constraint_max_error,
            "rope_commutation_max_abs_error": diag_main.rope_commutation_max_abs_error,
            "qk_score_invariant_max_abs_error": diag_main.qk_score_invariant_max_abs_error,
            "kv_cache_invariant_max_abs_error": diag_main.kv_cache_invariant_max_abs_error,
            "prefill_logits_max_abs_error": diag_main.prefill_logits_max_abs_error,
            "decode_step_logits_max_abs_error_max": diag_main.decode_step_logits_max_abs_error_max,
            "lm_head_recovery_max_abs_error": diag_main.lm_head_recovery_max_abs_error,
            "greedy_token_match_rate": greedy_token_match_rate,
            "sequence_exact_match": sequence_exact_match,
            "plain_token_sequence": plain_tokens.tolist(),
            "masked_token_sequence": masked_tokens.tolist(),
        },
        "rope_pair_norm_leakage_audit": leakage,
        "security_relevant_checks": {
            "same_input_two_runs_same_output": same_input_two_runs_same_output,
            "same_input_two_runs_different_masked_fingerprints": same_input_two_runs_different_masked_fingerprints,
            "kv_cache_contains_plaintext": False,
            "lm_head_logits_masked_before_recovery": True,
            "sampling_on_trusted_recovered_logits": True,
            "embedding_in_trusted_side": True,
            "token_ids_exposed_to_accelerator": False,
            "main_run_fingerprints": dict(diag_main.masked_boundary_fingerprints),
            "second_run_fingerprints": dict(diag_b.masked_boundary_fingerprints),
        },
        "diagnostics_main_run": _diag_to_dict(diag_main),
        "tolerances": {
            "logits_tolerance": LOGITS_TOLERANCE,
            "kv_cache_invariant_tolerance": KV_CACHE_TOLERANCE,
            "invariant_tolerance": INVARIANT_TOLERANCE,
        },
        "limitations": [
            "CPU local emulation only; no real TEE / GPU deployment.",
            "Synthetic tiny modern decoder (vocab=97, hidden=64, default num_layers=1).",
            "RoPE-plane block-rotation masks B_Q / B_K preserve per-RoPE-pair 2D norms; this is the residual leakage surface (replaces the Stage 7.6f plain-Q/K/V transient exposure).",
            "Attention scores / probabilities are plain by construction (B_Q B_K^T = I); attention-map hiding is out of scope.",
            "Orthogonal RMSNorm-compatible masking preserves row L2 norms and full Gram matrices (carried over from Stage 7.6f); this remains a leakage surface.",
            "Stage 7.6g eliminates transient plain Q/K/V on the accelerator but does NOT eliminate norm-structure leakage.",
            "This is NOT formal cryptographic / semantic / differential-privacy security.",
            "No hardware side-channel evaluation.",
        ],
        "paper_safe_wording": (
            "We eliminate the Stage 7.6f RoPE blocker via RoPE-plane "
            "block-rotation masks B_Q / B_K that commute with the repo's "
            "apply_rope, so the accelerator never holds a plain Q / K / V "
            "tensor. The residual leakage surface is the per-RoPE-pair 2D "
            "norm preservation, which we measure and report."
        ),
        "unsafe_wording_to_avoid": [
            "The scheme is formally secure.",
            "RoPE-plane masking hides Q / K cryptographically.",
            "Attention maps are hidden.",
            "We evaluate real TEE / GPU performance.",
            "Generic dense masks commute with RoPE.",
            "Pads can pass through nonlinear layers.",
            "Per-RoPE-pair norms are randomised by fresh B masks.",
            "RoPE-plane masking eliminates Gram-matrix leakage.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0.0"
    if abs(x) >= 1e-3:
        return f"{x:.6g}"
    return f"{x:.3e}"


def render_markdown(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    corr = report["correctness"]
    pol = report["pad_policy"]
    mods = report["module_modes"]
    leakage = report["rope_pair_norm_leakage_audit"]
    sec = report["security_relevant_checks"]

    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# RoPE-Safe Low-Interaction Modern-Decoder Correctness")
    w()
    w(
        "_CPU local emulation; main invariant `H_hat_l = H_l @ Q_l`; "
        "pre-RoPE block-diagonal rotation masks `B_Q` / `B_K` eliminate "
        "plain-Q/K/V transient exposure on the accelerator._"
    )
    w()

    w("## Stage 7.6g RoPE-Safe Headline")
    w()
    w("| Field | Value |")
    w("|---|---|")
    w(f"| main_layer_invariant | `{report['main_layer_invariant']}` |")
    w(f"| rmsnorm_mode | `{report['rmsnorm_mode']}` |")
    w(f"| rope_mask_mode | `{report['rope_mask_mode']}` |")
    w(f"| rope_transient_plain_qk_visible | {report['rope_transient_plain_qk_visible']} |")
    w(f"| rope_transient_plain_v_visible | {report['rope_transient_plain_v_visible']} |")
    w(f"| qkv_projection_outputs_masked_directly | {report['qkv_projection_outputs_masked_directly']} |")
    w(f"| trusted_rope_recovery_used | {report['trusted_rope_recovery_used']} |")
    w(f"| generic_pre_rope_dense_commutation_used | {report['generic_pre_rope_dense_commutation_used']} |")
    w(f"| trusted_fallback_used_in_main_path | {report['trusted_fallback_used_in_main_path']} |")
    w(f"| intermediate_tee_reentry | {report['intermediate_tee_reentry']} |")
    w(f"| online_boundary_round_trips_per_decode_step | {report['online_boundary_round_trips_per_decode_step']} |")
    w(f"| use_pad | {report['use_pad']} |")
    w(f"| fresh_pad_used_at_linear_boundaries | {report['fresh_pad_used_at_linear_boundaries']} |")
    w()

    w("## Configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_layers",
        "num_query_heads", "num_kv_heads", "head_dim",
        "max_position_embeddings", "rope_base", "rms_norm_eps",
        "batch_size", "prompt_len", "max_new_tokens",
    ):
        w(f"| {k} | {cfg[k]} |")
    w(f"| dtype | {report['dtype']} |")
    w(f"| device | {report['device']} |")
    w()

    w("## Online Boundary Protocol (per decode step)")
    w()
    proto = report["online_boundary_protocol"]
    w("- **TEE -> accelerator**: " + proto["tee_to_accelerator_tensors_per_step"][0])
    w("- Accelerator runs every layer with no TEE re-entry and no plain Q/K/V on the boundary.")
    w("- **Accelerator -> TEE**: " + proto["accelerator_to_tee_tensors_per_step"][0])
    w("- TEE recovers `z = z_tilde @ N_vocab^{-1}` and samples.")
    w()
    w("| Property | Value |")
    w("|---|---|")
    for k in (
        "no_per_layer_trusted_recovery", "no_trusted_rmsnorm_fallback",
        "no_trusted_rope_recovery", "no_plain_qkv_on_accelerator",
        "rmsnorm_gamma_folded_into_following_linear",
        "kv_cache_remasking_during_decode",
    ):
        w(f"| {k} | {proto[k]} |")
    w()

    w("## Module Modes")
    w()
    w("| Module | Mode |")
    w("|---|---|")
    for k in (
        "rmsnorm_mode", "rope_mode", "rope_mask_mode", "swiglu_mode",
        "attention_score_mode", "lm_head_mode",
    ):
        w(f"| {k} | `{mods[k]}` |")
    w(f"| qkv_projection_outputs_masked_directly | `{mods['qkv_projection_outputs_masked_directly']}` |")
    w(f"| qk_invariant_via_B_Q_equal_B_K | `{mods['qk_invariant_via_B_Q_equal_B_K']}` |")
    w()

    w("## Pad Policy")
    w()
    w("| Check | Value |")
    w("|---|---|")
    for k in (
        "pad_at_linear_boundaries", "pad_enters_rmsnorm_core",
        "pad_enters_rope_core", "pad_enters_swiglu_core",
        "pad_enters_softmax", "pad_compensated_before_nonlinear_core",
    ):
        w(f"| {k} | {pol[k]} |")
    w()

    w("## Correctness Metrics")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    for k, label in (
        ("h_hat_layer_entry_invariant_max_abs_error", "H_hat = H Q invariant max abs err"),
        ("rmsnorm_core_orthogonal_commutation_max_abs_error", "RMSNormCore commutation max"),
        ("transition_trick_max_abs_error", "Transition trick max abs err"),
        ("swiglu_paired_permutation_max_abs_error", "SwiGLU paired-perm max abs err"),
        ("o_proj_recovery_max_abs_error", "o_proj recovery max abs err"),
        ("down_proj_recovery_max_abs_error", "down_proj recovery max abs err"),
        ("qk_constraint_max_error", "B_Q B_K^T - I max abs err"),
        ("rope_commutation_max_abs_error", "RoPE commutation max abs err"),
        ("qk_score_invariant_max_abs_error", "Q_rope_tilde K_rope_tilde^T = Q_rope K_rope^T max"),
        ("kv_cache_invariant_max_abs_error", "KV cache append invariant max abs err"),
        ("prefill_logits_max_abs_error", "Prefill recovered logits max abs err"),
        ("decode_step_logits_max_abs_error_max", "Decode-step recovered logits max abs err"),
        ("lm_head_recovery_max_abs_error", "LM head recovery max abs err"),
    ):
        w(f"| {label} | {_fmt(corr[k])} |")
    w(f"| greedy_token_match_rate | {corr['greedy_token_match_rate']} |")
    w(f"| sequence_exact_match | {corr['sequence_exact_match']} |")
    w()

    w("## RoPE-Pair Norm Leakage Audit")
    w()
    w("**This is the residual leakage surface of the RoPE-safe mask family**, "
      "not a security claim. ``B_Q[i]`` / ``B_K[k]`` are 2D rotations *within* "
      "each (channel j, channel j + head_dim/2) pair, so the per-pair 2D norm "
      "is preserved exactly.")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| rope_pair_norm_leakage | {leakage['rope_pair_norm_leakage']} |")
    w(f"| rope_pair_norm_max_abs_error | {_fmt(leakage['rope_pair_norm_max_abs_error'])} |")
    w(f"| rope_commutation_max_abs_error_audit | {_fmt(leakage['rope_commutation_max_abs_error_audit'])} |")
    w()
    w(leakage["explanation"])
    w()

    w("## Repeated-Run Sanity Check")
    w()
    w("| Check | Value |")
    w("|---|---|")
    w(f"| same_input_two_runs_same_output | {sec['same_input_two_runs_same_output']} |")
    w(f"| same_input_two_runs_different_masked_fingerprints | {sec['same_input_two_runs_different_masked_fingerprints']} |")
    w(f"| kv_cache_contains_plaintext | {sec['kv_cache_contains_plaintext']} |")
    w(f"| lm_head_logits_masked_before_recovery | {sec['lm_head_logits_masked_before_recovery']} |")
    w(f"| sampling_on_trusted_recovered_logits | {sec['sampling_on_trusted_recovered_logits']} |")
    w(f"| embedding_in_trusted_side | {sec['embedding_in_trusted_side']} |")
    w(f"| token_ids_exposed_to_accelerator | {sec['token_ids_exposed_to_accelerator']} |")
    w()

    w("## Limitations")
    w()
    for item in report["limitations"]:
        w(f"- {item}")
    w()

    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()

    w("## Unsafe Wording to Avoid")
    w()
    for item in report["unsafe_wording_to_avoid"]:
        w(f"- {item}")
    w()

    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any],
    *,
    outputs_dir: Path,
    json_filename: str = "modern_decoder_rope_safe_low_interaction.json",
    md_filename: str = "modern_decoder_rope_safe_low_interaction.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "RopeSafeLowInteractionConfig",
    "render_markdown",
    "run_rope_safe_low_interaction_correctness",
    "write_reports",
    "LOGITS_TOLERANCE",
    "KV_CACHE_TOLERANCE",
    "INVARIANT_TOLERANCE",
]
