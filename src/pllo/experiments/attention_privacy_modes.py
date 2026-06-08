"""Stage 7.6i -- Attention-Map Protection / Secure-Attention Modes.

Stage 7.6h kept the QK invariant ``Q_tilde K_tilde^T == Q K^T`` and so
intentionally preserves the attention map on the accelerator boundary:
the accelerator can read ``S = Q K^T / sqrt(d_h) + causal_mask`` and
``P = softmax(S)`` plainly. That is fine for exact low-interaction
correctness but is *not* a private attention surface.

Stage 7.6i adds an ``attention_privacy_mode`` knob on top of the Stage
7.6h wrapper:

* ``exact_visible_attention`` -- the Stage 7.6h baseline; attention
  scores and probabilities are visible by construction. Exact, single-
  round-trip.
* ``trusted_softmax_attention`` -- the accelerator ships masked
  Q_tilde / K_tilde / V_tilde to the trusted side; trusted side
  recovers plain Q_rope / K_rope / V via the orthogonal masks it owns,
  runs softmax in trusted memory, and returns ``attn_out_tilde``.
  Exact but with *extra TEE round trips* per decode step. The
  attention map never appears in the accelerator transcript.
* ``score_blinding_experimental`` -- demonstration mode. Adds a
  *row-constant* shift ``c_i`` to scores; softmax is invariant under
  row-constant shifts so the output is exact, but ranking, relative
  margins, and attention topology are unchanged. We also record a
  *non-row-constant* random ``R`` shift on a separate copy so the
  report can show explicitly how that breaks softmax. This mode
  exists only to prove that trivial score blinding does NOT provide
  attention privacy.

This stage emits a comparison table across the three modes plus an
attention-leakage audit. It does NOT claim cryptographic security; it
explicitly documents the *exactness--hiding tension* of accelerator-
side softmax.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
)
from pllo.wrappers.low_interaction_modern_decoder_generation_wrapper import (
    LowInteractionDiagnostics,
    LowInteractionTinyModernDecoderWrapper,
)


# ---------------------------------------------------------------------------
# Config + tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionPrivacyModesConfig:
    seed: int = 2026
    weights_seed: int = 2026
    prompt_seed: int = 2027
    mask_seed: int = 2028
    batch_size: int = 2
    prompt_len: int = 6
    max_new_tokens: int = 3
    num_layers: int = 1
    use_pad: bool = True
    fresh_pad: bool = True
    fresh_mask: bool = True
    norm_mask_granularity: str = "sequence"
    norm_chunk_size: int = 1


LOGITS_TOLERANCE = 1e-9
INVARIANT_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Single mode run
# ---------------------------------------------------------------------------


def _diag_to_summary(diag: LowInteractionDiagnostics) -> Dict[str, Any]:
    return {
        # Stage 7.6i mode-specific fields.
        "attention_privacy_mode": diag.attention_privacy_mode,
        "attention_scores_visible": diag.attention_scores_visible,
        "attention_probs_visible": diag.attention_probs_visible,
        "attention_exact": diag.attention_exact,
        "attention_score_persistent_transcript_visible":
            diag.attention_score_persistent_transcript_visible,
        "attention_prob_persistent_transcript_visible":
            diag.attention_prob_persistent_transcript_visible,
        "attention_entropy_visible": diag.attention_entropy_visible,
        "attention_top1_index_visible": diag.attention_top1_index_visible,
        "attention_topk_indices_visible": diag.attention_topk_indices_visible,
        "attention_relative_margin_visible":
            diag.attention_relative_margin_visible,
        "attention_map_fingerprint_available":
            diag.attention_map_fingerprint_available,
        "trusted_softmax_used": diag.trusted_softmax_used,
        "attention_map_hidden_from_accelerator_transcript":
            diag.attention_map_hidden_from_accelerator_transcript,
        "row_constant_shift_used": diag.row_constant_shift_used,
        "hides_relative_attention": diag.hides_relative_attention,
        "attention_privacy_gain": diag.attention_privacy_gain,
        "attention_score_max_abs_error_vs_plain":
            diag.attention_score_max_abs_error_vs_plain,
        "attention_prob_max_abs_error_vs_plain":
            diag.attention_prob_max_abs_error_vs_plain,
        "attention_top1_match_rate": diag.attention_top1_match_rate,
        "row_constant_blinding_softmax_max_abs_error":
            diag.row_constant_blinding_softmax_max_abs_error,
        "nonconstant_blinding_softmax_max_abs_error":
            diag.nonconstant_blinding_softmax_max_abs_error,
        "attention_extra_tee_round_trips_per_layer":
            diag.attention_extra_tee_round_trips_per_layer,
        "requires_fused_kernel_assumption":
            diag.requires_fused_kernel_assumption,
        "not_cryptographic_security": diag.not_cryptographic_security,
        "attention_scores_ephemeral_inside_kernel":
            diag.attention_scores_ephemeral_inside_kernel,
        # Stage 7.6f / 7.6g / 7.6h carry-over.
        "main_layer_invariant": diag.main_layer_invariant,
        "rmsnorm_mode": diag.rmsnorm_mode,
        "rope_mask_mode": diag.rope_mask_mode,
        "rope_transient_plain_qk_visible":
            diag.rope_transient_plain_qk_visible,
        "rope_transient_plain_v_visible":
            diag.rope_transient_plain_v_visible,
        "qkv_projection_outputs_masked_directly":
            diag.qkv_projection_outputs_masked_directly,
        "intermediate_tee_reentry": diag.intermediate_tee_reentry,
        "online_boundary_round_trips_per_decode_step":
            diag.online_boundary_round_trips_per_decode_step,
        "trusted_fallback_used_in_main_path":
            diag.trusted_fallback_used_in_main_path,
        "use_pad": diag.use_pad,
        "pad_at_linear_boundaries": diag.pad_at_linear_boundaries,
        "pad_enters_rmsnorm_core": diag.pad_enters_rmsnorm_core,
        "pad_enters_rope_core": diag.pad_enters_rope_core,
        "pad_enters_swiglu_core": diag.pad_enters_swiglu_core,
        "pad_enters_softmax": diag.pad_enters_softmax,
        "fresh_pad_used_at_linear_boundaries":
            diag.fresh_pad_used_at_linear_boundaries,
        "norm_mask_granularity": diag.norm_mask_granularity,
        "norm_chunk_size": diag.norm_chunk_size,
        "norm_q_is_per_row": diag.norm_q_is_per_row,
        "h_hat_layer_entry_invariant_max_abs_error":
            diag.h_hat_layer_entry_invariant_max_abs_error,
        "rmsnorm_core_orthogonal_commutation_max_abs_error":
            diag.rmsnorm_core_orthogonal_commutation_max_abs_error,
        "transition_trick_max_abs_error":
            diag.transition_trick_max_abs_error,
        "swiglu_paired_permutation_max_abs_error":
            diag.swiglu_paired_permutation_max_abs_error,
        "o_proj_recovery_max_abs_error":
            diag.o_proj_recovery_max_abs_error,
        "down_proj_recovery_max_abs_error":
            diag.down_proj_recovery_max_abs_error,
        "qk_constraint_max_error": diag.qk_constraint_max_error,
        "rope_commutation_max_abs_error":
            diag.rope_commutation_max_abs_error,
        "qk_score_invariant_max_abs_error":
            diag.qk_score_invariant_max_abs_error,
        "kv_cache_invariant_max_abs_error":
            diag.kv_cache_invariant_max_abs_error,
        "prefill_logits_max_abs_error": diag.prefill_logits_max_abs_error,
        "decode_step_logits_max_abs_error_max":
            diag.decode_step_logits_max_abs_error_max,
        "lm_head_recovery_max_abs_error":
            diag.lm_head_recovery_max_abs_error,
    }


def _run_one_mode(
    model: TinyModernDecoderForCausalLM,
    cfg: AttentionPrivacyModesConfig,
    attention_privacy_mode: str,
    input_ids: torch.Tensor,
    mask_seed: int,
    fingerprint_prefix: str,
) -> Tuple[torch.Tensor, LowInteractionDiagnostics]:
    wrapper = LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=cfg.use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
        norm_mask_granularity=cfg.norm_mask_granularity,
        norm_chunk_size=cfg.norm_chunk_size,
        attention_privacy_mode=attention_privacy_mode,
    )
    g = torch.Generator(device="cpu").manual_seed(mask_seed)
    diag = LowInteractionDiagnostics()
    tokens, diag = wrapper.low_interaction_generate(
        input_ids, cfg.max_new_tokens, generator=g, diagnostics=diag,
        fingerprint_keys={
            "layer_entry_h_hat": f"{fingerprint_prefix}_layer_entry_h_hat",
            "lm_head_logits_tilde":
                f"{fingerprint_prefix}_lm_head_logits_tilde",
        },
    )
    return tokens, diag


# ---------------------------------------------------------------------------
# Mathematical analysis text (rendered into the report)
# ---------------------------------------------------------------------------


_EXACTNESS_VS_HIDING_DERIVATIONS: Dict[str, str] = {
    "current_exact_score_invariant": (
        "Given Q_tilde = Q B_Q and K_tilde = K B_K with B_Q B_K^T = I:\n"
        "  Q_tilde K_tilde^T = Q B_Q (K B_K)^T = Q B_Q B_K^T K^T = Q K^T.\n"
        "So S_tilde = S and softmax(S_tilde) = softmax(S). Exact correctness,\n"
        "attention scores / probabilities visible on the accelerator."
    ),
    "row_constant_shift_does_not_protect_attention": (
        "For any row-wise scalar c_i:\n"
        "  softmax(S_i + c_i * 1) = softmax(S_i).\n"
        "But (S_ij + c_i) - (S_ik + c_i) = S_ij - S_ik, so ranking,\n"
        "relative margins, entropy, and attention topology are unchanged.\n"
        "Row-constant additive blinding is NOT attention privacy."
    ),
    "general_additive_score_pad_breaks_exact_softmax": (
        "For arbitrary R: softmax(S + R) != softmax(S) unless R is row-\n"
        "constant. So additive score blinding with non-row-constant R\n"
        "cannot be used with ordinary accelerator-side softmax while\n"
        "preserving exactness."
    ),
    "exact_attention_hiding_requires_one_of": (
        "Exact attention map hiding therefore requires one of:\n"
        "  (i)   trusted / secure softmax,\n"
        "  (ii)  cryptographic protocol,\n"
        "  (iii) approximate attention,\n"
        "  (iv)  a changed threat model (fused confidential kernel)."
    ),
}


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def run_attention_privacy_modes(
    *, cfg: Optional[AttentionPrivacyModesConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = AttentionPrivacyModesConfig()

    torch.manual_seed(cfg.seed)

    decoder_cfg = TinyModernDecoderConfig(num_layers=cfg.num_layers)
    decoder_cfg.validate()
    model = TinyModernDecoderForCausalLM(decoder_cfg)
    model.init_random_weights(
        torch.Generator(device="cpu").manual_seed(cfg.weights_seed)
    )

    g_prompt = torch.Generator(device="cpu").manual_seed(cfg.prompt_seed)
    input_ids = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=g_prompt,
    )
    plain_tokens = model.greedy_generate(input_ids, cfg.max_new_tokens)

    modes = (
        "exact_visible_attention",
        "trusted_softmax_attention",
        "score_blinding_experimental",
    )
    per_mode_results: Dict[str, Any] = {}
    for mode in modes:
        tokens, diag = _run_one_mode(
            model, cfg, mode, input_ids,
            cfg.mask_seed, mode,
        )
        per_mode_results[mode] = {
            "mode": mode,
            "greedy_token_match_rate": float(
                (plain_tokens == tokens).float().mean().item()
            ),
            "sequence_exact_match": bool(torch.equal(plain_tokens, tokens)),
            "plain_token_sequence": plain_tokens.tolist(),
            "masked_token_sequence": tokens.tolist(),
            "diagnostics": _diag_to_summary(diag),
        }

    # Attention leakage audit table -- one bool / float per (mode, field).
    leakage_fields = (
        "attention_score_persistent_transcript_visible",
        "attention_prob_persistent_transcript_visible",
        "attention_entropy_visible",
        "attention_top1_index_visible",
        "attention_topk_indices_visible",
        "attention_relative_margin_visible",
        "attention_map_fingerprint_available",
    )
    leakage_audit: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        d = per_mode_results[mode]["diagnostics"]
        leakage_audit[mode] = {f: d[f] for f in leakage_fields}
        leakage_audit[mode]["attention_score_max_abs_error_vs_plain"] = (
            d["attention_score_max_abs_error_vs_plain"]
        )
        leakage_audit[mode]["attention_prob_max_abs_error_vs_plain"] = (
            d["attention_prob_max_abs_error_vs_plain"]
        )
        leakage_audit[mode]["attention_top1_match_rate"] = (
            d["attention_top1_match_rate"]
        )

    # Comparison table.
    comparison: Dict[str, Any] = {}
    for mode in modes:
        d = per_mode_results[mode]["diagnostics"]
        comparison[mode] = {
            "exact": bool(d["attention_exact"]),
            "one_round_trip": bool(
                d["online_boundary_round_trips_per_decode_step"] == 1
            ),
            "attention_hidden": bool(
                d["attention_map_hidden_from_accelerator_transcript"]
            ),
            "online_boundary_round_trips_per_decode_step":
                d["online_boundary_round_trips_per_decode_step"],
            "intermediate_tee_reentry": bool(d["intermediate_tee_reentry"]),
            "greedy_token_match_rate":
                per_mode_results[mode]["greedy_token_match_rate"],
            "sequence_exact_match":
                per_mode_results[mode]["sequence_exact_match"],
        }

    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.6i",
        "main_mode": "attention_privacy_modes",
        "modes_evaluated": list(modes),
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
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
            "weights_seed": cfg.weights_seed,
            "prompt_seed": cfg.prompt_seed,
            "mask_seed": cfg.mask_seed,
        },
        "stage_7_6h_inherited": {
            "use_pad": True,
            "rope_mask_mode": "pre_rope_block_diagonal_rotation",
            "rope_transient_plain_qk_visible": False,
            "qkv_projection_outputs_masked_directly": True,
            "norm_mask_granularity": cfg.norm_mask_granularity,
        },
        "per_mode_results": per_mode_results,
        "comparison": comparison,
        "attention_leakage_audit": leakage_audit,
        "attention_privacy_exactness_vs_hiding_tension":
            _EXACTNESS_VS_HIDING_DERIVATIONS,
        "topology_private_attention_experimental": {
            "implementation_status": "design_only_not_implemented",
            "reason": (
                "Approximate or alternative attention forms (kernelized, "
                "top-k hidden by trusted selection, noisy/rank-obfuscated "
                "scores) require dropping the exactness guarantee. They "
                "are not merged into the main protocol because the "
                "wrapper currently insists on attention_exact = true; "
                "any approximate attention must be reported with "
                "attention_exact = false."
            ),
        },
        "fused_kernel_transcript_hiding": {
            "implementation_status": "design_only_simulation_mode",
            "assumption": (
                "Adversary observes persistent accelerator tensors / "
                "global memory transcript, but does NOT observe "
                "registers or ephemeral values inside a fused "
                "confidential attention kernel."
            ),
            "requires_fused_kernel_assumption": True,
            "not_cryptographic_security": True,
            "attention_scores_persistent_visible": False,
            "attention_scores_ephemeral_inside_kernel": True,
        },
        "tolerances": {
            "logits_tolerance": LOGITS_TOLERANCE,
            "invariant_tolerance": INVARIANT_TOLERANCE,
        },
        "limitations": [
            "exact_visible_attention exposes the attention map by "
            "construction: the QK invariant Q_tilde K_tilde^T = Q K^T "
            "intentionally preserves scores on the accelerator side.",
            "trusted_softmax_attention adds extra TEE round trips per "
            "decode step (one per layer) and re-enters the trusted "
            "primitive once per attention block.",
            "score_blinding_experimental shows that row-constant score "
            "shifts preserve softmax exactly but do NOT hide ranking, "
            "relative margins, entropy, or attention topology.",
            "Non-row-constant additive score blinding breaks softmax "
            "exactness; the wrapper records the numerical error of that "
            "alternative to make the trade-off explicit.",
            "All modes are CPU local emulation only; no real TEE / GPU "
            "deployment.",
            "Synthetic tiny modern decoder; num_layers default = 1.",
            "This is NOT formal cryptographic / semantic / "
            "differential-privacy security.",
        ],
        "paper_safe_wording": (
            "Exact low-interaction attention with ordinary accelerator-"
            "side softmax exposes the attention map because the QK "
            "invariant intentionally preserves the score matrix. To "
            "hide attention maps exactly, the softmax computation "
            "must be moved to a trusted/secure primitive or replaced "
            "by an approximate/private attention mechanism. We "
            "therefore provide two modes: an exact low-interaction "
            "mode with visible attention maps, and an exact trusted-"
            "softmax mode that hides attention maps at the cost of "
            "additional trusted interaction."
        ),
        "unsafe_wording_to_avoid": [
            "The exact low-interaction mode hides attention maps.",
            "Row-wise score shifts provide attention privacy.",
            "This is cryptographic security.",
            "The trusted-softmax baseline preserves one round trip.",
            "Approximate attention is exact.",
        ],
        "attention_privacy_modes_completed": True,
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
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Attention-Map Protection / Secure-Attention Modes")
    w()
    w(
        "_Stage 7.6i: add an ``attention_privacy_mode`` knob to the "
        "Stage 7.6h low-interaction wrapper. Compares an exact low-"
        "interaction baseline (visible attention) against an exact "
        "trusted-softmax baseline (hidden attention, extra TEE round "
        "trips) and a row-constant score-blinding demonstration._"
    )
    w()

    w("## Inherited Stage 7.6h Guarantees")
    w()
    inh = report["stage_7_6h_inherited"]
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "use_pad", "rope_mask_mode", "rope_transient_plain_qk_visible",
        "qkv_projection_outputs_masked_directly",
        "norm_mask_granularity",
    ):
        w(f"| {k} | {inh[k]} |")
    w()

    w("## Configuration")
    w()
    cfg = report["config"]
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_layers",
        "num_query_heads", "num_kv_heads", "head_dim",
        "batch_size", "prompt_len", "max_new_tokens",
    ):
        w(f"| {k} | {cfg[k]} |")
    w(f"| dtype | {report['dtype']} |")
    w(f"| device | {report['device']} |")
    w()

    w("## Summary Comparison")
    w()
    w(
        "| Mode | exact | one_round_trip | attention_hidden | "
        "round_trips_per_decode_step | intermediate_tee_reentry | "
        "greedy_token_match_rate | sequence_exact_match |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for mode in report["modes_evaluated"]:
        c = report["comparison"][mode]
        w(
            f"| `{mode}` | {c['exact']} | {c['one_round_trip']} | "
            f"{c['attention_hidden']} | "
            f"{c['online_boundary_round_trips_per_decode_step']} | "
            f"{c['intermediate_tee_reentry']} | "
            f"{c['greedy_token_match_rate']} | "
            f"{c['sequence_exact_match']} |"
        )
    w()

    w("## Attention Privacy: Exactness vs Hiding Tension")
    w()
    der = report["attention_privacy_exactness_vs_hiding_tension"]
    for title, body in der.items():
        w(f"### {title.replace('_', ' ').title()}")
        w()
        w("```")
        w(body)
        w("```")
        w()

    w("## Per-Mode Correctness")
    w()
    w(
        "| Mode | greedy_match | seq_exact | "
        "lm_head_recovery_max | h_hat invariant max | "
        "qk_score invariant max | kv_cache invariant max |"
    )
    w("|---|---|---|---|---|---|---|")
    for mode in report["modes_evaluated"]:
        m = report["per_mode_results"][mode]
        d = m["diagnostics"]
        w(
            f"| `{mode}` | {m['greedy_token_match_rate']} | "
            f"{m['sequence_exact_match']} | "
            f"{_fmt(d['lm_head_recovery_max_abs_error'])} | "
            f"{_fmt(d['h_hat_layer_entry_invariant_max_abs_error'])} | "
            f"{_fmt(d['qk_score_invariant_max_abs_error'])} | "
            f"{_fmt(d['kv_cache_invariant_max_abs_error'])} |"
        )
    w()

    w("## Boundary Round-Trip Metrics")
    w()
    w(
        "| Mode | online_round_trips_per_decode_step | "
        "intermediate_tee_reentry | "
        "attention_extra_tee_round_trips_per_layer | "
        "trusted_softmax_used | "
        "trusted_fallback_used_in_main_path |"
    )
    w("|---|---|---|---|---|---|")
    for mode in report["modes_evaluated"]:
        d = report["per_mode_results"][mode]["diagnostics"]
        w(
            f"| `{mode}` | "
            f"{d['online_boundary_round_trips_per_decode_step']} | "
            f"{d['intermediate_tee_reentry']} | "
            f"{d['attention_extra_tee_round_trips_per_layer']} | "
            f"{d['trusted_softmax_used']} | "
            f"{d['trusted_fallback_used_in_main_path']} |"
        )
    w()

    w("## Attention Leakage Audit")
    w()
    w(
        "Each field is what the accelerator transcript exposes. "
        "``attention_map_fingerprint_available`` answers: can the "
        "accelerator-visible tensors recover the attention map?"
    )
    w()
    audit = report["attention_leakage_audit"]
    fields = (
        "attention_score_persistent_transcript_visible",
        "attention_prob_persistent_transcript_visible",
        "attention_entropy_visible",
        "attention_top1_index_visible",
        "attention_topk_indices_visible",
        "attention_relative_margin_visible",
        "attention_map_fingerprint_available",
    )
    w("| Mode | " + " | ".join(fields) + " |")
    w("|---" * (1 + len(fields)) + "|")
    for mode in report["modes_evaluated"]:
        row = [f"`{mode}`"] + [str(audit[mode][f]) for f in fields]
        w("| " + " | ".join(row) + " |")
    w()

    w("## Score-Blinding Experimental Detail")
    w()
    d_blind = (
        report["per_mode_results"]["score_blinding_experimental"]
        ["diagnostics"]
    )
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "row_constant_shift_used",
        "hides_relative_attention",
        "attention_privacy_gain",
        "row_constant_blinding_softmax_max_abs_error",
        "nonconstant_blinding_softmax_max_abs_error",
    ):
        val = d_blind[k]
        if isinstance(val, float):
            val_s = _fmt(val)
        else:
            val_s = str(val)
        w(f"| {k} | {val_s} |")
    w()
    w(
        "Row-constant shift: softmax exact, attention pattern fully "
        "preserved (privacy gain = none). Non-row-constant additive "
        "shift: softmax error is large (recorded as "
        "``nonconstant_blinding_softmax_max_abs_error``) -- proves "
        "that arbitrary additive score blinding cannot be combined "
        "with ordinary accelerator-side softmax while preserving "
        "exactness."
    )
    w()

    w("## Trusted-Softmax Detail")
    w()
    d_ts = (
        report["per_mode_results"]["trusted_softmax_attention"]
        ["diagnostics"]
    )
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "attention_privacy_mode",
        "attention_scores_visible",
        "attention_probs_visible",
        "attention_exact",
        "attention_map_hidden_from_accelerator_transcript",
        "trusted_softmax_used",
        "intermediate_tee_reentry",
        "online_boundary_round_trips_per_decode_step",
        "attention_extra_tee_round_trips_per_layer",
    ):
        w(f"| {k} | {d_ts[k]} |")
    w()

    w("## Topology-Private Attention (Experimental, Design-Only)")
    w()
    t = report["topology_private_attention_experimental"]
    w(f"- implementation_status: `{t['implementation_status']}`")
    w(f"- reason: {t['reason']}")
    w()

    w("## Fused-Kernel Transcript-Hiding Threat Model (Design-Only)")
    w()
    f = report["fused_kernel_transcript_hiding"]
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "implementation_status",
        "requires_fused_kernel_assumption",
        "not_cryptographic_security",
        "attention_scores_persistent_visible",
        "attention_scores_ephemeral_inside_kernel",
    ):
        w(f"| {k} | {f[k]} |")
    w()
    w(f"Assumption: {f['assumption']}")
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
    json_filename: str = "attention_privacy_modes.json",
    md_filename: str = "attention_privacy_modes.md",
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
    "AttentionPrivacyModesConfig",
    "LOGITS_TOLERANCE",
    "INVARIANT_TOLERANCE",
    "render_markdown",
    "run_attention_privacy_modes",
    "write_reports",
]
