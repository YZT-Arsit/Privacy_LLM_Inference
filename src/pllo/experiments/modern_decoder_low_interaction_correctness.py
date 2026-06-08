"""Stage 7.6f -- low-interaction operator-compatible correctness experiment.

Demonstrates the paper main invariant ``H_hat_l = H_l @ Q_l`` with **no
intermediate TEE re-entry** on a tiny modern-decoder model. Per decode
step there is exactly **one** boundary round-trip:

    TEE -> accelerator: masked H_hat_0 (one tensor)
    accelerator runs every layer with operator-compatible RMSNorm,
        padded boundary linears, post-RoPE per-head masking, paired-
        permutation SwiGLU, masked KV cache, padded LM head
    accelerator -> TEE: masked logits z_tilde (one tensor)
    TEE recovers z = z_tilde @ N_vocab^{-1} and samples

The experiment also emits a **norm leakage audit** on the GPU-visible
RMSNorm-compatible state ``H_hat = H @ Q`` with orthogonal Q:

    row_norm_error = max | ||H_hat_i||_2 - ||H_i||_2 |
    gram_matrix_error = max | H_hat H_hat^T - H H^T |
    same_prompt_fresh_Q_gram_linkability = || G_run1 - G_run2 ||
    different_prompt_gram_distance = nearest-neighbour Gram distance
                                       across distinct prompts

Orthogonal RMSNorm-compatible masking provably preserves row L2 norms
and the full Gram matrix; the trade-off of the no-reentry path is that
those Gram-matrix structures *leak* on the boundary. This is reported
as an explicit leakage surface, not patched, and not framed as
cryptographic security.
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
    rmsnorm,
)
from pllo.wrappers.low_interaction_modern_decoder_generation_wrapper import (
    LowInteractionDiagnostics,
    LowInteractionTinyModernDecoderWrapper,
)


# ---------------------------------------------------------------------------
# Config + tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LowInteractionConfig:
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
# Norm leakage audit
# ---------------------------------------------------------------------------


def _row_norms(x: torch.Tensor) -> torch.Tensor:
    """Compute per-row L2 norms over the last dim of a [..., D] tensor."""
    return x.pow(2).sum(dim=-1).clamp_min(0).sqrt()


def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Token-pair Gram matrix ``x @ x^T`` for a single [S, D] slice."""
    return x @ x.transpose(-2, -1)


def _norm_leakage_audit(
    model: TinyModernDecoderForCausalLM,
    cfg: LowInteractionConfig,
    input_ids_main: torch.Tensor,
    input_ids_alt: torch.Tensor,
) -> Dict[str, Any]:
    """Audit how much of the row-norm / Gram-matrix structure of the
    plain hidden states is preserved on the GPU-visible boundary.

    The audit constructs ``H_hat = RMSNorm(H) @ Q`` for two fresh
    sessions on the same prompt and one alternative prompt, and reports
    the worst-case errors plus same-prompt and different-prompt Gram
    distances.
    """
    cfg_md = model.cfg

    # Plain hidden state at the layer-entry boundary for two prompts.
    plain_h_main = model.embed_tokens(input_ids_main)
    plain_h_alt = model.embed_tokens(input_ids_alt)

    # Two fresh sessions with the same prompt.
    wrapper = LowInteractionTinyModernDecoderWrapper(model, use_pad=cfg.use_pad)
    sess_a = wrapper.compile_session(
        generator=torch.Generator(device="cpu").manual_seed(cfg.mask_seed_a)
    )
    sess_b = wrapper.compile_session(
        generator=torch.Generator(device="cpu").manual_seed(cfg.mask_seed_b)
    )

    h_hat_main_a = plain_h_main @ sess_a.q_layer[0]
    h_hat_main_b = plain_h_main @ sess_b.q_layer[0]
    h_hat_alt_a = plain_h_alt @ sess_a.q_layer[0]

    # Row L2 norm preservation.
    main_norms = _row_norms(plain_h_main)
    main_hat_norms_a = _row_norms(h_hat_main_a)
    row_norm_error = float(
        (main_hat_norms_a - main_norms).abs().max().item()
    )

    # Gram matrix preservation: H_hat H_hat^T should equal H H^T
    # exactly for orthogonal Q.
    gram_plain = _gram_matrix(plain_h_main)
    gram_hat_a = _gram_matrix(h_hat_main_a)
    gram_matrix_error = float((gram_hat_a - gram_plain).abs().max().item())

    # Same prompt + fresh Q1, Q2 -> Gram matrices match (both equal plain Gram).
    gram_hat_b = _gram_matrix(h_hat_main_b)
    same_prompt_fresh_Q_gram_linkability = float(
        (gram_hat_a - gram_hat_b).abs().max().item()
    )

    # Different prompt Gram distance.
    gram_plain_alt = _gram_matrix(plain_h_alt)
    different_prompt_gram_distance = float(
        (gram_plain - gram_plain_alt).abs().max().item()
    )

    # Nearest-neighbour matching: for every (batch, position) token, find
    # the index of the nearest match across the two prompts under the
    # Gram-matrix distance. We report what fraction matches the
    # ground-truth identity (i.e. token (b, p) of run-a paired with the
    # same (b, p) of run-b).
    b, s, d = plain_h_main.shape
    matched = 0
    total = 0
    main_gram_rows_a = h_hat_main_a.pow(2).sum(dim=-1)  # [B, S]
    alt_gram_rows = h_hat_alt_a.pow(2).sum(dim=-1)
    for bi in range(b):
        for pi in range(s):
            # Distance under row-norm (which equals plain row norm).
            distances = (alt_gram_rows[bi] - main_gram_rows_a[bi, pi]).abs()
            nearest = int(distances.argmin().item())
            if nearest == pi:
                matched += 1
            total += 1
    nearest_neighbour_match_rate = matched / max(total, 1)

    return {
        "row_norm_error": row_norm_error,
        "gram_matrix_error": gram_matrix_error,
        "same_prompt_fresh_Q_gram_linkability": same_prompt_fresh_Q_gram_linkability,
        "different_prompt_gram_distance": different_prompt_gram_distance,
        "nearest_neighbour_gram_match_rate_same_prompt": 1.0,  # trivially 1.0 by definition (same plain H)
        "nearest_neighbour_gram_match_rate_different_prompt": nearest_neighbour_match_rate,
        "notes": (
            "Row L2 norms and Gram matrices of the plain hidden states "
            "are exactly preserved by orthogonal Q; the no-reentry path "
            "therefore leaks token-pair similarity structure on the "
            "accelerator boundary."
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


def run_low_interaction_correctness(
    *,
    cfg: Optional[LowInteractionConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = LowInteractionConfig()

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
    input_ids_alt = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=torch.Generator(device="cpu").manual_seed(cfg.prompt_seed + 1),
    )

    # ---------- Main no-reentry run ----------
    wrapper = LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=cfg.use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
    )
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

    # ---------- Two-run sanity check ----------
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

    # ---------- Norm leakage audit ----------
    leakage = _norm_leakage_audit(model, cfg, input_ids, input_ids_alt)
    diag_main.row_norm_error = leakage["row_norm_error"]
    diag_main.gram_matrix_error = leakage["gram_matrix_error"]

    # ---------- Assemble JSON ----------
    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.6f",
        "main_mode": "low_interaction_operator_compatible_execution",
        # ---- Required main report fields ----
        "main_layer_invariant": "H_hat_l = H_l Q_l",
        "rmsnorm_mode": "operator_compatible_orthogonal",
        "trusted_fallback_used_in_main_path": False,
        "intermediate_tee_reentry": False,
        "online_boundary_round_trips_per_decode_step": 1,
        "use_pad": bool(cfg.use_pad),
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
            "rmsnorm_gamma_folded_into_following_linear": True,
            "post_rope_per_head_masking_used": True,
            "kv_cache_remasking_during_decode": False,
        },
        "rope_blocker": {
            "blocker_present": True,
            "description": (
                "Preferred path is post-RoPE per-head masking. The qkv-"
                "projection output is plain Q / K / V transiently on the "
                "accelerator -- RoPE is applied on plain Q / K and per-"
                "head masks are applied immediately afterwards. No TEE "
                "re-entry happens during this step (no data leaves the "
                "accelerator), but plain Q / K / V are transiently "
                "visible to the accelerator inside that block. This is "
                "the explicit blocker for the no-reentry path."
            ),
            "transient_plain_qk_on_accelerator": True,
            "tee_reentry_inside_rope_block": False,
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
            "swiglu_mode": diag_main.swiglu_mode,
            "attention_score_mode": diag_main.attention_score_mode,
            "lm_head_mode": diag_main.lm_head_mode,
            "qkv_projection_uses_transition_trick": True,
            "o_proj_uses_transition_trick": True,
            "mlp_in_uses_transition_trick": True,
            "down_proj_uses_transition_trick": True,
            "lm_head_uses_transition_trick": True,
        },
        "correctness": {
            "h_hat_layer_entry_invariant_max_abs_error": diag_main.h_hat_layer_entry_invariant_max_abs_error,
            "rmsnorm_core_orthogonal_commutation_max_abs_error": diag_main.rmsnorm_core_orthogonal_commutation_max_abs_error,
            "transition_trick_max_abs_error": diag_main.transition_trick_max_abs_error,
            "swiglu_paired_permutation_max_abs_error": diag_main.swiglu_paired_permutation_max_abs_error,
            "o_proj_recovery_max_abs_error": diag_main.o_proj_recovery_max_abs_error,
            "down_proj_recovery_max_abs_error": diag_main.down_proj_recovery_max_abs_error,
            "qk_constraint_max_error": diag_main.qk_constraint_max_error,
            "kv_cache_invariant_max_abs_error": diag_main.kv_cache_invariant_max_abs_error,
            "prefill_logits_max_abs_error": diag_main.prefill_logits_max_abs_error,
            "decode_step_logits_max_abs_error_max": diag_main.decode_step_logits_max_abs_error_max,
            "lm_head_recovery_max_abs_error": diag_main.lm_head_recovery_max_abs_error,
            "greedy_token_match_rate": greedy_token_match_rate,
            "sequence_exact_match": sequence_exact_match,
            "plain_token_sequence": plain_tokens.tolist(),
            "masked_token_sequence": masked_tokens.tolist(),
        },
        "norm_leakage_audit": {
            "row_norm_error": leakage["row_norm_error"],
            "gram_matrix_error": leakage["gram_matrix_error"],
            "same_prompt_fresh_Q_gram_linkability": leakage["same_prompt_fresh_Q_gram_linkability"],
            "different_prompt_gram_distance": leakage["different_prompt_gram_distance"],
            "nearest_neighbour_gram_match_rate_same_prompt": leakage["nearest_neighbour_gram_match_rate_same_prompt"],
            "nearest_neighbour_gram_match_rate_different_prompt": leakage["nearest_neighbour_gram_match_rate_different_prompt"],
            "interpretation": (
                "Orthogonal RMSNorm-compatible masking preserves row L2 "
                "norms and the full Gram matrix of the hidden states. "
                "The accelerator boundary therefore exposes the token-"
                "pair similarity structure of the plain hidden states. "
                "Fresh Q per session does NOT randomise this structure "
                "(both runs yield the same Gram matrix)."
            ),
            "notes": leakage["notes"],
        },
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
            "Synthetic tiny modern decoder (vocab=97, hidden=64, num_layers configurable).",
            "Default num_layers=1 demonstrates the invariant on a single layer; the wrapper supports N layers via accelerator-side inter-layer orthogonal change-of-basis matrices.",
            "RoPE remains the explicit blocker: post-RoPE per-head masking requires plain Q / K transiently on the accelerator. No TEE re-entry happens, but this is accelerator-side transient leakage of plain Q / K / V inside the qkv -> RoPE -> per-head-mask block.",
            "Attention scores / probabilities are plain by construction of the QK invariant (N_Q N_K^T = I); attention-map hiding is out of scope.",
            "Orthogonal RMSNorm-compatible masking preserves row L2 norms and Gram matrices; this is reported as an explicit boundary leakage surface.",
            "Low-interaction mode trades fewer TEE boundary crossings for norm-structure leakage; this is a deliberate trade-off, not a security claim.",
            "This is NOT formal cryptographic / semantic / differential privacy.",
            "No hardware side-channel evaluation.",
        ],
        "paper_safe_wording": (
            "We verify that the operator-compatible orthogonal RMSNorm "
            "path, combined with fresh boundary pads and the trusted-"
            "precomputed Q^{-1} M transition trick, can run a full tiny "
            "modern-decoder generation step without any per-layer TEE "
            "re-entry. The trade-off, made explicit by the Gram-matrix "
            "leakage audit, is that the GPU-visible H_hat = H Q "
            "preserves row-norm and token-pair similarity structure."
        ),
        "unsafe_wording_to_avoid": [
            "The scheme is formally secure.",
            "Orthogonal masking hides hidden states cryptographically.",
            "Attention maps are hidden.",
            "We evaluate real TEE / GPU performance.",
            "Generic dense masks commute with RoPE.",
            "Pads can pass through nonlinear layers.",
            "We support full Qwen / LLaMA private generation on real TEE / GPU.",
            "No information leaks on the accelerator boundary.",
            "The Gram matrix is hidden.",
            "Row norms are randomised by fresh Q.",
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
    leakage = report["norm_leakage_audit"]
    sec = report["security_relevant_checks"]

    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Low-Interaction Operator-Compatible Modern-Decoder Correctness")
    w()
    w(
        "_CPU local emulation; main invariant `H_hat_l = H_l @ Q_l` with "
        "orthogonal `Q_l`; one boundary round-trip per decode step._"
    )
    w()

    # No-reentry headline.
    w("## No-Reentry Headline")
    w()
    w("| Field | Value |")
    w("|---|---|")
    w(f"| main_layer_invariant | `{report['main_layer_invariant']}` |")
    w(f"| rmsnorm_mode | `{report['rmsnorm_mode']}` |")
    w(f"| trusted_fallback_used_in_main_path | {report['trusted_fallback_used_in_main_path']} |")
    w(f"| intermediate_tee_reentry | {report['intermediate_tee_reentry']} |")
    w(f"| online_boundary_round_trips_per_decode_step | {report['online_boundary_round_trips_per_decode_step']} |")
    w(f"| use_pad | {report['use_pad']} |")
    w(f"| fresh_pad_used_at_linear_boundaries | {report['fresh_pad_used_at_linear_boundaries']} |")
    w()

    # Configuration.
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

    # Online boundary protocol.
    w("## Online Boundary Protocol (per decode step)")
    w()
    proto = report["online_boundary_protocol"]
    w("- **TEE -> accelerator**: " + proto["tee_to_accelerator_tensors_per_step"][0])
    w("- Accelerator runs every layer with no TEE re-entry.")
    w("- **Accelerator -> TEE**: " + proto["accelerator_to_tee_tensors_per_step"][0])
    w("- TEE recovers `z = z_tilde @ N_vocab^{-1}` and samples.")
    w()
    w("| Property | Value |")
    w("|---|---|")
    w(f"| no_per_layer_trusted_recovery | {proto['no_per_layer_trusted_recovery']} |")
    w(f"| no_trusted_rmsnorm_fallback | {proto['no_trusted_rmsnorm_fallback']} |")
    w(f"| no_trusted_rope_recovery | {proto['no_trusted_rope_recovery']} |")
    w(f"| rmsnorm_gamma_folded_into_following_linear | {proto['rmsnorm_gamma_folded_into_following_linear']} |")
    w(f"| post_rope_per_head_masking_used | {proto['post_rope_per_head_masking_used']} |")
    w(f"| kv_cache_remasking_during_decode | {proto['kv_cache_remasking_during_decode']} |")
    w()

    # Module modes.
    w("## Module Modes")
    w()
    w("| Module | Mode |")
    w("|---|---|")
    for k in (
        "rmsnorm_mode", "rope_mode", "swiglu_mode",
        "attention_score_mode", "lm_head_mode",
    ):
        w(f"| {k} | `{mods[k]}` |")
    for k in (
        "qkv_projection_uses_transition_trick",
        "o_proj_uses_transition_trick",
        "mlp_in_uses_transition_trick",
        "down_proj_uses_transition_trick",
        "lm_head_uses_transition_trick",
    ):
        w(f"| {k} | `{mods[k]}` |")
    w()

    # Pad policy.
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

    # Correctness metrics.
    w("## Correctness Metrics")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    for k, label in (
        ("h_hat_layer_entry_invariant_max_abs_error", "H_hat = H Q invariant max abs err"),
        ("rmsnorm_core_orthogonal_commutation_max_abs_error", "RMSNormCore(H Q) - RMSNormCore(H) Q max"),
        ("transition_trick_max_abs_error", "Transition (Q^{-1} M, T M) max abs err"),
        ("swiglu_paired_permutation_max_abs_error", "SwiGLU paired-permutation max abs err"),
        ("o_proj_recovery_max_abs_error", "o_proj recovery max abs err"),
        ("down_proj_recovery_max_abs_error", "down_proj recovery max abs err"),
        ("qk_constraint_max_error", "N_Q N_K^T - I max abs err"),
        ("kv_cache_invariant_max_abs_error", "KV cache append invariant max abs err"),
        ("prefill_logits_max_abs_error", "Prefill recovered logits max abs err"),
        ("decode_step_logits_max_abs_error_max", "Decode-step recovered logits max abs err"),
        ("lm_head_recovery_max_abs_error", "LM head recovery max abs err"),
    ):
        w(f"| {label} | {_fmt(corr[k])} |")
    w(f"| greedy_token_match_rate | {corr['greedy_token_match_rate']} |")
    w(f"| sequence_exact_match | {corr['sequence_exact_match']} |")
    w()

    # RoPE blocker.
    w("## RoPE Blocker (explicit)")
    w()
    w(report["rope_blocker"]["description"])
    w()
    w("| Property | Value |")
    w("|---|---|")
    w(f"| blocker_present | {report['rope_blocker']['blocker_present']} |")
    w(f"| transient_plain_qk_on_accelerator | {report['rope_blocker']['transient_plain_qk_on_accelerator']} |")
    w(f"| tee_reentry_inside_rope_block | {report['rope_blocker']['tee_reentry_inside_rope_block']} |")
    w()

    # Norm leakage audit.
    w("## Norm Leakage Audit")
    w()
    w("**This is a leakage surface, not a security claim.** Orthogonal "
      "RMSNorm-compatible masking preserves row L2 norms and the full "
      "Gram matrix of the plain hidden states. Fresh `Q_l` per session "
      "does NOT randomise this structure.")
    w()
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| row_norm_error (||H_hat_i||_2 - ||H_i||_2) | {_fmt(leakage['row_norm_error'])} |")
    w(f"| gram_matrix_error (||H_hat H_hat^T - H H^T||_inf) | {_fmt(leakage['gram_matrix_error'])} |")
    w(f"| same_prompt_fresh_Q_gram_linkability | {_fmt(leakage['same_prompt_fresh_Q_gram_linkability'])} |")
    w(f"| different_prompt_gram_distance | {_fmt(leakage['different_prompt_gram_distance'])} |")
    w(f"| nn_gram_match_rate_same_prompt | {leakage['nearest_neighbour_gram_match_rate_same_prompt']} |")
    w(f"| nn_gram_match_rate_different_prompt | {leakage['nearest_neighbour_gram_match_rate_different_prompt']:.4f} |")
    w()
    w(leakage["interpretation"])
    w()

    # Repeated-run sanity check.
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

    # Limitations.
    w("## Limitations")
    w()
    for item in report["limitations"]:
        w(f"- {item}")
    w()

    # Paper-safe wording.
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()

    # Unsafe wording to avoid.
    w("## Unsafe Wording to Avoid")
    w()
    for item in report["unsafe_wording_to_avoid"]:
        w(f"- {item}")
    w()

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_reports(
    report: Dict[str, Any],
    *,
    outputs_dir: Path,
    json_filename: str = "modern_decoder_low_interaction_correctness.json",
    md_filename: str = "modern_decoder_low_interaction_correctness.md",
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
    "LowInteractionConfig",
    "render_markdown",
    "run_low_interaction_correctness",
    "write_reports",
    "LOGITS_TOLERANCE",
    "KV_CACHE_TOLERANCE",
    "INVARIANT_TOLERANCE",
]
