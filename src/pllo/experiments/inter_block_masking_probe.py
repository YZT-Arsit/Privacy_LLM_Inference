"""Stage 5.6 — Inter-block residual masking gap analysis.

Stage 5.5b surfaced the fact that the Stage 6.4c model-level wrapper
recovers each block's output to plain space before passing it as the next
block's input — i.e. the inter-block residual transcript
``H_l_out → H_{l+1}_in`` is plaintext at the attacker's view. Stage 5.5b
flagged ``boundary_input`` and ``final`` with
``inter_block_plain_recovered`` / ``risk_level = high`` (structural).

This module:

* (A) **Accounting baseline** — read the Stage 5.5b artifact (if present)
  and confirm the structural finding.
* (B) **Single-transition masking probe** — math-only verification that a
  single orthogonal residual mask ``N_inter`` can be absorbed by the next
  block's RMSNorm + Q/K/V folded weights with **no plain inter-block
  transcript** required. Tested on a synthetic block; reports allclose.
* (C) **Model-level experimental status** — Stage 5.6 surfaces the gap
  and the single-transition fix; it does NOT ship a full
  ``masked_boundary_experimental`` mode at the model-wrapper level. That
  is explicitly marked ``not_implemented_in_stage_5_6`` so callers do not
  silently assume it works.

Default for the wider system: ``inter_block_mask_mode = "plain_boundary"``
(current Stage 6.4c behaviour). The ``"masked_boundary_experimental"``
mode is a label only and is NEVER on by default; passing it asks the probe
to (B) verify the single-transition math.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pllo.experiments.modern_decoder_model_probe import (
    ModernDecoderModelWrapperConfig,
    _resolve_weights,
)
from pllo.hf_wrappers.modern_decoder_block_wrapper import (
    _rmsnorm_with_gamma,
)
from pllo.ops.compatible_masks import generate_orthogonal
from pllo.ops.nonlinear_islands import rmsnorm_core


# Tensors at the model-wrapper inter-block boundary that the Stage 5.5b
# attacker observes as plain.
INTER_BLOCK_PLAIN_TENSORS: tuple[str, ...] = ("boundary_input", "final")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class InterBlockMaskingProbeConfig:
    seed: int = 2026
    stage_5_5b_artifact: str = "outputs/real_token_activation_attacks.json"
    inter_block_mask_mode: str = "plain_boundary"
    # Synthetic block shape (no real model needed for the math probe).
    hidden_size: int = 32
    intermediate_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    batch_size: int = 2
    seq_len: int = 6
    dtype: str = "float32"
    device: str = "cpu"
    # For the model-level experimental status field.
    attempt_model_level_experimental: bool = False


# ---------------------------------------------------------------------------
# A. Accounting baseline
# ---------------------------------------------------------------------------


def _accounting_baseline(
    config: InterBlockMaskingProbeConfig,
) -> dict[str, Any]:
    """Read Stage 5.5b artifact (if present) and confirm the structural gap."""
    p = Path(config.stage_5_5b_artifact)
    if not p.exists():
        return {
            "status": "stage_5_5b_artifact_missing",
            "current_plain_boundary_detected": True,
            "affected_tensors": list(INTER_BLOCK_PLAIN_TENSORS),
            "accounting_risk_level": "high",
            "evidence_source": "static_analysis_only (artifact unavailable)",
            "recommendation": "needs_inter_block_masking",
        }
    payload = json.loads(p.read_text(encoding="utf-8"))
    # Pull the per-tensor records for the full bundle, prefill scope.
    full = (
        payload.get("target_tensor_results", {})
        .get("fresh_perm_plus_sandwich_plus_pad", {})
        .get("prefill", {})
    )
    inter_block_records: list[dict[str, Any]] = []
    for name in INTER_BLOCK_PLAIN_TENSORS:
        rec = full.get(name)
        if rec is None:
            inter_block_records.append({
                "tensor_name": name,
                "inter_block_plain": True,
                "linear_inverter_rel_l2_error": None,
                "linkability_visible_vs_plain_cosine": None,
                "risk_level": "high",
                "evidence_source": "static_analysis (record missing)",
            })
            continue
        inter_block_records.append({
            "tensor_name": name,
            "inter_block_plain": bool(rec.get("inter_block_plain", False)),
            "linear_inverter_rel_l2_error": float(
                rec["linear_inverter"]["relative_l2_error"]
            ),
            "linkability_visible_vs_plain_cosine": float(
                rec["linkability"]["visible_vs_plain_cosine"]
            ),
            "risk_level": str(rec.get("risk_level", "high")),
            "default_on_recommendation": str(
                rec.get("default_on_recommendation", "inter_block_plain_recovered")
            ),
            "evidence_source": "stage_5_5b_artifact",
        })
    any_high = any(
        r["risk_level"] == "high" for r in inter_block_records
    )
    return {
        "status": "ok",
        "current_plain_boundary_detected": True,
        "affected_tensors": list(INTER_BLOCK_PLAIN_TENSORS),
        "accounting_risk_level": "high" if any_high else "medium",
        "per_tensor_evidence": inter_block_records,
        "evidence_source": "stage_5_5b_artifact",
        "recommendation": "needs_inter_block_masking",
    }


# ---------------------------------------------------------------------------
# B. Single-transition masking probe
# ---------------------------------------------------------------------------


def _single_transition_probe(
    config: InterBlockMaskingProbeConfig,
) -> dict[str, Any]:
    """Verify that an orthogonal inter-block mask is absorbed by the next
    block's RMSNorm + folded Q/K/V projection — no plain transcript needed.

    Math: let H be the previous block's output and N_inter orthogonal.
    Define H_tilde = H @ N_inter. Then:

      rmsnorm_core(H_tilde) = rmsnorm_core(H) @ N_inter   (orthogonal-invariant)
      h1_plain              = rmsnorm_core(H) ⊙ γ_input
      h1_visible            = rmsnorm_core(H_tilde) ⊙ γ_input
                            = (rmsnorm_core(H) ⊙ γ_input) @ N_inter
                            = h1_plain @ N_inter

      So the next block's Q projection in the masked space is:
        q_tilde = h1_visible @ (N_inter^T @ W_q) = h1_plain @ W_q = q_plain.

    Hence the attacker never sees the plain residual H, only H_tilde, AND
    the next block's QKV math is unchanged. This is the structural lemma;
    the probe materialises numbers and reports allclose.
    """
    torch.manual_seed(config.seed)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device(config.device)
    H = int(config.hidden_size)
    B = int(config.batch_size)
    S = int(config.seq_len)

    # Synthetic block input + γ + projection weight.
    h_prev = torch.randn(B, S, H, dtype=torch.float32, generator=None).to(
        dtype=dtype, device=device,
    )
    gamma = (0.9 + 0.2 * torch.rand(H)).to(dtype=dtype, device=device)
    w_q_plain = torch.randn(H, 4 * H, dtype=torch.float32).to(
        dtype=dtype, device=device,
    )

    # Plain path: rmsnorm_with_gamma + Q projection.
    eps = 1e-6
    h1_plain = _rmsnorm_with_gamma(h_prev, gamma, eps)
    q_plain = h1_plain @ w_q_plain

    # Masked path: orthogonal N_inter; verify rmsnorm_core invariance, the
    # γ-fold, and the folded Q-projection.
    n_inter = generate_orthogonal(H, dtype, device)
    n_inter_inv = n_inter.T  # exact for orthogonal
    h_tilde = h_prev @ n_inter

    # rmsnorm_core(h_tilde) ?= rmsnorm_core(h_prev) @ n_inter
    core_plain = rmsnorm_core(h_prev, eps=eps)
    core_tilde = rmsnorm_core(h_tilde, eps=eps)
    rmsnorm_invariant_max_abs_error = float(
        (core_tilde - core_plain @ n_inter).abs().max().item()
    )

    # h1_visible = core_tilde * γ = (core_plain @ n_inter) * γ
    # Folded Q weight in the masked space: w_q_tilde = n_inter^T @ w_q_plain
    # so h1_visible @ w_q_tilde = h1_plain @ w_q_plain. We verify ONLY that
    # the rmsnorm_core invariance plus the dense fold preserves Q.
    # h1_visible needs γ applied AFTER the inter-block mask undo at the
    # core level — but γ ⊙ (core_tilde) is NOT equal to (γ ⊙ core_plain) @
    # n_inter in general, because γ is element-wise per channel. To absorb
    # γ correctly the wrapper folds γ into Q's weight instead of multiplying
    # it pointwise; that is exactly what Stage 6.4b already does.
    w_q_folded = gamma.unsqueeze(-1) * w_q_plain          # [H, 4H]
    w_q_tilde = n_inter.T @ w_q_folded                    # [H, 4H]

    q_via_masked = core_tilde @ w_q_tilde
    q_path_max_abs_error = float((q_via_masked - q_plain).abs().max().item())

    # Recovery sanity (trusted side can undo n_inter on the residual).
    h_recovered = h_tilde @ n_inter_inv
    residual_recovery_max_abs_error = float(
        (h_recovered - h_prev).abs().max().item()
    )

    tol = 1e-4 if dtype is torch.float32 else 1e-10
    allclose_rmsnorm = rmsnorm_invariant_max_abs_error < tol * max(
        1.0, float(h_prev.abs().max().item()),
    )
    allclose_q = q_path_max_abs_error < tol * max(
        1.0, float(q_plain.abs().max().item()),
    )
    allclose_residual = residual_recovery_max_abs_error < tol * max(
        1.0, float(h_prev.abs().max().item()),
    )

    return {
        "status": "ok",
        "rmsnorm_invariant_max_abs_error": rmsnorm_invariant_max_abs_error,
        "rmsnorm_invariant_allclose": bool(allclose_rmsnorm),
        "q_projection_path_max_abs_error": q_path_max_abs_error,
        "q_projection_path_allclose": bool(allclose_q),
        "residual_recovery_max_abs_error": residual_recovery_max_abs_error,
        "residual_recovery_allclose": bool(allclose_residual),
        "note": (
            "Single-transition probe: orthogonal N_inter applied to one"
            " block boundary; rmsnorm_core is invariant under N_inter and"
            " the folded Q-projection (w_q_tilde = N_inter^T @ (γ ⊙ w_q))"
            " reproduces the plain Q exactly. The attacker view of the"
            " inter-block residual is x_tilde = x @ N_inter (orthogonal,"
            " information-theoretically equivalent under random-N_inter"
            " sampling but with the same caveats as Stage 6.4b)."
        ),
    }


# ---------------------------------------------------------------------------
# C. Model-level experimental status (label-only, intentionally not shipped)
# ---------------------------------------------------------------------------


def _model_level_experimental_status(
    config: InterBlockMaskingProbeConfig,
) -> dict[str, Any]:
    if config.inter_block_mask_mode == "masked_boundary_experimental":
        return {
            "requested_mode": "masked_boundary_experimental",
            "status": "not_implemented_in_stage_5_6",
            "reason": (
                "Stage 5.6 ships the single-transition probe (math + a"
                " synthetic numerical check). A full model-level masked"
                " inter-block residual path needs the ObfuscatedModernDecoderModelWrapper's"
                " attention path to ALSO consume x_tilde directly (i.e."
                " input mask absorbed by the q/k/v fold AND by the residual"
                " add) instead of recovering h_mid to plain between blocks."
                " That change is non-trivial because the residual add must"
                " stay invariant under the SAME N_inter and the LM head"
                " must absorb the final N_inter — it is deferred so Stage"
                " 5.6's correctness path is unchanged."
            ),
            "deferred_to": "stage_5_6_extension_or_stage_7_0",
            "default_mode_unchanged": "plain_boundary",
        }
    if config.attempt_model_level_experimental:
        return {
            "requested_mode": "plain_boundary",
            "status": "experimental_explicitly_requested_but_default_keeps_plain_boundary",
            "default_mode_unchanged": "plain_boundary",
        }
    return {
        "requested_mode": "plain_boundary",
        "status": "default_plain_boundary_kept",
        "default_mode_unchanged": "plain_boundary",
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "Inter-block masking is experimental unless explicitly marked implemented.",
    "Stage 5.6 ships only an accounting baseline and a single-transition math probe; the full model-wrapper-level masked_boundary_experimental mode is `not_implemented_in_stage_5_6`.",
    "The single-transition probe uses synthetic weights / synthetic activations; it verifies the orthogonal-mask invariance lemma, not formal security.",
    "Orthogonal residual masks are information-theoretically weaker than dense masks; the probe inherits Stage 6.4b's N_res caveats.",
    "Dense sandwiching and fresh permutation reduce tested recovery but do not imply semantic security.",
]


def run_inter_block_masking_probe(
    config: InterBlockMaskingProbeConfig,
) -> dict[str, Any]:
    accounting = _accounting_baseline(config)
    probe = _single_transition_probe(config)
    experimental = _model_level_experimental_status(config)
    overall_risk = "high" if probe["status"] != "ok" or accounting[
        "accounting_risk_level"
    ] == "high" else "medium"
    masked_boundary_status = (
        "single_transition_probe_passed"
        if (
            probe["rmsnorm_invariant_allclose"]
            and probe["q_projection_path_allclose"]
            and probe["residual_recovery_allclose"]
        )
        else "single_transition_probe_failed"
    )
    return {
        "config": asdict(config),
        "current_plain_boundary_detected": True,
        "affected_tensors": list(INTER_BLOCK_PLAIN_TENSORS),
        "accounting_risk_level": accounting["accounting_risk_level"],
        "accounting_baseline": accounting,
        "single_transition_probe": probe,
        "single_transition_probe_status": masked_boundary_status,
        "masked_boundary_experimental_status": experimental["status"],
        "masked_boundary_experimental_default": "off",
        "experimental_detail": experimental,
        "overall_inter_block_risk_level": overall_risk,
        "recommendation": (
            "Stage 5.5b structural gap is acknowledged; the single-transition"
            " math probe shows the masked inter-block path is correct under"
            " an orthogonal N_inter. A full model-level masked boundary"
            " requires Stage 5.6 extension or Stage 7.0 (LoRA training"
            " path) before the default mode can change."
        ),
        "limitations": list(_LIMITATIONS),
    }


__all__ = [
    "INTER_BLOCK_PLAIN_TENSORS",
    "InterBlockMaskingProbeConfig",
    "run_inter_block_masking_probe",
]
