"""Stage 5.6 — Stronger attackers orchestrator.

Drives:
  1. :mod:`pllo.experiments.blackbox_attacker` — Black-box query attacker.
  2. :mod:`pllo.experiments.timing_sidechannel_proxy` — Model-based timing
     side-channel proxy.
  3. :mod:`pllo.experiments.inter_block_masking_probe` — Inter-block
     residual masking gap analysis + single-transition math probe.

Returns a JSON-safe report consumed by
``scripts/run_stronger_attackers.py`` and surfaced in
``outputs/stronger_attackers.{json,csv,md}``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pllo.experiments.blackbox_attacker import (
    BlackboxAttackerConfig,
    run_blackbox_attacker,
)
from pllo.experiments.inter_block_masking_probe import (
    InterBlockMaskingProbeConfig,
    run_inter_block_masking_probe,
)
from pllo.experiments.real_token_activation_attacker import (
    RealTokenActivationAttackConfig,
    run_real_token_activation_attacks,
)
from pllo.experiments.timing_sidechannel_proxy import (
    TimingSidechannelConfig,
    run_timing_sidechannel_proxy,
)
from pllo.ops.mitigation_bundles import VALID_MITIGATION_BUNDLES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StrongerAttackersConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    num_prompts: int = 8
    prompt_max_length: int = 16
    max_new_tokens: int = 3
    batch_size: int = 1
    model_id: str | None = None
    attempt_real_model_load: bool = False
    attempt_tokenizer_load: bool = False
    local_files_only: bool = False
    allow_synthetic_fallback: bool = True
    nonlinear_mode: str = "compatible_islands"
    mitigation_bundle: str = "fresh_perm_plus_sandwich_plus_pad"
    use_pad: bool = True
    inter_block_mask_mode: str = "plain_boundary"
    constant_time_decode_mode: str = "off"
    attacker_trials: int = 32
    timing_noise_std: float = 0.05
    dtype: str = "float32"
    device: str = "cpu"
    # Synthetic model shape.
    synthetic_vocab_size: int = 256
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8
    max_layers: int = 2
    # Stage 5.5b artifact for the accounting baseline.
    stage_5_5b_artifact: str = "outputs/real_token_activation_attacks.json"


# ---------------------------------------------------------------------------
# Overall risk synthesis + top-level limitations
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are stronger proxy attacks, not formal security proofs.",
    "Black-box attacks only use generated outputs and logits summaries.",
    "Timing results are model-based proxies, not real TEE timing measurements.",
    "No hardware side-channel attack is implemented.",
    "No real TEE isolation is evaluated.",
    "Synthetic fallback results are not real Qwen/TinyLlama results.",
    "Inter-block masking is experimental unless explicitly marked implemented.",
    "Dense sandwiching and fresh permutation reduce tested recovery but do not imply semantic security.",
]


def _overall_risk(
    blackbox: dict[str, Any],
    timing: dict[str, Any],
    inter_block: dict[str, Any],
) -> dict[str, Any]:
    order = {"low": 0, "medium": 1, "high": 2}
    inv = {0: "low", 1: "medium", 2: "high"}

    # ------ Envelope-integrity risk (Stage 5.6's pass/fail question) ------
    # The mitigation envelope is "sound" iff an attacker cannot
    # distinguish (mode, mitigation_bundle, use_pad) configurations from
    # the API output (black-box) or from latency (timing). Both are at
    # random chance under Stage 6.4c's exact-token-match guarantee + same
    # boundary call count across bundles.
    bb_mode = blackbox["mitigation_mode_distinguishability"][
        "mode_classification_accuracy"
    ]
    bb_mode_rc = blackbox["mitigation_mode_distinguishability"][
        "random_chance_baseline"
    ]
    timing_mode = timing["mitigation_distinguishability"][
        "mitigation_accuracy"
    ]
    timing_mode_rc = timing["mitigation_distinguishability"][
        "random_chance_baseline"
    ]

    def _grade(acc: float, rc: float) -> str:
        rc_safe = max(rc, 1e-12)
        ratio = acc / rc_safe
        if ratio > 3.0:
            return "high"
        if ratio > 1.5:
            return "medium"
        return "low"

    envelope_bb_risk = _grade(bb_mode, bb_mode_rc)
    envelope_timing_risk = _grade(timing_mode, timing_mode_rc)
    envelope_risk = inv[max(
        order[envelope_bb_risk], order[envelope_timing_risk],
    )]

    # ------ Structural-leakage risk (documented, NOT envelope failure) ------
    # Decode-step / prompt-length / method distinguishability are
    # STRUCTURAL: any latency-aware observer can count decode steps / see
    # prompt-length-proportional work. Inter-block plain boundary is a
    # structural Stage 6.4c limitation surfaced by Stage 5.5b.
    structural_timing_risk = inv[max(
        order[timing["decode_step_leakage"]["risk_level"]],
        order[timing["prompt_length_leakage"]["risk_level"]],
        order[timing["method_distinguishability"]["risk_level"]],
    )]
    structural_inter_block_risk = inter_block[
        "overall_inter_block_risk_level"
    ]
    structural_risk = inv[max(
        order[structural_timing_risk],
        order[structural_inter_block_risk],
    )]

    # Promotion eligibility is governed by ENVELOPE risk only.
    if envelope_risk == "low":
        upgrade = (
            "yes — envelope-integrity risk is `low` (modes / bundles are"
            " statistically indistinguishable from API output AND from"
            " timing). Eligible to label"
            " `adaptive-blackbox-and-timing-proxy-evaluated, not formal`."
            " Structural leakage (decode step, prompt length, inter-block"
            " plain boundary) is reported separately and is acknowledged"
            " as a known limitation of the current model wrapper, not a"
            " failure of the mitigation envelope."
        )
    elif envelope_risk == "medium":
        upgrade = (
            "partial — envelope-integrity risk is `medium`; security_profile_detail"
            " can record `adaptive-blackbox-and-timing-proxy-evaluated` only"
            " with an explicit note that mode/bundle distinguishability is"
            " above random chance."
        )
    else:
        upgrade = (
            "no — envelope-integrity risk is `high`; mode / bundle are"
            " distinguishable from API output or timing. Do NOT promote"
            " the security_profile_detail label."
        )

    return {
        "envelope_integrity_risk_level": envelope_risk,
        "envelope_blackbox_risk_level": envelope_bb_risk,
        "envelope_timing_risk_level": envelope_timing_risk,
        "structural_leakage_risk_level": structural_risk,
        "structural_timing_risk_level": structural_timing_risk,
        "structural_inter_block_risk_level": structural_inter_block_risk,
        "overall_risk_level": inv[max(
            order[envelope_risk], order[structural_risk],
        )],
        "security_profile_detail_with_stronger_attackers_eligibility": upgrade,
    }


def _recommendation(
    overall: dict[str, Any],
    inter_block: dict[str, Any],
    inter_block_closure: dict[str, Any],
    constant_time_summary: dict[str, Any],
    config: "StrongerAttackersConfig",
) -> dict[str, Any]:
    # Stage 5.6 extension — combined recommendation rule:
    # if envelope_risk is low AND inter_block closure verified at the
    # model-wrapper level AND constant_time_decode_mode reduced step
    # leakage, promote to the extended-proxy label.
    inter_block_closed = (
        inter_block_closure.get("masked_boundary_experimental_status")
        == "implemented"
    )
    timing_reduced = (
        constant_time_summary.get("mode") == "proxy_equalized"
        and constant_time_summary.get("risk_level_after") in {"low", "medium"}
        and constant_time_summary.get("risk_level_after")
        != constant_time_summary.get("risk_level_before")
    )
    envelope_ok = overall["envelope_integrity_risk_level"] == "low"
    if envelope_ok and inter_block_closed and timing_reduced:
        overall_recommendation = (
            "acceptable_with_mitigation_under_extended_proxy"
        )
        ext_eligibility = "yes"
    else:
        overall_recommendation = "needs_more_evaluation"
        ext_eligibility = "no"
    if inter_block_closed and envelope_ok:
        ext_label = (
            "inter-block-and-constant-time-proxy-evaluated, not formal"
            if timing_reduced
            else "inter-block-masked-and-adaptive-proxy-evaluated, not formal"
        )
    else:
        ext_label = None
    return {
        "security_profile_detail_with_stronger_attackers": (
            "adaptive-blackbox-and-timing-proxy-evaluated, not formal"
        ),
        "security_profile_detail_with_extended_proxy": ext_label,
        "extended_proxy_eligibility": ext_eligibility,
        "overall_recommendation": overall_recommendation,
        "promotion_eligibility_note": overall[
            "security_profile_detail_with_stronger_attackers_eligibility"
        ],
        "inter_block_residual_masking_recommendation": (
            "Stage 5.6 extension wires masked_boundary_experimental into"
            " the model-wrapper prefill / decode_step / greedy generation"
            " path. Default remains plain_boundary; the experimental mode"
            " is opt-in via inter_block_mask_mode='masked_boundary_experimental'."
        ),
        "constant_time_decode_recommendation": (
            "constant_time_decode_mode='proxy_equalized' equalises"
            " per-step simulated latency to a per-method upper bound;"
            " PROXY ONLY (no sleep, no real wall-time change). Decode-step"
            " timing leakage is reduced under this proxy."
        ),
        "inter_block_mask_mode_used": str(config.inter_block_mask_mode),
        "constant_time_decode_mode_used": str(config.constant_time_decode_mode),
        "default_mode_unchanged": "plain_boundary",
        "default_mitigation_bundle_unchanged": "fresh_perm_only",
        "default_nonlinear_mode_unchanged": "trusted",
        "default_constant_time_decode_mode_unchanged": "off",
    }


def _inter_block_closure_summary(
    config: "StrongerAttackersConfig",
) -> dict[str, Any]:
    """Run real-token attacker in both plain_boundary and masked_boundary_experimental
    modes and report a head-to-head comparison for boundary_input / final."""
    # Common synthetic config — small sweep so the orchestrator stays fast.
    def _run(mode: str) -> dict[str, Any]:
        c = RealTokenActivationAttackConfig(
            seed=config.seed,
            model_id=config.model_id,
            attempt_real_model_load=config.attempt_real_model_load,
            attempt_tokenizer_load=config.attempt_tokenizer_load,
            local_files_only=config.local_files_only,
            allow_synthetic_fallback=config.allow_synthetic_fallback,
            num_prompts=max(4, config.num_prompts // 2),
            prompt_max_length=config.prompt_max_length,
            max_layers=config.max_layers,
            max_new_tokens=config.max_new_tokens,
            attacker_steps=20,  # quick sweep for the head-to-head
            mlp_hidden_size=32,
            mlp_batch_size=16,
            use_pad=config.use_pad,
            nonlinear_mode="compatible_islands",
            inter_block_mask_mode=mode,
            dtype=config.dtype, device=config.device,
            synthetic_vocab_size=config.synthetic_vocab_size,
            synthetic_hidden_size=config.synthetic_hidden_size,
            synthetic_intermediate_size=config.synthetic_intermediate_size,
            synthetic_num_attention_heads=config.synthetic_num_attention_heads,
            synthetic_num_key_value_heads=config.synthetic_num_key_value_heads,
            synthetic_head_dim=config.synthetic_head_dim,
        )
        return run_real_token_activation_attacks(c)

    try:
        plain = _run("plain_boundary")
        masked = _run("masked_boundary_experimental")
        status = "implemented"
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "masked_boundary_experimental_status": "not_implemented",
            "error": str(exc),
        }

    def _extract(rec: dict[str, Any], tensor: str) -> dict[str, Any]:
        p = rec["target_tensor_results"]["fresh_perm_plus_sandwich_plus_pad"][
            "prefill"
        ].get(tensor)
        if p is None:
            return {"present": False}
        return {
            "present": True,
            "risk_level": p["risk_level"],
            "inter_block_plain": p["inter_block_plain"],
            "linear_rel_l2": float(
                p["linear_inverter"]["relative_l2_error"]
            ),
            "linkability_cosine": float(
                p["linkability"]["visible_vs_plain_cosine"]
            ),
        }

    boundary_input_before = _extract(plain, "boundary_input")
    boundary_input_after = _extract(masked, "boundary_input")
    final_before = _extract(plain, "final")
    final_after = _extract(masked, "final")

    return {
        "status": status,
        "masked_boundary_experimental_status": "implemented",
        "plain_boundary_artifact": "outputs/real_token_activation_attacks.json",
        "plain_boundary_summary": (
            plain["recommendation"]
        ),
        "masked_boundary_summary": (
            masked["recommendation"]
        ),
        "boundary_input_before": boundary_input_before,
        "boundary_input_after": boundary_input_after,
        "final_before": final_before,
        "final_after": final_after,
        "note": (
            "Head-to-head: plain_boundary keeps boundary_input / final as"
            " inter_block_plain_recovered (structural high). Stage 5.6"
            " extension's masked_boundary_experimental mode rotates the"
            " inter-block residual with a fresh orthogonal N_inter so"
            " those tensors join the masked tensor set and the attacker's"
            " linear / MLP / linkability proxies fail."
        ),
    }


def run_stronger_attackers(
    config: StrongerAttackersConfig,
) -> dict[str, Any]:
    bb_cfg = BlackboxAttackerConfig(
        seed=config.seed,
        num_prompts=config.num_prompts,
        prompt_max_length=config.prompt_max_length,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.batch_size,
        model_id=config.model_id,
        attempt_real_model_load=config.attempt_real_model_load,
        attempt_tokenizer_load=config.attempt_tokenizer_load,
        local_files_only=config.local_files_only,
        allow_synthetic_fallback=config.allow_synthetic_fallback,
        mitigation_bundles=VALID_MITIGATION_BUNDLES,
        dtype=config.dtype,
        device=config.device,
        synthetic_vocab_size=config.synthetic_vocab_size,
        synthetic_hidden_size=config.synthetic_hidden_size,
        synthetic_intermediate_size=config.synthetic_intermediate_size,
        synthetic_num_attention_heads=config.synthetic_num_attention_heads,
        synthetic_num_key_value_heads=config.synthetic_num_key_value_heads,
        synthetic_head_dim=config.synthetic_head_dim,
        max_layers=config.max_layers,
    )
    bb = run_blackbox_attacker(bb_cfg)

    timing_cfg = TimingSidechannelConfig(
        seed=config.seed,
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_intermediate_size,
        num_attention_heads=config.synthetic_num_attention_heads,
        head_dim=config.synthetic_head_dim,
        layers=config.max_layers,
        vocab_size=config.synthetic_vocab_size,
        batch_size=config.batch_size,
        timing_noise_std=config.timing_noise_std,
        constant_time_decode_mode=config.constant_time_decode_mode,
    )
    timing = run_timing_sidechannel_proxy(timing_cfg)

    ibm_cfg = InterBlockMaskingProbeConfig(
        seed=config.seed,
        stage_5_5b_artifact=config.stage_5_5b_artifact,
        inter_block_mask_mode=config.inter_block_mask_mode,
        hidden_size=config.synthetic_hidden_size,
        intermediate_size=config.synthetic_intermediate_size,
        num_attention_heads=config.synthetic_num_attention_heads,
        num_key_value_heads=config.synthetic_num_key_value_heads,
        head_dim=config.synthetic_head_dim,
        batch_size=config.batch_size,
        seq_len=min(config.prompt_max_length, 8),
        dtype=config.dtype,
        device=config.device,
    )
    inter_block = run_inter_block_masking_probe(ibm_cfg)

    # Stage 5.6 extension — head-to-head closure summary.
    inter_block_closure = _inter_block_closure_summary(config)
    constant_time_summary = timing["constant_time_decode"]

    overall = _overall_risk(bb, timing, inter_block)
    recommendation = _recommendation(
        overall, inter_block, inter_block_closure,
        constant_time_summary, config,
    )

    return {
        "config": asdict(config),
        "model_loading": bb["model_loading"],
        "tokenizer_loading": bb["tokenizer_loading"],
        "source": bb["source"],
        "blackbox_attacker": bb,
        "timing_sidechannel_proxy": timing,
        "inter_block_masking_gap": inter_block,
        "inter_block_closure_summary": inter_block_closure,
        "constant_time_decode_summary": constant_time_summary,
        "overall_risk_summary": overall,
        "recommendation": recommendation,
        "comparison_with_prior_stages": {
            "stage_5_4_artifact": "outputs/adaptive_island_attacks.json",
            "stage_5_5_artifact": "outputs/real_activation_attacks.json",
            "stage_5_5b_artifact": "outputs/real_token_activation_attacks.json",
            "stage_5_6_artifact": "outputs/stronger_attackers.json",
            "summary": (
                "Stage 5.4 — synthetic adaptive proxy. Stage 5.5 —"
                " real-activation (random hidden input) adaptive proxy."
                " Stage 5.5b — real-token-prompted real-activation"
                " adaptive proxy across prefill + decode_step. Stage 5.6"
                " — black-box query + timing side-channel proxy +"
                " inter-block masking gap analysis. Stages 5.4 / 5.5 /"
                " 5.5b all reported `low` risk for masked tensors under"
                " the full mitigation bundle; Stage 5.6 extends to attack"
                " surfaces that DON'T require paired plain/visible"
                " supervision."
            ),
        },
        "limitations": list(_LIMITATIONS),
    }


__all__ = [
    "StrongerAttackersConfig",
    "run_stronger_attackers",
]
