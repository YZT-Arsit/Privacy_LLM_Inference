"""Stage 7.5 — paper artifact consolidation.

Aggregates the ``outputs/*.json`` produced by Stage 1 through Stage 7.4
into paper-ready tables (CSV / Markdown / LaTeX) and a unified summary.
This module introduces NO new obfuscation primitives, NO new attack
algorithms, NO new probes. It is a pure aggregator.

The five tables built here are:

* ``artifact_inventory``  — every artifact's status (present / missing /
  json_error), size, top-level keys, and a one-line summary metric.
* ``correctness_summary`` — per-stage correctness metrics extracted from
  the LoRA / inference / island / multi-layer probe reports.
* ``security_proxy_summary`` — per-attack-family risk levels with the
  paper claim each row supports.
* ``workload_summary``    — per-method boundary / trusted / GPU op counts
  + projected / measured wall time from ``workload_profile.json``.
* ``lora_training_summary`` — per-stage LoRA training scope + correctness
  + rank-hiding state.
* ``limitations_summary`` — aggregated limitations strings cross-referenced
  to the affected paper claim.

All outputs are emitted to ``paper_results/{csv,latex,markdown,json}/``.
Reports publish summary metrics + fingerprints only; raw tensors / masks /
adapters / gradients / private data are NEVER emitted (Stage 7.0-7.4
contract preserved).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Artifact registry — list of all aggregator inputs.
# ---------------------------------------------------------------------------


_INFERENCE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("workload_profile", "outputs/workload_profile.json"),
    ("cross_architecture_summary", "outputs/cross_architecture_summary.json"),
    ("nonlinear_island_experiments", "outputs/nonlinear_island_experiments.json"),
    ("nonlinear_island_security", "outputs/nonlinear_island_security.json"),
    ("adaptive_island_attacks", "outputs/adaptive_island_attacks.json"),
    ("modern_decoder_probe", "outputs/modern_decoder_probe.json"),
    (
        "modern_decoder_block_wrapper_smoke",
        "outputs/modern_decoder_block_wrapper_smoke.json",
    ),
    (
        "modern_decoder_model_wrapper_smoke",
        "outputs/modern_decoder_model_wrapper_smoke.json",
    ),
    ("real_activation_attacks", "outputs/real_activation_attacks.json"),
    (
        "real_token_activation_attacks",
        "outputs/real_token_activation_attacks.json",
    ),
    ("stronger_attackers", "outputs/stronger_attackers.json"),
)


_LORA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("lora_training_experiments", "outputs/lora_training_experiments.json"),
    ("lora_security_proxy", "outputs/lora_security_proxy.json"),
    ("lora_backward_experiments", "outputs/lora_backward_experiments.json"),
    (
        "lora_gradient_security_proxy",
        "outputs/lora_gradient_security_proxy.json",
    ),
    (
        "lora_rank_padding_experiments",
        "outputs/lora_rank_padding_experiments.json",
    ),
    ("lora_rank_security_proxy", "outputs/lora_rank_security_proxy.json"),
    (
        "multilayer_lora_training_experiments",
        "outputs/multilayer_lora_training_experiments.json",
    ),
    (
        "multilayer_lora_security_proxy",
        "outputs/multilayer_lora_security_proxy.json",
    ),
    ("lora_training_timing_proxy", "outputs/lora_training_timing_proxy.json"),
    (
        "lora_stronger_dummy_experiments",
        "outputs/lora_stronger_dummy_experiments.json",
    ),
    (
        "lora_stronger_dummy_security_proxy",
        "outputs/lora_stronger_dummy_security_proxy.json",
    ),
)


@dataclass
class PaperArtifactConsolidationConfig:
    outputs_dir: str = "outputs"
    paper_results_dir: str = "paper_results"
    strict: bool = False
    include_missing_artifacts: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(
    path: Path, strict: bool,
) -> tuple[dict[str, Any] | None, str, str]:
    """Return ``(data, status, error)``.

    status ∈ {"present", "missing", "json_error"}.
    """
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"required artifact missing: {path}")
        return None, "missing", ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, "present", ""
    except json.JSONDecodeError as e:
        if strict:
            raise
        return None, "json_error", str(e)


def _top_level_keys(data: dict[str, Any] | None) -> list[str]:
    if not isinstance(data, dict):
        return []
    return sorted(data.keys())


def _safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _build_inventory(
    outputs_dir: Path, strict: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot, group in (("inference", _INFERENCE_ARTIFACTS),
                        ("lora", _LORA_ARTIFACTS)):
        for name, rel in group:
            path = outputs_dir.parent / rel if rel.startswith("outputs/") and outputs_dir.name == "outputs" else outputs_dir / Path(rel).name
            # Always resolve to <repo>/outputs/<basename>
            path = outputs_dir / Path(rel).name
            data, status, err = _load_json(path, strict)
            size_bytes = path.stat().st_size if path.exists() else 0
            rows.append({
                "slot": slot,
                "artifact_name": name,
                "artifact_path": str(rel),
                "status": status,
                "json_error": err,
                "size_bytes": int(size_bytes),
                "top_level_keys": "|".join(_top_level_keys(data)) if data else "",
            })
    return rows


def _build_correctness_summary(
    artifacts: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _row(stage, component, architecture, scope, metric, value,
             allclose, artifact_path, notes=""):
        rows.append({
            "stage": stage,
            "component": component,
            "architecture": architecture,
            "scope": scope,
            "metric": metric,
            "value": value,
            "allclose": allclose,
            "artifact_path": artifact_path,
            "notes": notes,
        })

    # ---- Linear / Attention / KV (inference) ----
    workload = artifacts.get("workload_profile") or {}
    if workload:
        # ours_current is the measured runtime baseline.
        ours = _safe_get(workload, "methods", "ours_current", default={})
        if ours:
            _row(
                "1-4", "ours_current_greedy",
                _safe_get(workload, "config", "model_id", default="gpt2"),
                "gpt2_model_level_greedy",
                "measured_wall_time_ms",
                ours.get("measured_wall_time_ms"),
                "see token_match in generation_correctness",
                "outputs/workload_profile.json",
                notes="GPT-2 model-level greedy measured wall-time.",
            )

    # Nonlinear islands.
    ni = artifacts.get("nonlinear_island_experiments") or {}
    if ni:
        max_err = _safe_get(ni, "summary", "max_max_abs_error")
        if max_err is None:
            # Walk through all sections to find the max abs error if present.
            max_err = _safe_get(ni, "max_max_abs_error", default="see artifact")
        _row(
            "5.2", "compatible_nonlinear_islands", "cross_architecture",
            "tensor_level", "max_max_abs_error", max_err,
            True if isinstance(max_err, (int, float)) and max_err < 1e-9 else None,
            "outputs/nonlinear_island_experiments.json",
        )

    # Modern decoder block / model.
    mblock = artifacts.get("modern_decoder_block_wrapper_smoke") or {}
    if mblock:
        max_err = _safe_get(mblock, "summary", "max_abs_error",
                            default=_safe_get(mblock, "max_abs_error",
                                              default="see artifact"))
        _row(
            "6.4b", "modern_decoder_block_wrapper", "modern_decoder",
            "block_level", "max_abs_error", max_err,
            None,
            "outputs/modern_decoder_block_wrapper_smoke.json",
        )
    mmodel = artifacts.get("modern_decoder_model_wrapper_smoke") or {}
    if mmodel:
        gen_match = _safe_get(
            mmodel, "summary", "sequence_exact_match",
            default="see artifact",
        )
        _row(
            "6.4c", "modern_decoder_model_wrapper", "modern_decoder",
            "model_level_greedy", "sequence_exact_match", gen_match,
            gen_match is True,
            "outputs/modern_decoder_model_wrapper_smoke.json",
        )

    # ---- LoRA Stage 7.0 ----
    lora_t = artifacts.get("lora_training_experiments") or {}
    if lora_t:
        tc = _safe_get(lora_t, "training_correctness", default={}) or {}
        _row(
            "7.0", "lora_forward", "synthetic_linear", "single_step",
            "max_loss_diff", tc.get("max_loss_diff"),
            tc.get("allclose"),
            "outputs/lora_training_experiments.json",
        )

    # ---- LoRA Stage 7.1 ----
    lora_b = artifacts.get("lora_backward_experiments") or {}
    if lora_b:
        tc = _safe_get(lora_b, "training_correctness", default={}) or {}
        _row(
            "7.1", "lora_backward", "synthetic_linear", "single_step",
            "max_grad_a_err", tc.get("max_grad_a_err"),
            tc.get("allclose"),
            "outputs/lora_backward_experiments.json",
        )
        _row(
            "7.1", "lora_backward", "synthetic_linear", "single_step",
            "max_grad_b_err", tc.get("max_grad_b_err"),
            tc.get("allclose"),
            "outputs/lora_backward_experiments.json",
        )

    # ---- LoRA Stage 7.2 ----
    lora_rp = artifacts.get("lora_rank_padding_experiments") or {}
    if lora_rp:
        rp = _safe_get(lora_rp, "rank_padding_correctness", default={}) or {}
        sh = _safe_get(lora_rp, "shape_level_rank_hiding", default={}) or {}
        _row(
            "7.2", "rank_padded_lora_forward", "synthetic_linear", "single_step",
            "max_forward_err",
            max(
                (s.get("forward_max_abs_err", 0.0)
                 for s in rp.get("per_step", [])),
                default=None,
            ),
            rp.get("allclose"),
            "outputs/lora_rank_padding_experiments.json",
        )
        _row(
            "7.2", "rank_padded_lora_backward", "synthetic_linear", "single_step",
            "max_grad_a_real_err",
            rp.get("max_grad_a_real_err"),
            rp.get("allclose"),
            "outputs/lora_rank_padding_experiments.json",
        )
        _row(
            "7.2", "rank_hiding", "synthetic_linear", "single_step",
            "true_rank_hidden_from_shape",
            sh.get("true_rank_hidden_from_shape"),
            sh.get("true_rank_hidden_from_shape") is True,
            "outputs/lora_rank_padding_experiments.json",
            notes="true_rank hidden from shape; padded_rank still visible.",
        )

    # ---- LoRA Stage 7.3 ----
    ml = artifacts.get("multilayer_lora_training_experiments") or {}
    if ml:
        tc = _safe_get(ml, "training_correctness", default={}) or {}
        spec = _safe_get(ml, "model_spec", default={}) or {}
        _row(
            "7.3", "multi_layer_lora_training",
            "synthetic_multi_layer_decoder",
            f"layers={spec.get('num_layers')}, modules={spec.get('total_lora_modules')}",
            "max_loss_diff", tc.get("max_loss_diff"),
            tc.get("allclose"),
            "outputs/multilayer_lora_training_experiments.json",
        )
        _row(
            "7.3", "multi_layer_lora_training",
            "synthetic_multi_layer_decoder",
            f"layers={spec.get('num_layers')}, modules={spec.get('total_lora_modules')}",
            "max_grad_a_real_err", tc.get("max_grad_a_real_err"),
            tc.get("allclose"),
            "outputs/multilayer_lora_training_experiments.json",
        )

    # ---- LoRA Stage 7.4 ----
    sd = artifacts.get("lora_stronger_dummy_experiments") or {}
    if sd:
        for entry in _safe_get(sd, "per_strategy", default=[]) or []:
            _row(
                "7.4", f"stronger_dummy::{entry.get('dummy_strategy')}",
                "synthetic_linear", "single_step",
                "max_forward_err", entry.get("max_forward_err"),
                entry.get("allclose"),
                "outputs/lora_stronger_dummy_experiments.json",
            )

    return rows


def _build_security_proxy_summary(
    artifacts: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _row(stage, attack_family, target, strategy, metric, value,
             risk_level, recommendation, artifact_path,
             claim_supported):
        rows.append({
            "stage": stage,
            "attack_family": attack_family,
            "target": target,
            "strategy": strategy,
            "metric": metric,
            "value": value,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "artifact_path": artifact_path,
            "claim_supported": claim_supported,
        })

    # ---- Stage 5.2 nonlinear island security ----
    nis = artifacts.get("nonlinear_island_security") or {}
    if nis:
        for entry in _safe_get(nis, "summary", "per_island", default=[]) or []:
            _row(
                "5.2", "activation_recovery",
                entry.get("island_name", "n/a"),
                entry.get("strategy", "n/a"),
                "risk_level",
                entry.get("risk_level"),
                entry.get("risk_level"),
                entry.get("recommendation", ""),
                "outputs/nonlinear_island_security.json",
                "proxy-supported only",
            )

    # ---- Stage 5.4 adaptive island attacks ----
    aia = artifacts.get("adaptive_island_attacks") or {}
    if aia:
        for sub in _safe_get(aia, "summary", "per_attacker", default=[]) or []:
            _row(
                "5.4", "adaptive_proxy_attacker",
                sub.get("island", "n/a"),
                sub.get("attacker_family", "n/a"),
                "best_metric",
                sub.get("best_metric"),
                sub.get("risk_level", "needs_more_evaluation"),
                sub.get("recommendation", ""),
                "outputs/adaptive_island_attacks.json",
                "proxy-supported only",
            )

    # ---- Stage 5.5 real activation attacker ----
    raa = artifacts.get("real_activation_attacks") or {}
    if raa:
        summary = _safe_get(raa, "summary", default={}) or {}
        _row(
            "5.5", "real_activation_adaptive_proxy",
            "modern_decoder_block",
            summary.get("attacker_family", "ridge_linear_plus_mlp"),
            "linkability_auc",
            summary.get("linkability_auc"),
            summary.get("risk_level", "needs_more_evaluation"),
            "fresh_perm + sandwich + pad",
            "outputs/real_activation_attacks.json",
            "proxy-supported only",
        )

    # ---- Stage 5.5b real-token activation attacker ----
    rta = artifacts.get("real_token_activation_attacks") or {}
    if rta:
        summary = _safe_get(rta, "summary", default={}) or {}
        _row(
            "5.5b", "real_token_real_activation_proxy",
            "modern_decoder_model_level",
            summary.get("attacker_family", "real_token_proxy"),
            "linkability_auc",
            summary.get("linkability_auc"),
            summary.get("risk_level", "needs_more_evaluation"),
            "fresh_perm + sandwich + pad",
            "outputs/real_token_activation_attacks.json",
            "proxy-supported only",
        )

    # ---- Stage 5.6 stronger attackers ----
    sa = artifacts.get("stronger_attackers") or {}
    if sa:
        bb = _safe_get(sa, "blackbox", default={}) or {}
        tm = _safe_get(sa, "timing_sidechannel", default={}) or {}
        _row(
            "5.6", "blackbox_query_attacker",
            "modern_decoder_model_level",
            bb.get("strategy", "n/a"),
            "best_distinguishability",
            bb.get("best_distinguishability"),
            bb.get("risk_level", "low"),
            "exact-token-match guarantee",
            "outputs/stronger_attackers.json",
            "proxy-supported only",
        )
        _row(
            "5.6", "timing_sidechannel_proxy",
            "modern_decoder_model_level",
            tm.get("constant_time_decode_mode", "off"),
            "best_classification_accuracy",
            tm.get("best_classification_accuracy"),
            tm.get("risk_level"),
            "constant_time_decode_mode=proxy_equalized",
            "outputs/stronger_attackers.json",
            "proxy-supported only (cost-model)",
        )

    # ---- Stage 7.0 LoRA adapter extraction ----
    lsp = artifacts.get("lora_security_proxy") or {}
    if lsp:
        interp = _safe_get(lsp, "interpretation", default={}) or {}
        _row(
            "7.0", "lora_adapter_extraction",
            "synthetic_lora_linear",
            "fresh_masks_fresh_u_with_pad",
            "linkability_auc_summary",
            interp.get("linkability_summary"),
            "needs_more_evaluation",
            "fresh masks + pad",
            "outputs/lora_security_proxy.json",
            "proxy-supported only",
        )

    # ---- Stage 7.1 LoRA gradient leakage ----
    gsp = artifacts.get("lora_gradient_security_proxy") or {}
    if gsp:
        interp = _safe_get(gsp, "interpretation", default={}) or {}
        _row(
            "7.1", "lora_gradient_leakage",
            "synthetic_lora_linear_backward",
            "fresh_masks_fresh_u_with_pad",
            "linkability_summary",
            interp.get("linkability_summary"),
            "needs_more_evaluation",
            "fresh masks + pad; rank still visible from shape",
            "outputs/lora_gradient_security_proxy.json",
            "proxy-supported only",
        )

    # ---- Stage 7.2 rank leakage ----
    rls = artifacts.get("lora_rank_security_proxy") or {}
    if rls:
        interp = _safe_get(rls, "interpretation", default={}) or {}
        spectral_summary = interp.get("spectral_inference_summary", "")
        gradient_summary = interp.get("gradient_inference_summary", "")
        _row(
            "7.2", "spectral_rank_inference",
            "rank_padded_lora",
            "paired_cancellation_dummy",
            "spectral_inference_summary",
            spectral_summary,
            _extract_risk(spectral_summary),
            "use stronger dummy distributions (Stage 7.4)",
            "outputs/lora_rank_security_proxy.json",
            "proxy-supported only",
        )
        _row(
            "7.2", "gradient_rank_inference",
            "rank_padded_lora",
            "paired_cancellation_dummy",
            "gradient_inference_summary",
            gradient_summary,
            _extract_risk(gradient_summary),
            "use stronger dummy distributions (Stage 7.4)",
            "outputs/lora_rank_security_proxy.json",
            "proxy-supported only",
        )

    # ---- Stage 7.3 multilayer security proxy ----
    mlsp = artifacts.get("multilayer_lora_security_proxy") or {}
    if mlsp:
        interp = _safe_get(mlsp, "interpretation", default={}) or {}
        _row(
            "7.3", "cross_layer_linkage",
            "multi_layer_rank_padded_lora",
            "fresh_masks_independent_u",
            "cross_layer_linkage_summary",
            interp.get("cross_layer_linkage_summary"),
            _extract_risk(interp.get("cross_layer_linkage_summary", "")),
            "fresh masks per module + paired cancellation",
            "outputs/multilayer_lora_security_proxy.json",
            "proxy-supported only",
        )

    # ---- Stage 7.3 training timing proxy ----
    tt = artifacts.get("lora_training_timing_proxy") or {}
    if tt:
        summary = _safe_get(tt, "summary", default={}) or {}
        _row(
            "7.3", "training_timing_sidechannel_proxy",
            "lora_training_step_cost_model",
            "proxy_equalized",
            "max_classification_accuracy_proxy_equalized",
            summary.get("max_classification_accuracy_proxy_equalized"),
            "low" if (
                summary.get("max_classification_accuracy_proxy_equalized") or 1.0
            ) < 0.6 else "needs_more_evaluation",
            "proxy_equalized constant-time mode",
            "outputs/lora_training_timing_proxy.json",
            "proxy-supported only (cost-model)",
        )

    # ---- Stage 7.4 stronger dummy security proxy ----
    sds = artifacts.get("lora_stronger_dummy_security_proxy") or {}
    if sds:
        interp = _safe_get(sds, "interpretation", default={}) or {}
        for k, label in (
            ("spectral_summary", "spectral_rank_inference"),
            ("gradient_summary", "gradient_rank_inference"),
            ("dummy_strategy_classification_summary",
             "dummy_strategy_classification"),
            ("cross_layer_linkage_summary", "cross_layer_linkage"),
        ):
            _row(
                "7.4", f"stronger_dummy::{label}",
                "rank_padded_lora_with_stronger_dummies",
                "ensemble",
                k,
                interp.get(k),
                _extract_risk(str(interp.get(k, ""))),
                "spectrum_matched_dummy / mixed_dummy_ensemble",
                "outputs/lora_stronger_dummy_security_proxy.json",
                "proxy-supported only",
            )

    return rows


def _extract_risk(text: str) -> str:
    """Pull the most-conservative risk word out of an interpretation string."""
    if not text:
        return "n/a"
    low = text.lower()
    for level in ("high", "medium", "needs_more_evaluation", "low"):
        if level in low:
            return level
    return "n/a"


def _build_workload_summary(
    artifacts: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workload = artifacts.get("workload_profile") or {}
    methods = _safe_get(workload, "methods", default={}) or {}
    for method_name, record in methods.items():
        rows.append({
            "method": method_name,
            "architecture": _safe_get(
                workload, "config", "model_id", default="gpt2"
            ),
            "integration_level": (
                "model_wrapper"
                if record.get("full_runtime_integrated") else
                ("model_level_smoke"
                 if record.get("measured_wall_time_scope")
                 else "projected")
            ),
            "boundary_calls": record.get("online_boundary_calls"),
            "trusted_compute": record.get("online_trusted_compute_ops"),
            "gpu_compute": record.get("online_gpu_ops"),
            "preprocessing_cost": record.get("preprocessing_trusted_ops"),
            "online_extra_matmul_count": record.get(
                "online_extra_matmul_count"
            ),
            "measured_wall_time_ms": record.get("measured_wall_time_ms"),
            "projected_wall_time_ms": record.get("projected_wall_time_ms"),
            "wall_time_source": record.get("wall_time_source"),
            "artifact_path": "outputs/workload_profile.json",
        })
    return rows


def _build_lora_training_summary(
    artifacts: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _row(stage, scope, num_layers, num_lora_modules, true_rank,
             padded_rank, optimizer, loss_diff, grad_error, update_error,
             rank_hidden, risk_level, artifact_path):
        rows.append({
            "stage": stage,
            "training_scope": scope,
            "num_layers": num_layers,
            "num_lora_modules": num_lora_modules,
            "true_rank": true_rank,
            "padded_rank": padded_rank,
            "optimizer": optimizer,
            "loss_diff": loss_diff,
            "grad_error": grad_error,
            "update_error": update_error,
            "rank_hidden_from_shape": rank_hidden,
            "risk_level": risk_level,
            "artifact_path": artifact_path,
        })

    lora_t = artifacts.get("lora_training_experiments") or {}
    if lora_t:
        cfg = _safe_get(lora_t, "config", default={}) or {}
        tc = _safe_get(lora_t, "training_correctness", default={}) or {}
        _row(
            "7.0", "single_linear", 1, 1,
            cfg.get("rank"), cfg.get("rank"), cfg.get("optimizer"),
            tc.get("max_loss_diff"),
            tc.get("max_grad_a_err"),
            tc.get("max_update_a_err"),
            False, "needs_more_evaluation",
            "outputs/lora_training_experiments.json",
        )

    lora_b = artifacts.get("lora_backward_experiments") or {}
    if lora_b:
        cfg = _safe_get(lora_b, "config", default={}) or {}
        tc = _safe_get(lora_b, "training_correctness", default={}) or {}
        _row(
            "7.1", "single_linear_masked_backward", 1, 1,
            cfg.get("rank"), cfg.get("rank"), cfg.get("optimizer"),
            tc.get("max_loss_diff"),
            tc.get("max_grad_a_err"),
            tc.get("max_update_a_err"),
            False, "needs_more_evaluation",
            "outputs/lora_backward_experiments.json",
        )

    lora_rp = artifacts.get("lora_rank_padding_experiments") or {}
    if lora_rp:
        cfg = _safe_get(lora_rp, "config", default={}) or {}
        rp = _safe_get(lora_rp, "rank_padding_correctness", default={}) or {}
        _row(
            "7.2", "single_linear_rank_padded", 1, 1,
            cfg.get("true_rank"), cfg.get("padded_rank"), cfg.get("optimizer"),
            rp.get("max_loss_diff"),
            rp.get("max_grad_a_real_err"),
            rp.get("final_adapter_a_update_err"),
            True, "needs_more_evaluation",
            "outputs/lora_rank_padding_experiments.json",
        )

    ml = artifacts.get("multilayer_lora_training_experiments") or {}
    if ml:
        cfg = _safe_get(ml, "config", default={}) or {}
        tc = _safe_get(ml, "training_correctness", default={}) or {}
        rp = _safe_get(ml, "rank_padding_summary", default={}) or {}
        _row(
            "7.3", "multi_layer_decoder",
            cfg.get("num_layers"),
            rp.get("num_lora_modules"),
            cfg.get("true_rank"),
            cfg.get("padded_rank"),
            cfg.get("optimizer"),
            tc.get("max_loss_diff"),
            tc.get("max_grad_a_real_err"),
            tc.get("max_update_a_err"),
            True, "needs_more_evaluation",
            "outputs/multilayer_lora_training_experiments.json",
        )

    sd = artifacts.get("lora_stronger_dummy_experiments") or {}
    if sd:
        cfg = _safe_get(sd, "config", default={}) or {}
        for entry in _safe_get(sd, "per_strategy", default=[]) or []:
            _row(
                f"7.4::{entry.get('dummy_strategy')}",
                "single_linear_stronger_dummy", 1, 1,
                cfg.get("true_rank"), cfg.get("padded_rank"),
                cfg.get("optimizer"),
                entry.get("max_loss_diff"),
                entry.get("max_grad_a_real_err"),
                entry.get("max_update_a_err"),
                True, "needs_more_evaluation",
                "outputs/lora_stronger_dummy_experiments.json",
            )

    return rows


def _build_limitations_summary(
    artifacts: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    severity_keywords = (
        ("real tee", "high"),
        ("formal", "high"),
        ("cryptographic", "high"),
        ("semantic security", "high"),
        ("padded rank", "medium"),
        ("optimizer state", "medium"),
        ("not full", "medium"),
        ("hardware side", "medium"),
    )

    def _severity(text: str) -> str:
        low = text.lower()
        for needle, sev in severity_keywords:
            if needle in low:
                return sev
        return "low"

    for name, data in artifacts.items():
        if not data:
            continue
        for lim in _safe_get(data, "limitations", default=[]) or []:
            text = str(lim)
            rows.append({
                "category": name,
                "limitation": text,
                "affected_claim": _affected_claim(text),
                "severity": _severity(text),
                "paper_wording": (
                    "Treat as a limitation in the paper's threat-model"
                    " section; do not advertise the property it disclaims."
                ),
                "artifact_support": f"outputs/{name}.json",
            })
    return rows


def _affected_claim(text: str) -> str:
    low = text.lower()
    if "real tee" in low:
        return "real_tee_wall_time"
    if "formal" in low or "cryptographic" in low or "semantic" in low:
        return "formal_security"
    if "padded rank" in low or "padded_rank" in low:
        return "padded_rank_hidden"
    if "optimizer" in low:
        return "optimizer_trusted_outsourced"
    if "qwen" in low or "tinyllama" in low or "llama" in low:
        return "full_model_lora_finetune"
    if "peft" in low or "deepspeed" in low or "vllm" in low or "flashattention" in low:
        return "peft_integration"
    if "side-channel" in low or "side channel" in low:
        return "hardware_sidechannel_security"
    if "adapter is never merged" in low:
        return "no_adapter_merge"
    return "other"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _md_table(
    rows: list[dict[str, Any]], columns: list[str], title: str,
) -> str:
    lines: list[str] = [f"# {title}\n"]
    if not rows:
        lines.append("_No rows — upstream artifacts missing._\n")
        return "\n".join(lines)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        values = []
        for c in columns:
            v = r.get(c, "")
            v_str = str(v)
            # Escape pipes that would break markdown table rendering.
            v_str = v_str.replace("|", "\\|").replace("\n", " ")
            if len(v_str) > 120:
                v_str = v_str[:117] + "..."
            values.append(v_str)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _latex_table(
    rows: list[dict[str, Any]], columns: list[str], title: str, label: str,
) -> str:
    def _esc(s: str) -> str:
        return (
            str(s)
            .replace("\\", "\\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

    lines: list[str] = [
        f"% Auto-generated by paper_artifact_consolidation",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        rf"\caption{{{_esc(title)}}}",
        rf"\label{{{_esc(label)}}}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(_esc(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    if rows:
        for r in rows:
            cells = []
            for c in columns:
                v = r.get(c, "")
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                cells.append(_esc(s))
            lines.append(" & ".join(cells) + r" \\")
    else:
        lines.append(
            r"\multicolumn{" + str(len(columns))
            + r"}{c}{\emph{No rows --- upstream artifacts missing.}} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _csv_lines(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue().splitlines(keepends=True)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


_CORRECTNESS_COLS = [
    "stage", "component", "architecture", "scope",
    "metric", "value", "allclose", "artifact_path", "notes",
]
_SECURITY_COLS = [
    "stage", "attack_family", "target", "strategy", "metric", "value",
    "risk_level", "recommendation", "artifact_path", "claim_supported",
]
_WORKLOAD_COLS = [
    "method", "architecture", "integration_level",
    "boundary_calls", "trusted_compute", "gpu_compute",
    "preprocessing_cost", "online_extra_matmul_count",
    "measured_wall_time_ms", "projected_wall_time_ms",
    "wall_time_source", "artifact_path",
]
_LORA_TRAINING_COLS = [
    "stage", "training_scope", "num_layers", "num_lora_modules",
    "true_rank", "padded_rank", "optimizer",
    "loss_diff", "grad_error", "update_error",
    "rank_hidden_from_shape", "risk_level", "artifact_path",
]
_LIMITATIONS_COLS = [
    "category", "limitation", "affected_claim", "severity",
    "paper_wording", "artifact_support",
]
_INVENTORY_COLS = [
    "slot", "artifact_name", "artifact_path", "status",
    "json_error", "size_bytes", "top_level_keys",
]


def run_paper_artifact_consolidation(
    config: PaperArtifactConsolidationConfig,
) -> dict[str, Any]:
    outputs_dir = Path(config.outputs_dir)
    paper_results_dir = Path(config.paper_results_dir)
    for sub in ("csv", "latex", "markdown", "json", "figures", "tables"):
        (paper_results_dir / sub).mkdir(parents=True, exist_ok=True)

    inventory = _build_inventory(outputs_dir, config.strict)

    # Load all artifacts once.
    artifacts: dict[str, dict[str, Any] | None] = {}
    for slot in (_INFERENCE_ARTIFACTS, _LORA_ARTIFACTS):
        for name, rel in slot:
            path = outputs_dir / Path(rel).name
            data, _, _ = _load_json(path, strict=False)
            artifacts[name] = data

    correctness = _build_correctness_summary(artifacts)
    security = _build_security_proxy_summary(artifacts)
    workload_table = _build_workload_summary(artifacts)
    lora_train = _build_lora_training_summary(artifacts)
    limitations = _build_limitations_summary(artifacts)

    missing = [
        r for r in inventory if r["status"] != "present"
    ]
    if config.strict and missing:
        raise FileNotFoundError(
            f"strict consolidation: {len(missing)} artifacts missing"
        )

    # Render outputs.
    sections = [
        ("artifact_inventory", inventory, _INVENTORY_COLS,
         "Artifact Inventory", "tab:artifact_inventory"),
        ("correctness_summary", correctness, _CORRECTNESS_COLS,
         "Correctness Summary", "tab:correctness_summary"),
        ("security_proxy_summary", security, _SECURITY_COLS,
         "Security Proxy Summary", "tab:security_proxy_summary"),
        ("workload_summary", workload_table, _WORKLOAD_COLS,
         "Workload Summary", "tab:workload_summary"),
        ("lora_training_summary", lora_train, _LORA_TRAINING_COLS,
         "LoRA Training Summary", "tab:lora_training_summary"),
        ("limitations_summary", limitations, _LIMITATIONS_COLS,
         "Limitations Summary", "tab:limitations_summary"),
    ]
    for slug, rows, cols, title, label in sections:
        csv_text = "".join(_csv_lines(rows, cols))
        (paper_results_dir / "csv" / f"{slug}.csv").write_text(
            csv_text, encoding="utf-8",
        )
        (paper_results_dir / "markdown" / f"{slug}.md").write_text(
            _md_table(rows, cols, title), encoding="utf-8",
        )
        (paper_results_dir / "latex" / f"{slug}.tex").write_text(
            _latex_table(rows, cols, title, label), encoding="utf-8",
        )

    report = {
        "config": asdict(config),
        "artifact_inventory": inventory,
        "correctness_summary": correctness,
        "security_proxy_summary": security,
        "workload_summary": workload_table,
        "lora_training_summary": lora_train,
        "limitations_summary": limitations,
        "missing_artifacts": missing,
        "paper_artifact_consolidation_status": "implemented",
        "security_profile": "proxy-evaluated, not formal",
        "limitations": [
            "Pure aggregation; no new ops, no new attacks, no new probes.",
            "Outputs are derived from existing outputs/*.json; if an upstream artifact is stale, the consolidated tables are stale.",
            "LaTeX tables use plain tabular; large tables may need manual layout.",
            "Figures are produced by run_paper_artifact_consolidation_figures (separate script); this module emits tables only.",
            "Reports publish summary metrics + fingerprints only; raw tensors / masks / adapters / gradients / private data are never emitted.",
            "No formal / cryptographic / semantic security is claimed.",
            "No real TEE wall-time is claimed.",
        ],
    }
    (paper_results_dir / "json" / "artifact_inventory.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    return report


__all__ = [
    "PaperArtifactConsolidationConfig",
    "run_paper_artifact_consolidation",
]
