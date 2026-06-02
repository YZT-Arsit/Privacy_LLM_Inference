"""Stage 6.3 — Cross-architecture coverage / correctness / workload aggregator.

Reads the JSON artifacts produced by earlier stages and emits one unified
summary across decoder-only / encoder-only / encoder-decoder. No model is
re-loaded and no probe is re-run by default; this is a pure aggregator over
existing ``outputs/*.json``.

If an upstream JSON file is missing the architecture is recorded with
``status="missing"`` (unless ``require_existing_outputs=True``), so the
report can be assembled even when only a subset of upstream probes have
been executed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CrossArchitectureSummaryConfig:
    output_dir: str = "outputs"
    require_existing_outputs: bool = False


# ---------------------------------------------------------------------------
# Per-architecture static metadata
# ---------------------------------------------------------------------------


ARCHITECTURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "architecture_type": "decoder_only",
        "model_id": "sshleifer/tiny-gpt2",
        "model_class_default": "GPT2LMHeadModel",
        "attention_kind": "causal_self_attention",
        "source_file": "attention_experiments.json",
        "cache_type": "autoregressive_kv_cache",
        "trusted_shortcuts": [
            "trusted_layernorm",
            "trusted_gelu",
            "lm_head_vocab_diag_mask_only",
        ],
        "limitations": [
            "LayerNorm runs inside SimulatedTEE (trusted shortcut).",
            "GELU runs inside SimulatedTEE (trusted shortcut).",
            "LM head uses a diagonal vocab output mask only, no full pad.",
            "Real TEE isolation is not implemented.",
        ],
    },
    {
        "architecture_type": "encoder_only",
        "model_id": "hf-internal-testing/tiny-bert",
        "model_class_default": "BertForMaskedLM",
        "attention_kind": "bidirectional_self_attention",
        "source_file": "encoder_attention_experiments.json",
        "cache_type": "none",
        "trusted_shortcuts": [
            "trusted_layernorm",
            "trusted_gelu",
            "no_mlm_head_obfuscation",
        ],
        "limitations": [
            "BERT obfuscated forward (LayerNorm / GELU / FFN / MLM head) is not implemented.",
            "Only first-layer self-attention Q / K / V / O is validated.",
            "Real TEE isolation is not implemented.",
        ],
    },
    {
        "architecture_type": "encoder_decoder",
        "model_id": "hf-internal-testing/tiny-random-t5",
        "model_class_default": "T5ForConditionalGeneration",
        "attention_kind": "cross_attention",
        "source_file": "cross_attention_experiments.json",
        "cache_type": "encoder_memory_cache",
        "trusted_shortcuts": [
            "trusted_layernorm",
            "trusted_ffn_activation",
            "no_decoder_self_attention_cache",
            "no_relative_position_bias_obfuscation",
        ],
        "limitations": [
            "T5/BART obfuscated forward (LayerNorm / FFN / activation / LM head) is not implemented.",
            "Decoder self-attention KV cache is not implemented.",
            "Encoder-decoder generation is not implemented.",
            "Relative position bias is not obfuscated.",
            "Real TEE isolation is not implemented.",
        ],
    },
)


# ---------------------------------------------------------------------------
# Upstream JSON readers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _safe_max(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return max(clean) if clean else None


def _aggregate_decoder_only(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the Stage 5.0 GPT-2 attention probe schema."""
    results = payload.get("results", [])
    num_cells = len(results)
    if not num_cells:
        return _empty_aggregation()

    output_errors: list[float] = []
    score_errors: list[float] = []
    prob_errors: list[float] = []
    cache_errors: list[float] = []
    all_allclose = True
    use_pad_values: set[bool] = set()
    model_ids: set[str] = set()
    num_rows = 0

    for r in results:
        cfg = r.get("config", {})
        use_pad_values.add(bool(cfg.get("use_pad")))
        model_ids.add(str(cfg.get("model_id")))
        full = r.get("full_attention", {})
        prefill = r.get("prefill_attention", {})
        decode = r.get("decode_attention", {})

        for block, key in (
            (full, "output_metrics"),
            (full, "score_metrics"),
            (full, "prob_metrics"),
            (prefill, "output_metrics"),
        ):
            m = block.get(key) or {}
            err = m.get("max_abs_error")
            if err is None:
                continue
            if key == "output_metrics":
                output_errors.append(err)
            elif key == "score_metrics":
                score_errors.append(err)
            elif key == "prob_metrics":
                prob_errors.append(err)

        for block_name, payload_block in (("prefill", prefill), ("decode", decode)):
            for cache_key in ("cache_key_metrics", "cache_value_metrics",
                              "cache_append_key_metrics", "cache_append_value_metrics"):
                m = payload_block.get(cache_key)
                if isinstance(m, dict):
                    err = m.get("max_abs_error")
                    if err is not None:
                        cache_errors.append(err)

        if not bool(full.get("allclose", True)):
            all_allclose = False
        if not bool(prefill.get("cache_invariant_allclose", True)):
            all_allclose = False
        if not bool(decode.get("cache_append_invariant_allclose", True)):
            all_allclose = False
        # one row per cell — decoder probe does not split by mask kind
        num_rows += 1

    return {
        "num_cells": num_cells,
        "num_rows": num_rows,
        "all_loaded_allclose": all_allclose,
        "max_output_error": _safe_max(output_errors),
        "max_score_error": _safe_max(score_errors),
        "max_prob_error": _safe_max(prob_errors),
        "max_cache_error": _safe_max(cache_errors),
        "use_pad_supported": sorted(use_pad_values),
        "padding_mask_supported": False,
        "bias_present": {"q": True, "k": True, "v": True, "o": True},
        "has_relative_attention_bias": False,
        "model_ids_seen": sorted(model_ids),
    }


def _aggregate_per_mask(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the Stage 6.1 / 6.2 ``results_per_mask`` schema."""
    results = payload.get("results", [])
    num_cells = len(results)
    if not num_cells:
        return _empty_aggregation()

    output_errors: list[float] = []
    score_errors: list[float] = []
    prob_errors: list[float] = []
    cache_errors: list[float] = []
    all_allclose = True
    loaded_any = False
    use_pad_values: set[bool] = set()
    model_ids: set[str] = set()
    bias_present: dict[str, bool] | None = None
    has_relative_attention_bias = False
    num_rows = 0

    for r in results:
        cfg = r.get("config", {})
        loading = r.get("model_loading", {})
        use_pad_values.add(bool(cfg.get("use_pad")))
        if loading.get("status") == "loaded":
            loaded_any = True
            model_ids.add(str(loading.get("model_id")))
            if bias_present is None and "bias_present" in loading:
                bias_present = dict(loading["bias_present"])
            has_relative_attention_bias = bool(
                loading.get("cross_attention_has_relative_bias",
                            has_relative_attention_bias)
            )
        per_mask = r.get("results_per_mask", {}) or {}
        for mk, m in per_mask.items():
            num_rows += 1
            for key, bucket in (
                ("score_metrics", score_errors),
                ("prob_metrics", prob_errors),
                ("output_metrics", output_errors),
            ):
                metric = m.get(key) or {}
                err = metric.get("max_abs_error")
                if err is not None:
                    bucket.append(err)
            if not bool(m.get("allclose", True)):
                all_allclose = False
        cache = r.get("encoder_memory_cache") or {}
        for key in ("key_metrics", "value_metrics"):
            metric = cache.get(key)
            if isinstance(metric, dict):
                err = metric.get("max_abs_error")
                if err is not None:
                    cache_errors.append(err)
        if isinstance(cache.get("allclose"), bool) and not cache["allclose"]:
            all_allclose = False

    return {
        "num_cells": num_cells,
        "num_rows": num_rows,
        "all_loaded_allclose": all_allclose if loaded_any else None,
        "max_output_error": _safe_max(output_errors),
        "max_score_error": _safe_max(score_errors),
        "max_prob_error": _safe_max(prob_errors),
        "max_cache_error": _safe_max(cache_errors),
        "use_pad_supported": sorted(use_pad_values),
        "padding_mask_supported": True,
        "bias_present": bias_present if bias_present is not None else {},
        "has_relative_attention_bias": has_relative_attention_bias,
        "model_ids_seen": sorted(model_ids),
    }


def _empty_aggregation() -> dict[str, Any]:
    return {
        "num_cells": 0,
        "num_rows": 0,
        "all_loaded_allclose": None,
        "max_output_error": None,
        "max_score_error": None,
        "max_prob_error": None,
        "max_cache_error": None,
        "use_pad_supported": [],
        "padding_mask_supported": False,
        "bias_present": {},
        "has_relative_attention_bias": False,
        "model_ids_seen": [],
    }


# ---------------------------------------------------------------------------
# Architecture coverage + workload readers
# ---------------------------------------------------------------------------


def _read_coverage(out_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(out_dir / "architecture_coverage.json") or {}
    coverage = payload.get("coverage", []) or []
    return {entry.get("architecture_key"): entry for entry in coverage if "architecture_key" in entry}


def _read_workload(out_dir: Path) -> dict[str, Any]:
    payload = _load_json(out_dir / "workload_profile.json") or {}
    if not payload:
        return {"status": "missing"}
    methods_payload = payload.get("methods", {}) or {}
    methods_summary: list[dict[str, Any]] = []
    for name, m in methods_payload.items():
        methods_summary.append(
            {
                "method": name,
                "implemented": m.get("implemented"),
                "online_boundary_calls": m.get("online_boundary_calls"),
                "boundary_calls_formula": m.get("boundary_calls_formula"),
                "online_trusted_compute_ops": m.get("online_trusted_compute_ops"),
                "online_gpu_ops": m.get("online_gpu_ops"),
                "preprocessing_trusted_ops": m.get("preprocessing_trusted_ops"),
                "measured_wall_time_ms": m.get("measured_wall_time_ms"),
                "wall_time_source": m.get("wall_time_source"),
                "online_extra_matmul_count": m.get("online_extra_matmul_count"),
                "uses_compatible_nonlinear_islands": m.get(
                    "uses_compatible_nonlinear_islands"
                ),
                "security_profile": m.get("security_profile"),
            }
        )
    return {
        "status": "loaded",
        "config": payload.get("config"),
        "calibration": payload.get("calibration"),
        "methods": methods_summary,
        "paper_metrics": payload.get("paper_metrics"),
        "wrapper_integration_status": payload.get("wrapper_integration_status"),
        "method_records_full": methods_payload,
    }


def _compatible_island_projection(
    architectures: list[dict[str, Any]],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Per-architecture projection for ``ours_compatible_nonlinear_islands``.

    Each architecture row records the current "trusted-shortcuts" method
    (Stage 4.x for decoder_only, Stage 6.x probe + trusted shortcuts for
    encoder_only / encoder_decoder) alongside the projected island method.
    Status is ``projected_from_probe`` for every architecture — Stage 5.3
    is the wrapper integration step.
    """
    methods = workload.get("methods", []) or []
    ours_current_method = next(
        (m for m in methods if m["method"] == "ours_current"), None
    )
    islands_method = next(
        (m for m in methods if m["method"] == "ours_compatible_nonlinear_islands"),
        None,
    )
    paper_metrics = (workload.get("paper_metrics") or {}).get(
        "ours_compatible_nonlinear_islands", {}
    )

    if islands_method is None or ours_current_method is None:
        return {
            "status": "workload_missing_or_method_absent",
            "per_architecture": [],
        }

    trusted_reduction = paper_metrics.get(
        "trusted_compute_reduction_vs_ours_current"
    )
    boundary_reduction = paper_metrics.get(
        "boundary_call_reduction_vs_ours_current"
    )
    per_architecture: list[dict[str, Any]] = []
    for arch in architectures:
        per_architecture.append(
            {
                "architecture_type": arch["architecture_type"],
                "model_id": arch["model_id"],
                "attention_kind": arch["attention_kind"],
                "current_method": "ours_current"
                if arch["architecture_type"] == "decoder_only"
                else "stage6_probe_plus_trusted_shortcuts",
                "compatible_island_method": "ours_compatible_nonlinear_islands",
                "current_boundary_formula": ours_current_method.get(
                    "boundary_calls_formula"
                ),
                "compatible_boundary_formula": islands_method.get(
                    "boundary_calls_formula"
                ),
                "trusted_compute_reduction": trusted_reduction,
                "boundary_call_reduction": boundary_reduction,
                "online_extra_matmul_count": islands_method.get(
                    "online_extra_matmul_count", 0
                ),
                "security_proxy_status": islands_method.get(
                    "security_profile"
                ),
                "status": "projected_from_probe",
                "limitations": [
                    "Compatible mask families are weaker than unrestricted"
                    " dense masks inside nonlinear islands.",
                    "Fresh permutation, dense sandwiching, and pad at Linear"
                    " boundaries are required mitigations.",
                    "Not yet integrated into the GPT-2 / BERT / T5 wrappers.",
                    "Projected, not measured. No real TEE isolation.",
                ],
            }
        )
    return {
        "status": "available",
        "per_architecture": per_architecture,
        "method_record": {
            "title": islands_method.get("method"),
            "implemented": islands_method.get("implemented"),
            "boundary_calls_formula": islands_method.get("boundary_calls_formula"),
            "online_boundary_calls": islands_method.get("online_boundary_calls"),
            "online_trusted_compute_ops": islands_method.get(
                "online_trusted_compute_ops"
            ),
            "online_extra_matmul_count": islands_method.get(
                "online_extra_matmul_count", 0
            ),
            "security_profile": islands_method.get("security_profile"),
            "wall_time_source": islands_method.get("wall_time_source"),
        },
        "paper_metrics": paper_metrics,
    }


# ---------------------------------------------------------------------------
# Stage 5.3c — Compatible Island Integration Status (per architecture)
# ---------------------------------------------------------------------------


_ARCH_TO_INTEGRATION_KEYS: dict[str, tuple[str, ...]] = {
    "decoder_only": ("gpt2_model_level", "gpt2_single_block"),
    "encoder_only": ("bert",),
    "encoder_decoder": ("t5",),
}


_STATUS_TO_LEVEL: dict[str, str] = {
    "implemented": "model_level",
    "implemented_probe_level": "probe_level",
    "implemented_block_level": "block_level",
    "not_yet": "not_yet",
    "missing": "not_yet",
}


def _resolve_integration_level(arch_key: str, wrapper_status: dict[str, Any]) -> str:
    keys = _ARCH_TO_INTEGRATION_KEYS.get(arch_key, ())
    for k in keys:
        raw = str(wrapper_status.get(k, "not_yet"))
        level = _STATUS_TO_LEVEL.get(raw, raw)
        if level != "not_yet":
            return level
    return "not_yet"


def _compatible_island_integration_status(
    architectures: list[dict[str, Any]],
    workload: dict[str, Any],
) -> dict[str, Any]:
    """Stage 5.3c — per-architecture compatible-island integration table.

    Reads ``workload_profile.json``'s ``wrapper_integration_status`` for
    ``ours_compatible_nonlinear_islands`` and emits one row per architecture
    with the integration level (``model_level`` / ``probe_level`` /
    ``not_yet``), available nonlinear modes, pad support, online extra
    matmul count, the Stage 5.2b security proxy status, and the
    documented per-architecture limitations.
    """
    method_records = workload.get("method_records_full") or {}
    method = method_records.get("ours_compatible_nonlinear_islands", {}) or {}
    wrapper_status = method.get("wrapper_integration_status") or {}
    top_status = (
        workload.get("wrapper_integration_status") or {}
    ).get("ours_compatible_nonlinear_islands") or {}
    # Prefer per-method wrapper status; fall back to top-level mirror.
    effective_status = {**top_status, **wrapper_status}

    if not effective_status:
        return {
            "status": "workload_missing_or_method_absent",
            "per_architecture": [],
            "measured_integration_scope": None,
            "full_runtime_integrated": None,
            "all_architecture_probe_level_implemented": None,
            "security_profile": method.get("security_profile"),
        }

    online_extra_matmul = int(method.get("online_extra_matmul_count", 0) or 0)
    security_profile = method.get("security_profile") or top_status.get(
        "security_profile"
    )

    per_architecture: list[dict[str, Any]] = []
    for arch in architectures:
        arch_key = arch["architecture_type"]
        level = _resolve_integration_level(arch_key, effective_status)
        if arch_key == "decoder_only":
            nonlinear_modes = ["trusted", "compatible_islands"]
            limitations = [
                "GPT-2 model-level integration is measured smoke, not a real TEE measurement.",
                "LayerNorm remains trusted.",
                "Default mode remains trusted; compatible_islands is gated behind a feature flag.",
            ]
        elif arch_key == "encoder_only":
            nonlinear_modes = ["trusted", "compatible_islands"]
            limitations = [
                "BERT is probe-level integration, not a full BERT wrapper.",
                "MLM head, pooler, and classifier are not modified.",
                "LayerNorm remains trusted.",
                "Default mode remains trusted; compatible_islands is gated behind a feature flag.",
            ]
        elif arch_key == "encoder_decoder":
            nonlinear_modes = ["trusted", "compatible_islands"]
            limitations = [
                "T5 / BART is probe-level integration, not a full wrapper.",
                "LM head and encoder-decoder generation are not modified.",
                "Cross-attention probe invariants (Stage 6.2) are not modified.",
                "Gated-GELU is not yet supported (Stage 5.2a only covers SiLU gated MLP island).",
                "Default mode remains trusted; compatible_islands is gated behind a feature flag.",
            ]
        else:
            nonlinear_modes = []
            limitations = []
        per_architecture.append(
            {
                "architecture_type": arch_key,
                "model_id": arch["model_id"],
                "integration_level": level,
                "nonlinear_mode_available": nonlinear_modes,
                "use_pad_supported": True,
                "online_extra_matmul_count": online_extra_matmul,
                "security_proxy_status": security_profile,
                "limitations": limitations,
            }
        )

    modern_decoder_status = (
        effective_status.get("qwen_or_modern_decoder")
        or effective_status.get("modern_decoder_probe")
    )
    modern_decoder_block_wrapper_status = effective_status.get(
        "modern_decoder_block_wrapper"
    )
    modern_decoder_model_wrapper_status = effective_status.get(
        "modern_decoder_model_wrapper"
    )
    modern_decoder_row = None
    if modern_decoder_status:
        is_model_level = (
            str(modern_decoder_status) == "implemented"
            or modern_decoder_model_wrapper_status == "implemented"
        )
        is_block_level = (
            str(modern_decoder_status) == "implemented_block_level"
            or modern_decoder_block_wrapper_status == "implemented"
        )
        modern_decoder_row = {
            # Logical type stays ``decoder_only`` (per architecture taxonomy);
            # ``modern_decoder_only`` is a display label so the integration
            # table can distinguish this row from the GPT-2 / decoder_only row.
            "architecture_type": "modern_decoder_only",
            "logical_architecture_type": "decoder_only",
            "model_id": "qwen_like / llama_like / synthetic_modern_decoder",
            "model_family": "qwen_like / llama_like / synthetic_modern_decoder",
            "nonlinear_mode_available": ["trusted", "compatible_islands"],
            "use_pad_supported": True,
            "integration_level": _STATUS_TO_LEVEL.get(
                str(modern_decoder_status), str(modern_decoder_status)
            ),
            "norm_type": "rmsnorm",
            "activation_type": "swiglu",
            "position_encoding_type": "rotary",
            "attention_variant": "mha/gqa/mqa",
            "online_extra_matmul_count": online_extra_matmul,
            "security_proxy_status": (
                method.get("security_profile_detail")
                or security_profile
            ),
            "modern_decoder_probe_status": str(
                effective_status.get("modern_decoder_probe", "not_yet")
            ),
            "modern_decoder_block_wrapper_status": str(
                modern_decoder_block_wrapper_status or "not_yet"
            ),
            "modern_decoder_model_wrapper_status": str(
                modern_decoder_model_wrapper_status or "not_yet"
            ),
            "block_level_correctness_artifact": (
                "outputs/modern_decoder_block_wrapper_smoke.json"
                if is_block_level else None
            ),
            "model_level_correctness_artifact": (
                "outputs/modern_decoder_model_wrapper_smoke.json"
                if is_model_level else None
            ),
            "modern_decoder_generation_status": str(
                effective_status.get("modern_decoder_generation_status")
                or method.get("modern_decoder_generation_status")
                or "not_yet"
            ),
            "modern_decoder_kv_cache_status": str(
                effective_status.get("modern_decoder_kv_cache_status")
                or method.get("modern_decoder_kv_cache_status")
                or "not_yet"
            ),
            "real_activation_attacker_status": str(
                effective_status.get("real_activation_attacker_status")
                or method.get("real_activation_attacker_status")
                or "not_yet"
            ),
            "real_activation_attacker_scope": str(
                effective_status.get("real_activation_attacker_scope")
                or method.get("real_activation_attacker_scope")
                or "not_yet"
            ),
            "real_activation_attacker_artifact": (
                effective_status.get("real_activation_attacker_artifact")
                or method.get("real_activation_attacker_artifact")
            ),
            "real_token_activation_attacker_status": str(
                effective_status.get("real_token_activation_attacker_status")
                or method.get("real_token_activation_attacker_status")
                or "not_yet"
            ),
            "real_token_activation_attacker_scope": str(
                effective_status.get("real_token_activation_attacker_scope")
                or method.get("real_token_activation_attacker_scope")
                or "not_yet"
            ),
            "real_token_activation_attacker_artifact": (
                effective_status.get("real_token_activation_attacker_artifact")
                or method.get("real_token_activation_attacker_artifact")
            ),
            "security_profile_detail_with_real_token_activation": (
                effective_status.get(
                    "security_profile_detail_with_real_token_activation"
                )
                or method.get("security_profile_detail_with_real_token_activation")
            ),
            "stronger_attackers_status": str(
                effective_status.get("stronger_attackers_status")
                or method.get("stronger_attackers_status")
                or "not_yet"
            ),
            "stronger_attackers_artifact": (
                effective_status.get("stronger_attackers_artifact")
                or method.get("stronger_attackers_artifact")
            ),
            "blackbox_proxy_status": str(
                effective_status.get("blackbox_proxy_status")
                or method.get("blackbox_proxy_status")
                or "not_yet"
            ),
            "timing_sidechannel_proxy_status": str(
                effective_status.get("timing_sidechannel_proxy_status")
                or method.get("timing_sidechannel_proxy_status")
                or "not_yet"
            ),
            "inter_block_masking_gap_status": str(
                effective_status.get("inter_block_masking_gap_status")
                or method.get("inter_block_masking_gap_status")
                or "not_yet"
            ),
            "inter_block_masking_experimental_status": str(
                effective_status.get("inter_block_masking_experimental_status")
                or method.get("inter_block_masking_experimental_status")
                or "not_yet"
            ),
            "security_profile_detail_with_stronger_attackers": (
                effective_status.get(
                    "security_profile_detail_with_stronger_attackers"
                )
                or method.get("security_profile_detail_with_stronger_attackers")
            ),
            "inter_block_mask_mode_supported": bool(
                effective_status.get("inter_block_mask_mode_supported")
                or method.get("inter_block_mask_mode_supported")
                or False
            ),
            "masked_boundary_experimental_status": str(
                effective_status.get("masked_boundary_experimental_status")
                or method.get("masked_boundary_experimental_status")
                or "not_yet"
            ),
            "constant_time_decode_proxy_status": str(
                effective_status.get("constant_time_decode_proxy_status")
                or method.get("constant_time_decode_proxy_status")
                or "not_yet"
            ),
            "extended_proxy_status": str(
                effective_status.get("extended_proxy_status")
                or method.get("extended_proxy_status")
                or "not_yet"
            ),
            "extended_proxy_artifact": (
                effective_status.get("extended_proxy_artifact")
                or method.get("extended_proxy_artifact")
            ),
            "security_profile_detail_with_extended_proxy": (
                effective_status.get(
                    "security_profile_detail_with_extended_proxy"
                )
                or method.get("security_profile_detail_with_extended_proxy")
            ),
            # Stage 7.0 — LoRA private training prototype metadata.
            "lora_private_training_status": str(
                effective_status.get("lora_private_training_status")
                or method.get("lora_private_training_status")
                or "not_yet"
            ),
            "lora_forward_masking_status": str(
                effective_status.get("lora_forward_masking_status")
                or method.get("lora_forward_masking_status")
                or "not_yet"
            ),
            "lora_training_step_status": str(
                effective_status.get("lora_training_step_status")
                or method.get("lora_training_step_status")
                or "not_yet"
            ),
            "lora_security_proxy_status": str(
                effective_status.get("lora_security_proxy_status")
                or method.get("lora_security_proxy_status")
                or "not_yet"
            ),
            "lora_training_artifact": (
                effective_status.get("lora_training_artifact")
                or method.get("lora_training_artifact")
            ),
            "lora_security_artifact": (
                effective_status.get("lora_security_artifact")
                or method.get("lora_security_artifact")
            ),
            "lora_merge_adapter_into_w": bool(
                effective_status.get("lora_merge_adapter_into_w")
                if "lora_merge_adapter_into_w" in effective_status
                else method.get("lora_merge_adapter_into_w", False)
            ),
            "security_profile_detail_with_lora": (
                effective_status.get("security_profile_detail_with_lora")
                or method.get("security_profile_detail_with_lora")
            ),
            # Stage 7.1 — masked backward / gradient-side obfuscation
            "lora_backward_status": str(
                effective_status.get("lora_backward_status")
                or method.get("lora_backward_status")
                or "not_yet"
            ),
            "lora_loss_status": str(
                effective_status.get("lora_loss_status")
                or method.get("lora_loss_status")
                or "not_yet"
            ),
            "lora_optimizer_status": str(
                effective_status.get("lora_optimizer_status")
                or method.get("lora_optimizer_status")
                or "not_yet"
            ),
            "lora_gradient_security_proxy_status": str(
                effective_status.get("lora_gradient_security_proxy_status")
                or method.get("lora_gradient_security_proxy_status")
                or "not_yet"
            ),
            "lora_backward_artifact": (
                effective_status.get("lora_backward_artifact")
                or method.get("lora_backward_artifact")
            ),
            "lora_gradient_security_artifact": (
                effective_status.get("lora_gradient_security_artifact")
                or method.get("lora_gradient_security_artifact")
            ),
            "security_profile_detail_with_lora_backward": (
                effective_status.get(
                    "security_profile_detail_with_lora_backward"
                )
                or method.get("security_profile_detail_with_lora_backward")
            ),
            # Stage 7.2 — rank padding / hidden-rank LoRA
            "lora_rank_padding_status": str(
                effective_status.get("lora_rank_padding_status")
                or method.get("lora_rank_padding_status")
                or "not_yet"
            ),
            "lora_hidden_rank_status": str(
                effective_status.get("lora_hidden_rank_status")
                or method.get("lora_hidden_rank_status")
                or "not_yet"
            ),
            "lora_true_rank_hidden_from_shape": bool(
                effective_status.get("lora_true_rank_hidden_from_shape")
                if "lora_true_rank_hidden_from_shape" in effective_status
                else method.get("lora_true_rank_hidden_from_shape", False)
            ),
            "lora_padded_rank_visible": bool(
                effective_status.get("lora_padded_rank_visible")
                if "lora_padded_rank_visible" in effective_status
                else method.get("lora_padded_rank_visible", False)
            ),
            "lora_rank_padding_artifact": (
                effective_status.get("lora_rank_padding_artifact")
                or method.get("lora_rank_padding_artifact")
            ),
            "lora_rank_security_artifact": (
                effective_status.get("lora_rank_security_artifact")
                or method.get("lora_rank_security_artifact")
            ),
            "security_profile_detail_with_lora_rank_padding": (
                effective_status.get(
                    "security_profile_detail_with_lora_rank_padding"
                )
                or method.get("security_profile_detail_with_lora_rank_padding")
            ),
            # Stage 7.3 — multi-layer LoRA end-to-end training prototype +
            # cross-layer security proxy + LoRA training timing-side proxy.
            "lora_multilayer_training_status": str(
                effective_status.get("lora_multilayer_training_status")
                or method.get("lora_multilayer_training_status")
                or "not_yet"
            ),
            "lora_multilayer_training_artifact": (
                effective_status.get("lora_multilayer_training_artifact")
                or method.get("lora_multilayer_training_artifact")
            ),
            "lora_multilayer_security_proxy_status": str(
                effective_status.get("lora_multilayer_security_proxy_status")
                or method.get("lora_multilayer_security_proxy_status")
                or "not_yet"
            ),
            "lora_multilayer_security_artifact": (
                effective_status.get("lora_multilayer_security_artifact")
                or method.get("lora_multilayer_security_artifact")
            ),
            "lora_training_timing_proxy_status": str(
                effective_status.get("lora_training_timing_proxy_status")
                or method.get("lora_training_timing_proxy_status")
                or "not_yet"
            ),
            "lora_training_timing_artifact": (
                effective_status.get("lora_training_timing_artifact")
                or method.get("lora_training_timing_artifact")
            ),
            "security_profile_detail_with_lora_multilayer": (
                effective_status.get(
                    "security_profile_detail_with_lora_multilayer"
                )
                or method.get(
                    "security_profile_detail_with_lora_multilayer"
                )
            ),
            "limitations": (
                [
                    "Model-level wrapper smoke is allclose vs plain reference;"
                    " real TEE wall-time is not measured.",
                    "Greedy generation only; beam / top-k / top-p not implemented.",
                    "Real Qwen / TinyLlama loading is opt-in; pytest stays synthetic.",
                    "Inherits Stage 5.4 mitigation requirements.",
                    "Default mode remains trusted; compatible_islands is gated behind a feature flag.",
                ] if is_model_level else [
                    (
                        "Block-level integration only; not a full Qwen / TinyLlama"
                        " model-level wrapper."
                        if is_block_level else
                        "Probe-level migration only; not a full Qwen / TinyLlama wrapper."
                    ),
                    "No generation / decode_step / KV cache runtime is implemented.",
                    "RoPE is handled by masking after RoPE.",
                    "GQA / MQA is tensor-level only, not full runtime KV cache.",
                    "Inherits Stage 5.4 mitigation requirements.",
                    "Default mode remains trusted; compatible_islands is gated behind a feature flag.",
                ]
            ),
        }
        per_architecture.append(modern_decoder_row)
    # Stage 5.3e — surface the mitigation bundle support table.
    mitigation_bundle_selectable = bool(
        method.get("mitigation_bundle_selectable")
        or top_status.get("mitigation_bundle_selectable")
    )
    default_bundle = (
        method.get("default_mitigation_bundle")
        or top_status.get("default_mitigation_bundle")
        or "fresh_perm_only"
    )
    recommended_bundle = (
        method.get("recommended_default_on_bundle")
        or top_status.get("recommended_default_on_bundle")
        or "fresh_perm_plus_sandwich_plus_pad"
    )
    recommended_status = (
        method.get("recommended_default_on_status")
        or top_status.get("recommended_default_on_status")
        or "acceptable_with_mitigation_under_adaptive_proxy"
    )
    mitigation_bundle_support: list[dict[str, Any]] = []
    if mitigation_bundle_selectable:
        for entry in per_architecture:
            mitigation_bundle_support.append(
                {
                    "architecture": entry["architecture_type"],
                    "integration_level": entry["integration_level"],
                    "fresh_perm_only": "supported",
                    "fresh_perm_plus_sandwich_plus_pad": "supported",
                    "use_pad_supported": entry.get("use_pad_supported", True),
                    "dense_sandwich_enabled": True,
                    "online_extra_matmul_count": entry.get(
                        "online_extra_matmul_count", 0
                    ),
                    "default_on_candidate": recommended_bundle,
                    "security_profile": entry.get(
                        "security_proxy_status", security_profile
                    ),
                }
            )

    return {
        "status": "available",
        "per_architecture": per_architecture,
        "modern_decoder_row": modern_decoder_row,
        "measured_integration_scope": (
            method.get("measured_integration_scope")
            or top_status.get("measured_integration_scope")
        ),
        "full_runtime_integrated": (
            method.get("full_runtime_integrated")
            if "full_runtime_integrated" in method
            else top_status.get("full_runtime_integrated")
        ),
        "all_architecture_probe_level_implemented": (
            method.get("all_architecture_probe_level_implemented")
            if "all_architecture_probe_level_implemented" in method
            else top_status.get("all_architecture_probe_level_implemented")
        ),
        "mitigation_bundle_selectable": mitigation_bundle_selectable,
        "default_mitigation_bundle": default_bundle,
        "recommended_default_on_bundle": recommended_bundle,
        "recommended_default_on_status": recommended_status,
        "mitigation_bundle_support": mitigation_bundle_support,
        "security_profile": security_profile,
        "note": top_status.get("note"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cross_architecture_summary(
    config: CrossArchitectureSummaryConfig,
) -> dict[str, Any]:
    """Aggregate upstream Stage 5.0 / 6.0 / 6.1 / 6.2 JSON artifacts."""
    out_dir = Path(config.output_dir)
    coverage_map = _read_coverage(out_dir)
    workload = _read_workload(out_dir)

    if (
        config.require_existing_outputs
        and (workload.get("status") == "missing" or not coverage_map)
    ):
        raise FileNotFoundError(
            "require_existing_outputs=True but architecture_coverage.json or"
            " workload_profile.json is missing under "
            f"{out_dir!s}"
        )

    architectures: list[dict[str, Any]] = []
    for spec in ARCHITECTURE_SPECS:
        source_path = out_dir / spec["source_file"]
        upstream = _load_json(source_path)
        if upstream is None:
            if config.require_existing_outputs:
                raise FileNotFoundError(
                    f"Missing upstream output for {spec['architecture_type']}:"
                    f" {source_path}"
                )
            agg = _empty_aggregation()
            status = "missing"
        else:
            if spec["architecture_type"] == "decoder_only":
                agg = _aggregate_decoder_only(upstream)
            else:
                agg = _aggregate_per_mask(upstream)
            status = "aggregated"

        coverage_entry = coverage_map.get(spec["architecture_type"], {})
        coverage_spec = coverage_entry.get("spec", {}) if coverage_entry else {}
        module_paths = coverage_entry.get("module_paths", {}) if coverage_entry else {}

        architectures.append(
            {
                "architecture_type": spec["architecture_type"],
                "status": status,
                "model_id": spec["model_id"],
                "model_class": coverage_spec.get(
                    "model_class", spec["model_class_default"]
                ),
                "attention_kind": spec["attention_kind"],
                "cache_type": spec["cache_type"],
                "source_file": spec["source_file"],
                "trusted_shortcuts": list(spec["trusted_shortcuts"]),
                "limitations": list(spec["limitations"]),
                **agg,
                "coverage_spec": {
                    "has_encoder": coverage_spec.get("has_encoder"),
                    "has_decoder": coverage_spec.get("has_decoder"),
                    "has_cross_attention": coverage_spec.get("has_cross_attention"),
                    "supports_past_key_values": coverage_spec.get(
                        "supports_past_key_values"
                    ),
                    "vocab_size": coverage_spec.get("vocab_size"),
                    "hidden_size": coverage_spec.get("hidden_size"),
                    "num_layers": coverage_spec.get("num_layers"),
                    "num_heads": coverage_spec.get("num_heads"),
                },
                "module_paths": module_paths,
            }
        )

    compatible_island_projection = _compatible_island_projection(
        architectures, workload
    )
    compatible_island_integration_status = _compatible_island_integration_status(
        architectures, workload
    )

    return {
        "config": asdict(config),
        "architectures": architectures,
        "workload": workload,
        "compatible_island_projection": compatible_island_projection,
        "compatible_island_integration_status": compatible_island_integration_status,
        "global_summary": {
            "num_architectures": len(architectures),
            "num_aggregated": sum(
                1 for a in architectures if a["status"] == "aggregated"
            ),
            "num_missing": sum(
                1 for a in architectures if a["status"] == "missing"
            ),
            "all_architectures_allclose": all(
                a["all_loaded_allclose"] for a in architectures
                if a["all_loaded_allclose"] is not None
            )
            if any(
                a["all_loaded_allclose"] is not None for a in architectures
            )
            else None,
            "compatible_island_projection_available": (
                compatible_island_projection.get("status") == "available"
            ),
            "compatible_island_integration_status_available": (
                compatible_island_integration_status.get("status") == "available"
            ),
            "compatible_island_full_runtime_integrated": (
                compatible_island_integration_status.get("full_runtime_integrated")
                if compatible_island_integration_status.get("status") == "available"
                else None
            ),
            "compatible_island_all_architecture_probe_level_implemented": (
                compatible_island_integration_status.get(
                    "all_architecture_probe_level_implemented"
                )
                if compatible_island_integration_status.get("status") == "available"
                else None
            ),
        },
        "stage_note": (
            "Cross-architecture summary aggregates Stage 5.0 (decoder-only),"
            " Stage 6.1 (encoder-only) and Stage 6.2 (encoder-decoder cross-"
            "attention) probe outputs plus the Stage 5.0.1 / 5.2c workload"
            " profile. It does not re-execute any probe."
        ),
    }


__all__ = [
    "ARCHITECTURE_SPECS",
    "CrossArchitectureSummaryConfig",
    "run_cross_architecture_summary",
]
