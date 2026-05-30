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

    return {
        "config": asdict(config),
        "architectures": architectures,
        "workload": workload,
        "compatible_island_projection": compatible_island_projection,
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
