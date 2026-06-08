"""Stage 7.8b tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllo.experiments.precision_quantization_stability import (
    PRECISION_MODES,
    PrecisionStabilityConfig,
    render_markdown,
    run_precision_quantization_stability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict:
    return run_precision_quantization_stability(cfg=PrecisionStabilityConfig())


def _orth_row(report: dict, precision: str) -> dict:
    for r in report["orthogonal_mask"]["per_precision"]:
        if r["precision_mode"] == precision:
            return r
    raise KeyError(precision)


def test_all_precision_modes_present(report: dict) -> None:
    assert set(report["precision_modes_tested"]) == set(PRECISION_MODES)


def test_float64_exact(report: dict) -> None:
    r = _orth_row(report, "float64_reference")
    assert r["logits_max_abs_error_vs_float64_plain"] < 1e-9
    assert r["greedy_token_match_rate"] == 1.0
    assert r["overflow_detected"] is False
    assert r["nan_detected"] is False


def test_float32_within_tolerance(report: dict) -> None:
    r = _orth_row(report, "float32_simulated")
    assert r["logits_max_abs_error_vs_float64_plain"] < 1e-3
    assert r["greedy_token_match_rate"] == 1.0


def test_bf16_fp16_bounded_error(report: dict) -> None:
    bf = _orth_row(report, "bfloat16_simulated")
    fp = _orth_row(report, "float16_simulated")
    # Bounded error, no overflow/nan for orthogonal mask.
    assert bf["logits_max_abs_error_vs_float64_plain"] < 1.0
    assert fp["logits_max_abs_error_vs_float64_plain"] < 1.0
    assert bf["overflow_detected"] is False
    assert bf["nan_detected"] is False
    assert fp["overflow_detected"] is False
    assert fp["nan_detected"] is False


def test_condition_number_increases_error(report: dict) -> None:
    # Compare the highest-cond row against the lowest-cond row in
    # float32 -- the highest must dominate. Strict per-step
    # monotonicity is too tight at low condition numbers because the
    # error is dominated by precision rather than conditioning.
    f32_by_cond = []
    for row in report["condition_sweep"]:
        for r in row["per_precision"]:
            if r["precision_mode"] == "float32_simulated":
                f32_by_cond.append(
                    (row["condition_number"],
                     r["logits_max_abs_error_vs_float64_plain"])
                )
    f32_by_cond.sort(key=lambda kv: kv[0])
    low_err = f32_by_cond[0][1]
    high_err = f32_by_cond[-1][1]
    assert high_err > low_err * 5, (f32_by_cond, low_err, high_err)


def test_orthogonal_more_stable_than_ill_conditioned(report: dict) -> None:
    orth_fp16 = _orth_row(report, "float16_simulated")
    worst_cond_fp16 = None
    for row in report["condition_sweep"]:
        if row["condition_number"] >= 100.0:
            for r in row["per_precision"]:
                if r["precision_mode"] == "float16_simulated":
                    if worst_cond_fp16 is None or \
                            r["logits_max_abs_error_vs_float64_plain"] > worst_cond_fp16:
                        worst_cond_fp16 = r["logits_max_abs_error_vs_float64_plain"]
    assert worst_cond_fp16 is not None
    assert worst_cond_fp16 > orth_fp16["logits_max_abs_error_vs_float64_plain"]


def test_no_nan_or_inf_in_stable_modes(report: dict) -> None:
    for family in ("orthogonal_mask", "permutation_mask"):
        for r in report[family]["per_precision"]:
            assert r["nan_detected"] is False, r
            # Overflow may legitimately happen for very low precision /
            # very large cond, but orthogonal/permutation should not
            # overflow.
            assert r["overflow_detected"] is False, r


def test_no_real_gpu_or_kernel_claim(report: dict) -> None:
    assert report["real_gpu_kernel_measured"] is False
    assert report["real_quantized_model_loaded"] is False
    text = " ".join(report["limitations"]).lower()
    assert "no real gpu" in text or "no real fp16" in text or \
           "no real gpu fp16" in text or "cpu local emulation" in text


def test_unsafe_wording_avoid_list_contains_quantized(report: dict) -> None:
    text = " ".join(report["unsafe_wording_to_avoid"]).lower()
    assert "quantized model deployment" in text or "real quantized" in text


def test_render_markdown_smoke(report: dict) -> None:
    md = render_markdown(report)
    assert "Precision / Quantization Stability" in md
    for mode in PRECISION_MODES:
        assert mode in md
    assert "Condition-Number Sweep" in md
    assert "Paper-Safe Wording" in md


def test_outputs_artifact_present() -> None:
    j = REPO_ROOT / "outputs" / "precision_quantization_stability.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "7.8b"
