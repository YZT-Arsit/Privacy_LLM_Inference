"""Tests for Stage 5.0.1 calibrated workload profiler."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from pllo.experiments import (
    INTERACTION_CATEGORIES,
    MODULE_CATEGORIES,
    WORKLOAD_METHODS,
    WorkloadProfileConfig,
    run_workload_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "run_workload_profile.py"


def _try_run(config: WorkloadProfileConfig):
    try:
        return run_workload_profile(config)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if (
            "huggingface" in msg
            or "not found" in msg
            or "connection" in msg
            or "tiny-gpt2" in msg
        ):
            pytest.skip(f"sshleifer/tiny-gpt2 unavailable: {exc}")
        raise


@pytest.fixture(scope="module")
def profile():
    return _try_run(
        WorkloadProfileConfig(
            batch_size=1,
            prompt_len=4,
            max_new_tokens=2,
            warmup=1,
            repeat=2,
        )
    )


# ---------------------------------------------------------------------------
# Schema coverage
# ---------------------------------------------------------------------------


REQUIRED_METHOD_FIELDS = {
    "title",
    "summary",
    "implemented",
    "implementation_note",
    "citation_caveat",
    "online_boundary_calls",
    "online_trusted_compute_ops",
    "online_trusted_transfer_bytes",
    "online_gpu_ops",
    "preprocessing_trusted_ops",
    "preprocessing_transfer_bytes",
    "measured_wall_time_ms",
    "projected_wall_time_ms",
    "wall_time_source",
    "boundary_calls_formula",
}


def test_profile_contains_all_required_methods(profile) -> None:
    expected = {m.name for m in WORKLOAD_METHODS}
    assert set(profile["methods"].keys()) == expected
    assert "amulet_style_reference" in expected, "Stage 5.0.1 must include the Amulet reference method"


def test_each_method_has_full_new_schema(profile) -> None:
    for name, m in profile["methods"].items():
        missing = REQUIRED_METHOD_FIELDS - set(m.keys())
        assert not missing, f"method {name} missing fields: {missing}"


def test_all_proxies_are_non_negative(profile) -> None:
    for name, m in profile["methods"].items():
        assert m["online_boundary_calls"] >= 0, name
        assert m["online_trusted_compute_ops"] >= 0, name
        assert m["online_trusted_transfer_bytes"] >= 0, name
        assert m["online_gpu_ops"] >= 0, name
        assert m["preprocessing_trusted_ops"] >= 0, name
        assert m["preprocessing_transfer_bytes"] >= 0, name


# ---------------------------------------------------------------------------
# Cost-model separation (the headline Stage 5.0.1 fix)
# ---------------------------------------------------------------------------


def test_preprocessing_and_online_compute_are_separated(profile) -> None:
    ours = profile["methods"]["ours_current"]
    # Static weight obfuscation must show up as preprocessing, not online.
    assert ours["preprocessing_trusted_ops"] > 0, (
        "ours_current should report nonzero preprocessing_trusted_ops "
        "(W_tilde generation)"
    )
    # Preprocessing should be at least the LM head transform bytes.
    assert ours["preprocessing_transfer_bytes"] > 0
    # Plain and TSLP do not preprocess.
    assert profile["methods"]["plain_hf_gpu"]["preprocessing_trusted_ops"] == 0
    assert (
        profile["methods"]["tslp_trusted_nonlinear_baseline"]["preprocessing_trusted_ops"]
        == 0
    )


def test_mask_state_bookkeeping_not_counted_as_boundary(profile) -> None:
    """The pre-5.0.1 profiler inflated ours_current to ~10x the architectural
    formula (4L + 1 per forward) by counting per-linear mask-state creation as
    a boundary call. After the cleanup, ours_current should sit close to its
    documented formula and on the same order as TSLP, not an order of
    magnitude above it."""
    ours = profile["methods"]["ours_current"]["online_boundary_calls"]
    tslp = profile["methods"]["tslp_trusted_nonlinear_baseline"]["online_boundary_calls"]
    assert ours > 0
    assert tslp > 0
    # Sanity: ours_current must NOT be more than 2× TSLP. The old profiler
    # had ours_current ≈ 2.1× TSLP because of bookkeeping; the calibrated
    # version is within ~12.5% architecturally.
    assert ours <= 2 * tslp, (
        f"ours_current boundary calls {ours} more than 2x TSLP {tslp} — "
        "likely bookkeeping inflation has crept back in"
    )


def test_ours_ideal_has_fewer_boundary_calls_than_ours_current(profile) -> None:
    ideal = profile["methods"]["ours_ideal_gpu_nonlinear"]["online_boundary_calls"]
    current = profile["methods"]["ours_current"]["online_boundary_calls"]
    assert ideal < current, (
        f"ours_ideal_gpu_nonlinear ({ideal}) should have fewer boundary calls "
        f"than ours_current ({current})"
    )


def test_plain_method_has_zero_online_trusted_load(profile) -> None:
    plain = profile["methods"]["plain_hf_gpu"]
    assert plain["online_boundary_calls"] == 0
    assert plain["online_trusted_compute_ops"] == 0
    assert plain["online_trusted_transfer_bytes"] == 0
    assert plain["wall_time_source"] == "measured"


def test_implementation_flags(profile) -> None:
    assert profile["methods"]["plain_hf_gpu"]["implemented"] is True
    assert profile["methods"]["ours_current"]["implemented"] is True
    assert profile["methods"]["tslp_trusted_nonlinear_baseline"]["implemented"] is False
    assert profile["methods"]["ours_ideal_gpu_nonlinear"]["implemented"] is False
    assert profile["methods"]["amulet_style_reference"]["implemented"] is False


def test_unimplemented_methods_are_projected_only(profile) -> None:
    for name in (
        "tslp_trusted_nonlinear_baseline",
        "ours_ideal_gpu_nonlinear",
        "amulet_style_reference",
    ):
        m = profile["methods"][name]
        assert m["wall_time_source"] == "projected_from_op_counts"
        assert m["measured_wall_time_ms"] is None
        assert m["projected_wall_time_ms"] is not None


# ---------------------------------------------------------------------------
# Interaction breakdown (new in 5.0.1)
# ---------------------------------------------------------------------------


def test_interaction_breakdown_has_all_categories(profile) -> None:
    assert set(profile["interaction_breakdown"].keys()) == set(INTERACTION_CATEGORIES)
    for category, payload in profile["interaction_breakdown"].items():
        assert set(payload.keys()) == {m.name for m in WORKLOAD_METHODS}, category
        for method_name, counts in payload.items():
            for key in (
                "online_boundary_calls",
                "online_trusted_transfer_bytes",
                "online_trusted_compute_ops",
                "notes",
            ):
                assert key in counts, f"{category}/{method_name} missing {key}"


def test_tslp_layernorm_interaction_records_boundary_calls(profile) -> None:
    ln_for_tslp = profile["interaction_breakdown"]["trusted_layernorm"][
        "tslp_trusted_nonlinear_baseline"
    ]
    assert ln_for_tslp["online_boundary_calls"] > 0
    assert ln_for_tslp["online_trusted_compute_ops"] > 0


def test_ours_current_layernorm_interaction_has_compute_but_no_extra_boundary(profile) -> None:
    """Trusted shortcut: LN runs in trusted side without a separate ECALL."""
    ln_for_ours = profile["interaction_breakdown"]["trusted_layernorm"]["ours_current"]
    assert ln_for_ours["online_trusted_compute_ops"] > 0
    assert ln_for_ours["online_boundary_calls"] == 0


# ---------------------------------------------------------------------------
# Module breakdown still works with the new fields
# ---------------------------------------------------------------------------


def test_module_breakdown_uses_new_field_names(profile) -> None:
    assert set(profile["module_breakdown"].keys()) == set(MODULE_CATEGORIES)
    for category, payload in profile["module_breakdown"].items():
        for method in WORKLOAD_METHODS:
            entry = payload[method.name]
            for key in (
                "online_gpu_ops",
                "online_trusted_compute_ops",
                "online_boundary_calls",
                "online_trusted_transfer_bytes",
                "location",
            ):
                assert key in entry, f"module {category} method {method.name} missing {key}"


def test_layernorm_module_moves_between_methods(profile) -> None:
    ln = profile["module_breakdown"]["layernorm"]
    assert ln["plain_hf_gpu"]["online_trusted_compute_ops"] == 0
    assert ln["plain_hf_gpu"]["online_gpu_ops"] > 0
    assert ln["tslp_trusted_nonlinear_baseline"]["online_trusted_compute_ops"] > 0
    assert ln["ours_current"]["online_trusted_compute_ops"] > 0
    assert ln["ours_ideal_gpu_nonlinear"]["online_trusted_compute_ops"] == 0
    assert ln["ours_ideal_gpu_nonlinear"]["online_gpu_ops"] > 0


# ---------------------------------------------------------------------------
# Paper metrics
# ---------------------------------------------------------------------------


def test_paper_metrics_fields_present(profile) -> None:
    pm = profile["paper_metrics"]
    for key in (
        "boundary_call_reduction_vs_tslp",
        "trusted_transfer_reduction_vs_tslp",
        "online_trusted_compute_reduction_vs_tslp",
        "gpu_offload_ratio",
        "preprocessing_amortized",
        "boundary_calls_per_forward",
    ):
        assert key in pm, key
    assert 0.0 <= pm["gpu_offload_ratio"] <= 1.0
    assert pm["preprocessing_amortized"] is True


def test_boundary_call_reduction_not_inflated_by_bookkeeping(profile) -> None:
    """After the Stage 5.0.1 cleanup, ours_current vs TSLP should differ by
    at most one boundary call per layer per forward — not the ~10× inflation
    the previous schema reported."""
    pm = profile["paper_metrics"]
    # Architecturally ours_current crosses 4× per layer vs TSLP's 3×, so the
    # reduction will be slightly negative. It should NOT be huge negative.
    assert pm["boundary_call_reduction_vs_tslp"] > -0.5, (
        f"boundary_call_reduction_vs_tslp = "
        f"{pm['boundary_call_reduction_vs_tslp']:.2%} — too negative, "
        "suggests bookkeeping inflation"
    )


def test_boundary_calls_per_forward_documented(profile) -> None:
    formulas = profile["paper_metrics"]["boundary_calls_per_forward"]
    assert formulas["plain_hf_gpu"] == 0
    assert formulas["ours_ideal_gpu_nonlinear"] == 1
    assert formulas["amulet_style_reference"] == 1
    # ours_current should be 4L + 1, TSLP should be 3L + 2 (L=2 → 9, 8)
    assert formulas["ours_current"] == formulas["tslp_trusted_nonlinear_baseline"] + 1


# ---------------------------------------------------------------------------
# Interpretation + limitations
# ---------------------------------------------------------------------------


def test_interpretation_warns_about_simulated_cost_model(profile) -> None:
    interp = profile["interpretation"]
    assert "cost_model_warning" in interp
    assert "simulated" in interp["cost_model_warning"].lower()
    assert "real sgx" in interp["cost_model_warning"].lower()


def test_limitations_section_is_populated(profile) -> None:
    lims = profile["limitations"]
    assert isinstance(lims, list)
    assert len(lims) >= 3
    assert any("not real SGX" in l.lower() or "simulated tee" in l.lower() for l in lims)
    assert any("amulet" in l.lower() for l in lims)


# ---------------------------------------------------------------------------
# End-to-end script smoke
# ---------------------------------------------------------------------------


def test_workload_script_emits_all_three_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(WORKLOAD_SCRIPT),
            "--batch-size",
            "1",
            "--prompt-len",
            "4",
            "--max-new-tokens",
            "2",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for filename in (
        "workload_profile.json",
        "workload_profile.csv",
        "workload_profile.md",
    ):
        assert (tmp_path / filename).exists(), filename
    payload = json.loads((tmp_path / "workload_profile.json").read_text(encoding="utf-8"))
    for key in ("methods", "module_breakdown", "interaction_breakdown", "paper_metrics"):
        assert key in payload, key
    md = (tmp_path / "workload_profile.md").read_text(encoding="utf-8")
    assert "Method comparison" in md
    assert "Interaction breakdown" in md
    assert "Paper metrics" in md
    assert "Limitations" in md
    assert "main_online_bottleneck" in result.stdout
