"""Stage 5.5 — real activation adaptive attacker tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.real_activation_attacker import (
    EXTENDED_BUNDLES,
    PERMUTATION_TARGET_TENSORS,
    RealActivationAttackConfig,
    run_real_activation_attacks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_real_activation_attacks.py"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "real_activation_attacks.json"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "real_activation_attacks.csv"
OUTPUT_MD = PROJECT_ROOT / "outputs" / "real_activation_attacks.md"

_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


@pytest.fixture(scope="module")
def fast_report() -> dict:
    cfg = RealActivationAttackConfig(
        num_samples=256, attacker_steps=10, mlp_hidden_size=64,
        batch_size=2, seq_len=8,
        synthetic_hidden_size=32, synthetic_intermediate_size=64,
        synthetic_num_attention_heads=4, synthetic_num_key_value_heads=2,
        synthetic_head_dim=8,
    )
    return run_real_activation_attacks(cfg)


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_report_top_level_keys_present(fast_report) -> None:
    for k in (
        "config", "model_loading", "source", "block_spec",
        "trace_summary", "target_tensor_results",
        "bundle_comparison", "attacker_summary", "recommendation",
        "limitations",
    ):
        assert k in fast_report, f"missing top-level key {k!r}"


def test_bundles_evaluated_includes_full_bundle(fast_report) -> None:
    assert "fresh_perm_plus_sandwich_plus_pad" in fast_report["target_tensor_results"]
    assert "fresh_perm_only" in fast_report["target_tensor_results"]


def test_target_tensor_results_cover_required_tensors(fast_report) -> None:
    by_bundle = fast_report["target_tensor_results"]
    for bundle in ("fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad"):
        tensors = set(by_bundle[bundle].keys())
        assert {"gate", "up", "swiglu_intermediate", "post_island"} <= tensors


def test_per_tensor_payload_has_required_subsections(fast_report) -> None:
    by_bundle = fast_report["target_tensor_results"]
    for bundle, by_tensor in by_bundle.items():
        for tensor, payload in by_tensor.items():
            for k in ("linear_inverter", "mlp_inverter", "linkability",
                      "risk_level", "default_on_recommendation"):
                assert k in payload, f"missing {k!r} on {tensor}/{bundle}"


def test_permutation_recovery_only_for_swiglu_tensors(fast_report) -> None:
    by_bundle = fast_report["target_tensor_results"]
    for bundle, by_tensor in by_bundle.items():
        for tensor, payload in by_tensor.items():
            if tensor in PERMUTATION_TARGET_TENSORS:
                assert payload["permutation_recovery"] is not None
            else:
                assert payload.get("permutation_recovery") is None


# ---------------------------------------------------------------------------
# Bundle comparison
# ---------------------------------------------------------------------------


def test_bundle_comparison_has_required_columns(fast_report) -> None:
    comparison = fast_report["bundle_comparison"]
    assert len(comparison) >= 4
    for row in comparison:
        for k in (
            "tensor_name",
            "linear_rel_l2_delta", "mlp_rel_l2_delta",
            "linkability_cosine_delta",
            "risk_level_fresh_only", "risk_level_full_bundle",
            "math_equivalence_note",
        ):
            assert k in row


def test_recommendation_top_level_fields(fast_report) -> None:
    rec = fast_report["recommendation"]
    assert "default_on_recommendation_full_bundle" in rec
    assert "default_on_recommendation_fresh_only" in rec
    assert "security_profile_detail_with_real_activation" in rec
    assert "real-activation" in rec["security_profile_detail_with_real_activation"]
    assert "not formal" in rec["security_profile_detail_with_real_activation"]


# ---------------------------------------------------------------------------
# Conservative outcome under proxy
# ---------------------------------------------------------------------------


def test_full_bundle_not_worse_than_fresh_only(fast_report) -> None:
    """The full bundle's per-tensor risk_level should be ≤ fresh_only's risk_level.

    Because the two bundles produce numerically identical traces under the
    Stage 6.4b wrapper, the two risk_levels must match exactly per tensor.
    """
    order = {"low": 0, "medium": 1, "high": 2}
    fresh = fast_report["target_tensor_results"]["fresh_perm_only"]
    full = fast_report["target_tensor_results"]["fresh_perm_plus_sandwich_plus_pad"]
    for tensor in fresh:
        if tensor not in full:
            continue
        assert order[full[tensor]["risk_level"]] <= order[fresh[tensor]["risk_level"]]


def test_full_bundle_recommendation_is_acceptable_or_conservative(fast_report) -> None:
    rec = fast_report["recommendation"]["default_on_recommendation_full_bundle"]
    assert rec in {
        "acceptable_with_mitigation_under_real_activation_proxy",
        "needs_more_evaluation_under_real_activation_proxy",
    }


# ---------------------------------------------------------------------------
# Fixed-permutation debug baseline (opt-in)
# ---------------------------------------------------------------------------


def test_fixed_permutation_debug_shows_recovery() -> None:
    cfg = RealActivationAttackConfig(
        num_samples=256, attacker_steps=10, mlp_hidden_size=64,
        batch_size=2, seq_len=8,
        synthetic_hidden_size=32, synthetic_intermediate_size=64,
        synthetic_num_attention_heads=4, synthetic_num_key_value_heads=2,
        synthetic_head_dim=8,
        mitigation_bundles=(
            "fresh_perm_only",
            "fresh_perm_plus_sandwich_plus_pad",
            "fixed_permutation_debug",
        ),
    )
    r = run_real_activation_attacks(cfg)
    fresh = r["target_tensor_results"]["fresh_perm_only"]
    fixed = r["target_tensor_results"]["fixed_permutation_debug"]
    # Under fixed masks, the linear inverter on SwiGLU intermediates
    # collapses recovery error to ≈ 0 (well below fresh-mask values).
    for tensor in ("gate", "up", "swiglu_intermediate"):
        if tensor not in fixed:
            continue
        assert (
            fixed[tensor]["linear_inverter"]["relative_l2_error"]
            < fresh[tensor]["linear_inverter"]["relative_l2_error"]
        )
        assert fixed[tensor]["risk_level"] == "high"


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------


def test_report_does_not_contain_secret_tensors(fast_report) -> None:
    blob = json.dumps(
        {
            k: v
            for k, v in fast_report.items()
            if k != "traces"  # in-memory only; not present at this level
        }
    )
    assert "tensor(" not in blob
    assert "torch.Tensor" not in blob
    assert _LONG_NUMBER_ARRAY.search(blob) is None


# ---------------------------------------------------------------------------
# CLI script smoke
# ---------------------------------------------------------------------------


def test_script_emits_json_csv_markdown(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-samples", "192",
            "--attacker-steps", "5",
            "--mlp-hidden-size", "32",
            "--batch-size", "2", "--seq-len", "6",
            "--hidden-size", "32", "--intermediate-size", "64",
            "--num-query-heads", "4", "--num-kv-heads", "2", "--head-dim", "8",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    json_path = tmp_path / "real_activation_attacks.json"
    csv_path = tmp_path / "real_activation_attacks.csv"
    md_path = tmp_path / "real_activation_attacks.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "target_tensor_results" in payload
    assert "bundle_comparison" in payload
    assert "attacker_summary" in payload
    assert "recommendation" in payload

    md = md_path.read_text(encoding="utf-8")
    for required in (
        "real-activation adaptive proxy attacks, not formal security proofs",
        "Dense sandwiching reduces tested recovery but does not imply"
        " semantic security",
        "No real TEE isolation is evaluated",
        "Experiment Scope",
        "Model Loading Status",
        "Trace Collection Summary",
        "Target Tensor Inventory",
        "Linear Inverter on Real Activations",
        "Small MLP Inverter on Real Activations",
        "Permutation Recovery on Real Activations",
        "Linkability on Real Activations",
        "Mitigation Bundle Comparison",
        "Comparison with Stage 5.4 Synthetic Adaptive Attacker",
        "Recommendation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert required in md, f"missing markdown phrase: {required!r}"

    # No secret tensor in any of the three.
    for path in (json_path, csv_path, md_path):
        text = path.read_text(encoding="utf-8")
        assert "tensor(" not in text
        assert _LONG_NUMBER_ARRAY.search(text) is None, path


# ---------------------------------------------------------------------------
# Committed outputs sanity (skip if absent)
# ---------------------------------------------------------------------------


def test_committed_outputs_present_or_skipped() -> None:
    if not OUTPUT_JSON.exists():
        pytest.skip(
            "outputs/real_activation_attacks.json missing — run "
            "`python scripts/run_real_activation_attacks.py` first."
        )
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert "target_tensor_results" in payload
    assert "recommendation" in payload


def test_committed_markdown_present_or_skipped() -> None:
    if not OUTPUT_MD.exists():
        pytest.skip("outputs/real_activation_attacks.md missing")
    md = OUTPUT_MD.read_text(encoding="utf-8")
    assert "Stage 5.5" in md or "Real-Activation Adaptive Attacker" in md
