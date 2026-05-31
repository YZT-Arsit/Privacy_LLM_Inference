"""Stage 5.4 — Adaptive permutation / linkability attacker tests.

The unit-test config uses small ``num_train_samples`` and ``attacker_steps``
so the suite runs quickly on CPU. The full configuration is exercised by
``scripts/run_adaptive_island_attacks.py``.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.adaptive_island_attacker import (
    STRATEGIES,
    AdaptiveIslandAttackConfig,
    generate_structured_channel_data,
    run_adaptive_island_attacks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_adaptive_island_attacks.py"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "adaptive_island_attacks.json"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "adaptive_island_attacks.csv"
OUTPUT_MD = PROJECT_ROOT / "outputs" / "adaptive_island_attacks.md"


# ---------------------------------------------------------------------------
# Fast unit-test config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_report() -> dict:
    cfg = AdaptiveIslandAttackConfig(
        hidden_size=32,
        num_train_samples=256,
        num_test_samples=128,
        num_sessions=8,
        samples_per_session=32,
        permutation_pool_size=4,
        attacker_steps=10,        # keep MLP training tiny
        mlp_hidden_size=64,
        mlp_batch_size=32,
        soft_assignment_iters=20,
    )
    return run_adaptive_island_attacks(cfg)


# ---------------------------------------------------------------------------
# Section structure
# ---------------------------------------------------------------------------


def test_report_has_four_top_level_sections(fast_report) -> None:
    for section in (
        "linear_inverter",
        "mlp_inverter",
        "permutation_recovery",
        "mitigation_summary",
    ):
        assert section in fast_report, f"missing section: {section}"


def test_report_covers_all_six_strategies_in_inverters(fast_report) -> None:
    for section in ("linear_inverter", "mlp_inverter"):
        keys = set(fast_report[section]["strategies"].keys())
        assert set(STRATEGIES) <= keys, (section, keys)
    mitigation_strategies = {
        row["strategy"] for row in fast_report["mitigation_summary"]["per_strategy"]
    }
    assert set(STRATEGIES) <= mitigation_strategies


def test_permutation_recovery_has_both_attacks(fast_report) -> None:
    rec = fast_report["permutation_recovery"]
    assert "signature_matching" in rec
    assert "soft_assignment" in rec
    # Stage 5.4 baseline (4 strategies). Stage 5.3e additionally evaluates
    # the recommended ``fresh_perm_plus_sandwich_plus_pad`` bundle.
    baseline = {
        "fixed_permutation",
        "fresh_permutation_per_session",
        "permutation_pool",
        "dense_sandwich",
    }
    assert baseline <= set(rec["signature_matching"].keys())
    assert baseline <= set(rec["soft_assignment"].keys())


# ---------------------------------------------------------------------------
# Attacker effectiveness expectations
# ---------------------------------------------------------------------------


def test_linear_inverter_fixed_easier_than_dense_sandwich(fast_report) -> None:
    fixed = fast_report["linear_inverter"]["strategies"]["fixed_permutation"]
    sandwich = fast_report["linear_inverter"]["strategies"]["dense_sandwich"]
    assert fixed["relative_l2_error"] < sandwich["relative_l2_error"]
    assert fixed["cosine_similarity"] > sandwich["cosine_similarity"]


def test_permutation_recovery_fixed_above_dense_sandwich(fast_report) -> None:
    """The strongest perm-recovery attacker on fixed must beat dense sandwich."""
    sig = fast_report["permutation_recovery"]["signature_matching"]
    soft = fast_report["permutation_recovery"]["soft_assignment"]
    fixed_best = max(
        sig["fixed_permutation"]["top1_recovery_rate"],
        soft["fixed_permutation"]["top1_recovery_rate"],
    )
    sandwich_best = max(
        sig["dense_sandwich"]["top1_recovery_rate"],
        soft["dense_sandwich"]["top1_recovery_rate"],
    )
    assert fixed_best > sandwich_best


def test_boundary_pad_protects_boundary_view_only(fast_report) -> None:
    """Boundary view rec is hard; activation view under fixed perm is trivial."""
    linear = fast_report["linear_inverter"]["strategies"]
    boundary = linear["boundary_pad_only_boundary_view"]
    activation = linear["boundary_pad_only_activation_view"]
    # Boundary view: linear inverter cannot recover (fresh pad + mask per session).
    assert boundary["relative_l2_error"] > 0.5
    # Activation view: fixed permutation ⇒ trivial linear inverse.
    assert activation["relative_l2_error"] < 0.1
    # Cosine ordering matches.
    assert boundary["cosine_similarity"] < activation["cosine_similarity"]


# ---------------------------------------------------------------------------
# Mitigation decisions
# ---------------------------------------------------------------------------


def _decision(report: dict, strategy: str) -> dict:
    for row in report["mitigation_summary"]["per_strategy"]:
        if row["strategy"] == strategy:
            return row
    raise AssertionError(f"no decision row for {strategy!r}")


def test_fixed_permutation_marked_unsafe_default_on(fast_report) -> None:
    row = _decision(fast_report, "fixed_permutation")
    assert row["risk_level"] == "high"
    assert row["default_on_recommendation"] == "unsafe_default_on"


def test_dense_sandwich_marked_low_or_acceptable(fast_report) -> None:
    row = _decision(fast_report, "dense_sandwich")
    assert row["risk_level"] == "low"
    assert row["default_on_recommendation"] == "acceptable_with_mitigation"


def test_boundary_pad_activation_view_unsafe(fast_report) -> None:
    row = _decision(fast_report, "boundary_pad_only_activation_view")
    assert row["risk_level"] == "high"
    assert row["default_on_recommendation"] == "unsafe_default_on"


def test_fresh_permutation_not_marked_low(fast_report) -> None:
    """Fresh permutation alone is not safe under the adaptive attacker."""
    row = _decision(fast_report, "fresh_permutation_per_session")
    assert row["risk_level"] in {"medium", "high"}
    assert row["default_on_recommendation"] != "acceptable_with_mitigation"


def test_recommended_default_on_candidate_lists_three_mitigations(fast_report) -> None:
    candidate = fast_report["mitigation_summary"][
        "recommended_default_on_candidate"
    ].lower()
    assert "fresh" in candidate
    assert "sandwich" in candidate
    assert "pad" in candidate


def test_default_on_caveat_disclaims_formal_security(fast_report) -> None:
    text = fast_report["mitigation_summary"]["default_on_caveat"].lower()
    assert "not a formal security" in text or "not formal" in text
    assert "tee" in text


# ---------------------------------------------------------------------------
# Comparison with Stage 5.2b naive proxy
# ---------------------------------------------------------------------------


def test_comparison_with_naive_proxy_has_uplift_field(fast_report) -> None:
    comp = fast_report["comparison_with_naive_proxy"]["per_strategy"]
    # Stage 5.4 baseline; Stage 5.3e adds ``fresh_perm_plus_sandwich_plus_pad``.
    assert {
        "fixed_permutation",
        "fresh_permutation_per_session",
        "permutation_pool",
        "dense_sandwich",
    } <= set(comp.keys())
    for r in comp.values():
        assert "naive_signature_matching_top1" in r
        assert "adaptive_soft_assignment_top1" in r
        assert "absolute_uplift" in r


# ---------------------------------------------------------------------------
# Output safety — never emit secrets / full mask tensors
# ---------------------------------------------------------------------------


_LONG_NUMBER_ARRAY = re.compile(
    r"\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*){32,}-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\]"
)


def test_report_does_not_contain_secret_tensors(fast_report) -> None:
    blob = json.dumps(fast_report)
    assert "tensor(" not in blob
    assert "torch.Tensor" not in blob
    # No numeric array with >= hidden_size (32 in fast config) entries.
    assert _LONG_NUMBER_ARRAY.search(blob) is None


# ---------------------------------------------------------------------------
# Script / output artifacts (full config)
# ---------------------------------------------------------------------------


def test_script_generates_json_csv_markdown(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--hidden-size",
            "32",
            "--num-train-samples",
            "128",
            "--num-test-samples",
            "64",
            "--num-sessions",
            "4",
            "--samples-per-session",
            "16",
            "--attacker-steps",
            "5",
            "--mlp-hidden-size",
            "32",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    json_path = tmp_path / "adaptive_island_attacks.json"
    csv_path = tmp_path / "adaptive_island_attacks.csv"
    md_path = tmp_path / "adaptive_island_attacks.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    # CSV must be long-format and contain a mitigation_decision row.
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(
        r["section"] == "mitigation_decision" and r["strategy"] == "fixed_permutation"
        for r in rows
    )
    md = md_path.read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Threat Model",
        "Structured Synthetic Activation Distribution",
        "Learned Linear Inverter",
        "Small MLP Inverter",
        "Adaptive Permutation Recovery",
        "Mitigation Decision Table",
        "Comparison with Stage 5.2b Naive Proxy",
        "Limitations",
        "Next Stage Plan",
        "These are adaptive/proxy attacks, not formal security proofs",
        "Dense sandwiching reduces tested recovery but does not imply"
        " semantic security",
        "Default-on recommendations are conditional",
    ):
        assert phrase in md, f"missing phrase: {phrase!r}"
    # No secret tensor in any of the three.
    for path in (json_path, csv_path, md_path):
        text = path.read_text(encoding="utf-8")
        assert "tensor(" not in text
        assert _LONG_NUMBER_ARRAY.search(text) is None, path


def test_full_outputs_artifacts_present_or_skipped() -> None:
    """Sanity check on the committed outputs/ artifacts."""
    if not OUTPUT_JSON.exists():
        pytest.skip(
            "outputs/adaptive_island_attacks.json missing — run "
            "`python scripts/run_adaptive_island_attacks.py` first."
        )
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert "mitigation_summary" in payload


# ---------------------------------------------------------------------------
# Data generator helper
# ---------------------------------------------------------------------------


def test_structured_channel_data_shape_and_column_means() -> None:
    X = generate_structured_channel_data(2048, 32, seed=2026)
    assert X.shape == (2048, 32)
    means = X.mean(dim=0)
    # Monotonic mean profile from -2 → 2 (within sample noise).
    assert float(means[0]) < float(means[-1])
    # Per-column std varies meaningfully.
    stds = X.std(dim=0)
    assert float(stds.max() - stds.min()) > 0.5
