"""Tests for the Stage 5.2b nonlinear island security proxy."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments import (
    MASK_FAMILY_ACCOUNTING,
    NonlinearIslandSecurityConfig,
    compute_channel_signature,
    recover_permutation_by_signature,
    run_nonlinear_island_security,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_nonlinear_island_security.py"


# ---------------------------------------------------------------------------
# Shared fixture — one realistic run reused across most tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def security_payload():
    cfg = NonlinearIslandSecurityConfig(
        num_sessions=16,
        samples_per_session=32,
        batch_size=2,
        seq_len=4,
        hidden_size=64,
        seed=2026,
    )
    return run_nonlinear_island_security(cfg)


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------


def test_report_has_three_sections(security_payload) -> None:
    for section in (
        "permutation_recovery",
        "island_linkability",
        "mask_family_accounting",
        "global_summary",
        "limitations",
        "threat_model",
    ):
        assert section in security_payload, f"missing section: {section}"


# ---------------------------------------------------------------------------
# Proxy 1 — permutation recovery
# ---------------------------------------------------------------------------


def test_permutation_recovery_lists_four_strategies(security_payload) -> None:
    per = security_payload["permutation_recovery"]["per_strategy"]
    assert set(per.keys()) == {
        "fixed_permutation",
        "fresh_permutation_per_session",
        "permutation_pool",
        "dense_sandwich_reference",
    }
    for name, m in per.items():
        for field in (
            "top1_recovery_rate",
            "top5_recovery_rate",
            "mean_correct_rank",
            "mean_signature_error",
            "expected_risk_level",
            "interpretation",
        ):
            assert field in m, f"strategy {name} missing field {field}"


def test_fixed_recovery_higher_than_dense_sandwich(security_payload) -> None:
    p = security_payload["permutation_recovery"]["per_strategy"]
    assert (
        p["fixed_permutation"]["top1_recovery_rate"]
        > p["dense_sandwich_reference"]["top1_recovery_rate"]
    )


def test_fixed_recovery_at_least_as_strong_as_fresh(security_payload) -> None:
    """Fixed permutation must be more recoverable than fresh by top1 or rank."""
    p = security_payload["permutation_recovery"]["per_strategy"]
    fixed_top1 = p["fixed_permutation"]["top1_recovery_rate"]
    fresh_top1 = p["fresh_permutation_per_session"]["top1_recovery_rate"]
    fixed_rank = p["fixed_permutation"]["mean_correct_rank"]
    fresh_rank = p["fresh_permutation_per_session"]["mean_correct_rank"]
    assert fixed_top1 > fresh_top1 or fixed_rank < fresh_rank, (
        f"expected fixed > fresh in top1 ({fixed_top1} vs {fresh_top1}) "
        f"or fixed < fresh in rank ({fixed_rank} vs {fresh_rank})"
    )


def test_signature_and_recover_helpers_are_exposed() -> None:
    """Helpers must be reachable from the package surface."""
    import torch

    x = torch.randn(64, 16)
    sig = compute_channel_signature(x)
    assert sig.shape == (16, 6)
    metrics = recover_permutation_by_signature(sig, sig)
    # Identical signatures ⇒ perfect recovery.
    assert metrics["top1_recovery_rate"] == pytest.approx(1.0)
    assert metrics["mean_correct_rank"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Proxy 2 — island linkability
# ---------------------------------------------------------------------------


def test_island_linkability_lists_four_strategies(security_payload) -> None:
    per = security_payload["island_linkability"]["per_strategy"]
    assert set(per.keys()) == {
        "fixed_perm_no_pad",
        "fixed_perm_with_linear_boundary_pad",
        "fresh_perm_with_linear_boundary_pad",
        "dense_to_perm_to_dense_sandwich",
    }


def test_fixed_no_pad_more_linkable_than_fresh_with_pad(security_payload) -> None:
    l = security_payload["island_linkability"]["per_strategy"]
    fixed_cos = l["fixed_perm_no_pad"]["mean_pairwise_cosine"]
    fresh_pad_cos = l["fresh_perm_with_linear_boundary_pad"][
        "activation_input_visible"
    ]["mean_pairwise_cosine"]
    assert fixed_cos > fresh_pad_cos
    assert fixed_cos > 0.99  # identical visible tensor for fixed P + same X


def test_fixed_perm_with_pad_dual_view_exposes_pad_only_helps_boundary(
    security_payload,
) -> None:
    """Pad protects the boundary view; activation view stays linkable under fixed P."""
    s = security_payload["island_linkability"]["per_strategy"][
        "fixed_perm_with_linear_boundary_pad"
    ]
    assert "boundary_input_visible" in s
    assert "activation_input_visible" in s
    # boundary view: low linkability
    assert s["boundary_input_visible"]["mean_pairwise_cosine"] < 0.5
    # activation view: still highly linkable
    assert s["activation_input_visible"]["mean_pairwise_cosine"] > 0.99


def test_dense_sandwich_is_lowest_linkability(security_payload) -> None:
    main = security_payload["island_linkability"]["main_metric_per_strategy"][
        "values"
    ]
    assert (
        main["dense_to_perm_to_dense_sandwich"]
        <= main["fresh_perm_with_linear_boundary_pad"]
    )
    assert main["dense_to_perm_to_dense_sandwich"] < 0.1


# ---------------------------------------------------------------------------
# Proxy 3 — mask family security accounting
# ---------------------------------------------------------------------------


def test_mask_family_accounting_includes_permutation_multiset_note(
    security_payload,
) -> None:
    by_family = {
        e["mask_family"]: e
        for e in security_payload["mask_family_accounting"]["table"]
    }
    assert "permutation" in by_family
    text = " ".join(by_family["permutation"].values()).lower()
    assert "coordinate-value multiset" in text


def test_mask_family_accounting_records_orthogonal_norm_preservation(
    security_payload,
) -> None:
    by_family = {
        e["mask_family"]: e
        for e in security_payload["mask_family_accounting"]["table"]
    }
    assert "orthogonal" in by_family
    text = " ".join(by_family["orthogonal"].values()).lower()
    assert "row l2 norm" in text or "row norm" in text


def test_mask_family_accounting_records_mean_preserving_invariants(
    security_payload,
) -> None:
    by_family = {
        e["mask_family"]: e
        for e in security_payload["mask_family_accounting"]["table"]
    }
    assert "mean_preserving_orthogonal" in by_family
    text = " ".join(by_family["mean_preserving_orthogonal"].values()).lower()
    assert "row mean" in text
    assert "centered" in text


def test_mask_family_accounting_static_list_is_exported() -> None:
    families = {e["mask_family"] for e in MASK_FAMILY_ACCOUNTING}
    assert {
        "dense_invertible",
        "orthogonal",
        "mean_preserving_orthogonal",
        "permutation",
        "paired_permutation",
    }.issubset(families)


# ---------------------------------------------------------------------------
# Limitations + threat model text
# ---------------------------------------------------------------------------


def test_global_limitations_recorded(security_payload) -> None:
    text = " ".join(security_payload["limitations"]).lower()
    assert "security proxies" in text
    assert "not formal security proofs" in text
    assert "compatible mask families are weaker than unrestricted dense masks" in text
    assert "permutation islands hide channel identity" in text
    assert "real tee" in text


# ---------------------------------------------------------------------------
# JSON / CSV / Markdown emitter + secret-tensor leakage check
# ---------------------------------------------------------------------------


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--num-sessions",
            "12",
            "--samples-per-session",
            "32",
            "--hidden-size",
            "64",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    for filename in (
        "nonlinear_island_security.json",
        "nonlinear_island_security.csv",
        "nonlinear_island_security.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "nonlinear_island_security.md").read_text(encoding="utf-8")
    for section in (
        "Threat Model for Proxy Experiments",
        "Permutation Recovery Proxy",
        "Island Linkability Proxy",
        "Mask Family Security Accounting",
        "Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert section in md, f"missing section: {section}"
    # Spec-mandated phrases inside Limitations.
    for phrase in (
        "Compatible mask families are weaker than unrestricted dense masks",
        "Permutation islands hide channel identity but do not hide coordinate-value multisets",
        "These experiments are security proxies, not formal security proofs",
    ):
        assert phrase in md, f"missing limitations phrase: {phrase!r}"

    payload = json.loads(
        (tmp_path / "nonlinear_island_security.json").read_text(encoding="utf-8")
    )
    assert {
        "permutation_recovery",
        "island_linkability",
        "mask_family_accounting",
        "global_summary",
        "limitations",
        "threat_model",
    }.issubset(payload.keys())
    assert "fixed_top1" in result.stdout


def test_outputs_contain_no_full_mask_tensors(tmp_path) -> None:
    """The JSON / CSV / Markdown must not embed any full-mask tensor.

    We forbid: explicit ``tensor(`` strings, and any numeric array of length
    >= hidden_size (which would betray a row of N or T) appearing as a JSON
    list. The genuine outputs only carry scalars and short strings.
    """
    hidden = 32
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--num-sessions",
            "6",
            "--samples-per-session",
            "16",
            "--hidden-size",
            str(hidden),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    json_text = (tmp_path / "nonlinear_island_security.json").read_text(
        encoding="utf-8"
    )
    csv_text = (tmp_path / "nonlinear_island_security.csv").read_text(
        encoding="utf-8"
    )
    md_text = (tmp_path / "nonlinear_island_security.md").read_text(
        encoding="utf-8"
    )
    for body in (json_text, csv_text, md_text):
        assert "tensor(" not in body
        assert "torch.Tensor" not in body
    # Reject any JSON array of >= hidden numbers (would be a mask row).
    json_payload = json.loads(json_text)

    def _walk(node):
        if isinstance(node, list):
            if len(node) >= hidden and all(
                isinstance(x, (int, float)) for x in node
            ):
                raise AssertionError(
                    f"JSON contains a numeric array of length {len(node)}"
                    f" >= hidden_size={hidden} (possible mask leak)"
                )
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)

    _walk(json_payload)
