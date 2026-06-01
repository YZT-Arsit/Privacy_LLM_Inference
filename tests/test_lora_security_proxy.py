"""Stage 7.0 — tests for the LoRA security proxy."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.lora_security_proxy import (
    LoRASecurityProxyConfig,
    VALID_STRATEGIES,
    run_lora_security_proxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lora_security_proxy.py"


def _cfg(**overrides) -> LoRASecurityProxyConfig:
    base = dict(
        seed=2026, d_in=16, d_out=8, rank=3, alpha=1.0,
        num_trials=4, pad_scale=1.0,
        membership_trials_per_sample=3, dtype="float64",
    )
    base.update(overrides)
    return LoRASecurityProxyConfig(**base)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_lora_security_proxy(
        LoRASecurityProxyConfig(
            seed=2026, d_in=16, d_out=8, rank=3, alpha=1.0,
            num_trials=8, membership_trials_per_sample=4, dtype="float64",
        )
    )


# ---------------------------------------------------------------------------
# 1. run_lora_security_proxy small config runs
# ---------------------------------------------------------------------------


def test_run_lora_security_proxy_runs(report: dict) -> None:
    assert report["lora_security_proxy_status"] == "implemented"
    assert set(report["strategies"]) == set(VALID_STRATEGIES)


# ---------------------------------------------------------------------------
# 2. adapter extraction section exists
# ---------------------------------------------------------------------------


def test_adapter_extraction_section_present(report: dict) -> None:
    section = report["adapter_extraction_proxy"]
    assert isinstance(section, list)
    assert len(section) == len(VALID_STRATEGIES)
    strategies_seen = {e["strategy"] for e in section}
    assert strategies_seen == set(VALID_STRATEGIES)
    # The unmasked baseline must achieve essentially-exact recovery.
    baseline = next(
        e for e in section if e["strategy"] == "unmasked_adapter_baseline"
    )
    assert baseline["delta_w_recovery_rel_l2_mean"] < 1e-6
    # Strong strategies must show large recovery error (proxy claim, not proof).
    strong = next(
        e for e in section
        if e["strategy"] == "fresh_masks_fresh_u_with_pad"
    )
    assert strong["delta_w_recovery_rel_l2_mean"] > 0.5


# ---------------------------------------------------------------------------
# 3. gradient leakage accounting exists
# ---------------------------------------------------------------------------


def test_gradient_leakage_accounting_present(report: dict) -> None:
    table = report["gradient_leakage_accounting"]
    for strategy in VALID_STRATEGIES:
        assert strategy in table
    rows = table["fresh_masks_fresh_u_with_pad"]
    names = {r["name"] for r in rows}
    assert "grad_A" in names
    assert "grad_B" in names
    assert "optimizer_state (SGD momentum / AdamW m, v)" in names
    assert "adapter_A" in names
    assert "adapter_B" in names
    # grad_A / grad_B must be trusted in Stage 7.0.
    grad_a = next(r for r in rows if r["name"] == "grad_A")
    grad_b = next(r for r in rows if r["name"] == "grad_B")
    assert grad_a["visibility"] == "trusted"
    assert grad_b["visibility"] == "trusted"
    assert grad_a["stage_7_0_status"] == "trusted_backward_prototype"


# ---------------------------------------------------------------------------
# 4. membership-style proxy exists
# ---------------------------------------------------------------------------


def test_membership_style_proxy_present(report: dict) -> None:
    membership = report["membership_style_linkability_proxy"]
    assert len(membership) == len(VALID_STRATEGIES)
    seen = {m["strategy"] for m in membership}
    assert seen == set(VALID_STRATEGIES)


# ---------------------------------------------------------------------------
# 5. fresh masks reduce linkability vs fixed baseline (or conservative)
# ---------------------------------------------------------------------------


def test_fresh_masks_reduce_linkability_or_report_conservative(report: dict) -> None:
    membership = report["membership_style_linkability_proxy"]
    fixed = next(
        m for m in membership if m["strategy"] == "fixed_masks_fixed_u"
    )
    fresh = next(
        m for m in membership if m["strategy"] == "fresh_masks_fresh_u"
    )
    fresh_pad = next(
        m for m in membership
        if m["strategy"] == "fresh_masks_fresh_u_with_pad"
    )
    interp = report["interpretation"]["linkability_summary"]
    if fresh["membership_auc_proxy"] < fixed["membership_auc_proxy"] - 0.10:
        assert "reduce" in interp.lower()
    elif fresh_pad["membership_auc_proxy"] < fixed["membership_auc_proxy"] - 0.10:
        assert "reduce" in interp.lower()
    else:
        assert "needs_more_evaluation" in interp.lower() or (
            "did not" in interp.lower() or "did not clearly" in interp.lower()
        )


# ---------------------------------------------------------------------------
# 6. rank leakage is explicitly reported
# ---------------------------------------------------------------------------


def test_rank_leakage_reported(report: dict) -> None:
    for e in report["adapter_extraction_proxy"]:
        assert e["rank_visible_in_a_tilde_shape"] is True
        assert e["configured_rank"] >= 1
        assert e["rank_signature_a"] >= 1
    note = report["interpretation"]["rank_visibility_note"]
    assert "rank" in note.lower()
    assert "padding" in note.lower() or "padded" in note.lower()


# ---------------------------------------------------------------------------
# 7. JSON / CSV / Markdown generated
# ---------------------------------------------------------------------------


def test_runner_script_generates_artifacts(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--output-dir", str(tmp_path),
            "--num-trials", "4",
            "--membership-trials-per-sample", "3",
            "--d-in", "16", "--d-out", "8", "--rank", "3",
        ],
        cwd=PROJECT_ROOT, capture_output=True, check=True, text=True,
    )
    md = (tmp_path / "lora_security_proxy.md").read_text(encoding="utf-8")
    for phrase in (
        "Experiment Scope",
        "Threat Model",
        "Adapter Extraction Proxy",
        "Gradient Leakage Accounting",
        "Membership-Style Linkability Proxy",
        "Interpretation",
        "Limitations",
        "Next Stage Plan",
    ):
        assert phrase in md, f"missing markdown section: {phrase!r}"
    with (tmp_path / "lora_security_proxy.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sections = {r["section"] for r in rows}
    assert "adapter_extraction" in sections
    assert "gradient_leakage" in sections
    assert "membership_linkability" in sections


# ---------------------------------------------------------------------------
# 8. no raw adapter / mask / private data in outputs
# ---------------------------------------------------------------------------


def test_outputs_do_not_expose_raw_adapter_mask_or_private_data(report: dict) -> None:
    text = json.dumps(report, default=str)
    assert "tensor(" not in text
    # No top-level dense matrix dumps (only fingerprints + scalar metrics)
    for entry in report["adapter_extraction_proxy"]:
        # No tensor list under any extraction key
        for k, v in entry.items():
            assert not isinstance(v, list) or all(not isinstance(x, list) for x in v)


def test_security_profile_is_proxy_evaluated(report: dict) -> None:
    assert report["security_profile"] == "proxy-evaluated, not formal"
    assert (
        report["security_profile_detail_with_lora"]
        == "private-adapter-trusted-backward, not formal"
    )


def test_limitations_include_no_formal_security(report: dict) -> None:
    lims = " ".join(report["limitations"]).lower()
    assert "not formal" in lims or "not a formal" in lims
    assert "no real tee" in lims
    assert "rank" in lims  # rank visibility flagged
