"""Tests for the Stage 6.3 security proxy experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pllo.experiments import (
    GPU_VISIBLE_TENSORS,
    MASK_AUDIT_SPECS,
    SecurityProxyConfig,
    TRUSTED_ONLY_TENSORS,
    run_security_proxy_experiments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_security_proxy_experiments.py"


@pytest.fixture(scope="module")
def proxy_payload():
    cfg = SecurityProxyConfig(
        num_trials=12,
        batch_size=2,
        seq_len=4,
        hidden_size=16,
        seed=4242,
    )
    return run_security_proxy_experiments(cfg)


# ---------------------------------------------------------------------------
# Proxy 1: pad linkability
# ---------------------------------------------------------------------------


def test_pad_linkability_has_four_strategy_groups(proxy_payload) -> None:
    per = proxy_payload["pad_linkability_proxy"]["per_strategy"]
    assert set(per.keys()) == {
        "fixed_mask_no_pad",
        "fresh_mask_no_pad",
        "fixed_mask_fresh_pad",
        "fresh_mask_fresh_pad",
    }
    for name, m in per.items():
        for metric in (
            "mean_pairwise_cosine",
            "max_pairwise_cosine",
            "min_pairwise_cosine",
            "mean_pairwise_l2",
            "max_pairwise_l2",
            "min_pairwise_l2",
            "interpretation",
        ):
            assert metric in m, f"missing {metric} for strategy {name}"


def test_fixed_mask_no_pad_is_more_linkable_than_fresh_mask_fresh_pad(
    proxy_payload,
) -> None:
    per = proxy_payload["pad_linkability_proxy"]["per_strategy"]
    assert (
        per["fixed_mask_no_pad"]["mean_pairwise_cosine"]
        > per["fresh_mask_fresh_pad"]["mean_pairwise_cosine"]
    )
    # fixed_mask_no_pad: identical visible tensor across trials ⇒ cosine ≈ 1.
    assert per["fixed_mask_no_pad"]["mean_pairwise_cosine"] > 0.99
    # fresh_mask_fresh_pad: visible tensor decorrelates ⇒ |cosine| small.
    assert abs(per["fresh_mask_fresh_pad"]["mean_pairwise_cosine"]) < 0.2


# ---------------------------------------------------------------------------
# Proxy 2: mask freshness audit
# ---------------------------------------------------------------------------


def test_mask_freshness_audit_does_not_leak_mask_tensors(proxy_payload) -> None:
    audit = proxy_payload["mask_freshness_audit"]
    # No "mask" / "tensor" / "values" field in the per-mask records:
    for entry in audit["per_mask"]:
        for forbidden in ("mask", "tensor", "values", "raw"):
            assert forbidden not in entry, (
                f"per_mask record exposes {forbidden!r}: {entry}"
            )


def test_mask_freshness_audit_records_expected_policies(proxy_payload) -> None:
    by_name = {m["mask_name"]: m for m in proxy_payload["mask_freshness_audit"]["per_mask"]}
    expected_names = {spec["mask_name"] for spec in MASK_AUDIT_SPECS}
    assert set(by_name.keys()) == expected_names
    # Input masks must be fresh across every trial under the declared policy.
    for name in ("input_mask", "output_mask", "pad"):
        record = by_name[name]
        assert record["expected_policy"] == "fresh_across_trials"
        assert record["num_unique_fingerprints"] == record["num_generated"]
        assert record["unexpected_reuse_count"] == 0
    # KV / encoder masks reuse exactly once per session, fresh across sessions.
    for name in ("kv_cache_mask", "encoder_memory_mask"):
        record = by_name[name]
        assert record["num_unique_fingerprints"] == record["num_generated"] // 2
        assert record["unexpected_reuse_count"] == 0


# ---------------------------------------------------------------------------
# Proxy 3: boundary leakage accounting
# ---------------------------------------------------------------------------


def test_boundary_accounting_marks_compensation_terms_gpu_visible(
    proxy_payload,
) -> None:
    accounting = proxy_payload["boundary_leakage_accounting"]
    gpu_visible_names = {it["name"] for it in accounting["gpu_visible"]}
    trusted_only_names = {it["name"] for it in accounting["trusted_only"]}
    assert "compensation_terms" in gpu_visible_names
    assert "obfuscated_kv_cache" in gpu_visible_names
    assert "obfuscated_encoder_memory_cache" in gpu_visible_names
    assert "plaintext_input" in trusted_only_names
    assert "masks" in trusted_only_names
    assert "pads" in trusted_only_names
    # No tensor is both GPU-visible and trusted-only.
    assert gpu_visible_names.isdisjoint(trusted_only_names)


def test_boundary_accounting_summary_notes_mention_compensation_and_no_security_proof(
    proxy_payload,
) -> None:
    notes = proxy_payload["boundary_leakage_accounting"]["summary_notes"]
    joined = " ".join(notes).lower()
    assert "compensation_terms" in joined
    assert "security proxy" in joined or "semantic security" in joined
    assert "real tee" in joined or "tee isolation" in joined


def test_boundary_accounting_static_lists_are_exported() -> None:
    """The static GPU-visible / trusted-only lists are exposed at the package root."""
    assert any(it["name"] == "compensation_terms" for it in GPU_VISIBLE_TENSORS)
    assert any(it["name"] == "plaintext_input" for it in TRUSTED_ONLY_TENSORS)


# ---------------------------------------------------------------------------
# Proxy 4: cache leakage proxy
# ---------------------------------------------------------------------------


def test_cache_leakage_proxy_has_required_metric_fields(proxy_payload) -> None:
    cache = proxy_payload["cache_leakage_proxy"]
    for kind in ("kv_cache", "encoder_memory_cache"):
        for matching in ("plain_to_plain_baseline", "obfuscated_to_plain"):
            m = cache[kind][matching]
            for metric in (
                "top1_match_rate",
                "mean_correct_rank",
                "mean_cosine_correct_pair",
                "mean_cosine_best_wrong_pair",
            ):
                assert metric in m


def test_cache_leakage_proxy_obfuscated_top1_below_baseline(proxy_payload) -> None:
    cache = proxy_payload["cache_leakage_proxy"]
    for kind in ("kv_cache", "encoder_memory_cache"):
        baseline = cache[kind]["plain_to_plain_baseline"]["top1_match_rate"]
        obf = cache[kind]["obfuscated_to_plain"]["top1_match_rate"]
        assert baseline == pytest.approx(1.0)
        # Naive matching against masked cache should be near random.
        assert obf < 0.25, (
            f"{kind} obfuscated_to_plain top1={obf} should be near random;"
            f" anything above 25% breaks the proxy claim."
        )


# ---------------------------------------------------------------------------
# Global limitations + script smoke
# ---------------------------------------------------------------------------


def test_global_limitations_are_recorded(proxy_payload) -> None:
    lims = proxy_payload["global_limitations"]
    text = " ".join(lims).lower()
    assert "security proxies" in text
    assert "not formal security proofs" in text
    assert "adaptive attacks" in text
    assert "learned inversion" in text
    assert "real tee" in text


def test_script_emits_all_three_artifacts(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--num-trials",
            "8",
            "--hidden-size",
            "16",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    for filename in (
        "security_proxy_experiments.json",
        "security_proxy_experiments.csv",
        "security_proxy_experiments.md",
    ):
        assert (tmp_path / filename).exists(), filename

    md = (tmp_path / "security_proxy_experiments.md").read_text(encoding="utf-8")
    for section in (
        "Pad linkability proxy",
        "Mask freshness audit",
        "Boundary leakage accounting",
        "Cache leakage proxy",
        "Interpretation",
        "Limitations",
        "Next stage plan",
    ):
        assert section in md, f"missing section: {section}"

    payload = json.loads(
        (tmp_path / "security_proxy_experiments.json").read_text(encoding="utf-8")
    )
    assert "pad_linkability_proxy" in payload
    assert "mask_freshness_audit" in payload
    assert "boundary_leakage_accounting" in payload
    assert "cache_leakage_proxy" in payload
    assert "num_trials=8" in result.stdout
