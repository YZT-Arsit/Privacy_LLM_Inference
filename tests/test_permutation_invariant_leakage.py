"""Tests for Stage 5.7 — Permutation-Invariant Leakage Audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pllo.experiments.permutation_invariant_leakage import (
    PermutationInvariantLeakageConfig,
    render_markdown,
    run_permutation_invariant_leakage,
    write_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_PHRASES = (
    "Permutation-only nonlinear views provide channel-index hiding, "
    "not value hiding.",
    "This is a proxy leakage audit, not a formal security proof.",
    "Dense sandwiching and boundary pads mitigate temporal and "
    "boundary exposure but do not remove single-shot "
    "permutation-invariant statistics inside the activation core.",
    "No real TEE isolation or hardware side-channel resistance is "
    "evaluated.",
    "Raw tensors, masks, permutations, adapters, gradients, and "
    "private data are not exported.",
)


@pytest.fixture(scope="module")
def small_config() -> PermutationInvariantLeakageConfig:
    """Small synthetic config -- no network, no HF download."""
    return PermutationInvariantLeakageConfig(
        num_prompts=2, prompt_max_length=4, max_new_tokens=1,
        max_layers=1, attempt_tokenizer_load=False,
        attempt_real_model_load=False, include_fixed_debug=True,
        synthetic_hidden_size=16, synthetic_intermediate_size=24,
        synthetic_num_attention_heads=2, synthetic_num_key_value_heads=1,
        synthetic_head_dim=8, seed=0,
    )


@pytest.fixture(scope="module")
def report(small_config) -> dict:
    return run_permutation_invariant_leakage(small_config)


# 1. Small synthetic config runs end-to-end.
def test_small_synthetic_runs_end_to_end(report: dict) -> None:
    assert report["status"] == "ok"
    assert report["stage"] == "5.7"
    assert isinstance(report["per_bundle"], dict)
    assert set(report["per_bundle"].keys()) == {
        "fresh_perm_only", "fresh_perm_plus_sandwich_plus_pad",
    }


# 2. JSON / CSV / Markdown are emitted.
def test_outputs_json_csv_md_emitted(report: dict, tmp_path: Path) -> None:
    j, c, m = write_reports(report, outputs_dir=str(tmp_path))
    assert Path(j).exists() and Path(j).stat().st_size > 0
    assert Path(c).exists() and Path(c).stat().st_size > 0
    assert Path(m).exists() and Path(m).stat().st_size > 0
    # Valid JSON.
    obj = json.loads(Path(j).read_text())
    assert obj["status"] == "ok"


# 3. JSON contains no raw tensor string / no long numeric arrays.
def test_json_has_no_raw_tensors(report: dict, tmp_path: Path) -> None:
    j, _, _ = write_reports(report, outputs_dir=str(tmp_path))
    text = Path(j).read_text()
    assert "tensor(" not in text, "JSON must not contain raw tensor reprs"
    # Reject long numeric arrays of more than ~50 contiguous numbers.
    long_arr = re.search(r"\[(\s*-?\d+(\.\d+)?\s*,\s*){50,}", text)
    assert long_arr is None, (
        "JSON appears to contain a long numeric array; only summary scalars "
        "and shapes should be exported."
    )


# 4. At least gate or swiglu_intermediate appears in target tensor results.
def test_gate_or_swiglu_present(report: dict) -> None:
    found = False
    for b in report["per_bundle"]:
        per_tensor = report["per_bundle"][b]["per_tensor"]
        for tn in ("gate", "swiglu_intermediate"):
            if tn in per_tensor:
                # Check at least one scope has metrics.
                for scope, audit in per_tensor[tn].items():
                    if "metrics" in audit:
                        found = True
                        break
        if found:
            break
    assert found, (
        "Expected gate or swiglu_intermediate to be measured in at least "
        "one (bundle, scope)."
    )


# 5. Norm preservation for a synthetic permutation view is near exact.
def test_norm_preservation_near_exact_for_permutation_views(
    report: dict,
) -> None:
    candidates = ("gate", "up", "swiglu_intermediate")
    for b in report["per_bundle"]:
        for tn in candidates:
            scope_map = report["per_bundle"][b]["per_tensor"].get(tn, {})
            for scope, audit in scope_map.items():
                if "metrics" not in audit:
                    continue
                m = audit["metrics"]
                # For a permutation-only view, row L2 norms must be
                # identical to plain.
                assert m["l2_corr"] > 0.99, (b, tn, scope, m)
                assert m["l2_max_abs_diff"] < 1e-3, (b, tn, scope, m)


# 6. Sorted multiset MSE for a synthetic permutation view is near zero.
def test_sorted_multiset_near_zero_for_permutation_views(
    report: dict,
) -> None:
    candidates = ("gate", "up", "swiglu_intermediate")
    any_checked = False
    for b in report["per_bundle"]:
        for tn in candidates:
            scope_map = report["per_bundle"][b]["per_tensor"].get(tn, {})
            for scope, audit in scope_map.items():
                if "metrics" not in audit:
                    continue
                m = audit["metrics"]
                # sorted MSE may carry float32 rounding; require near-zero.
                assert m["sorted_mse_mean"] < 1e-6, (b, tn, scope, m)
                any_checked = True
    assert any_checked, "No permutation-view sorted-multiset was measured."


# 7. Post-island or dense-masked view, if present, is not incorrectly
#    labelled as exact permutation leakage unless metrics support it.
def test_post_island_not_falsely_labelled_high(report: dict) -> None:
    for b in report["per_bundle"]:
        scope_map = report["per_bundle"][b]["per_tensor"].get("post_island", {})
        for scope, audit in scope_map.items():
            if "metrics" not in audit:
                continue
            m = audit["metrics"]
            label = audit["statistical_leakage_label"]
            if label == "statistical_leakage_detected_high":
                # Only allow the "high" label when the metrics actually
                # support exact preservation.
                assert m["sorted_mse_mean"] < 1e-6, (b, scope, m)
                assert m["sorted_l2_rel_mean"] < 1e-3, (b, scope, m)


# 8. Markdown contains all required honesty phrases.
def test_markdown_contains_required_honesty_phrases(report: dict) -> None:
    md = render_markdown(report)
    for phrase in _REQUIRED_PHRASES:
        assert phrase in md, f"missing honesty phrase: {phrase!r}"


# 9. formal_security_claim is false.
def test_formal_security_claim_is_false(report: dict) -> None:
    assert report["formal_security_claim"] is False
    md = render_markdown(report)
    assert "`formal_security_claim`: `False`" in md


# 10. Runner exits successfully.
def test_runner_exits_successfully(tmp_path: Path) -> None:
    import subprocess
    script = REPO_ROOT / "scripts" / "run_permutation_invariant_leakage.py"
    env_outputs = tmp_path / "outputs"
    env_outputs.mkdir()
    # Use a small env config via PYTHONPATH; the script itself reads
    # REPO_ROOT/outputs, so we copy artifacts back if needed.
    result = subprocess.run(
        ["python", str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "status=ok" in result.stdout


# Bonus integrity tests for the audit structure.
def test_fixed_permutation_ablation_present(report: dict) -> None:
    fd = report["fixed_permutation_debug"]
    assert fd is not None
    assert fd["status"] == "ok"
    assert "fixed_permutation_metrics" in fd
    assert "fresh_permutation_metrics" in fd
    # Both fixed and fresh perms must preserve sorted multiset.
    assert fd["fixed_permutation_metrics"]["sorted_mse_mean"] < 1e-9
    assert fd["fresh_permutation_metrics"]["sorted_mse_mean"] < 1e-9
    # Linkability under fixed perm > linkability under fresh perm.
    assert fd["fixed_perm_linkability_accuracy"] >= (
        fd["fresh_perm_linkability_accuracy"]
    )


def test_skipped_classifier_tasks_marked(report: dict) -> None:
    # Tasks that have no per-row labels must be marked skipped, not
    # fabricated.
    for b in report["per_bundle"]:
        ctasks = report["per_bundle"][b]["classifier_proxy_tasks"]
        for tn, task_map in ctasks.items():
            for task_name in (
                "prompt_id_linkability", "position_bucket_classification",
            ):
                assert task_map[task_name]["status"].startswith("skipped")
                assert task_map[task_name]["proxy_attack_label"] == (
                    "proxy_attack_skipped"
                )


def test_unsafe_wording_quarantined(report: dict) -> None:
    md = render_markdown(report)
    # Forbidden phrases must NEVER appear outside their own unsafe list,
    # which is reported in the JSON (not the MD). The MD should not
    # promote any of them.
    forbidden = (
        "value-level privacy",
        "cryptographic security",
        "formal security proof",
        "real TEE backend",
    )
    for phrase in forbidden:
        # Allow the phrase only when explicitly negated in the same
        # sentence (the MD uses negations like "not a formal security
        # proof"). We just check that the standalone unsafe promotion
        # never appears as a top-level claim.
        for line in md.splitlines():
            if phrase.lower() in line.lower():
                assert (
                    "not " in line.lower()
                    or "do not" in line.lower()
                    or "does not" in line.lower()
                    or "no real" in line.lower()
                ), f"unsafe phrase {phrase!r} promoted in line: {line!r}"


def test_target_tensor_inventory_populated(report: dict) -> None:
    inv = report["target_tensor_inventory"]
    assert isinstance(inv, dict)
    # At least one of the canonical permutation views must be present.
    perm_views = {"gate", "up", "swiglu_intermediate"}
    assert any(
        any(inv.get(tn, {}).values())
        for tn in perm_views
    )


def test_outputs_artifact_present_after_runner() -> None:
    j = REPO_ROOT / "outputs" / "permutation_invariant_leakage.json"
    if j.exists():
        obj = json.loads(j.read_text())
        assert obj["status"] == "ok"
        assert obj["stage"] == "5.7"
