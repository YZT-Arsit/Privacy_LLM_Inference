"""Task B: E10 LoRA utility preservation + extended validate_lora_effect.
stdlib only.

Run: python -m pytest tests/test_lora_utility_e10.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.lora_utility import build_lora_utility_report  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


_SEC = {"worker_has_raw_lora": False, "worker_has_mask_secrets": False,
        "tee_used_on_gpu": False, "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": [], "audit_passed": True}


def _e9(metric, **extra):
    d = dict(_SEC, stage="e9_task_utility_benchmark", dataset="sst2",
             task_type="classification", metric_name="accuracy",
             metric_value=metric, dry_run=False, paper_ready=True)
    d.update(extra)
    return d


def test_e10_preserves_gain_and_security() -> None:
    verify = {"contains_raw_lora": False, "contains_optimizer_state": False,
              "contains_training_data": False, "contains_mask_secrets": False}
    rep = build_lora_utility_report({
        "base": _e9(0.70), "plaintext_lora": _e9(0.90),
        "folded_lora": _e9(0.895), "lora_verify": verify,
        "effect": {"tokens_differ": True, "no_lora_token_ids": [1, 2],
                   "lora_token_ids": [1, 3]},
        "preserve_threshold": 0.9})
    assert rep["lora_gain_plaintext"] == 0.2
    assert abs(rep["lora_gain_folded"] - 0.195) < 1e-9
    assert rep["lora_gain_preserved_ratio"] >= 0.9
    assert rep["folded_lora_preserves_gain"] is True
    assert rep["lora_differs_from_no_lora"] is True
    assert rep["security_ok"] is True
    assert rep["utility_preserved"] is True
    assert rep["paper_ready"] is True


def test_e10_not_preserved_when_folded_gain_collapses() -> None:
    rep = build_lora_utility_report({
        "base": _e9(0.70), "plaintext_lora": _e9(0.90),
        "folded_lora": _e9(0.71), "preserve_threshold": 0.9})
    assert rep["folded_lora_preserves_gain"] is False
    assert rep["utility_preserved"] is False


def test_e10_dry_run_not_paper_ready() -> None:
    rep = build_lora_utility_report({
        "base": _e9(0.7, dry_run=True, paper_ready=False),
        "plaintext_lora": _e9(0.8, dry_run=True, paper_ready=False),
        "folded_lora": _e9(0.8, dry_run=True, paper_ready=False)})
    assert rep["dry_run"] is True
    assert rep["paper_ready"] is False


def test_e10_script(tmp_path) -> None:
    mod = _load("e10", "scripts/run_e10_lora_utility_benchmark.py")
    paths = {}
    for k, m in (("base", 0.70), ("plain", 0.90), ("folded", 0.895)):
        p = tmp_path / (k + ".json")
        p.write_text(json.dumps(_e9(m)))
        paths[k] = p
    vfy = tmp_path / "verify.json"
    vfy.write_text(json.dumps({"contains_raw_lora": False,
                               "contains_optimizer_state": False,
                               "contains_training_data": False,
                               "contains_mask_secrets": False}))
    oj = tmp_path / "e10.json"
    rc = _main(mod, ["x", "--base-json", str(paths["base"]),
                     "--plaintext-lora-json", str(paths["plain"]),
                     "--folded-lora-json", str(paths["folded"]),
                     "--lora-verify-json", str(vfy),
                     "--output-json", str(oj),
                     "--output-md", str(tmp_path / "e10.md")])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["folded_lora_preserves_gain"] is True
    assert r["utility_preserved"] is True


def test_validate_lora_effect_metric_improved(tmp_path) -> None:
    mod = _load("vle", "scripts/validate_lora_effect.py")
    nl = tmp_path / "nl.json"
    ll = tmp_path / "ll.json"
    nl.write_text(json.dumps({"package_token_ids": [1, 2], "metric_value": 0.70}))
    ll.write_text(json.dumps({"package_token_ids": [1, 3], "metric_value": 0.88}))
    out = tmp_path / "v.json"
    rc = _main(mod, ["x", "--no-lora-json", str(nl), "--lora-json", str(ll),
                     "--require-effect", "true", "--output-json", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["metric_improved"] is True
    assert r["no_lora_metric"] == 0.70
    assert r["lora_metric"] == 0.88
    assert r["lora_has_effect"] is True
