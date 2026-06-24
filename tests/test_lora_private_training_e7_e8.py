"""E7 (private LoRA update prototype) + E8 (LoRA final report) -- numpy only,
no H800/TDX/CUDA/torch/checkpoint.

Run: python -m pytest tests/test_lora_private_training_e7_e8.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e8_lora_report import (  # noqa: E402
    build_e8_report,
    render_e8_md,
)
from pllo.training.lora_private_trainer import run_private_lora_training  # noqa: E402


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


# ---------------------------------------------------------------------------
# E7
# ---------------------------------------------------------------------------


def test_private_lora_training_decreases_loss_and_audit_clean() -> None:
    r = run_private_lora_training(["q_proj", "v_proj"], rank=8, alpha=16.0,
                                  steps=5, seed=0)
    assert r["stage"] == "private_lora_training_probe"
    assert r["loss_decreased"] is True
    assert r["loss_after"] < r["loss_before"]
    assert r["adapter_delta_norm"] > 0.0
    # security: nothing sensitive reached the GPU
    assert r["raw_lora_visible_to_gpu"] is False
    assert r["optimizer_state_visible_to_gpu"] is False
    assert r["training_data_visible_to_gpu"] is False
    assert r["labels_visible_to_gpu"] is False
    assert r["gradients_visible_to_gpu"] is False
    assert r["worker_has_mask_secrets"] is False
    assert r["tee_used_on_gpu"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["audit_passed"] is True
    assert len(r["limitations"]) >= 4


def test_private_lora_training_single_module() -> None:
    r = run_private_lora_training(["q_proj"], rank=4, alpha=8.0, steps=3, seed=1)
    assert r["target_modules"] == ["q_proj"]
    assert r["loss_decreased"] is True
    assert r["audit_passed"] is True


def test_e7_script(tmp_path) -> None:
    mod = _load("e7", "scripts/run_lora_private_training_tiny_probe.py")
    js = tmp_path / "e7.json"
    rc = _main(mod, ["x", "--target-modules", "q_proj,v_proj", "--rank", "8",
                     "--alpha", "16", "--steps", "5",
                     "--output-json", str(js), "--output-md",
                     str(tmp_path / "e7.md")])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["loss_decreased"] is True
    assert r["audit_passed"] is True


# ---------------------------------------------------------------------------
# E8
# ---------------------------------------------------------------------------


def _e8_inputs():
    sec = {"worker_has_raw_lora": False, "worker_has_mask_secrets": False,
           "tee_used_on_gpu": False, "gpu_visible_plaintext_fields": [],
           "leaked_secret_fields": []}
    return {
        "local": dict(sec, stage="qwen7b_lora_folded_local_probe",
                      allclose=True, max_abs_error=2.98e-07,
                      relative_l2_error=4.5e-07, top1_match=True,
                      tokens_exact_match=True, token_match_rate=1.0,
                      latency_s=0.5, peak_gpu_memory_mb=2200.0),
        "remote": dict(sec, stage="qwen7b_lora_folded_remote_decode_probe",
                       lora_enabled=True, folded_lora_loaded=True,
                       folded_lora_valid=True, tokens_exact_match=True,
                       token_match_rate=1.0, latency_s=60.0,
                       peak_gpu_memory_mb=2160.0),
        "attested": dict(sec, stage="qwen7b_lora_folded_remote_decode_probe",
                         tokens_exact_match=True, audit_passed=True,
                         boundary_attested=True, runtime_hash_bound=True),
        "lora_build": {"size_gb": 0.05, "build_time_s": 12.3, "rank": 8,
                       "target_modules": ["q_proj", "k_proj", "v_proj",
                                          "o_proj"]},
        "base_decode": {"latency_s": 59.0, "peak_gpu_memory_mb": 2140.0},
        "training": run_private_lora_training(["q_proj"], rank=8, alpha=16.0,
                                              steps=5, seed=0),
    }


def test_e8_build_report_sections() -> None:
    report = build_e8_report(_e8_inputs())
    assert report["experiment"] == "E8"
    assert len(report["correctness"]) == 3
    assert all(r["pass"] for r in report["correctness"] if r["provided"])
    assert len(report["security_matrix"]["matrix"]) == 6
    assert report["security_matrix"]["audit_cross_check_ok"] is True
    c = report["cost"]
    assert c["folded_lora_package_size_gb"] == 0.05
    # latency overhead = lora(remote 60) - base(59) = 1.0
    assert c["decode_latency_overhead_s"] == pytest.approx(1.0)
    assert c["memory_overhead_mb"] == pytest.approx(20.0)
    assert report["training"]["loss_decreased"] is True
    md = render_e8_md(report)
    for h in ("1. LoRA inference correctness", "2. LoRA security matrix",
              "3. LoRA cost", "4. LoRA training prototype"):
        assert h in md


def test_e8_missing_inputs_not_assumed() -> None:
    report = build_e8_report({})
    assert all(v is False for v in report["inputs_provided"].values())
    for r in report["correctness"]:
        assert r["provided"] is False
        assert r["pass"] is None
    assert report["security_matrix"]["audit_cross_check_ok"] is None
    assert report["training"]["provided"] is False


def test_e8_leak_flips_cross_check() -> None:
    inp = _e8_inputs()
    inp = dict(inp)
    bad = dict(inp["remote"])
    bad["worker_has_raw_lora"] = True
    inp["remote"] = bad
    report = build_e8_report(inp)
    assert report["security_matrix"]["audit_cross_check_ok"] is False


def test_e8_script(tmp_path) -> None:
    inp = _e8_inputs()
    paths = {}
    for key in ("local", "remote", "attested", "lora_build", "base_decode",
                "training"):
        p = tmp_path / (key + ".json")
        p.write_text(json.dumps(inp[key], default=str))
        paths[key] = p
    mod = _load("e8", "scripts/run_e8_lora_final_report.py")
    oj = tmp_path / "e8.json"
    rc = _main(mod, ["x", "--local-json", str(paths["local"]),
                     "--remote-json", str(paths["remote"]),
                     "--attested-json", str(paths["attested"]),
                     "--lora-build-json", str(paths["lora_build"]),
                     "--base-decode-json", str(paths["base_decode"]),
                     "--training-json", str(paths["training"]),
                     "--output-json", str(oj),
                     "--output-md", str(tmp_path / "e8.md")])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["experiment"] == "E8"
    assert r["security_matrix"]["audit_cross_check_ok"] is True
