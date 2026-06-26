"""Tests for the Dolly LoRA final-summary aggregation (pure, no model)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_S = _load("sum_dolly", "scripts/summarize_dolly_lora_experiment.py")


def _stages():
    train = {"num_train_examples": 1000, "num_valid_examples": 100,
             "lora_rank": 16, "lora_alpha": 32.0, "lora_dropout": 0.05,
             "target_modules": "q_proj,k_proj", "trainable_param_ratio": 0.004,
             "adapter_param_count": 1234, "train_loss_final": 1.23,
             "valid_loss_final": 1.45, "train_runtime_s": 99.0,
             "paper_ready": True}
    base_gen = {"online_generation_latency_s": 10.0, "prompt_format": "chat"}
    lora_gen = {"online_generation_latency_s": 11.0, "prompt_format": "chat"}
    ev = {"num_examples": 200,
          "base": {"avg_words": 50.0, "unigram_f1_avg": 0.2,
                   "empty_response_count": 0, "contains_human_marker_count": 1},
          "lora": {"avg_words": 70.0, "unigram_f1_avg": 0.4,
                   "rouge_l_like_lcs_avg": 0.3, "empty_response_count": 0,
                   "contains_human_marker_count": 0},
          "response_length_delta_lora_minus_base": 20.0}
    folded = {"lora_package_valid": True, "contains_raw_lora": False,
              "contains_optimizer_state": False, "contains_training_data": False,
              "contains_mask_secrets": False, "worker_has_raw_lora": False,
              "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
              "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
              "audit_passed": True, "local_allclose": True,
              "local_max_abs_error": 1e-6, "tokens_exact_match": True,
              "token_match_rate": 1.0, "latency_s": 0.44, "trusted_bytes": 100,
              "gpu_bytes": 200, "boundary_calls": 16, "peak_gpu_memory_mb": 26000}
    return train, base_gen, lora_gen, ev, folded


def test_build_summary_fields() -> None:
    s = _S.build_summary(*_stages())
    assert s["dataset"] == "databricks-dolly-15k"
    assert s["model"] == "Qwen2.5-7B-Instruct"
    assert s["train_size"] == 1000 and s["valid_size"] == 100
    assert s["test_size"] == 200
    assert s["lora_rank"] == 16 and s["target_modules"] == "q_proj,k_proj"
    assert s["train_loss_final"] == 1.23 and s["valid_loss_final"] == 1.45
    assert s["response_length_delta_lora_minus_base"] == 20.0
    # security clean -> paper_ready True
    assert s["contains_raw_lora"] is False and s["worker_has_raw_lora"] is False
    assert s["audit_passed"] is True
    assert s["paper_ready"] is True
    # required security fields present
    for k in ("contains_optimizer_state", "contains_training_data",
              "contains_mask_secrets", "worker_has_mask_secrets",
              "gpu_visible_plaintext_fields", "leaked_secret_fields",
              "local_allclose", "local_max_abs_error", "tokens_exact_match",
              "token_match_rate", "latency_s", "trusted_bytes", "gpu_bytes",
              "boundary_calls", "peak_gpu_memory_mb"):
        assert k in s


def test_paper_ready_false_when_leak() -> None:
    train, base_gen, lora_gen, ev, folded = _stages()
    folded["contains_raw_lora"] = True              # a leak
    s = _S.build_summary(train, base_gen, lora_gen, ev, folded)
    assert s["paper_ready"] is False


def test_paper_ready_false_when_audit_failed() -> None:
    train, base_gen, lora_gen, ev, folded = _stages()
    folded["audit_passed"] = False
    s = _S.build_summary(train, base_gen, lora_gen, ev, folded)
    assert s["paper_ready"] is False


def test_missing_stages_are_tolerated() -> None:
    s = _S.build_summary({}, {}, {}, {}, {})
    assert s["dataset"] == "databricks-dolly-15k"
    assert s["train_size"] is None and s["paper_ready"] is False


def test_writes_json_md_csv(tmp_path) -> None:
    train, base_gen, lora_gen, ev, folded = _stages()
    for name, obj in (("t.json", train), ("bg.json", base_gen),
                      ("lg.json", lora_gen), ("e.json", ev), ("f.json", folded)):
        (tmp_path / name).write_text(json.dumps(obj))
    old = sys.argv
    try:
        sys.argv = ["x", "--training-json", str(tmp_path / "t.json"),
                    "--base-gen-json", str(tmp_path / "bg.json"),
                    "--lora-gen-json", str(tmp_path / "lg.json"),
                    "--eval-json", str(tmp_path / "e.json"),
                    "--folded-json", str(tmp_path / "f.json"),
                    "--out-dir", str(tmp_path / "out")]
        rc = _S.main()
    finally:
        sys.argv = old
    assert rc == 0
    out = tmp_path / "out"
    j = json.loads((out / "dolly_lora_final_summary.json").read_text())
    assert j["paper_ready"] is True
    assert (out / "dolly_lora_final_summary.md").read_text().startswith("# Dolly")
    with open(out / "dolly_lora_final_summary.csv") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["field", "value"]
    assert any(r[0] == "lora_rank" for r in rows)
