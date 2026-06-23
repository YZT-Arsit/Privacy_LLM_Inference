"""Stage 8.2 summary-parser tests (no network, no real checkpoint)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import summarize_stage8_2_real_checkpoint_results as S  # noqa: E402


def _report(model_id, dtype, folding, runtime, recovery, tok, rec_max,
            rec_mean=None, rel_l2=None):
    return {
        "stage": "8.2_modelscope_real_checkpoint",
        "config": {"model_id": model_id, "dtype": dtype,
                   "folding_dtype": folding,
                   "folded_weight_runtime_dtype": runtime,
                   "recovery_dtype": recovery, "compare_dtype": "float32",
                   "prefill_seq_len": 16, "decode_steps": 8, "max_layers": 1,
                   "mask_mode": "signed_permutation",
                   "residual_mask_strategy": "shared",
                   "cache_dir": "/root/autodl-tmp/modelscope_cache"},
        "resolved_dtypes": {"model": dtype, "folding": folding,
                            "folded_weight_runtime": runtime,
                            "recovery": recovery, "compare": "float32"},
        "environment": {"device_name": "NVIDIA GeForce RTX 5090",
                        "cuda_version": "12.8", "torch_version": "2.8.0+cu128"},
        "attention_mask_explicit": True,
        "status": "ok", "model_id": model_id, "total_layers": 24,
        "max_layers": 1, "mask": {"mask_mode": "signed_permutation",
                                  "residual_mask_strategy": "shared"},
        "masked_runtime": {"token_match_rate_vs_extracted": tok,
                           "recovered_logits_max_abs_error": rec_max,
                           "peak_cuda_memory": {"max_allocated_mb": 2048.0},
                           "latency_s_with_reference": 0.5},
        "bf16_diagnostics": {"recovered_logits_mean_abs_err": rec_mean,
                             "recovered_logits_relative_l2_err": rel_l2},
    }


def test_parameter_scale() -> None:
    assert S.parameter_scale("Qwen/Qwen2.5-0.5B-Instruct") == "0.5B"
    assert S.parameter_scale("Qwen/Qwen2.5-1.5B-Instruct") == "1.5B"
    assert S.parameter_scale("Qwen/Qwen2.5-3B-Instruct") == "3B"
    assert S.parameter_scale("nope") == "?"


def test_classify_precision_mode() -> None:
    c = S.classify_precision_mode
    assert c({"dtype": "float32", "folding_dtype": "float32",
              "folded_weight_runtime_dtype": "float32",
              "recovery_dtype": "float32"}, {}) == "float32"
    assert c({"dtype": "bfloat16", "folding_dtype": "float32",
              "folded_weight_runtime_dtype": "float32",
              "recovery_dtype": "float32"}, {}) == "bf16_mixed_safe"
    assert c({"dtype": "bfloat16", "folding_dtype": "float32",
              "folded_weight_runtime_dtype": "bfloat16",
              "recovery_dtype": "float32"}, {}) == "bf16_runtime_cast"
    assert c({"dtype": "bfloat16", "folding_dtype": "bfloat16",
              "folded_weight_runtime_dtype": "bfloat16",
              "recovery_dtype": "bfloat16"}, {}) == "bf16_all"


def test_extract_row_fields() -> None:
    r = S.extract_row(
        _report("Qwen/Qwen2.5-3B-Instruct", "float32", "float32", "float32",
                "float32", 1.0, 1.47e-04, 1e-5, 2e-4), "f.json")
    assert r["parameter_scale"] == "3B"
    assert r["precision_mode"] == "float32"
    assert r["token_match_rate_vs_extracted"] == 1.0
    assert r["recovered_logits_max_abs_err"] == 1.47e-04
    assert r["attention_mask_explicit"] is True


def test_summarize_and_render(tmp_path: Path) -> None:
    (tmp_path / "modelscope_qwen2_5_0_5b_stage8_2_float32.json").write_text(
        json.dumps(_report("Qwen/Qwen2.5-0.5B-Instruct", "float32", "float32",
                           "float32", "float32", 1.0, 3.27e-04)))
    (tmp_path / "modelscope_qwen2_5_0_5b_bf16_mixed_safe.json").write_text(
        json.dumps(_report("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", "float32",
                           "float32", "float32", 1.0, 3.3e-04)))
    (tmp_path / "modelscope_qwen2_5_0_5b_bf16_runtime_cast.json").write_text(
        json.dumps(_report("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", "float32",
                           "bfloat16", "float32", 0.777, 7.44)))
    summary = S.summarize(str(tmp_path))
    assert summary["num_reports_found"] == 3
    modes = {r["precision_mode"] for r in summary["all_rows"]}
    assert modes == {"float32", "bf16_mixed_safe", "bf16_runtime_cast"}
    # main table = canonical MAIN_FILES (present rows have status ok).
    assert len(summary["main_correctness_rows"]) == len(S.MAIN_FILES)
    present_main = [r for r in summary["main_correctness_rows"]
                    if r.get("status") == "ok"]
    assert len(present_main) == 2  # 0.5B float32 + 0.5B mixed-safe present
    # ablation includes the present runtime-cast row.
    rc = [r for r in summary["ablation_rows"]
          if r.get("precision_mode") == "bf16_runtime_cast"]
    assert rc and rc[0]["token_match_rate_vs_extracted"] == 0.777
    md = S.render_markdown(summary)
    assert "Main correctness table" in md
    assert "Claim audit" in md
    assert "Semantic security" in md  # disallowed list
    assert len(md) < 100_000
    # CSV writes one row per report, compact.
    S.write_csv(summary, str(tmp_path / "t.csv"))
    with open(tmp_path / "t.csv") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1 + 3


def test_missing_files_marked_not_fatal(tmp_path: Path) -> None:
    summary = S.summarize(str(tmp_path))  # empty dir
    assert summary["num_reports_found"] == 0
    assert all(s == "missing" for s in summary["expected_files"].values())
    # render still works
    assert "Stage 8.2" in S.render_markdown(summary)
