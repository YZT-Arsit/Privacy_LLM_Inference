"""Tests for the E1/E2 paper-table aggregator (validation + table assembly).

Run: python -m pytest tests/test_summarize_qwen_e1_e2.py -q
"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "summ", REPO_ROOT / "scripts" / "summarize_qwen_e1_e2_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e1(context_mode="natural_prompt", padded_seq_len=None, **over):
    rec = {
        "context_mode": context_mode, "seq_len_requested": 128,
        "effective_prompt_len": 39, "padded_seq_len": padded_seq_len,
        "max_new_tokens": 64, "attention_mask_used": True,
        "teacher_forced_top1_match_rate_hf_plain": 1.0,
        "teacher_forced_top1_match_rate_hf_masked": 1.0,
        "teacher_forced_top1_match_rate_plain_masked": 1.0,
        "plain_vs_masked_token_match_rate": 1.0, "topk_overlap": 0.9875,
        "logits_max_abs_error": 0.3367, "logits_mean_abs_error": 0.01,
        "logits_relative_l2_error": 0.02, "latency_s": 182.0,
        "peak_gpu_memory_mb": 31550.0, "trusted_bytes": 123, "gpu_bytes": 456,
        "tee_used_on_gpu": False, "leaked_secret_fields": [],
        "gpu_visible_plaintext_fields": [],
    }
    rec.update(over)
    return rec


def _e2(context_mode="natural_prompt", padded_seq_len=None):
    row = {
        "context_mode": context_mode, "max_new_tokens": 64,
        "seq_len_requested": 128, "effective_prompt_len": 39,
        "padded_seq_len": padded_seq_len,
        "teacher_forced_top1_match_rate_hf_masked": 1.0,
        "teacher_forced_top1_match_rate_plain_masked": 1.0,
        "plain_vs_masked_token_match_rate": 1.0, "topk_overlap": 0.9875,
        "logits_max_abs_error": 0.3367, "latency_s": 182.0,
        "peak_gpu_memory_mb": 31550.0, "trusted_bytes": 1, "gpu_bytes": 2,
        "tee_used_on_gpu": False, "attention_mask_used": True,
        "leaked_secret_fields": [], "gpu_visible_plaintext_fields": [],
    }
    return {"stage": "E2_token_scaling", "context_mode": context_mode,
            "rows": [row, {**row, "max_new_tokens": 1}]}


def _ns(tmp, *, e1n=None, e1p=None, e2n=None, e2p=None):
    def w(name, obj):
        if obj is None:
            return None
        p = tmp / name
        p.write_text(json.dumps(obj), encoding="utf-8")
        return str(p)
    return Namespace(e1_natural=w("e1n.json", e1n), e1_padded=w("e1p.json", e1p),
                     e2_natural=w("e2n.json", e2n), e2_padded=w("e2p.json", e2p))


def test_valid_inputs_pass_and_build_tables(tmp_path) -> None:
    s = _load_script()
    ns = _ns(tmp_path, e1n=_e1("natural_prompt", None),
             e1p=_e1("fixed_padded", 128), e2n=_e2("natural_prompt", None))
    e1_rows, e2_rows, problems = s.collect(ns)
    assert problems == []
    assert len(e1_rows) == 2 and len(e2_rows) == 2
    assert e1_rows[0]["context_mode"] == "natural_prompt"
    assert all(f in e1_rows[0] for f in s.E1_FIELDS)
    assert all(f in e2_rows[0] for f in s.E2_FIELDS)


def test_natural_padded_none_ok_but_fixed_padded_none_fails(tmp_path) -> None:
    s = _load_script()
    # natural with padded_seq_len None -> OK
    _, _, ok = s.collect(_ns(tmp_path, e1n=_e1("natural_prompt", None)))
    assert ok == []
    # fixed_padded with padded_seq_len None -> violation
    _, _, bad = s.collect(_ns(tmp_path, e1p=_e1("fixed_padded", None)))
    assert any("padded_seq_len" in p for p in bad)


def test_none_metric_fails(tmp_path) -> None:
    s = _load_script()
    _, _, bad = s.collect(_ns(
        tmp_path, e1n=_e1(plain_vs_masked_token_match_rate=None)))
    assert any("plain_vs_masked_token_match_rate" in p for p in bad)


def test_tee_true_and_mask_false_and_leak_fail(tmp_path) -> None:
    s = _load_script()
    _, _, t = s.collect(_ns(tmp_path, e1n=_e1(tee_used_on_gpu=True)))
    assert any("tee_used_on_gpu" in p for p in t)
    _, _, m = s.collect(_ns(tmp_path, e1n=_e1(attention_mask_used=False)))
    assert any("attention_mask_used" in p for p in m)
    _, _, lk = s.collect(_ns(tmp_path, e1n=_e1(leaked_secret_fields=["seed"])))
    assert any("leaked_secret_fields" in p for p in lk)
    _, _, pv = s.collect(_ns(
        tmp_path, e1n=_e1(gpu_visible_plaintext_fields=["input_ids"])))
    assert any("gpu_visible_plaintext_fields" in p for p in pv)


def test_main_writes_outputs_and_exits_nonzero_on_failure(tmp_path) -> None:
    s = _load_script()
    md = tmp_path / "t.md"
    csv_ = tmp_path / "t.csv"
    js = tmp_path / "t.json"
    ns = _ns(tmp_path, e1n=_e1("natural_prompt", None), e2n=_e2())
    import sys
    argv = ["prog", "--e1-natural", ns.e1_natural, "--e2-natural", ns.e2_natural,
            "--output-md", str(md), "--output-csv", str(csv_),
            "--output-json", str(js)]
    old = sys.argv
    try:
        sys.argv = argv
        assert s.main() == 0
    finally:
        sys.argv = old
    assert md.exists() and csv_.exists() and js.exists()
    out = json.loads(js.read_text())
    assert out["validation_passed"] is True
    assert len(out["e1_table"]) == 1 and len(out["e2_table"]) == 2
    assert "Table 1" in md.read_text() and "Table 2" in md.read_text()

    # failing input -> SystemExit(1)
    bad = _ns(tmp_path, e1n=_e1(tee_used_on_gpu=True))
    try:
        sys.argv = ["prog", "--e1-natural", bad.e1_natural,
                    "--output-md", str(tmp_path / "b.md"),
                    "--output-csv", str(tmp_path / "b.csv"),
                    "--output-json", str(tmp_path / "b.json")]
        with pytest.raises(SystemExit) as ei:
            s.main()
        assert ei.value.code == 1
    finally:
        sys.argv = old
