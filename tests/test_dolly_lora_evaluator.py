"""Tests for the Dolly LoRA held-out evaluator metrics (pure, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_E = _load("eval_dolly", "scripts/evaluate_dolly_lora_outputs.py")


def test_unigram_f1() -> None:
    assert _E.unigram_f1("a b c", "a b c") == 1.0
    assert _E.unigram_f1("", "") == 1.0
    assert _E.unigram_f1("a b", "") == 0.0
    assert 0.0 < _E.unigram_f1("a b c d", "a b x y") < 1.0


def test_lcs_and_rouge() -> None:
    assert _E.lcs_len("a b c d".split(), "a c d".split()) == 3
    assert _E.rouge_l_like("a b c", "a b c") == 1.0
    assert _E.rouge_l_like("x y z", "a b c") == 0.0


def test_response_features() -> None:
    f = _E.response_features("Human: hi\nNote: careful")
    assert f["contains_human_marker"] and f["contains_note"]
    assert _E.response_features("   ")["empty"] is True
    assert _E.response_features("assistant: ok")["contains_assistant_marker"]


def test_evaluate_end_to_end() -> None:
    data = {"a": {"response": "the cat sat", "category": "qa"},
            "b": {"response": "dogs run fast", "category": "brainstorm"}}
    base = {"a": {"response": "the cat sat", "finish_reason": "eos"},
            "b": {"response": "x", "finish_reason": "length"}}
    lora = {"a": {"response": "the cat sat on a mat", "finish_reason": "eos"},
            "b": {"response": "dogs run fast", "finish_reason": "eos"}}
    rep = _E.evaluate(data, base, lora)
    assert rep["num_examples"] == 2
    assert rep["category_counts"] == {"brainstorm": 1, "qa": 1}
    # lora matches references better than base on example b
    assert rep["lora"]["unigram_f1_avg"] > rep["base"]["unigram_f1_avg"]
    # lora produced more words on average -> positive delta
    assert rep["response_length_delta_lora_minus_base"] > 0
    assert rep["base"]["finish_reason_counts"] == {"eos": 1, "length": 1}
    assert rep["lora"]["finish_reason_counts"] == {"eos": 2}


def test_exact_duplicate_and_empty() -> None:
    data = {"a": {"response": "ref"}, "b": {"response": "ref"},
            "c": {"response": "ref"}}
    lora = {"a": {"response": "same"}, "b": {"response": "same"},
            "c": {"response": "  "}}
    rep = _E.evaluate(data, {}, lora)
    assert rep["lora"]["empty_response_count"] == 1
    assert rep["lora"]["exact_duplicate_rate"] > 0.0   # "same" appears twice


def test_markdown_renders() -> None:
    data = {"a": {"response": "r", "category": "qa"}}
    rep = _E.evaluate(data, {"a": {"response": "r", "finish_reason": "eos"}},
                      {"a": {"response": "r", "finish_reason": "eos"}})
    md = _E._markdown(rep)
    assert "Dolly LoRA held-out statistics" in md and "| metric | base | lora |" in md
