"""Tests for the IFEval strict-gap analyzer (text + evaluator results only)."""

from __future__ import annotations

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


_MOD = _load("strict_gap", "scripts/analyze_ifeval_strict_gap.py")


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _rec(key, prompt, response, iids, follow):
    return {"key": key, "prompt": prompt, "response": response,
            "instruction_id_list": iids, "follow_instruction_list": follow,
            "follow_all_instructions": all(follow)}


def test_features_detect_format_issues() -> None:
    f = _MOD._features("  Hello, World.\n\n- a\n- b ", "Hello")
    assert f["has_leading_whitespace"] and f["has_trailing_whitespace"]
    assert f["has_uppercase"] and not f["is_all_lowercase"]
    assert f["bullet_count"] == 2 and f["paragraph_count"] == 2
    assert f["comma_count"] == 1
    g = _MOD._features("all lowercase no caps", "x")
    assert g["is_all_lowercase"] and not g["has_uppercase"]


def test_quote_wrap_and_echo() -> None:
    assert _MOD._features('"wrapped"', "x")["quote_wrapped"] is True
    assert _MOD._features("answer", "x")["quote_wrapped"] is False
    echo = _MOD._features("make a tweet without capitals and stuff here",
                          "make a tweet without capitals")
    assert echo["echoes_prompt"] is True


def test_strict_gap_end_to_end(tmp_path) -> None:
    # ex "cap": plaintext all-lowercase (passes), folded has uppercase (fails)
    ps = tmp_path / "ps.jsonl"
    fs = tmp_path / "fs.jsonl"
    _write(ps, [
        _rec("cap", "no capitals please", "all good lowercase",
             ["change_case:english_lowercase"], [True]),
        _rec("ok", "say hi", "hi", ["keywords:existence"], [True]),
    ])
    _write(fs, [
        _rec("cap", "no capitals please", "All Good With Caps",
             ["change_case:english_lowercase"], [False]),
        _rec("ok", "say hi", "hi", ["keywords:existence"], [True]),
    ])
    out_md = tmp_path / "g.md"
    out_json = tmp_path / "g.json"
    old = sys.argv
    try:
        sys.argv = ["x", "--plaintext-strict", str(ps), "--folded-strict", str(fs),
                    "--output-md", str(out_md), "--output-json", str(out_json)]
        rc = _MOD.main()
    finally:
        sys.argv = old
    assert rc == 0
    r = json.loads(out_json.read_text())
    s = r["strict"]
    assert s["plaintext_pass_folded_fail"] == ["cap"]
    assert s["folded_pass_plaintext_fail"] == []
    assert s["failed_instruction_category_counts"] == {"change_case": 1}
    assert s["plaintext_pass_rate"] == 1.0 and s["folded_pass_rate"] == 0.5
    # the focus example's folded response shows the casing violation
    cap = next(e for e in s["per_example"] if e["id"] == "cap")
    assert cap["folded_features"]["has_uppercase"] is True
    assert cap["plaintext_features"]["is_all_lowercase"] is True
    assert r["no_secret_fields"] is True
    assert "IFEval strict-gap" in out_md.read_text()


def test_loose_section_optional(tmp_path) -> None:
    ps = tmp_path / "ps.jsonl"
    fs = tmp_path / "fs.jsonl"
    _write(ps, [_rec("a", "p", "r", ["x:y"], [True])])
    _write(fs, [_rec("a", "p", "r", ["x:y"], [True])])
    out_json = tmp_path / "g.json"
    old = sys.argv
    try:
        sys.argv = ["x", "--plaintext-strict", str(ps), "--folded-strict", str(fs),
                    "--output-md", str(tmp_path / "g.md"),
                    "--output-json", str(out_json)]
        _MOD.main()
    finally:
        sys.argv = old
    r = json.loads(out_json.read_text())
    assert r["loose"] is None
    assert r["strict"]["plaintext_pass_folded_fail"] == []
