"""Tests for offline IFEval scoring of existing responses (no official package,
no model). Pure helpers + main() driven with injected fake strict/loose scorers.

Run:
    PYTHONPATH=$PWD/src pytest tests/test_evaluate_existing_ifeval_responses.py -q
"""

from __future__ import annotations

import collections
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


_E = _load("eval_ifeval", "scripts/evaluate_existing_ifeval_responses.py")

# mimic the official evaluation_lib.OutputExample
_Out = collections.namedtuple(
    "OutputExample", ["instruction_id_list", "prompt", "response",
                      "follow_all_instructions", "follow_instruction_list"])


def _fake_strict(inp, p2r):
    """Follow an instruction iff its id starts with 'good'."""
    resp = p2r[inp.prompt]
    fl = [str(i).startswith("good") for i in inp.instruction_id_list]
    return _Out(list(inp.instruction_id_list), inp.prompt, resp, all(fl), fl)


def _fake_loose(inp, p2r):
    """Looser: also accept ids starting with 'ok'."""
    resp = p2r[inp.prompt]
    fl = [str(i).startswith(("good", "ok")) for i in inp.instruction_id_list]
    return _Out(list(inp.instruction_id_list), inp.prompt, resp, all(fl), fl)


# ---- loading / joining ----------------------------------------------------


def test_load_prompts_requires_instruction_metadata(tmp_path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text("\n".join([
        json.dumps({"key": 1, "prompt": "P1",
                    "instruction_id_list": ["good:a", "bad:b"],
                    "kwargs": [{}, {"n": 2}]}),
        json.dumps({"key": 2, "prompt": "P2"}),            # no metadata -> skip
        json.dumps({"key": 3, "prompt": "P3",
                    "instruction_id_list": ["good:a"]}),   # no kwargs -> filled
        "",                                                 # blank
        "{bad json",                                        # skip
    ]) + "\n")
    inputs, skipped = _E.load_prompts(p)
    assert [i.key for i in inputs] == [1, 3]
    assert inputs[0].instruction_id_list == ["good:a", "bad:b"]
    assert inputs[0].kwargs == [{}, {"n": 2}]
    assert inputs[1].kwargs == [{}]                        # filled to match
    assert skipped == 2                                    # P2 (no metadata) + bad json


def test_load_responses_and_join(tmp_path) -> None:
    r = tmp_path / "resp.jsonl"
    r.write_text("\n".join([
        json.dumps({"id": 1, "prompt": "P1", "response": "R1"}),
        json.dumps({"key": 2, "prompt": "PX", "response": "R2"}),  # join by key
    ]) + "\n")
    by_prompt, by_key = _E.load_responses(r)
    assert by_prompt["P1"] == "R1"
    assert by_key["2"] == "R2"
    Inp = _E.InputExample
    inputs = [Inp(1, ["good:a"], "P1", [{}]),
              Inp(2, ["good:a"], "P2", [{}]),    # prompt miss -> fallback key
              Inp(9, ["good:a"], "PZ", [{}])]    # missing entirely
    p2r, matched, missing = _E.build_prompt_to_response(inputs, by_prompt, by_key)
    assert p2r["P1"] == "R1" and p2r["P2"] == "R2"
    assert matched == [1, 2] and missing == [9]


# ---- aggregation ----------------------------------------------------------


def test_aggregate_scores_math() -> None:
    outs_strict = [
        _Out(["good:a", "bad:b"], "P1", "R1", False, [True, False]),
        _Out(["good:c"], "P2", "R2", True, [True]),
    ]
    outs_loose = [
        _Out(["good:a", "bad:b"], "P1", "R1", True, [True, True]),
        _Out(["good:c"], "P2", "R2", True, [True]),
    ]
    s = _E.aggregate_scores(outs_strict, outs_loose)
    assert s["num_prompts"] == 2
    assert s["num_instructions"] == 3
    # strict: prompts all-followed = 1/2; instructions followed = 2/3
    assert s["strict_prompt_accuracy"] == 0.5
    assert s["strict_instruction_accuracy"] == round(2 / 3, 6)
    # loose: both prompts all-followed -> 1.0; all 3 instructions followed
    assert s["loose_prompt_accuracy"] == 1.0
    assert s["loose_instruction_accuracy"] == 1.0
    # per-category strict: good=2/2, bad=0/1
    pc = s["per_instruction_category"]
    assert pc["good"]["strict_instruction_accuracy"] == 1.0
    assert pc["bad"]["strict_instruction_accuracy"] == 0.0


def test_aggregate_empty() -> None:
    s = _E.aggregate_scores([], [])
    assert s["num_prompts"] == 0 and s["num_instructions"] == 0
    assert s["strict_prompt_accuracy"] is None


# ---- evaluate + records ---------------------------------------------------


def test_evaluate_skips_unanswered() -> None:
    Inp = _E.InputExample
    inputs = [Inp(1, ["good:a"], "P1", [{}]),
              Inp(2, ["good:a"], "P2", [{}])]   # no response for P2
    p2r = {"P1": "R1"}
    so, lo = _E.evaluate(inputs, p2r, _fake_strict, _fake_loose)
    assert len(so) == 1 and len(lo) == 1
    assert so[0].prompt == "P1"


def test_build_records_analyzer_compatible() -> None:
    Inp = _E.InputExample
    scored = [Inp(7, ["good:a", "bad:b"], "P1", [{}, {}])]
    p2r = {"P1": "R1"}
    so, lo = _E.evaluate(scored, p2r, _fake_strict, _fake_loose)
    combined, strict_recs, loose_recs = _E.build_records(scored, so, lo)
    # combined carries both modes
    assert combined[0]["key"] == 7
    assert combined[0]["strict_follow_all_instructions"] is False
    # per-mode records have the exact keys analyze_ifeval_strict_gap.py reads
    for rec in (strict_recs[0], loose_recs[0]):
        assert set(["key", "id", "prompt", "response", "instruction_id_list",
                    "follow_instruction_list",
                    "follow_all_instructions"]).issubset(rec)
    assert strict_recs[0]["follow_instruction_list"] == [True, False]
    assert loose_recs[0]["follow_instruction_list"] == [True, False]


# ---- main() ---------------------------------------------------------------


class _FakeEvLib:
    test_instruction_following_strict = staticmethod(_fake_strict)
    test_instruction_following_loose = staticmethod(_fake_loose)


def _write(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join([
        json.dumps({"key": 1, "prompt": "P1",
                    "instruction_id_list": ["good:a", "bad:b"],
                    "kwargs": [{}, {}]}),
        json.dumps({"key": 2, "prompt": "P2",
                    "instruction_id_list": ["good:c"], "kwargs": [{}]}),
    ]) + "\n")
    resp = tmp_path / "resp.jsonl"
    resp.write_text("\n".join([
        json.dumps({"id": 1, "prompt": "P1", "response": "R1"}),
        json.dumps({"id": 2, "prompt": "P2", "response": "R2"}),
    ]) + "\n")
    return prompts, resp


def test_main_end_to_end_with_injected_evlib(tmp_path, monkeypatch) -> None:
    prompts, resp = _write(tmp_path)
    monkeypatch.setattr(_E, "import_ifeval_eval_lib", lambda: _FakeEvLib)
    oj = tmp_path / "eval.json"
    ojl = tmp_path / "rec.jsonl"
    om = tmp_path / "eval.md"
    rc = _E.main(["--input-jsonl", str(prompts), "--response-jsonl", str(resp),
                  "--output-json", str(oj), "--output-jsonl", str(ojl),
                  "--output-md", str(om)])
    assert rc == 0
    report = json.loads(oj.read_text())
    assert report["num_prompts"] == 2
    assert report["num_instructions"] == 3
    assert report["model_rerun"] is False
    assert report["evaluator"] == "official_instruction_following_eval"
    # strict: P1 fails (bad:b), P2 passes -> prompt acc 0.5
    assert report["strict_prompt_accuracy"] == 0.5
    # combined + sibling strict/loose files exist and are analyzer-ready
    assert len(ojl.read_text().splitlines()) == 2
    strict_sib = ojl.with_name(ojl.stem + "_strict.jsonl")
    loose_sib = ojl.with_name(ojl.stem + "_loose.jsonl")
    assert strict_sib.exists() and loose_sib.exists()
    srec = json.loads(strict_sib.read_text().splitlines()[0])
    assert "follow_instruction_list" in srec and "follow_all_instructions" in srec
    assert om.read_text().startswith("# IFEval offline evaluation")


def test_main_errors_without_official_package(tmp_path, monkeypatch) -> None:
    prompts, resp = _write(tmp_path)
    monkeypatch.setattr(_E, "import_ifeval_eval_lib", lambda: None)
    rc = _E.main(["--input-jsonl", str(prompts), "--response-jsonl", str(resp),
                  "--output-json", str(tmp_path / "j.json"),
                  "--output-jsonl", str(tmp_path / "r.jsonl"),
                  "--output-md", str(tmp_path / "m.md")])
    assert rc == 3                                   # no home-grown fallback


def test_main_errors_without_scoreable_prompts(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": 1, "prompt": "P1"}) + "\n")  # no metadata
    resp = tmp_path / "r.jsonl"
    resp.write_text(json.dumps({"id": 1, "prompt": "P1", "response": "R"}) + "\n")
    monkeypatch.setattr(_E, "import_ifeval_eval_lib", lambda: _FakeEvLib)
    rc = _E.main(["--input-jsonl", str(prompts), "--response-jsonl", str(resp),
                  "--output-json", str(tmp_path / "j.json"),
                  "--output-jsonl", str(tmp_path / "rec.jsonl"),
                  "--output-md", str(tmp_path / "m.md")])
    assert rc == 3
