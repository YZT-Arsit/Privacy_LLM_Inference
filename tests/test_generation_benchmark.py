"""Generation utility benchmark tests (stdlib, no model/worker/H800).

Covers the task-aware metrics (numeric EM, ROUGE + fallback, edit similarity),
the per-backend runner (stub + injected fake predictor), the pairwise
preservation + drop/exact-output/latency/audit logic, the summary table, the
current-only guard, and all three scripts (prepare / utility / pairwise) via
importlib + argv swap.

Run: python -m pytest tests/test_generation_benchmark.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks import generation_metrics as gm  # noqa: E402


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


# ---- metrics --------------------------------------------------------------

def test_extract_number_and_numeric_match() -> None:
    assert gm.extract_number("The answer is #### 42") == "42"
    assert gm.numeric_exact_match("so the total is 42 apples", "#### 42") is True
    assert gm.numeric_exact_match("about 41", "#### 42") is False
    assert gm.numeric_exact_match("no digits here", "#### 42") is False


def test_score_example_gsm8k_marker() -> None:
    # the runner path: score_example -> extracted_number must NOT be null for a
    # response containing "#### 3", and numeric_exact_match must be True vs gold.
    row = gm.score_example("generation_exact", "#### 3", "#### 3", [1, 2, 3])
    assert row["extracted_number"] == "3"
    assert row["numeric_exact_match"] is True
    # trailing explanation numbers must not override the marker
    row2 = gm.score_example("generation_exact",
                            "5+7=12 #### 12\nrecheck 6+6", "#### 12", None)
    assert row2["extracted_number"] == "12"
    assert row2["numeric_exact_match"] is True
    # decimal equivalence
    row3 = gm.score_example("generation_exact", "#### 3.0", "#### 3", None)
    assert row3["numeric_exact_match"] is True


def test_rouge_scores_fields_and_fallback() -> None:
    r = gm.rouge_scores("the cat sat on the mat", "the cat sat on the mat")
    for k in ("rouge1", "rouge2", "rougeL", "rouge_unavailable"):
        assert k in r
    assert isinstance(r["rouge_unavailable"], bool)
    assert r["rouge1"] == 1.0 and r["rougeL"] == 1.0
    r2 = gm.rouge_scores("totally unrelated words", "the cat sat on the mat")
    assert 0.0 <= r2["rouge1"] <= 1.0 and r2["rougeL"] < 0.5


def test_output_length_tokens() -> None:
    assert gm.output_length_tokens([1, 2, 3]) == 3
    assert gm.output_length_tokens(None) is None


def test_score_example_per_task() -> None:
    g = gm.score_example("generation_exact", "the result is 42", "#### 42",
                         [1, 2, 3])
    assert g["numeric_exact_match"] is True and g["extracted_number"] == "42"
    assert g["output_length_tokens"] == 3
    s = gm.score_example("summarization", "a short summary", "a short summary",
                         [1, 2])
    assert s["rougeL"] == 1.0 and isinstance(s["rouge_unavailable"], bool)
    o = gm.score_example("open_ended", "hello world", "hello world", [1, 2])
    assert o["exact_text_match"] is True
    assert o["normalized_edit_similarity"] == 1.0
    o2 = gm.score_example("open_ended", "hello", None, [1])
    assert o2["exact_text_match"] is None


# ---- current-only guard ---------------------------------------------------

def test_current_only_guard() -> None:
    assert gm.assert_current_only("current") == "current"
    with pytest.raises(ValueError):
        gm.assert_current_only("trusted_shortcut")
    with pytest.raises(ValueError):
        gm.assert_current_only("amulet_migrated")


# ---- runner ---------------------------------------------------------------

_GSM8K = [
    {"id": "g1", "dataset_name": "gsm8k", "task_type": "generation_exact",
     "prompt": "Problem: 40 + 2 =\nAnswer:", "reference": "#### 42"},
    {"id": "g2", "dataset_name": "gsm8k", "task_type": "generation_exact",
     "prompt": "Problem: 1 + 1 =\nAnswer:", "reference": "#### 2"},
]


class _Fake:
    def __init__(self, fn, *, audit=True):
        self._fn = fn
        self._audit = audit

    def generate(self, prompt):
        t = self._fn(prompt)
        return {"text": t, "token_ids": [ord(c) for c in t]}

    def stats(self):
        return {"audit_passed": self._audit, "tee_used_on_gpu": False,
                "worker_has_mask_secrets": False}


def test_runner_stub_dry_run() -> None:
    rep = gm.run_generation_utility_benchmark(
        _GSM8K, backend="plaintext_local", predictor=None, max_new_tokens=8)
    assert rep["stage"] == gm.STAGE_BENCHMARK
    assert rep["dry_run"] is True and rep["paper_ready"] is False
    assert rep["task_type"] == "generation_exact"
    assert rep["metric_name"] == "numeric_exact_match"
    assert rep["max_new_tokens"] == 8 and rep["dataset_name"] == "gsm8k"


def test_runner_gsm8k_numeric_match() -> None:
    # a fake that always answers "42" -> matches g1 only
    pred = _Fake(lambda p: "the answer is 42")
    rep = gm.run_generation_utility_benchmark(
        _GSM8K, backend="folded_remote", predictor=pred, max_new_tokens=16)
    assert rep["paper_ready"] is True and rep["audit_passed"] is True
    assert rep["metric_name"] == "numeric_exact_match"
    assert rep["metric_value"] == 0.5            # 1 of 2 correct
    assert rep["nonlinear_backend"] == "current"


def test_runner_refuses_trusted_shortcut() -> None:
    with pytest.raises(ValueError):
        gm.run_generation_utility_benchmark(
            _GSM8K, backend="folded_remote", predictor=None,
            nonlinear_backend="trusted_shortcut")


# ---- pairwise + summary ---------------------------------------------------

_SUMM = [
    {"id": "s1", "dataset_name": "cnndm", "task_type": "summarization",
     "prompt": "Article: cats are great. Summary:", "reference": "cats are great"},
    {"id": "s2", "dataset_name": "cnndm", "task_type": "summarization",
     "prompt": "Article: dogs are loyal. Summary:",
     "reference": "dogs are loyal"},
]


def _report(backend, fn, examples, *, audit=True, dry_run=False):
    rep = gm.run_generation_utility_benchmark(
        examples, backend=backend, predictor=_Fake(fn, audit=audit),
        max_new_tokens=16, dry_run=dry_run)
    return rep


def test_pairwise_identical_preserved() -> None:
    fn = lambda p: "cats are great" if "cats" in p else "dogs are loyal"  # noqa
    base = _report("plaintext_local", fn, _SUMM)
    cand = _report("folded_remote", fn, _SUMM)
    r = gm.pairwise_generation_preservation(base, cand)
    assert r["metric_name"] == "rougeL"
    assert r["metric_abs_drop"] == 0.0
    assert r["exact_output_match_rate"] == 1.0
    assert r["length_delta_mean"] == 0
    assert r["within_threshold"] is True
    assert r["utility_preserved"] is True and r["paper_ready"] is True


def test_pairwise_metric_drop_breaks_preservation() -> None:
    base = _report("plaintext_local",
                   lambda p: "cats are great" if "cats" in p
                   else "dogs are loyal", _SUMM)
    cand = _report("folded_remote", lambda p: "completely wrong text", _SUMM)
    r = gm.pairwise_generation_preservation(base, cand)
    assert r["metric_abs_drop"] > 0.05
    assert r["within_threshold"] is False
    assert r["utility_preserved"] is False


def test_pairwise_failed_audit_blocks() -> None:
    fn = lambda p: "cats are great" if "cats" in p else "dogs are loyal"  # noqa
    base = _report("plaintext_local", fn, _SUMM)
    cand = _report("folded_remote", fn, _SUMM, audit=False)
    r = gm.pairwise_generation_preservation(base, cand)
    assert r["audit_passed"] is False
    assert r["utility_preserved"] is False


def test_pairwise_dry_run_not_paper_ready() -> None:
    fn = lambda p: "cats are great" if "cats" in p else "dogs are loyal"  # noqa
    base = _report("plaintext_local", fn, _SUMM, dry_run=True)
    cand = _report("folded_remote", fn, _SUMM)
    r = gm.pairwise_generation_preservation(base, cand)
    assert r["paper_ready"] is False and r["utility_preserved"] is False


def test_pairwise_refuses_trusted_shortcut_candidate() -> None:
    base = _report("plaintext_local", lambda p: "x", _SUMM)
    cand = _report("folded_remote", lambda p: "x", _SUMM)
    cand["nonlinear_backend"] = "trusted_shortcut"
    with pytest.raises(ValueError):
        gm.pairwise_generation_preservation(base, cand)


def test_summary_table() -> None:
    fn = lambda p: "cats are great" if "cats" in p else "dogs are loyal"  # noqa
    pr = gm.pairwise_generation_preservation(
        _report("plaintext_local", fn, _SUMM), _report("folded_remote", fn,
                                                       _SUMM))
    s = gm.summarize_pairwise([pr])
    assert s["stage"] == gm.STAGE_SUMMARY and s["num_datasets"] == 1
    assert s["all_utility_preserved"] is True
    assert s["rows"][0]["dataset_name"] == "cnndm"


# ---- prepare script -------------------------------------------------------

def _w_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    return str(path)


def test_prepare_cnndm_and_card(tmp_path) -> None:
    prep = _load("prep_cnndm", "scripts/prepare_generation_benchmark_jsonl.py")
    raw = _w_jsonl(tmp_path / "cnndm_raw.jsonl",
                   [{"article": "Cats are great pets.", "highlights": "cats"},
                    {"article": "Dogs are loyal.", "highlights": "dogs"}])
    out = tmp_path / "cnndm_gen.jsonl"
    rc = _main(prep, ["x", "--dataset", "cnndm", "--input", raw,
                      "--output", str(out)])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert all(r["task_type"] == "summarization" for r in rows)
    assert all(r["dataset_name"] == "cnndm" for r in rows)
    assert all("Summarize" in r["prompt"] and r["reference"] for r in rows)
    card = json.loads((tmp_path / "cnndm_gen.jsonl.card.json").read_text())
    assert card["task_type"] == "summarization" and card["no_downloads"] is True
    assert card["sample_count"] == 2 and card["output_file_sha256"]


def test_prepare_gsm8k_and_xsum_and_custom(tmp_path) -> None:
    prep = _load("prep_multi", "scripts/prepare_generation_benchmark_jsonl.py")
    g_raw = _w_jsonl(tmp_path / "g.jsonl",
                     [{"question": "2+2?", "answer": "#### 4"}])
    g_out = tmp_path / "g_gen.jsonl"
    assert _main(prep, ["x", "--dataset", "gsm8k", "--input", g_raw,
                        "--output", str(g_out)]) == 0
    gr = json.loads(g_out.read_text().splitlines()[0])
    assert gr["task_type"] == "generation_exact"
    assert gr["numeric_reference"] == "4"

    x_raw = _w_jsonl(tmp_path / "x.jsonl",
                     [{"document": "long doc", "summary": "short"}])
    x_out = tmp_path / "x_gen.jsonl"
    assert _main(prep, ["x", "--dataset", "xsum", "--input", x_raw,
                        "--output", str(x_out)]) == 0
    assert json.loads(x_out.read_text().splitlines()[0])["dataset_name"] == "xsum"

    c_raw = _w_jsonl(tmp_path / "c.jsonl",
                     [{"id": "c1", "prompt": "Write a haiku.",
                       "category": "creative"}])
    c_out = tmp_path / "c_gen.jsonl"
    assert _main(prep, ["x", "--dataset", "custom", "--input", c_raw,
                        "--output", str(c_out)]) == 0
    cr = json.loads(c_out.read_text().splitlines()[0])
    assert cr["task_type"] == "open_ended" and cr["category"] == "creative"


# ---- runner + pairwise + summary scripts ----------------------------------

def test_utility_and_pairwise_and_summary_scripts(tmp_path) -> None:
    bench = _load("util_sc", "scripts/run_generation_utility_benchmark.py")
    pair = _load("pair_sc", "scripts/run_generation_pairwise_preservation.py")
    ds = _w_jsonl(tmp_path / "gsm.jsonl", _GSM8K)
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    assert _main(bench, ["x", "--dataset-jsonl", ds, "--backend",
                         "plaintext_local", "--max-new-tokens", "8",
                         "--output-json", str(base),
                         "--output-csv", str(tmp_path / "b.csv"),
                         "--output-md", str(tmp_path / "b.md")]) == 0
    assert _main(bench, ["x", "--dataset-jsonl", ds, "--backend",
                         "folded_remote", "--nonlinear-backend", "current",
                         "--max-new-tokens", "8", "--output-json",
                         str(cand)]) == 0
    assert (tmp_path / "b.csv").is_file() and (tmp_path / "b.md").is_file()

    pj = tmp_path / "pair.json"
    assert _main(pair, ["x", "--baseline-json", str(base), "--candidate-json",
                        str(cand), "--output-json", str(pj),
                        "--output-md", str(tmp_path / "pair.md")]) == 0
    pr = json.loads(pj.read_text())
    assert pr["stage"] == gm.STAGE_PAIRWISE
    assert pr["paper_ready"] is False            # dry-run inputs

    sj = tmp_path / "summary.json"
    assert _main(pair, ["x", "--summary-input", str(pj), "--output-json",
                        str(sj), "--output-md", str(tmp_path / "sum.md"),
                        "--output-csv", str(tmp_path / "sum.csv")]) == 0
    sm = json.loads(sj.read_text())
    assert sm["stage"] == gm.STAGE_SUMMARY and sm["num_datasets"] == 1


def test_utility_script_refuses_trusted_shortcut(tmp_path) -> None:
    bench = _load("util_ts", "scripts/run_generation_utility_benchmark.py")
    ds = _w_jsonl(tmp_path / "gsm.jsonl", _GSM8K)
    rc = _main(bench, ["x", "--dataset-jsonl", ds, "--backend", "folded_remote",
                       "--nonlinear-backend", "trusted_shortcut",
                       "--output-json", str(tmp_path / "o.json")])
    assert rc == 3
