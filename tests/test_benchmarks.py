"""Public benchmark dataset support (Task A) -- pure python / stdlib only.

No downloads, no torch, no model. Run:
    python -m pytest tests/test_benchmarks.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks import metrics as M  # noqa: E402
from pllo.benchmarks.prompt_templates import build_prompt  # noqa: E402
from pllo.benchmarks.public_dataset_converters import (  # noqa: E402
    CONVERTERS,
    build_dataset_card,
    convert_agnews_csv,
    convert_boolq_jsonl,
    convert_ceval_csv,
    convert_gsm8k_jsonl,
    convert_mmlu_csv,
    convert_sst2,
    convert_summarization_jsonl,
    deterministic_sample,
    sha256_file,
)
from pllo.benchmarks.runners import run_benchmark  # noqa: E402
from pllo.benchmarks.task_schemas import assert_valid, validate_example  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "benchmarks"


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
# Converters
# ---------------------------------------------------------------------------


def test_convert_mmlu_valid():
    ex = list(convert_mmlu_csv(FIX / "mmlu_tiny.csv"))
    assert len(ex) == 4
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["task_type"] == "multiple_choice"
    assert ex[0]["metric"] == "accuracy"
    assert ex[0]["answer"] == "B"
    assert ex[0]["choices"][1] == "B. 4"
    assert ex[0]["id"] == "mmlu-test-0"


def test_convert_ceval_valid():
    ex = list(convert_ceval_csv(FIX / "ceval_tiny.csv"))
    assert len(ex) == 4
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["dataset"] == "ceval"
    assert ex[0]["task_type"] == "multiple_choice"


def test_convert_gsm8k_valid():
    ex = list(convert_gsm8k_jsonl(FIX / "gsm8k_tiny.jsonl"))
    assert len(ex) == 4
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["task_type"] == "generation_exact"
    assert ex[0]["metric"] == "numeric_exact_match"
    assert ex[0]["numeric_answer"] == "72"


def test_convert_boolq_valid():
    ex = list(convert_boolq_jsonl(FIX / "boolq_tiny.jsonl"))
    assert len(ex) == 4
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["answer"] == "yes"
    assert ex[1]["answer"] == "no"
    assert ex[0]["label_space"] == ["yes", "no"]


def test_convert_agnews_valid():
    ex = list(convert_agnews_csv(FIX / "agnews_tiny.csv"))
    assert len(ex) == 5
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["task_type"] == "classification"
    assert ex[0]["metric"] == "macro_f1"
    assert ex[0]["label"] == "Business"
    assert ex[0]["text"].startswith("Markets rally on earnings.")


def test_convert_sst2_valid():
    ex = list(convert_sst2(FIX / "sst2_tiny.tsv"))
    assert len(ex) == 5
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["label"] == "positive"
    assert ex[1]["label"] == "negative"
    assert ex[0]["label_space"] == ["negative", "positive"]


def test_convert_summarization_valid():
    ex = list(convert_summarization_jsonl(FIX / "cnndm_tiny.jsonl",
                                          dataset="cnndm"))
    assert len(ex) == 4
    for e in ex:
        assert validate_example(e) == []
    assert ex[0]["task_type"] == "summarization"
    assert ex[0]["metric"] == "rouge_l"
    assert ex[0]["dataset"] == "cnndm"


def test_registry_has_all_datasets():
    for k in ("mmlu", "ceval", "cmmlu", "gsm8k", "boolq", "agnews", "sst2",
              "cnndm", "xsum"):
        assert k in CONVERTERS


def test_deterministic_sample_reproducible():
    ex = list(convert_mmlu_csv(FIX / "mmlu_tiny.csv"))
    a = deterministic_sample(ex, 3, seed=7)
    b = deterministic_sample(ex, 3, seed=7)
    c = deterministic_sample(ex, 3, seed=8)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len(a) == 3
    # different seed -> (very likely) different order for this tiny set
    assert [x["id"] for x in a] != [x["id"] for x in c] or len(ex) <= 1


def test_dataset_card_has_sha256_fields(tmp_path):
    out = tmp_path / "mmlu.jsonl"
    ex = list(convert_mmlu_csv(FIX / "mmlu_tiny.csv"))
    out.write_text("\n".join(json.dumps(e) for e in ex), encoding="utf-8")
    card = build_dataset_card(
        source_name="mmlu", split="test", sample_count=len(ex),
        task_type="multiple_choice", metric="accuracy", sampling_seed=0,
        input_file_sha256=sha256_file(FIX / "mmlu_tiny.csv"),
        output_file_sha256=sha256_file(out))
    assert len(card["input_file_sha256"]) == 64
    assert len(card["output_file_sha256"]) == 64
    assert card["sample_count"] == 4
    assert card["task_type"] == "multiple_choice"
    assert card["metric"] == "accuracy"


# ---------------------------------------------------------------------------
# Metrics (hand-computed expectations)
# ---------------------------------------------------------------------------


def test_accuracy_and_exact_match():
    assert M.accuracy(["A", "B", "C"], ["A", "X", "C"]) == pytest.approx(2 / 3)
    assert M.exact_match(["yes", "No"], ["yes", "no"]) == pytest.approx(1.0)


def test_extract_numeric_answer():
    assert M.extract_numeric_answer("#### 72") == "72"
    assert M.extract_numeric_answer("the total is $1,234") == "1234"
    assert M.extract_numeric_answer("answer: 3.5") == repr(3.5)
    assert M.extract_numeric_answer("no number here") is None


def test_extract_numeric_answer_gsm8k_marker():
    # required GSM8K marker cases
    assert M.extract_numeric_answer("#### 3") == "3"
    assert M.extract_numeric_answer("#### 18") == "18"
    assert M.extract_numeric_answer("#### -2") == "-2"
    assert M.extract_numeric_answer("#### 1,234") == "1234"      # commas stripped
    assert M.extract_numeric_answer("#### 3.5") == "3.5"
    assert M.extract_numeric_answer("$18\n\n#### 18") == "18"


def test_extract_numeric_answer_marker_beats_other_numbers():
    # the FINAL #### marker wins over earlier reasoning AND trailing explanation
    assert M.extract_numeric_answer(
        "Step 1: 5 apples, step 2: 7 more, 5+7=12\n#### 12") == "12"
    assert M.extract_numeric_answer(
        "#### 3 (since 2+1=3, checked 2 times)") == "3"
    assert M.extract_numeric_answer(
        "first 5 then 7 #### 4 ... 1 2 3") == "4"


def test_extract_numeric_answer_fallback_without_marker():
    # only when there is NO #### marker do we use the last number in the text
    assert M.extract_numeric_answer("the answer is 5 then finally 7") == "7"
    assert M.extract_numeric_answer("the total is $1,234.00") == "1234"


def test_numeric_values_equal_decimal():
    assert M.numeric_values_equal("3", "3.0") is True
    assert M.numeric_values_equal("3.5", "3.50") is True
    assert M.numeric_values_equal("1234", "1234") is True
    assert M.numeric_values_equal("3", "4") is False
    assert M.numeric_values_equal(None, "3") is False


def test_numeric_exact_match():
    preds = ["the answer is 72", "I think 16", "result 6"]
    golds = ["#### 72", "#### 15", "#### 6"]
    assert M.numeric_exact_match(preds, golds) == pytest.approx(2 / 3)
    # decimal equivalence: "3" matches "3.0"
    assert M.numeric_exact_match(["#### 3"], ["#### 3.0"]) == 1.0


def test_macro_f1():
    # labels P/N; perfect prediction -> 1.0
    assert M.macro_f1(["P", "N"], ["P", "N"], ["P", "N"]) == pytest.approx(1.0)
    # 2 pos, 2 neg gold; pred all pos -> pos: P=2/4 R=1 F1=2/3 ; neg: 0 -> macro=1/3
    preds = ["P", "P", "P", "P"]
    golds = ["P", "P", "N", "N"]
    assert M.macro_f1(preds, golds, ["P", "N"]) == pytest.approx(1 / 3)


def test_rouge_l():
    # LCS("the cat sat", "the cat sat on the mat") = 3 ; P=1, R=0.5, F1=2/3
    assert M.rouge_l("the cat sat", "the cat sat on the mat") == pytest.approx(
        2 / 3)
    assert M.rouge_l("identical text", "identical text") == pytest.approx(1.0)
    assert M.rouge_l("", "anything") == pytest.approx(0.0)


def test_token_match_rate():
    assert M.token_match_rate([1, 2, 3], [1, 9, 3]) == pytest.approx(2 / 3)
    assert M.token_match_rate([], [1]) == pytest.approx(0.0)


def test_compute_metric_dispatch():
    assert M.compute_metric("accuracy", ["a"], ["a"]) == pytest.approx(1.0)
    assert M.compute_metric("macro_f1", ["P"], ["P"],
                            labels=["P", "N"]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        M.compute_metric("nope", [], [])


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_build_prompt_dispatch():
    mc = list(convert_mmlu_csv(FIX / "mmlu_tiny.csv"))[0]
    p = build_prompt(mc)
    assert "Answer:" in p and "\nB. 4" in p
    g = list(convert_gsm8k_jsonl(FIX / "gsm8k_tiny.jsonl"))[0]
    assert build_prompt(g).endswith("Answer:")
    b = list(convert_boolq_jsonl(FIX / "boolq_tiny.jsonl"))[0]
    assert "Answer (yes/no):" in build_prompt(b)
    s = list(convert_sst2(FIX / "sst2_tiny.tsv"))[0]
    assert "Label (one of: negative, positive):" in build_prompt(s)


# ---------------------------------------------------------------------------
# prepare script
# ---------------------------------------------------------------------------


def test_prepare_script(tmp_path):
    mod = _load("prep", "scripts/prepare_public_benchmark_jsonl.py")
    out = tmp_path / "mmlu.jsonl"
    card = tmp_path / "mmlu.card.json"
    rc = _main(mod, ["x", "--input-path", str(FIX / "mmlu_tiny.csv"),
                     "--dataset-name", "mmlu", "--split", "test",
                     "--max-examples", "3", "--seed", "0",
                     "--output-jsonl", str(out),
                     "--dataset-card-json", str(card)])
    assert rc == 0
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    for l in lines:
        assert_valid(json.loads(l))
    c = json.loads(card.read_text())
    assert c["sample_count"] == 3
    assert len(c["input_file_sha256"]) == 64
    assert len(c["output_file_sha256"]) == 64


# ---------------------------------------------------------------------------
# Runner dry-run
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path, examples):
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    return p


def test_runner_dry_run(tmp_path):
    ex = list(convert_mmlu_csv(FIX / "mmlu_tiny.csv"))
    ds = _write_jsonl(tmp_path, ex)
    rep = run_benchmark(ds, backend="plaintext_local",
                        task_type="multiple_choice", max_examples=4)
    assert rep["dry_run"] is True
    assert rep["paper_ready"] is False
    assert rep["backend"] == "plaintext_local"
    assert rep["stage"] == "e9_task_utility_benchmark"
    assert rep["metric_value"] is not None
    assert rep["gpu_visible_plaintext_fields"] == []
    assert rep["leaked_secret_fields"] == []
    assert rep["worker_has_mask_secrets"] is False
    assert rep["tee_used_on_gpu"] is False
    # stub answers "A"; gold answers vary -> accuracy in [0,1]
    assert 0.0 <= rep["metric_value"] <= 1.0


def test_runner_summarization_stub(tmp_path):
    ex = list(convert_summarization_jsonl(FIX / "cnndm_tiny.jsonl",
                                          dataset="cnndm"))
    ds = _write_jsonl(tmp_path, ex)
    rep = run_benchmark(ds, backend="folded_remote",
                        task_type="summarization")
    assert rep["dry_run"] is True
    assert rep["paper_ready"] is False
    assert rep["rouge_l"] is not None
    assert rep["metric_name"] == "rouge_l"


def test_run_e9_script(tmp_path):
    ex = list(convert_boolq_jsonl(FIX / "boolq_tiny.jsonl"))
    ds = _write_jsonl(tmp_path, ex)
    mod = _load("e9", "scripts/run_e9_task_utility_benchmark.py")
    oj = tmp_path / "e9.json"
    om = tmp_path / "e9.md"
    oc = tmp_path / "e9.csv"
    rc = _main(mod, ["x", "--dataset-jsonl", str(ds),
                     "--backend", "tdx_attested_remote",
                     "--task-type", "yes_no",
                     "--output-json", str(oj), "--output-md", str(om),
                     "--output-csv", str(oc)])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["dry_run"] is True
    assert r["paper_ready"] is False
    assert r["backend"] == "tdx_attested_remote"
    assert "E9" in om.read_text()
    assert "dry_run" in oc.read_text()
