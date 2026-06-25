"""Open-ended generation preservation benchmark tests (stdlib, no model).

Covers the objective metrics, the per-backend runner (stub + injected fake
predictor), the pairwise comparison + preservation decision, the current-only
guard (trusted_shortcut refused), and the two scripts via importlib + argv swap.

Run: python -m pytest tests/test_generation_preservation.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks import generation_preservation as gp  # noqa: E402


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

def test_levenshtein_and_edit_similarity() -> None:
    assert gp.levenshtein("abc", "abc") == 0
    assert gp.levenshtein("abc", "abd") == 1
    assert gp.levenshtein("", "abc") == 3
    assert gp.normalized_edit_similarity("abc", "abc") == 1.0
    assert gp.normalized_edit_similarity("", "") == 1.0
    assert 0.6 < gp.normalized_edit_similarity("abcde", "abxde") < 0.85


def test_exact_text_and_token_match() -> None:
    assert gp.exact_text_match("hi there", "hi there") is True
    assert gp.exact_text_match("hi", "hi ") is False
    assert gp.exact_token_match([1, 2, 3], [1, 2, 3]) is True
    assert gp.exact_token_match([1, 2], [1, 3]) is False
    assert gp.exact_token_match(None, [1]) is None      # unavailable


def test_compare_generation_row() -> None:
    b = {"id": "1", "category": "qa", "text": "the cat sat",
         "token_ids": [10, 11, 12]}
    c = {"id": "1", "category": "qa", "text": "the cat sat",
         "token_ids": [10, 11, 12]}
    row = gp.compare_generation(b, c)
    assert row["exact_text_match"] is True
    assert row["exact_token_match"] is True
    assert row["normalized_edit_similarity"] == 1.0
    assert row["output_char_length_delta"] == 0
    assert row["output_token_length_delta"] == 0


# ---- current-only guard ---------------------------------------------------

def test_current_only_guard_refuses_trusted_shortcut() -> None:
    assert gp.assert_current_only("current") == "current"
    with pytest.raises(ValueError):
        gp.assert_current_only("trusted_shortcut")
    with pytest.raises(ValueError):
        gp.assert_current_only("amulet_migrated")    # alias of trusted_shortcut


# ---- per-backend runner (stub + fake predictor) ---------------------------

_EXAMPLES = [
    {"id": "a1", "prompt": "Write a haiku about privacy.", "category": "creative"},
    {"id": "a2", "prompt": "Explain TEEs in one line.", "category": "explain"},
]


def test_runner_stub_is_dry_run() -> None:
    rep = gp.run_generation_benchmark(
        _EXAMPLES, backend="plaintext_local", predictor=None, max_new_tokens=8)
    assert rep["stage"] == gp.STAGE_BENCHMARK
    assert rep["dry_run"] is True and rep["paper_ready"] is False
    assert rep["num_examples"] == 2
    assert rep["token_ids_available"] is True
    assert all(g["token_ids"] is not None for g in rep["generations"])


class _FakePredictor:
    """Deterministic fake generator (no torch/model)."""

    def __init__(self, transform):
        self._t = transform

    def generate(self, prompt):
        text = self._t(prompt)
        return {"text": text, "token_ids": [ord(c) for c in text]}

    def stats(self):
        return {"audit_passed": True, "tee_used_on_gpu": False,
                "worker_has_mask_secrets": False}


def test_runner_real_predictor_paper_ready() -> None:
    pred = _FakePredictor(lambda p: "OUT:" + p)
    rep = gp.run_generation_benchmark(
        _EXAMPLES, backend="folded_remote", predictor=pred,
        nonlinear_backend="current", max_new_tokens=8)
    assert rep["dry_run"] is False and rep["paper_ready"] is True
    assert rep["audit_passed"] is True
    assert rep["nonlinear_backend"] == "current"
    assert rep["generations"][0]["text"] == "OUT:" + _EXAMPLES[0]["prompt"]


def test_runner_refuses_trusted_shortcut() -> None:
    with pytest.raises(ValueError):
        gp.run_generation_benchmark(
            _EXAMPLES, backend="folded_remote", predictor=None,
            nonlinear_backend="trusted_shortcut")


# ---- pairwise preservation ------------------------------------------------

def _report(backend, transform, *, dry_run=False, audit=True):
    pred = _FakePredictor(transform)
    rep = gp.run_generation_benchmark(
        _EXAMPLES, backend=backend, predictor=pred, nonlinear_backend="current",
        max_new_tokens=8, dry_run=dry_run)
    rep["audit_passed"] = audit
    return rep


def test_pairwise_identical_outputs_preserved() -> None:
    same = lambda p: "ANSWER " + p           # noqa: E731
    base = _report("plaintext_local", same)
    cand = _report("folded_remote", same)
    r = gp.pairwise_generation_preservation(base, cand)
    a = r["aggregate"]
    assert a["exact_text_match_rate"] == 1.0
    assert a["exact_token_match_rate"] == 1.0
    assert a["mean_normalized_edit_similarity"] == 1.0
    assert a["mean_output_char_length_delta"] == 0
    assert r["generation_preserved"] is True
    assert r["paper_ready"] is True
    assert set(a["by_category"]) == {"creative", "explain"}


def test_pairwise_divergent_outputs_not_preserved() -> None:
    base = _report("plaintext_local", lambda p: "the cat sat on the mat here")
    cand = _report("folded_remote", lambda p: "a totally different sentence!!")
    r = gp.pairwise_generation_preservation(base, cand)
    assert r["aggregate"]["exact_token_match_rate"] == 0.0
    assert r["generation_preserved"] is False


def test_pairwise_dry_run_never_paper_ready() -> None:
    same = lambda p: "X " + p                # noqa: E731
    base = _report("plaintext_local", same, dry_run=True)
    cand = _report("folded_remote", same, dry_run=False)
    r = gp.pairwise_generation_preservation(base, cand)
    # identical text, but a dry-run baseline can never back a preserved claim
    assert r["paper_ready"] is False
    assert r["generation_preserved"] is False


def test_pairwise_failed_audit_blocks_preservation() -> None:
    same = lambda p: "Z " + p                # noqa: E731
    base = _report("plaintext_local", same)
    cand = _report("folded_remote", same, audit=False)   # candidate audit failed
    r = gp.pairwise_generation_preservation(base, cand)
    assert r["candidate_audit_passed"] is False
    assert r["generation_preserved"] is False


def test_pairwise_refuses_trusted_shortcut_candidate() -> None:
    base = _report("plaintext_local", lambda p: p)
    cand = _report("folded_remote", lambda p: p)
    cand["nonlinear_backend"] = "trusted_shortcut"
    with pytest.raises(ValueError):
        gp.pairwise_generation_preservation(base, cand)


# ---- scripts via importlib + argv swap ------------------------------------

def _write_jsonl(path):
    path.write_text("\n".join(json.dumps(e) for e in _EXAMPLES) + "\n",
                    encoding="utf-8")
    return str(path)


def test_benchmark_script_dry_run(tmp_path) -> None:
    mod = _load("genbench", "scripts/run_generation_preservation_benchmark.py")
    ds = _write_jsonl(tmp_path / "p.jsonl")
    oj = tmp_path / "b.json"
    rc = _main(mod, ["x", "--dataset-jsonl", ds, "--backend", "plaintext_local",
                     "--max-new-tokens", "8", "--output-json", str(oj),
                     "--output-csv", str(tmp_path / "b.csv"),
                     "--output-md", str(tmp_path / "b.md")])
    assert rc == 0
    rep = json.loads(oj.read_text())
    assert rep["stage"] == gp.STAGE_BENCHMARK and rep["dry_run"] is True
    assert (tmp_path / "b.csv").is_file() and (tmp_path / "b.md").is_file()


def test_benchmark_script_refuses_trusted_shortcut(tmp_path) -> None:
    mod = _load("genbench2", "scripts/run_generation_preservation_benchmark.py")
    ds = _write_jsonl(tmp_path / "p.jsonl")
    rc = _main(mod, ["x", "--dataset-jsonl", ds, "--backend", "folded_remote",
                     "--nonlinear-backend", "trusted_shortcut",
                     "--output-json", str(tmp_path / "b.json")])
    assert rc == 3


def test_pairwise_script(tmp_path) -> None:
    bench = _load("genbench3", "scripts/run_generation_preservation_benchmark.py")
    pair = _load("genpair", "scripts/run_generation_preservation_pairwise.py")
    ds = _write_jsonl(tmp_path / "p.jsonl")
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    # two dry-run reports (identical stub text) -> compared but not paper_ready
    assert _main(bench, ["x", "--dataset-jsonl", ds, "--backend",
                         "plaintext_local", "--max-new-tokens", "8",
                         "--output-json", str(base)]) == 0
    assert _main(bench, ["x", "--dataset-jsonl", ds, "--backend",
                         "folded_remote", "--nonlinear-backend", "current",
                         "--max-new-tokens", "8", "--output-json",
                         str(cand)]) == 0
    oj = tmp_path / "pair.json"
    rc = _main(pair, ["x", "--baseline-json", str(base), "--candidate-json",
                      str(cand), "--output-json", str(oj),
                      "--output-csv", str(tmp_path / "pair.csv"),
                      "--output-md", str(tmp_path / "pair.md")])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["stage"] == gp.STAGE_PAIRWISE
    assert r["aggregate"]["exact_text_match_rate"] == 1.0
    assert r["paper_ready"] is False          # dry-run inputs
    assert (tmp_path / "pair.csv").is_file() and (tmp_path / "pair.md").is_file()
