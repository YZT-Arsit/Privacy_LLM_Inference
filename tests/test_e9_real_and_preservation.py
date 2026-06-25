"""Gaps 1/2/4/5: E9 --require-real gating, pairwise+aggregate utility
preservation, tightened claim validator, and the preflight checker. stdlib only.

Run: python -m pytest tests/test_e9_real_and_preservation.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.utility_preservation import (  # noqa: E402
    aggregate_preservation,
    pairwise_preservation,
)
from pllo.experiments.claim_validator import build_claim_report  # noqa: E402


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


def _mc_jsonl(tmp_path):
    p = tmp_path / "mc.jsonl"
    p.write_text(json.dumps({"id": "mmlu-test-0", "dataset": "mmlu",
                             "task_type": "multiple_choice", "metric": "accuracy",
                             "question": "2+2?", "choices": ["A. 3", "B. 4",
                                                             "C. 5", "D. 6"],
                             "answer": "B"}) + "\n")
    return p


# --------------------------------------------------------------------------
# Gap 1: --require-real
# --------------------------------------------------------------------------


def test_e9_require_real_fails_without_model(tmp_path) -> None:
    mod = _load("e9", "scripts/run_e9_task_utility_benchmark.py")
    rc = _main(mod, ["x", "--dataset-jsonl", str(_mc_jsonl(tmp_path)),
                     "--task-type", "multiple_choice", "--backend",
                     "plaintext_local", "--require-real", "--output-json",
                     str(tmp_path / "e9.json")])
    assert rc == 3                                    # no fallback to stub


def test_e9_require_real_remote_fails_without_worker(tmp_path) -> None:
    mod = _load("e9b", "scripts/run_e9_task_utility_benchmark.py")
    rc = _main(mod, ["x", "--dataset-jsonl", str(_mc_jsonl(tmp_path)),
                     "--task-type", "multiple_choice", "--backend",
                     "tdx_attested_remote", "--require-real", "--output-json",
                     str(tmp_path / "e9.json")])
    assert rc == 3


def test_e9_stub_still_works_without_require_real(tmp_path) -> None:
    mod = _load("e9c", "scripts/run_e9_task_utility_benchmark.py")
    js = tmp_path / "e9.json"
    rc = _main(mod, ["x", "--dataset-jsonl", str(_mc_jsonl(tmp_path)),
                     "--task-type", "multiple_choice", "--backend",
                     "folded_remote", "--output-json", str(js)])
    assert rc == 0
    r = json.loads(js.read_text())
    assert r["dry_run"] is True
    assert r["paper_ready"] is False


# --------------------------------------------------------------------------
# Gap 2: pairwise + aggregate utility preservation
# --------------------------------------------------------------------------


def _e9(metric, *, backend, dataset="mmlu", paper_ready=True, dry_run=False):
    return {"stage": "e9_task_utility_benchmark", "dataset": dataset,
            "task_type": "multiple_choice", "metric_name": "accuracy",
            "metric_value": metric, "backend": backend,
            "paper_ready": paper_ready, "dry_run": dry_run}


def test_pairwise_within_threshold_preserved() -> None:
    base = _e9(0.80, backend="plaintext_local")
    cand = _e9(0.79, backend="tdx_attested_remote")
    r = pairwise_preservation(base, cand, max_abs_drop=0.02, max_rel_drop=0.05)
    assert r["delta_abs"] == 0.01
    assert r["within_threshold"] is True
    assert r["utility_preserved"] is True
    assert r["paper_ready"] is True


def test_pairwise_exceeds_threshold_not_preserved() -> None:
    base = _e9(0.80, backend="plaintext_local")
    cand = _e9(0.50, backend="folded_remote")
    r = pairwise_preservation(base, cand, max_abs_drop=0.02, max_rel_drop=0.05)
    assert r["within_threshold"] is False
    assert r["utility_preserved"] is False


def test_pairwise_dry_run_not_paper_ready() -> None:
    base = _e9(0.80, backend="plaintext_local", dry_run=True, paper_ready=False)
    cand = _e9(0.80, backend="folded_remote", dry_run=True, paper_ready=False)
    r = pairwise_preservation(base, cand)
    assert r["within_threshold"] is True
    assert r["paper_ready"] is False
    assert r["utility_preserved"] is False          # gated on paper_ready


def _pw(dataset, base_m, cand_m):
    return pairwise_preservation(
        _e9(base_m, backend="plaintext_local", dataset=dataset),
        _e9(cand_m, backend="tdx_attested_remote", dataset=dataset),
        max_abs_drop=0.03, max_rel_drop=0.05, dataset=dataset)


def test_aggregate_all_required_preserved() -> None:
    rows = [_pw("mmlu", 0.62, 0.61), _pw("gsm8k", 0.40, 0.39),
            _pw("boolq", 0.80, 0.79), _pw("sst2", 0.92, 0.91)]
    agg = aggregate_preservation(rows)
    assert agg["missing_datasets"] == []
    assert agg["utility_preserved"] is True


def test_aggregate_missing_dataset_fails() -> None:
    rows = [_pw("mmlu", 0.62, 0.61), _pw("gsm8k", 0.40, 0.39)]
    agg = aggregate_preservation(rows)
    assert "boolq" in agg["missing_datasets"]
    assert agg["utility_preserved"] is False


def test_pairwise_script(tmp_path) -> None:
    mod = _load("pw", "scripts/run_e9_pairwise_utility_preservation.py")
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(_e9(0.80, backend="plaintext_local")))
    c.write_text(json.dumps(_e9(0.79, backend="tdx_attested_remote")))
    oj = tmp_path / "pw.json"
    rc = _main(mod, ["x", "--baseline-json", str(b), "--candidate-json", str(c),
                     "--max-abs-drop", "0.02", "--max-rel-drop", "0.05",
                     "--output-json", str(oj)])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["stage"] == "e9_pairwise_utility_preservation"
    assert r["utility_preserved"] is True


# --------------------------------------------------------------------------
# Gap 4: claim validator tightening
# --------------------------------------------------------------------------


def test_single_e9_metric_does_not_support_utility_claim() -> None:
    single = _e9(0.62, backend="tdx_attested_remote")
    rep = build_claim_report([{"file": "e9.json", "report": single}])
    assert "public_benchmark_utility_preserved" in rep["unsupported_claims"]
    risks = {(o["claim"], tuple(o["reasons"])) for o in rep["overclaim_risks"]}
    assert any(c == "public_benchmark_utility_preserved"
               and "single_e9_metric_not_preservation" in rs
               for c, rs in risks)


def test_aggregate_supports_utility_claim() -> None:
    agg = {"stage": "e9_aggregate_utility_preservation",
           "utility_preserved": True, "paper_ready": True, "dry_run": False}
    rep = build_claim_report([{"file": "agg.json", "report": agg}])
    assert "public_benchmark_utility_preserved" in rep["supported_claims"]


def test_aggregate_dry_run_does_not_support_utility_claim() -> None:
    agg = {"stage": "e9_aggregate_utility_preservation",
           "utility_preserved": True, "paper_ready": False, "dry_run": True}
    rep = build_claim_report([{"file": "agg.json", "report": agg}])
    assert "public_benchmark_utility_preserved" in rep["unsupported_claims"]


# --------------------------------------------------------------------------
# Gap 5: preflight
# --------------------------------------------------------------------------


def _preflight():
    return _load("pf", "scripts/preflight_real_eval.py")


def test_preflight_missing_evidence_blocks(tmp_path) -> None:
    pf = _preflight()
    rep = pf.run_preflight({
        "backend": "tdx_attested_remote", "require_attested": True,
        "model_path": str(tmp_path), "output_dir": str(tmp_path / "out"),
        "attestation_evidence": None})
    assert rep["preflight_passed"] is False
    assert any("attestation_evidence_exists" in b for b in rep["blockers"])


def test_preflight_stale_evidence_blocks(tmp_path) -> None:
    pf = _preflight()
    ev = tmp_path / "evidence.json"
    ev.write_text(json.dumps({
        "tee": "tdx", "td_attributes": {"debug": False}, "jwt": "a.b.c",
        "report_data": "00" * 64, "mr_td": "MRTD"}))
    rep = pf.run_preflight({
        "backend": "tdx_attested_remote", "require_attested": True,
        "model_path": str(tmp_path), "embedding_artifact_path": str(tmp_path),
        "gpu_worker_url": "http://127.0.0.1:9",
        "attestation_evidence": str(ev), "expected_mr_td": "MRTD",
        "output_dir": str(tmp_path / "out")})
    assert rep["preflight_passed"] is False
    chk = {c["name"]: c["ok"] for c in rep["checks"]}
    assert chk.get("runtime_hash_matches_evidence") is False
    assert any("runtime_hash_matches_evidence" in b for b in rep["blockers"])


def test_preflight_script_runs(tmp_path) -> None:
    pf = _preflight()
    rc = _main(pf, ["x", "--backend", "plaintext_local", "--model-path",
                    str(tmp_path), "--output-dir", str(tmp_path / "o"),
                    "--output-json", str(tmp_path / "pf.json"),
                    "--output-md", str(tmp_path / "pf.md")])
    # plaintext_local with an existing model_path dir -> require_real check ok;
    # rc is 0 unless other blockers (output dir writable, etc.)
    assert rc in (0, 1)
    r = json.loads((tmp_path / "pf.json").read_text())
    assert "preflight_passed" in r
    assert isinstance(r["commands_to_run_next"], list)
