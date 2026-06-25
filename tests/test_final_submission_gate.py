"""Fixture tests for the final submission gate. stdlib only.

Run: python -m pytest tests/test_final_submission_gate.py -q
"""

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


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# fake report builders -- each genuinely passes the matching claim predicate
# ---------------------------------------------------------------------------


def _attestation():
    return {"tee_type": "tdx", "verified": True, "available": True,
            "mr_td_match": True}


def _e9_bench(dataset, backend="current"):
    return {"stage": "e9_task_utility_benchmark", "dataset": dataset,
            "metric_name": "accuracy", "metric_value": 0.8,
            "paper_ready": True, "dry_run": False, "nonlinear_backend": backend}


def _aggregate(backend="current"):
    return {"stage": "e9_aggregate_utility_preservation",
            "utility_preserved": True, "paper_ready": True, "dry_run": False,
            "nonlinear_backend": backend}


def _tdx_attested_no_lora_decode(backend="current"):
    return {"stage": "package_backed_decode_tdx_attested",
            "dry_run": False, "lora_enabled": False,
            "gpu_backend": "qwen7b_folded_package", "gpu_worker_remote": True,
            "package_backed_decode": True, "tokens_exact_match": True,
            "folded_package_loaded": True, "folded_package_valid": True,
            "attestation": _attestation(), "boundary_attested": True,
            "runtime_hash_bound": True, "gpu_name": "NVIDIA H800",
            "boundary_mode": "full_reference",
            "nonlinear_backend": backend}


def _folded_lora_h800_decode(backend="current"):
    return {"stage": "folded_lora_decode", "dry_run": False,
            "lora_enabled": True, "lora_mode": "hf",
            "gpu_backend": "qwen7b_folded_package", "gpu_worker_remote": True,
            "package_backed_decode": True, "tokens_exact_match": True,
            "folded_lora_loaded": True, "folded_lora_valid": True,
            "gpu_name": "NVIDIA H800", "nonlinear_backend": backend}


def _security_negative():
    return {"stage": "security_negative_tests", "all_passed": True,
            "nonlinear_backend": "current"}


def _transcript_scan():
    return {"stage": "security_transcript_scan", "passed": True, "leaks": []}


def _latency(backend="current"):
    return {"stage": "latency_baselines", "nonlinear_backend": backend,
            "rows": [{"name": "decode", "paper_ready": True, "dry_run": False,
                      "latency_s": 1.0}]}


def _write(p, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _complete_set(tmp_path, backend="current"):
    files = []
    files.append(_write(tmp_path / "e9_mmlu.json", _e9_bench("mmlu", backend)))
    files.append(_write(tmp_path / "e9_gsm8k.json", _e9_bench("gsm8k", backend)))
    files.append(_write(tmp_path / "e9_boolq.json", _e9_bench("boolq", backend)))
    files.append(_write(tmp_path / "agg.json", _aggregate(backend)))
    files.append(_write(tmp_path / "tdx_nolora.json",
                        _tdx_attested_no_lora_decode(backend)))
    files.append(_write(tmp_path / "lora_h800.json",
                        _folded_lora_h800_decode(backend)))
    files.append(_write(tmp_path / "secneg.json", _security_negative()))
    files.append(_write(tmp_path / "transcript.json", _transcript_scan()))
    files.append(_write(tmp_path / "latency.json", _latency(backend)))
    return files


def _tar(tmp_path):
    p = tmp_path / "paper_artifacts.tar"
    p.write_bytes(b"fake tar contents")
    return str(p)


# ---------------------------------------------------------------------------
# (1) empty/insufficient -> gate fails
# ---------------------------------------------------------------------------


def test_gate_fails_with_no_results(tmp_path) -> None:
    mod = _load("gate1", "scripts/final_submission_gate.py")
    oj = tmp_path / "gate.json"
    rc = _main(mod, ["x", "--output-json", str(oj)])
    assert rc == 1
    r = json.loads(oj.read_text())
    assert r["gate_passed"] is False
    names = {c["name"] for c in r["checks"] if not c["ok"]}
    assert "three_paper_ready_public_benchmarks" in names
    assert "no_lora_tdx_attested_remote_package_decode" in names
    assert "folded_lora_h800_real_validated" in names
    assert "final_artifact_tar_exists" in names
    assert r["blockers"]


# ---------------------------------------------------------------------------
# (2) complete synthetic set -> gate passes
# ---------------------------------------------------------------------------


def test_gate_passes_with_complete_set(tmp_path) -> None:
    mod = _load("gate2", "scripts/final_submission_gate.py")
    files = _complete_set(tmp_path)
    argv = ["x"]
    for f in files:
        argv += ["--result-json", f]
    argv += ["--final-artifact-tar", _tar(tmp_path)]
    oj = tmp_path / "gate.json"
    om = tmp_path / "gate.md"
    argv += ["--output-json", str(oj), "--output-md", str(om)]
    rc = _main(mod, argv)
    r = json.loads(oj.read_text())
    failed = [c["name"] for c in r["checks"] if not c["ok"]]
    assert failed == [], "unexpected failed checks: %s" % failed
    assert r["gate_passed"] is True
    assert rc == 0
    assert om.is_file()


# ---------------------------------------------------------------------------
# (3) per-backend: evidence only tagged 'current' -> trusted_shortcut fails
# ---------------------------------------------------------------------------


def test_gate_per_backend_missing_design_fails(tmp_path) -> None:
    mod = _load("gate3", "scripts/final_submission_gate.py")
    files = _complete_set(tmp_path, backend="current")
    argv = ["x"]
    for f in files:
        argv += ["--result-json", f]
    argv += ["--final-artifact-tar", _tar(tmp_path),
             "--nonlinear-backends", "current,trusted_shortcut",
             "--output-json", str(tmp_path / "gate.json")]
    rc = _main(mod, argv)
    r = json.loads((tmp_path / "gate.json").read_text())
    assert rc == 1
    assert r["gate_passed"] is False
    pb = r["per_backend"]
    assert pb["current"]["ok"] is True
    assert pb["trusted_shortcut"]["ok"] is False
    assert pb["trusted_shortcut"]["missing"]
    # the backend-specific check should be a blocker
    names = {c["name"] for c in r["checks"] if not c["ok"]}
    assert "nonlinear_backend_trusted_shortcut" in names
