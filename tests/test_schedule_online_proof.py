"""run_ifeval_generation.py online-deterministic schedule proof (mock path):
full coverage + no leakage, no default CUDA secret-tensor materialization.

Run:
    PYTHONPATH=$PWD/src pytest tests/test_schedule_online_proof.py -q
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


_R = _load("ifeval_runner_sched", "scripts/run_ifeval_generation.py")


def _prompts(tmp_path, n=2):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": "ex%d" % i, "prompt": "Q%d?" % i})
        for i in range(n)) + "\n", encoding="utf-8")
    return p


def _run(tmp_path, extra):
    inp = _prompts(tmp_path)
    rj, rep = tmp_path / "r.jsonl", tmp_path / "r.json"
    argv = ["x", "--input-jsonl", str(inp), "--backend", "folded_remote",
            "--mock-runtime", "--max-new-tokens", "4",
            "--output-response-jsonl", str(rj),
            "--output-report-json", str(rep)] + extra
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    return rc, (json.loads(rep.read_text()) if rep.exists() else None)


def test_online_deterministic_no_secret_tensors(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16",
        "--schedule-proof-mode", "online_deterministic"])
    assert rc == 0
    assert report["schedule_proof_mode"] == "online_deterministic"
    assert report["secret_tensor_precompute_performed"] is False
    assert report["schedule_materialized_on_gpu"] is False
    assert report["schedule_precompute_device"] == "cpu"


def test_full_coverage_verified_and_no_leak(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16"])
    assert rc == 0
    assert report["schedule_full_coverage_verified"] is True
    assert report["schedule_secret_leaked_to_gpu"] is False
    assert report["schedule_materialized_on_gpu"] is False
    assert report["online_remask_still_performed"] is True
    assert (report["schedule_slots_required_total"]
            == report["schedule_slots_consumed_total"])
    cov = report["schedule_coverage_per_example"]
    assert len(cov) == 2
    for r in cov:
        assert r["slots_consumed_matches_generated_tokens"] is True
        assert r["slots_required"] == r["slots_consumed"] == r["generated_tokens"]
        assert r["schedule_secret_leaked_to_gpu"] is False
        assert r["schedule_materialized_on_gpu"] is False
        assert r["schedule_seed_commitment"]


def test_legacy_precompute_flag_no_default_cuda_materialize(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "16",
        "--device", "cuda"])             # OLD command, device cuda
    assert rc == 0
    assert report["precompute_obfuscation_schedule"] is True
    assert report["secret_tensor_precompute_performed"] is False
    assert report["schedule_materialized_on_gpu"] is False
    assert report["schedule_proof_mode"] == "online_deterministic"


def test_precompute_secret_tensors_requires_strong_confirm(tmp_path, capsys):
    rc, report = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-max-steps", "8",
        "--schedule-proof-mode", "precompute_secret_tensors"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "precompute_secret_tensors" in err and "falling back" in err
    assert report["secret_tensor_precompute_performed"] is False


def test_proof_mode_none_disables_schedule(tmp_path) -> None:
    rc, report = _run(tmp_path, [
        "--precompute-obfuscation-schedule", "--schedule-proof-mode", "none"])
    assert rc == 0
    assert report["precompute_obfuscation_schedule"] is False
    assert report["schedule_slots_required_total"] == 0
    assert report["schedule_full_coverage_verified"] is False
