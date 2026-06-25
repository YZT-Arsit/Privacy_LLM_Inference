"""Precomputed per-step obfuscation schedule tests (local, no GPU/model/server).

Covers: per-step freshness, no-GPU-leakage audit, seed reproducibility, the
generation report schedule fields, backward-compat when disabled, double-consume
refusal, secret non-persistence, and the probe / ifeval scripts in mock mode.

Run: python -m pytest tests/test_precomputed_obfuscation_schedule.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.runtime.obfuscation_schedule import (  # noqa: E402
    PrecomputedMaskSchedule,
    ScheduleSecretLeak,
    ScheduleSlotAlreadyConsumed,
    audit_gpu_payload_no_schedule_secrets,
    audit_schedule_trusted_only,
    audit_worker_package_no_schedule_secrets,
    default_schedule_report_fields,
    schedule_report_fields,
)

_REPORT_FIELDS = set(default_schedule_report_fields())


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


def _sched(max_steps=8, seed=2035, secrets=False):
    return PrecomputedMaskSchedule.precompute(
        max_steps=max_steps, hidden_size=16, seed=seed,
        with_secret_tensors=secrets, strict_audit=True)


# ---- 1. freshness ---------------------------------------------------------

def test_precomputed_schedule_freshness() -> None:
    s = _sched(max_steps=10)
    mask_ids = [sl.mask_id for sl in s.slots]
    domain_ids = [sl.domain_id for sl in s.slots]
    # every decode step gets a DISTINCT fresh obfuscation domain (no fixed mask)
    assert len(set(mask_ids)) == 10
    assert len(set(domain_ids)) == 10
    assert all(sl.remask_meta["kind"] == "fresh_per_step" for sl in s.slots)
    # double-consume is refused (reusing a domain across steps is forbidden)
    s.consume(3)
    with pytest.raises(ScheduleSlotAlreadyConsumed):
        s.consume(3)
    assert s.consume(4).step_id == 4


# ---- 2. no GPU leakage ----------------------------------------------------

def test_schedule_no_gpu_leakage() -> None:
    s = _sched(max_steps=4, secrets=True)
    # the schedule's serializable surface holds no secret-named fields / tensors
    audit_schedule_trusted_only(s)
    json.dumps(s.to_dict())                      # public surface is plain JSON

    # a legitimate masked GPU request passes
    audit_gpu_payload_no_schedule_secrets(
        {"session_id": "x", "masked_embedding": [0.0, 0.1], "step": 1,
         "obfuscation_mask_id": s.slots[1].mask_id, "seq_len": 8,
         "mask_family": "pairwise_complex_scaling"})
    # any schedule secret on a GPU payload is rejected loudly
    for bad in ({"mask_secret": [1, 2]}, {"meta": {"pad_value": [0.1]}},
                {"prg_seed": 7}, {"schedule_secret_tensor": [0.0]}):
        with pytest.raises(ScheduleSecretLeak):
            audit_gpu_payload_no_schedule_secrets(bad)
    # worker package / manifest audit
    audit_worker_package_no_schedule_secrets({"num_shards": 30, "ok": True})
    with pytest.raises(ScheduleSecretLeak):
        audit_worker_package_no_schedule_secrets({"mask_inverse": [1]})


def test_remote_post_audit_blocks_schedule_secret() -> None:
    # the live-wire client guard rejects forbidden/secret fields before send
    remote = importlib.import_module("pllo.protocol.remote")
    assert "schedule_secret" in remote.FORBIDDEN_WIRE_FIELDS
    assert remote.forbidden_fields_in_payload({"mask_secret": [1]})
    assert remote.forbidden_fields_in_payload(
        {"session_id": "x", "masked_embedding": [0.1], "seq_len": 8}) == []


# ---- 3. reproducibility ---------------------------------------------------

def test_schedule_reproducibility_with_seed() -> None:
    a = _sched(max_steps=6, seed=2035)
    b = _sched(max_steps=6, seed=2035)
    c = _sched(max_steps=6, seed=999)
    assert [s.mask_id for s in a.slots] == [s.mask_id for s in b.slots]
    assert [s.domain_id for s in a.slots] == [s.domain_id for s in b.slots]
    assert [s.mask_id for s in a.slots] != [s.mask_id for s in c.slots]
    assert a.public_metadata()["session_fingerprint"] == \
        b.public_metadata()["session_fingerprint"]


# ---- 4. report fields -----------------------------------------------------

def test_schedule_report_fields_complete_and_defaulted() -> None:
    # disabled defaults
    d = default_schedule_report_fields()
    assert d["precompute_obfuscation_schedule"] is False
    assert d["schedule_secret_leaked_to_gpu"] is False
    assert d["gpu_request_contains_schedule_secret"] is False
    # enabled fields
    s = _sched(max_steps=8)
    for i in range(5):
        s.consume(i)
    f = schedule_report_fields(s, enabled=True,
                               online_generation_latency_s=12.0,
                               boundary_calls=10, generated_tokens=5)
    assert set(f) == _REPORT_FIELDS
    assert f["precompute_obfuscation_schedule"] is True
    assert f["schedule_slots_precomputed"] == 8
    assert f["schedule_slots_consumed"] == 5
    assert f["boundary_calls_per_generated_token"] == 2.0
    assert f["schedule_used_for_metadata_only"] is True
    assert f["online_remask_still_performed"] is True
    assert f["schedule_secret_leaked_to_gpu"] is False


def test_generation_report_contains_schedule_fields(tmp_path) -> None:
    mod = _load("ifeval1", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Write a poem."}) + "\n",
                  encoding="utf-8")
    rep_j = tmp_path / "rep.json"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend",
                     "folded_remote", "--mock-runtime", "--max-new-tokens", "6",
                     "--precompute-obfuscation-schedule", "--schedule-max-steps",
                     "8", "--schedule-seed", "2035", "--report-schedule-stats",
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep_j)])
    assert rc == 0
    rep = json.loads(rep_j.read_text())
    assert _REPORT_FIELDS.issubset(set(rep))     # all schedule fields present
    assert rep["precompute_obfuscation_schedule"] is True
    assert rep["schedule_slots_precomputed"] >= 6
    assert rep["schedule_slots_consumed"] >= 1
    assert rep["schedule_used_for_metadata_only"] is True
    assert rep["online_remask_still_performed"] is True
    assert rep["schedule_secret_leaked_to_gpu"] is False
    assert rep["max_new_tokens"] == 6 and rep["seq_len"] == 1024
    assert "schedule_stats" in rep
    # responses written
    assert (tmp_path / "r.jsonl").is_file()


# ---- 5. backward compat ---------------------------------------------------

def test_backward_compat_without_precompute(tmp_path) -> None:
    mod = _load("ifeval2", "scripts/run_ifeval_generation.py")
    ds = tmp_path / "p.jsonl"
    ds.write_text(json.dumps({"id": "a", "prompt": "Hello."}) + "\n",
                  encoding="utf-8")
    rep_j = tmp_path / "rep.json"
    rc = _main(mod, ["x", "--input-jsonl", str(ds), "--backend",
                     "plaintext_local", "--mock-runtime", "--max-new-tokens", "4",
                     "--output-response-jsonl", str(tmp_path / "r.jsonl"),
                     "--output-report-json", str(rep_j)])
    assert rc == 0
    rep = json.loads(rep_j.read_text())
    # disabled: schedule fields present but defaulted -> old path unchanged
    assert rep["precompute_obfuscation_schedule"] is False
    assert rep["schedule_slots_precomputed"] == 0
    assert rep["schedule_slots_consumed"] == 0
    assert rep["schedule_precompute_latency_s"] is None
    assert rep["dry_run"] is True and rep["paper_ready"] is False


# ---- to_disk: secrets never persisted by default --------------------------

def test_to_disk_no_secret_persist(tmp_path) -> None:
    s = _sched(max_steps=4, secrets=True)
    p = s.to_disk(tmp_path / "sched.json")
    data = json.loads(Path(p).read_text())
    assert data["secret_tensors_persisted"] is False
    audit_schedule_trusted_only(  # reloaded public surface is still secret-free
        s)
    # explicit secret persist refused unless allow_secret_persist
    with pytest.raises(ScheduleSecretLeak):
        s.to_disk(tmp_path / "s2.json", save_secret_tensors=True)


# ---- probe script (mock) --------------------------------------------------

def test_probe_script_mock(tmp_path) -> None:
    mod = _load("probe", "scripts/run_precomputed_schedule_probe.py")
    oj = tmp_path / "probe.json"
    rc = _main(mod, ["x", "--hidden-size", "32", "--seq-len", "16",
                     "--max-new-tokens", "6", "--dtype", "float32", "--device",
                     "cpu", "--precompute-obfuscation-schedule",
                     "--schedule-max-steps", "16", "--schedule-seed", "2035",
                     "--mock-runtime", "--report-schedule-stats",
                     "--output-json", str(oj)])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["audit_passed"] is True
    assert r["schedule_secret_leaked_to_gpu"] is False
    assert r["gpu_request_contains_schedule_secret"] is False
    assert r["schedule_slots_precomputed"] == 16
    assert r["schedule_slots_consumed"] == 6
    assert r["boundary_calls"] == 1 + 2 * 5      # prefill + 5 decode steps
    assert r["gpu_calls"] == 6
    assert r["schedule_used_for_metadata_only"] is True
    assert r["online_remask_still_performed"] is True


def test_probe_requires_mock_runtime(tmp_path) -> None:
    mod = _load("probe2", "scripts/run_precomputed_schedule_probe.py")
    rc = _main(mod, ["x", "--output-json", str(tmp_path / "x.json")])
    assert rc == 3                               # refuses non-mock locally
