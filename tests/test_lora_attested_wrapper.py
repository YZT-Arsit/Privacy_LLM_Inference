"""Gap 3: the folded-LoRA remote decode wrapper attaches + verifies attestation.

Uses a stub demo (canned report) so no model / worker / torch is needed; the REAL
attestation verifier still runs over real boundary-source-file hashes, so a stale
report_data is correctly rejected (binding fails). stdlib only.

Run: python -m pytest tests/test_lora_attested_wrapper.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
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


def _canned_report():
    return {
        "stage": "qwen7b_folded_remote_package_decode",
        "boundary_backend": "process", "gpu_backend": "qwen7b_folded_package",
        "lora_enabled": True, "folded_lora_loaded": True,
        "folded_lora_valid": True, "worker_has_raw_lora": False,
        "package_backed_prefill": True, "package_backed_decode": True,
        "reference_token_ids": None, "package_token_ids": [1, 2, 3, 4],
        "tokens_exact_match": None, "token_match_rate": None,
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "audit_passed": True, "max_new_tokens": 4,
    }


def _fake_demo():
    """A demo namespace with the real attestation helpers + a canned builder."""
    real = _load("demo_real", "scripts/run_tee_gpu_protocol_demo.py")
    ns = types.SimpleNamespace(
        build_remote_folded_package_decode_report=lambda a, b, **k:
            dict(_canned_report()),
        attach_attestation=real.attach_attestation,
        boundary_manifest_metadata=real.boundary_manifest_metadata,
        write_runtime_manifest=real.write_runtime_manifest,
        _write_remote_folded_md=real._write_remote_folded_md)
    return ns


def _run_wrapper(tmp_path, extra, monkeypatch):
    mod = _load("loraw", "scripts/run_qwen7b_lora_folded_remote_decode_probe.py")
    monkeypatch.setattr(mod, "_load_demo", _fake_demo)
    oj = tmp_path / "out.json"
    argv = ["x", "--gpu-worker-url", "http://127.0.0.1:9",
            "--embedding-path", str(tmp_path), "--input-ids", "1,2,3,4",
            "--seq-len", "4", "--max-new-tokens", "4",
            "--output-json", str(oj)] + extra
    rc = _main(mod, argv)
    return rc, json.loads(oj.read_text())


def test_no_evidence_makes_no_attestation_claim(tmp_path, monkeypatch) -> None:
    rc, rep = _run_wrapper(tmp_path, [], monkeypatch)
    assert "attestation" not in rep
    assert "boundary_attested" not in rep
    assert rc == 0                                    # lora ok, no attest demanded


def test_stale_evidence_attaches_and_fails(tmp_path, monkeypatch) -> None:
    ev = tmp_path / "evidence.json"
    ev.write_text(json.dumps({
        "tee": "tdx", "td_attributes": {"debug": False}, "jwt": "a.b.c",
        "report_data": "00" * 64, "mr_td": "MRTD"}))
    rc, rep = _run_wrapper(
        tmp_path, ["--attestation-evidence", str(ev), "--expected-mr-td",
                   "MRTD"], monkeypatch)
    # attestation fields are attached when evidence is supplied
    for k in ("attestation", "boundary_tee_type", "boundary_attested",
              "runtime_hash", "expected_runtime_hash", "evidence_report_data",
              "runtime_hash_bound", "binding_mismatch_reason", "mr_td"):
        assert k in rep, "missing attestation field %r" % k
    # stale report_data -> binding fails -> wrapper FAILS
    assert rep["runtime_hash_bound"] is not True
    assert rep["boundary_attested"] is False
    assert rep["binding_mismatch_reason"]
    assert rc == 1
