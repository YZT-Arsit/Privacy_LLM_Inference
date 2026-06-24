"""Tests for the current-boundary TDX attestation evidence generator (--simulate).

Off-TDX, ``--simulate`` exercises the full plumbing with an unsigned token. The
binding logic IS the bug being fixed, so we assert: the generated runtime hash
equals what the demo verifies, the evidence verifies as bound + attested, all
artifacts are written, and feeding the evidence through the demo's
``attach_attestation`` yields ``runtime_hash_bound`` / ``boundary_attested`` true
(and false when the expected_mr_td differs -> stale-binding detection).

Run: python -m pytest tests/test_generate_tdx_attestation_evidence.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.protocol.attestation import (  # noqa: E402
    boundary_manifest_metadata,
    boundary_runtime_hash,
)

MRTD = ("e0199499baacb2e4f4bc73046f25bedf674d42defbe4e854242bd6554a9d155e"
        "df7f3bff8e6202e63ed230e59ab2568a")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen(tmp_path, mr_td=MRTD):
    gen = _load("gentdx", "scripts/generate_tdx_attestation_evidence.py")
    out_dir = tmp_path / "att"
    out_ev = tmp_path / "evidence.json"
    argv = ["prog", "--boundary-backend", "process", "--gpu-backend", "mock",
            "--expected-mr-td", mr_td, "--simulate",
            "--output-dir", str(out_dir), "--output-evidence", str(out_ev)]
    old = sys.argv
    try:
        sys.argv = argv
        rc = gen.main()
    finally:
        sys.argv = old
    return rc, out_dir, out_ev


def test_simulate_generates_bound_evidence_and_artifacts(tmp_path) -> None:
    rc, out_dir, out_ev = _gen(tmp_path)
    assert rc == 0
    # all required artifacts written
    for f in ("runtime_hash.hex", "trusted_boundary_manifest.json",
              "td_quote.bin", "attestation.jwt", "claims.json",
              "attest_request.json", "evidence.json"):
        assert (out_dir / f).exists(), f
    ev = json.loads(out_ev.read_text())
    rh = (out_dir / "runtime_hash.hex").read_text().strip()
    assert ev["tee"] == "tdx"
    assert ev["report_data"] == rh
    assert ev["mr_td"] == MRTD
    assert ev["simulated_unsigned"] is True
    assert ev["jwt"].count(".") == 2          # 3-part token


def test_runtime_hash_matches_what_demo_verifies(tmp_path) -> None:
    _, out_dir, _ = _gen(tmp_path)
    rh = (out_dir / "runtime_hash.hex").read_text().strip()
    # identical recipe to the demo's attach_attestation
    expected = boundary_runtime_hash(
        boundary_manifest_metadata("process", "mock", MRTD))
    assert rh == expected


def test_evidence_passes_demo_attach_attestation(tmp_path) -> None:
    _, _, out_ev = _gen(tmp_path)
    demo = _load("rtgpd", "scripts/run_tee_gpu_protocol_demo.py")
    report = {"boundary_backend": "process", "gpu_backend": "mock"}
    demo.attach_attestation(report, evidence=str(out_ev), expected_mr_td=MRTD)
    assert report["boundary_tee_type"] == "tdx"
    assert report["boundary_attested"] is True
    assert report["runtime_hash_bound"] is True
    assert report["mr_td"] == MRTD
    assert report["expected_runtime_hash"] == report["evidence_report_data"]
    assert report["binding_mismatch_reason"] is None


def test_wrong_expected_mr_td_breaks_binding(tmp_path) -> None:
    # evidence bound to MRTD; verifying with a DIFFERENT mr_td recomputes a
    # different runtime hash -> stale/wrong binding is detected.
    _, _, out_ev = _gen(tmp_path)
    other = "ab" * 48
    demo = _load("rtgpd2", "scripts/run_tee_gpu_protocol_demo.py")
    report = {"boundary_backend": "process", "gpu_backend": "mock"}
    demo.attach_attestation(report, evidence=str(out_ev), expected_mr_td=other)
    assert report["runtime_hash_bound"] is False
    assert report["boundary_attested"] is False
    assert report["binding_mismatch_reason"]


def test_exit_nonzero_when_no_jwt_source_offtdx(tmp_path) -> None:
    # without --simulate and without any quote/jwt source off-TDX, generation
    # cannot proceed -> non-zero (the operator must supply a real source).
    gen = _load("gentdx2", "scripts/generate_tdx_attestation_evidence.py")
    argv = ["prog", "--boundary-backend", "process", "--gpu-backend", "mock",
            "--expected-mr-td", MRTD, "--quote-file", str(tmp_path / "nope.bin"),
            "--output-dir", str(tmp_path / "a"),
            "--output-evidence", str(tmp_path / "e.json")]
    old = sys.argv
    try:
        sys.argv = argv
        raised = False
        try:
            gen.main()
        except Exception:                      # noqa: BLE001 - missing quote file
            raised = True
    finally:
        sys.argv = old
    assert raised
