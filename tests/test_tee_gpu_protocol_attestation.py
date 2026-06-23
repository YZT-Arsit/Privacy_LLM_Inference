"""Tests for the trusted-boundary TDX attestation + runtime-hash binding.

Verifies the evidence-checking logic against the shape of the real Alibaba Cloud
TDX evidence (tee=tdx, debug=false, 3-part signed JWT, mr_td, report_data bound
to the runtime hash). numpy/stdlib only -- no quote is generated here.

Run: python -m pytest tests/test_tee_gpu_protocol_attestation.py -q
"""

from __future__ import annotations

import json

from pllo.protocol import (
    attest_boundary,
    build_trusted_boundary_manifest,
    compute_runtime_hash,
    compute_runtime_hash_from_manifest,
    runtime_report_data_hex,
    verify_evidence,
    write_runtime_hash,
    write_runtime_manifest,
)
from pllo.protocol.attestation import REPORT_DATA_BYTES

# The verified mr_td from the Alibaba Cloud TDX VM (SHA-384, 48 bytes).
REAL_MR_TD = ("e0199499baacb2e4f4bc73046f25bedf674d42defbe4e854242bd6554a9d155e"
              "df7f3bff8e6202e63ed230e59ab2568a")

COMPONENTS = {
    "component": "pllo-tee-boundary", "stage": "8.5", "version": "1",
    "boundary_backend": "process", "gpu_backend": "mock",
    "mask_mode": "signed_permutation", "hidden_size": 64, "vocab_size": 500,
}


def _evidence(report_data: str, *, debug=False, tee="tdx", jwt="aaa.bbb.ccc",
              mr_td=REAL_MR_TD) -> dict:
    return {"tee": tee, "tdx": {"td_attributes": {"debug": debug}},
            "mr_td": mr_td, "report_data": report_data, "jwt": jwt}


def test_runtime_hash_deterministic_and_64_bytes() -> None:
    h1 = compute_runtime_hash(COMPONENTS)
    h2 = compute_runtime_hash(dict(reversed(list(COMPONENTS.items()))))
    assert h1 == h2                                   # key order independent
    assert len(h1) == 64
    rd = runtime_report_data_hex(h1)
    assert len(rd) == REPORT_DATA_BYTES * 2 == 128
    # different identity -> different hash
    assert compute_runtime_hash({**COMPONENTS, "version": "2"}) != h1


def test_verify_evidence_success_with_binding_and_mr_td() -> None:
    h = compute_runtime_hash(COMPONENTS)
    ev = verify_evidence(_evidence(runtime_report_data_hex(h)), h,
                         expected_mr_td=REAL_MR_TD)
    assert ev.tee_type == "tdx"
    assert ev.verified is True
    assert ev.runtime_hash_bound is True
    assert ev.debug is False
    assert ev.jwt_present is True and ev.jwt_parts == 3
    assert ev.mr_td_match is True
    assert ev.quote_available is True
    assert ev.quote_status == "verified"


def test_verify_evidence_wrong_report_data_fails_binding() -> None:
    h = compute_runtime_hash(COMPONENTS)
    bad = runtime_report_data_hex(compute_runtime_hash(
        {**COMPONENTS, "version": "999"}))
    ev = verify_evidence(_evidence(bad), h)
    assert ev.runtime_hash_bound is False
    assert ev.verified is False
    assert ev.quote_status == "evidence_check_failed"


def test_verify_evidence_debug_true_fails() -> None:
    h = compute_runtime_hash(COMPONENTS)
    ev = verify_evidence(_evidence(runtime_report_data_hex(h), debug=True), h)
    assert ev.debug is True
    assert ev.verified is False


def test_verify_evidence_jwt_must_have_three_parts() -> None:
    h = compute_runtime_hash(COMPONENTS)
    ev = verify_evidence(_evidence(runtime_report_data_hex(h),
                                   jwt="header.payload"), h)
    assert ev.jwt_parts == 2
    assert ev.verified is False


def test_verify_evidence_mr_td_mismatch_fails() -> None:
    h = compute_runtime_hash(COMPONENTS)
    ev = verify_evidence(_evidence(runtime_report_data_hex(h), mr_td="deadbeef"),
                         h, expected_mr_td=REAL_MR_TD)
    assert ev.mr_td_match is False
    assert ev.verified is False


def test_attest_boundary_offline_is_simulated_but_exposes_runtime_hash() -> None:
    # No evidence + no TDX guest device in CI -> simulated, not verified, but the
    # runtime hash the boundary would bind is still reported.
    ev = attest_boundary(COMPONENTS, tdx_guest_device="/nonexistent/tdx_guest")
    assert ev.tee_type == "simulated"
    assert ev.verified is False
    assert ev.quote_status == "simulated_no_tdx"
    assert ev.runtime_hash_hex == runtime_report_data_hex(
        compute_runtime_hash(COMPONENTS))
    assert ev.tdx_guest_device_present is False


def test_attest_boundary_with_evidence_dict_verifies() -> None:
    h = compute_runtime_hash(COMPONENTS)
    ev = attest_boundary(COMPONENTS,
                         evidence=_evidence(runtime_report_data_hex(h)),
                         expected_mr_td=REAL_MR_TD)
    assert ev.verified is True
    assert ev.mr_td == REAL_MR_TD


# --- manifest-based runtime hash (binds the actual code artifact) -----------

MD = {"protocol_version": "8.5", "boundary_backend": "process",
      "allowed_gpu_backend": "mock", "expected_mr_td": None}


def _tmp_boundary(tmp_path):
    (tmp_path / "a.py").write_text("print('boundary a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('boundary b')\n", encoding="utf-8")
    return ["a.py", "b.py"]


def test_manifest_same_files_and_metadata_identical_hash() -> None:
    h1 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=MD))
    h2 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=dict(MD)))
    assert h1 == h2
    assert len(h1) == 128                              # 64-byte SHA-512 hex


def test_manifest_changes_when_trusted_source_changes(tmp_path) -> None:
    paths = _tmp_boundary(tmp_path)
    h1 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    # rebuild unchanged -> identical
    h1b = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    assert h1 == h1b
    # change one trusted-boundary source file -> hash changes
    (tmp_path / "a.py").write_text("print('boundary a MODIFIED')\n",
                                   encoding="utf-8")
    h2 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    assert h2 != h1


def test_prompt_and_input_do_not_change_hash(tmp_path) -> None:
    paths = _tmp_boundary(tmp_path)
    h1 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    # prompts / input_ids / generated tokens are simply never inputs to the
    # recipe, so building again (same code + metadata) is invariant to them.
    h2 = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    assert h1 == h2
    blob = json.dumps(build_trusted_boundary_manifest(paths, MD, base=tmp_path))
    assert "Explain why privacy matters" not in blob   # sample prompt text


def test_manifest_contains_no_secrets_or_plaintext() -> None:
    m = build_trusted_boundary_manifest(metadata=MD)
    assert set(m) == {"kind", "manifest_version", "files", "runtime_identity",
                      "excludes"}
    # file entries carry only path/digest/size metadata
    for e in m["files"]:
        assert set(e) <= {"path", "sha256", "size", "missing"}
    # runtime identity carries only public identity, no per-request/secret keys
    forbidden = {"raw_prompt", "prompt", "input_ids", "generated_token_ids",
                 "recovered_logits", "residual_perm", "vocab_perm",
                 "vocab_scale", "residual_signs", "seed"}
    assert not (set(m["runtime_identity"]) & forbidden)
    # the excludes list documents what is intentionally NOT measured
    for name in ("raw_prompt", "input_ids", "generated_token_ids",
                 "recovered_logits", "mask_secrets", "model_weights"):
        assert name in m["excludes"]


def test_default_manifest_measures_expected_trusted_files() -> None:
    m = build_trusted_boundary_manifest(metadata=MD)
    paths = {e["path"] for e in m["files"]}
    assert "src/pllo/protocol/attestation.py" in paths
    assert "src/pllo/protocol/tee_gpu_messages.py" in paths
    assert "src/pllo/protocol/security_audit.py" in paths
    assert "scripts/run_tee_gpu_protocol_demo.py" in paths
    assert any(p.startswith("src/pllo/tee/") for p in paths)
    # all measured files exist (real digests, nothing missing)
    for e in m["files"]:
        assert e.get("missing") is not True
        assert e["sha256"] and len(e["sha256"]) == 64


def test_write_runtime_manifest_and_hash(tmp_path) -> None:
    mpath = tmp_path / "manifest.json"
    hpath = tmp_path / "runtime_hash.txt"
    manifest = write_runtime_manifest(mpath, metadata=MD)
    rh = write_runtime_hash(hpath, metadata=MD)
    assert mpath.exists() and hpath.exists()
    assert json.loads(mpath.read_text())["kind"] == "trusted_boundary_manifest"
    assert rh == compute_runtime_hash_from_manifest(manifest)
    assert hpath.read_text().strip() == rh


def test_attest_boundary_with_manifest_hash_binds_and_verifies() -> None:
    manifest = build_trusted_boundary_manifest(metadata=MD)
    rh = compute_runtime_hash_from_manifest(manifest)        # hex
    ev = attest_boundary(runtime_hash=rh,
                         evidence=_evidence(rh), expected_mr_td=REAL_MR_TD)
    assert ev.runtime_hash_hex == rh
    assert ev.runtime_hash_bound is True
    assert ev.verified is True


def test_attest_boundary_requires_a_hash_source() -> None:
    import pytest
    with pytest.raises(ValueError):
        attest_boundary()
