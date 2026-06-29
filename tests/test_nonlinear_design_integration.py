"""Cross-cutting tests for the two-design nonlinear dimension.

Covers the foundation + integration points that the per-module agent tests do
not: registry normalization, manifest records the design, package verifier
mismatch, LoRA/base design-compatibility, claim-validator cross-backend refusal,
preflight per-backend, runtime-hash binds the design, and stale (design-A)
attestation evidence failing for design B. stdlib only -- no torch / GPU / model.

Run: python -m pytest tests/test_nonlinear_design_integration.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments import nonlinear_designs as nd  # noqa: E402


# ---------------------------------------------------------------------------
# 1. registry normalization
# ---------------------------------------------------------------------------

def test_registry_normalization_and_aliases() -> None:
    assert nd.list_nonlinear_backends() == [
        "current", "trusted_shortcut", "A_rightmul"]
    assert nd.normalize_nonlinear_backend("amulet_migrated") == "trusted_shortcut"
    assert nd.normalize_nonlinear_backend("TEE-Shortcut-Nonlinear") == \
        "trusted_shortcut"
    assert nd.normalize_nonlinear_backend("baseline_nonlinear") == "current"
    assert nd.normalize_nonlinear_backend("design_b") == "trusted_shortcut"
    assert nd.op_backend_for_design("trusted_shortcut") == "amulet_migrated"
    assert nd.op_backend_for_design("current") == "current"
    assert nd.parse_nonlinear_backends("current, amulet_migrated, current") == \
        ["current", "trusted_shortcut"]
    try:
        nd.normalize_nonlinear_backend("does_not_exist")
        assert False, "expected UnknownNonlinearBackend"
    except nd.UnknownNonlinearBackend:
        pass


def test_design_hashes_distinct_and_stable() -> None:
    a = nd.nonlinear_design_metadata_hash("current")
    b = nd.nonlinear_design_metadata_hash("trusted_shortcut")
    assert a != b
    assert a == nd.nonlinear_design_metadata_hash("current")          # stable
    assert b == nd.nonlinear_design_metadata_hash("amulet_migrated")  # alias
    fields = nd.nonlinear_design_report_fields("amulet_migrated")
    assert fields["nonlinear_backend"] == "trusted_shortcut"
    assert fields["nonlinear_design_metadata_hash"] == b


# ---------------------------------------------------------------------------
# 2. manifest records the design + validate detects a tampered hash
# ---------------------------------------------------------------------------

def _manifest(backend):
    from pllo.deployment import build_manifest
    return build_manifest(
        package_type="base_model", model_name="m", model_path_or_id="/m",
        num_layers=1, dtype="bfloat16", nonlinear_backend=backend,
        created_by="test", build_command="python build.py",
        shard_index=[{"name": "head", "path": "head.pt",
                      "sha256": "0" * 64, "nbytes": 4}])


def test_manifest_records_design_and_validates() -> None:
    from pllo.deployment import validate_manifest
    m = _manifest("trusted_shortcut")
    assert m.nonlinear_backend == "trusted_shortcut"
    assert m.nonlinear_design_metadata_hash == \
        nd.nonlinear_design_metadata_hash("trusted_shortcut")
    assert m.nonlinear_design_version == "1.0"
    assert m.build_command == "python build.py"
    ok, problems = validate_manifest(m)
    assert ok, problems


def test_manifest_tampered_design_hash_fails_validation() -> None:
    from pllo.deployment import validate_manifest
    m = _manifest("current")
    m.nonlinear_design_metadata_hash = "deadbeef" * 8      # wrong
    ok, problems = validate_manifest(m)
    assert not ok
    assert any("nonlinear_design_metadata_hash" in p for p in problems)


# ---------------------------------------------------------------------------
# 3. verifier catches a nonlinear mismatch (base + LoRA/base compat)
# ---------------------------------------------------------------------------

def test_check_nonlinear_backend_mismatch() -> None:
    from pllo.deployment import check_nonlinear_backend
    m = _manifest("current")
    ok, _ = check_nonlinear_backend(m, "current")
    assert ok
    ok2, probs = check_nonlinear_backend(m, "trusted_shortcut")
    assert not ok2 and probs
    # alias still matches
    ok3, _ = check_nonlinear_backend(_manifest("trusted_shortcut"),
                                     "amulet_migrated")
    assert ok3


def test_lora_base_nonlinear_compat() -> None:
    from pllo.deployment import check_lora_base_nonlinear_compatibility
    base = _manifest("current")
    lora_ok = _manifest("current")
    lora_bad = _manifest("trusted_shortcut")
    ok, _ = check_lora_base_nonlinear_compatibility(lora_ok, base)
    assert ok
    bad, probs = check_lora_base_nonlinear_compatibility(lora_bad, base)
    assert not bad and probs


# ---------------------------------------------------------------------------
# 7. claim validator refuses cross-backend evidence
# ---------------------------------------------------------------------------

def _pairwise_report(backend, executed=True):
    r = {"stage": "e9_pairwise_utility_preservation", "nonlinear_backend":
         backend, "utility_preserved": True, "paper_ready": True,
         "dry_run": False, "dataset": "mmlu", "delta_abs": 0.0}
    # trusted_shortcut needs genuine Amulet-lift execution evidence to count;
    # current is executed by definition (trusted-boundary inline).
    if backend == "trusted_shortcut" and executed:
        r.update({"nonlinear_op_backend": "amulet_migrated",
                  "amulet_lift_executed": True, "lifted_nonlinear_ops_count": 56,
                  "lift_k": 4, "lifted_gpu_bytes": 123456})
    return r


def test_claim_validator_cross_backend_refusal() -> None:
    from pllo.experiments.claim_validator import build_claim_report
    results = [{"file": "cur.json", "report": _pairwise_report("current")}]
    # required claim tagged for the OTHER design must NOT be supported
    rep = build_claim_report(results, required_claims=[
        "public_benchmark_utility_preserved[trusted_shortcut]"])
    assert rep["all_required_supported"] is False
    assert "public_benchmark_utility_preserved[current]" in \
        rep["backend_tagged_supported"]
    assert "public_benchmark_utility_preserved[trusted_shortcut]" not in \
        rep["backend_tagged_supported"]
    assert rep["nonlinear_designs_evaluated"] == ["current"]
    assert "trusted_shortcut" in rep["nonlinear_designs_not_evaluated"]
    # the same claim tagged for the matching design IS supported
    rep2 = build_claim_report(results, required_claims=[
        "public_benchmark_utility_preserved[current]"])
    assert rep2["all_required_supported"] is True


def test_claim_validator_both_designs_supported() -> None:
    from pllo.experiments.claim_validator import build_claim_report
    results = [{"file": "c.json", "report": _pairwise_report("current")},
               {"file": "t.json", "report": _pairwise_report("trusted_shortcut")}]
    rep = build_claim_report(results)
    assert rep["both_nonlinear_designs_supported"] is True
    assert set(rep["nonlinear_designs_evaluated"]) == {"current",
                                                       "trusted_shortcut"}
    assert rep["trusted_shortcut_executed_in_real_path"] is True


def test_claim_validator_refuses_tag_only_trusted_shortcut() -> None:
    from pllo.experiments.claim_validator import build_claim_report
    # tag-only trusted_shortcut (no Amulet-lift execution evidence)
    results = [{"file": "t.json",
                "report": _pairwise_report("trusted_shortcut", executed=False)}]
    rep = build_claim_report(results, required_claims=[
        "public_benchmark_utility_preserved[trusted_shortcut]"])
    assert rep["all_required_supported"] is False
    assert "public_benchmark_utility_preserved[trusted_shortcut]" not in \
        rep["backend_tagged_supported"]
    assert rep["trusted_shortcut_executed_in_real_path"] is False
    assert rep["trusted_shortcut_tag_only_files"] == ["t.json"]
    assert any("trusted_shortcut_not_executed_in_real_path" in w
               for w in rep["warnings"])


# ---------------------------------------------------------------------------
# 8. preflight reports blockers separately by backend
# ---------------------------------------------------------------------------

def _load_script(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_matrix_per_backend(tmp_path) -> None:
    pf = _load_script("pf", "scripts/preflight_real_eval.py")
    rep = pf.run_preflight_matrix({
        "backend": "tdx_attested_remote",
        "nonlinear_backends": "current,trusted_shortcut",
        "output_dir": str(tmp_path)})
    assert rep["nonlinear_backends"] == ["current", "trusted_shortcut"]
    assert set(rep["preflight_passed_by_backend"]) == {"current",
                                                       "trusted_shortcut"}
    # no resources provided -> each design has its own blockers
    assert rep["blockers_by_backend"]["current"]
    assert rep["blockers_by_backend"]["trusted_shortcut"]
    assert rep["preflight_passed"] is False


# ---------------------------------------------------------------------------
# 9 + 10. runtime hash binds the design; design-A evidence fails for design B
# ---------------------------------------------------------------------------

def _runtime_hash(backend):
    from pllo.protocol.attestation import (
        boundary_manifest_metadata, build_trusted_boundary_manifest,
        compute_runtime_hash_from_manifest)
    md = boundary_manifest_metadata("process", "qwen7b_folded_package",
                                    "mrtd", nonlinear_backend=backend)
    return compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=md))


def test_runtime_hash_differs_by_design() -> None:
    rc = _runtime_hash("current")
    rt = _runtime_hash("trusted_shortcut")
    assert rc != rt
    assert _runtime_hash("amulet_migrated") == rt           # alias identical


def test_stale_design_A_evidence_fails_for_design_B() -> None:
    from pllo.protocol.attestation import verify_evidence
    rh_current = _runtime_hash("current")
    rh_ts = _runtime_hash("trusted_shortcut")
    # evidence bound to design A (current)
    evidence = {"tee": "tdx", "tdx": {"td_attributes": {"debug": False}},
                "jwt": "a.b.c", "report_data": rh_current, "mr_td": "MRTD"}
    # used for design A -> binds
    res_a = verify_evidence(evidence, bytes.fromhex(rh_current),
                            expected_mr_td="MRTD")
    assert res_a.runtime_hash_bound is True
    assert res_a.verified is True
    # used for design B -> NOT bound (cannot replay design A evidence)
    res_b = verify_evidence(evidence, bytes.fromhex(rh_ts),
                            expected_mr_td="MRTD")
    assert res_b.runtime_hash_bound is not True
    assert res_b.verified is False
