"""Paper-facing nonlinear contract: A_rightmul compatible-mask verification,
amulet_secure_R secure-R execution, validator paper-facing gating + trusted-call
rejection, TDX nonlinear binding, and linear-pad paper-facing coverage.

CPU + float64; no CUDA, no real 7B weights.
"""

from __future__ import annotations

import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch.nn.functional as F  # noqa: E402

from pllo.experiments import nonlinear_designs as nd  # noqa: E402
from pllo.experiments.claim_validator import build_claim_report  # noqa: E402
from pllo.nonlinear.amulet_secure_r_backend import (  # noqa: E402
    AmuletSecureRNonlinearBackend, SecureRViolation, secure_r_activation)
from pllo.ops.compatible_mask_verify import (  # noqa: E402
    CompatibleMaskViolation, verify_compatible_masks, is_signed_permutation,
    assert_signed_permutation, assert_qk_compatible,
    assert_shared_channel_permutation)


# ---------------------------------------------------------------------------
# A_rightmul compatible-mask correctness + incompatible dense mask must FAIL
# ---------------------------------------------------------------------------


def _signed_perm(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.float64) * 2 - 1)
    m = torch.zeros(n, n, dtype=torch.float64)
    m[torch.arange(n), perm] = signs
    return m


def test_compatible_masks_accept_valid_families() -> None:
    n_res = _signed_perm(8, 1)
    assert is_signed_permutation(n_res)
    # Q/K compatible: Nk = (Nq^{-1})^T  => Nq Nk^T = I
    nq = _signed_perm(8, 2)
    nk = torch.linalg.inv(nq).T.contiguous()
    p = torch.eye(8, dtype=torch.float64)[torch.randperm(
        8, generator=torch.Generator().manual_seed(3))]
    out = verify_compatible_masks(n_res=n_res, nq=nq, nk=nk, p_gate=p, p_up=p)
    assert out["compatible_masks_verified"] is True
    assert out["residual_mask_is_signed_permutation"] is True
    assert out["attention_qk_scores_preserved"] is True
    assert out["swiglu_shared_channel_permutation"] is True


def test_incompatible_dense_mask_must_fail() -> None:
    dense = torch.randn(8, 8, dtype=torch.float64)        # arbitrary dense mask
    with pytest.raises(CompatibleMaskViolation):
        assert_signed_permutation(dense, name="N_res")
    # Q/K that do NOT preserve scores
    nq = torch.randn(8, 8, dtype=torch.float64)
    nk = torch.randn(8, 8, dtype=torch.float64)
    with pytest.raises(CompatibleMaskViolation):
        assert_qk_compatible(nq, nk)
    # distinct (non-shared) channel masks for SwiGLU
    eye = torch.eye(6, dtype=torch.float64)
    p_gate = eye[torch.randperm(6, generator=torch.Generator().manual_seed(4))]
    p_up = eye[torch.randperm(6, generator=torch.Generator().manual_seed(5))]
    with pytest.raises(CompatibleMaskViolation):
        assert_shared_channel_permutation(p_gate, p_up)


def test_compatible_mask_verifier_actually_proves_commutation() -> None:
    # signed-permutation residual mask: RMSNorm core commutes with it
    n_res = _signed_perm(16, 7)
    x = torch.randn(4, 16, dtype=torch.float64)

    def rms(z):
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(rms(x @ n_res), rms(x) @ n_res, atol=1e-10)


# ---------------------------------------------------------------------------
# amulet_secure_R: no visible selector / no zero decoy / no trusted calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("act", [F.gelu, F.silu])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_secure_r_exact_and_no_trusted(act, k) -> None:
    x = torch.randn(5, 12, dtype=torch.float64)
    out, audit = secure_r_activation(x, act, k=k, seed=11)
    assert torch.allclose(out, act(x), atol=1e-9)
    assert audit["trusted_calls"] == 0
    assert audit["zero_decoys"] is False
    assert audit["selector_visible_to_gpu"] is False
    assert audit["valid_channel_observable"] is False
    assert audit["secure_R_enabled"] is True


def test_secure_r_backend_zero_trusted_calls() -> None:
    be = AmuletSecureRNonlinearBackend(lift_k=2, seed=1)
    for r in (be.silu(torch.randn(3, 8, dtype=torch.float64)),
              be.gelu(torch.randn(3, 8, dtype=torch.float64)),
              be.softmax(torch.randn(2, 2, 4, 4, dtype=torch.float64)),
              be.rmsnorm(torch.randn(3, 8, dtype=torch.float64))):
        assert r.trusted_calls == 0
        assert r.tee_used_on_gpu is False
    assert be.security_status == "claimed_under_secure_R_assumption"


def test_secure_r_violation_raises_on_zero_decoy() -> None:
    from pllo.nonlinear.amulet_secure_r_backend import _assert_secure_rbar
    rbar = torch.tensor([[1.0, 0.0], [0.7, 0.9]], dtype=torch.float64)  # zero decoy
    with pytest.raises(SecureRViolation):
        _assert_secure_rbar(rbar, 0, 0)
    onehot = torch.tensor([[1.0, 1.0], [0.7, 0.9]], dtype=torch.float64)
    with pytest.raises(SecureRViolation):
        _assert_secure_rbar(onehot, 0, 0)


# ---------------------------------------------------------------------------
# validator: reject current in paper-facing mode + reject trusted_calls>0
# ---------------------------------------------------------------------------


def _decode(backend, **extra):
    r = {"stage": "qwen7b_folded_remote_package_decode",
         "nonlinear_backend": backend, "package_backed_decode": True,
         "tokens_exact_match": True, "paper_ready": True, "dry_run": False,
         "tee_real": True, "attestation_verified": True,
         "runtime_hash_bound": True, "gpu_worker_remote": True}
    r.update(extra)
    return r


def _pairwise(backend, **extra):
    r = {"stage": "e9_pairwise_utility_preservation",
         "nonlinear_backend": backend, "utility_preserved": True,
         "paper_ready": True, "dry_run": False, "dataset": "mmlu",
         "delta_abs": 0.0}
    r.update(extra)
    return r


def test_validator_rejects_current_in_paper_facing_mode() -> None:
    rep = build_claim_report(
        [{"file": "cur.json", "report": _pairwise("current")}],
        required_claims=["public_benchmark_utility_preserved[current]"],
        paper_facing=True)
    assert rep["all_required_supported"] is False
    assert any("non_paper_facing_design_current" in rsn
               for o in rep["overclaim_risks"] for rsn in o.get("reasons", []))
    # without paper_facing the legacy design is still tracked (back-compat)
    rep2 = build_claim_report(
        [{"file": "cur.json", "report": _pairwise("current")}])
    assert "public_benchmark_utility_preserved[current]" in \
        rep2["backend_tagged_supported"]


def test_validator_rejects_trusted_calls_gt_zero() -> None:
    # an A_rightmul-tagged decode that (illegally) reports a trusted nonlinear
    # crossing must be rejected and never back a per-backend claim.
    bad = _decode("A_rightmul", nonlinear_op_backend="compatible_right_multiply",
                  right_multiply_nonlinear_executed=True,
                  right_multiply_nonlinear_ops_count=10,
                  trusted_nonlinear_ops_count=3, nonlinear_trusted_calls=3)
    rep = build_claim_report(
        [{"file": "bad.json", "report": bad}],
        required_claims=[
            "no_lora_tdx_attested_remote_package_decode[A_rightmul]"])
    assert rep["all_required_supported"] is False
    assert "bad.json" in rep["nonlinear_trusted_calls_violation_files"]
    assert rep["nonlinear_trusted_calls_clean"] is False


def test_validator_clean_a_rightmul_supported() -> None:
    good = _pairwise("A_rightmul",
                     nonlinear_op_backend="compatible_right_multiply",
                     right_multiply_nonlinear_executed=True,
                     right_multiply_nonlinear_ops_count=10,
                     trusted_nonlinear_ops_count=0, nonlinear_trusted_calls=0)
    rep = build_claim_report(
        [{"file": "good.json", "report": good}],
        required_claims=["public_benchmark_utility_preserved[A_rightmul]"],
        paper_facing=True)
    assert rep["all_required_supported"] is True
    assert rep["nonlinear_trusted_calls_clean"] is True


# ---------------------------------------------------------------------------
# TDX: nonlinear backend binds into the runtime hash
# ---------------------------------------------------------------------------


def test_tdx_runtime_hash_binds_nonlinear_backend() -> None:
    from pllo.protocol.attestation import (
        boundary_manifest_metadata, build_trusted_boundary_manifest,
        compute_runtime_hash_from_manifest)

    def _hash(nb):
        md = boundary_manifest_metadata(
            "process", "qwen7b", None, nonlinear_backend=nb)
        return compute_runtime_hash_from_manifest(
            build_trusted_boundary_manifest(metadata=md))

    h_arm = _hash("A_rightmul")
    h_sec = _hash("amulet_secure_R")
    h_none = compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(metadata=boundary_manifest_metadata(
            "process", "qwen7b", None)))
    assert h_arm != h_sec               # different designs -> different hashes
    assert h_arm != h_none and h_sec != h_none
    assert _hash("A_rightmul") == h_arm  # deterministic / same design -> same


# ---------------------------------------------------------------------------
# integration: amulet_secure_R + A_rightmul really execute in the worker path
# ---------------------------------------------------------------------------


def _tiny_pkg(nonlinear_backend):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pllo.deployment import (FoldedPackageWriter, build_manifest,
                                 write_manifest)
    from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
    from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig

    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(mc).eval()
    ids = torch.randint(0, mc.vocab_size, (1, 8))
    cfg = MemoryOptimizedConfig(
        num_layers=2, batch_size=1, seq_len=8, max_new_tokens=2, device="cpu",
        dtype="float32", folding_dtype="float64", folded_weight_device="cpu",
        mlp_down_chunk_size=64, seed=2035, use_linear_boundary_pad=True,
        linear_pad_scale=0.3)
    sess = MaskedQwenSession(model, mc, cfg)
    pkg = tempfile.mkdtemp()
    w = FoldedPackageWriter(pkg)
    for ell in range(2):
        w.add_shard(f"layer_{ell:03d}", sess.export_folded_layer_tensors(ell))
    w.add_shard("head", sess.export_folded_head_tensors())
    write_manifest(build_manifest(
        package_type="base_model", model_name="tiny", model_path_or_id=None,
        num_layers=2, dtype="float64", nonlinear_backend=nonlinear_backend,
        created_by="test", shard_index=w.shard_index, hidden_size=128,
        vocab_size=256, mask_schedule_id="s-n2"), pkg)
    return sess, pkg, ids, mc


def _run_layer(nonlinear_backend):
    from pllo.hf_wrappers.qwen_memory_optimized import _cfg_to
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import BoundaryInitRequest

    sess, pkg, ids, mc = _tiny_pkg(nonlinear_backend)
    be = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=pkg, device="cpu", dtype="float64",
        nonlinear_backend=nonlinear_backend)
    be.init(BoundaryInitRequest(
        session_id="t", hidden_size=128, vocab_size=256, num_layers=2,
        dtype="float64", gpu_backend="qwen7b_folded_package"))
    cfg_c = _cfg_to(sess.layer_configs[0], torch.device("cpu"))
    h = sess.mask_embeddings(ids).double()
    out = be.run_single_layer_prefill(h, 0, cfg_c, sess._cos, sess._sin,
                                      float(sess.eps))
    return be, out["y_tilde"]


def test_amulet_secure_r_worker_executes_and_exact() -> None:
    be, y_sec = _run_layer("amulet_secure_R")
    ev = be.nonlinear_execution_evidence()
    assert ev["nonlinear_op_backend"] == "amulet_secure_R"
    assert ev["secure_right_multiply_executed"] is True
    assert ev["secure_right_multiply_ops_count"] > 0
    assert ev["trusted_nonlinear_ops_count"] == 0
    assert ev["nonlinear_trusted_calls"] == 0
    assert ev["nonlinear_execution_status"] == "secure_right_multiply_on_accelerator"
    assert ev["zero_decoys"] is False
    assert ev["selector_visible_to_gpu"] is False
    desc = be.describe()
    assert desc["nonlinear_backend"] == "amulet_secure_R"
    assert desc["tee_used_on_gpu"] is False
    # exact vs current (only accounting differs)
    _bec, y_cur = _run_layer("current")
    assert torch.allclose(y_sec.double(), y_cur.double(), atol=1e-9)


def test_no_tag_only_secure_r_report() -> None:
    be, _ = _run_layer("amulet_secure_R")
    rep = nd.nonlinear_design_report_fields("amulet_secure_R")
    rep.update(be.nonlinear_execution_evidence())
    rep["stage"] = "qwen7b_folded_package_prefill_probe"
    assert nd.nonlinear_tag_only(rep) is False
    assert nd.report_has_secure_right_multiply_execution(rep) is True


# ---------------------------------------------------------------------------
# end-to-end validator (scripts/validate_tee_gpu_e2e.py)
# ---------------------------------------------------------------------------


def _load_e2e():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "e2e_val", repo / "scripts" / "validate_tee_gpu_e2e.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_e2e_validator_passes_clean_and_fails_dirty() -> None:
    e2e = _load_e2e()
    arm = {"file": "arm.json", "report": {
        "stage": "qwen7b_folded_package_prefill_probe",
        "nonlinear_backend": "A_rightmul",
        "nonlinear_op_backend": "compatible_right_multiply",
        "nonlinear_real_path_executed": True,
        "nonlinear_execution_status": "right_multiply_on_accelerator",
        "right_multiply_nonlinear_executed": True,
        "right_multiply_nonlinear_ops_count": 4,
        "trusted_nonlinear_ops_count": 0, "nonlinear_trusted_calls": 0,
        "nonlinear_single_tee_entry_exit": True,
        "linear_pad_coverage": {m: True for m in (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj", "lm_head")},
        "base_linear_pad_all_modules_covered": True}}
    rep = e2e.validate([arm], expected_mr_td=None,
                       require=["nonlinear_exec", "linear_pad"])
    assert rep["passed"] is True

    # a report tagged paper-facing but with a trusted nonlinear call -> fail
    dirty = {"file": "bad.json", "report": dict(arm["report"],
             nonlinear_trusted_calls=2, trusted_nonlinear_ops_count=2)}
    rep2 = e2e.validate([dirty], expected_mr_td=None,
                        require=["nonlinear_exec"])
    assert rep2["passed"] is False

    # a legacy current report is not paper-facing -> fail
    legacy = {"file": "cur.json", "report": {
        "nonlinear_backend": "current", "nonlinear_real_path_executed": True,
        "nonlinear_execution_status": "executed_trusted_boundary_inline",
        "trusted_nonlinear_ops_count": 4, "nonlinear_trusted_calls": 4}}
    rep3 = e2e.validate([legacy], expected_mr_td=None,
                        require=["nonlinear_exec"])
    assert rep3["passed"] is False


def test_e2e_validator_rejects_simulated_tdx() -> None:
    e2e = _load_e2e()
    sim = {"file": "sim.json", "report": {
        "tee": "tdx", "simulated_unsigned": True, "paper_facing": False,
        "report_data": "ab", "runtime_hash": "ab",
        "jwt": "a.b.c", "tdx": {"td_attributes": {"debug": False}}}}
    rep = e2e.validate([sim], expected_mr_td=None, require=["tdx_quote"])
    assert rep["passed"] is False
