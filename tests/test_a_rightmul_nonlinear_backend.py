"""A_rightmul (compatible right-multiply) nonlinear backend.

Covers:
  A. registry/design recognise the A_rightmul aliases + op backend.
  B. the folded nonlinear runner EXECUTES right-multiply and never falls back to
     the 'current' trusted path (zero trusted nonlinear crossings).
  C. an execution-bearing A_rightmul report passes the non-tag-only validation,
     and a tag-only one is flagged.
  D. B/Amulet behaviour is unchanged (trusted_shortcut -> amulet_migrated).
  E. the real Qwen folded worker (1-layer) emits measured A_rightmul evidence and
     is numerically identical to the 'current' backend.

CPU + float64; no CUDA, no real 7B weights.
"""

from __future__ import annotations

import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
from pllo.experiments import nonlinear_designs as nd
from pllo.nonlinear.registry import available_backends, make_nonlinear_backend

_ALIASES = ["A_rightmul", "a_rightmul", "right_multiply", "RIGHT-MULTIPLY",
            "compatible_right_multiply", "compatible_nonlinear_islands"]


# ---------------------------------------------------------------------------
# A. registry / design recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", _ALIASES)
def test_A_aliases_normalize(alias) -> None:
    assert nd.normalize_nonlinear_backend(alias) == "A_rightmul"


def test_A_op_backend_and_registry() -> None:
    assert nd.op_backend_for_design("A_rightmul") == "compatible_right_multiply"
    assert "compatible_right_multiply" in available_backends()
    be = make_nonlinear_backend("compatible_right_multiply")
    assert be.name == "compatible_right_multiply"
    # security under development (NOT a formal claim)
    assert be.security_status == "under_development"
    assert be.security_claim_status == "under_development"
    # B/Amulet must remain mapped to amulet_migrated (unchanged)
    assert nd.op_backend_for_design("trusted_shortcut") == "amulet_migrated"
    assert nd.normalize_nonlinear_backend("amulet_migrated") == "trusted_shortcut"


def test_A_real_path_executes_not_prototype() -> None:
    assert nd.real_path_executes("A_rightmul") is True
    assert nd.real_path_execution_status("A_rightmul") == \
        "right_multiply_on_accelerator"


# ---------------------------------------------------------------------------
# B. runner executes right-multiply, no fallback to current
# ---------------------------------------------------------------------------


def test_B_runner_no_fallback_to_current() -> None:
    r = make_folded_nonlinear_runner("A_rightmul")
    # the rightmul backend is built; the trusted/amulet paths are NOT used
    assert r._rightmul is not None
    assert r._amulet is None
    assert r.op_backend == "compatible_right_multiply"

    x = torch.randn(2, 4, 8, dtype=torch.float64)
    scores = torch.randn(2, 2, 4, 4, dtype=torch.float64)
    out_silu = r.silu(x)
    r.gelu(x)
    r.rmsnorm_core(x, 1e-6)
    r.softmax(scores)
    ev = r.execution_evidence()

    assert ev["nonlinear_backend"] == "A_rightmul"
    assert ev["nonlinear_op_backend"] == "compatible_right_multiply"
    assert ev["nonlinear_real_path_executed"] is True
    assert ev["right_multiply_nonlinear_executed"] is True
    assert ev["right_multiply_nonlinear_ops_count"] == 4
    # the defining property: ZERO trusted nonlinear crossings (single TEE I/O)
    assert ev["trusted_nonlinear_ops_count"] == 0
    assert ev["nonlinear_trusted_calls"] == 0
    assert ev["nonlinear_execution_status"] == "right_multiply_on_accelerator"
    assert ev["nonlinear_masking_mode"] == \
        "compatible_right_multiply_or_permutation"
    assert ev["linear_boundary_pad"] is True

    # numerically identical to the 'current' backend (only accounting differs)
    rc = make_folded_nonlinear_runner("current")
    assert torch.equal(out_silu, rc.silu(x))
    assert torch.equal(r.softmax(scores), rc.softmax(scores))


def test_B_current_and_trusted_shortcut_unchanged() -> None:
    rc = make_folded_nonlinear_runner("current")
    rc.silu(torch.randn(2, 8, dtype=torch.float64))
    evc = rc.execution_evidence()
    assert evc["nonlinear_op_backend"] == "current"
    assert evc["trusted_nonlinear_ops_count"] == 1   # trusted-island accounting
    assert evc["nonlinear_execution_status"] == "executed_trusted_boundary_inline"

    rb = make_folded_nonlinear_runner("trusted_shortcut")
    assert rb._amulet is not None and rb._rightmul is None
    rb.silu(torch.randn(2, 8, dtype=torch.float64))
    evb = rb.execution_evidence()
    assert evb["nonlinear_op_backend"] == "amulet_migrated"
    assert evb["amulet_lift_executed"] is True


# ---------------------------------------------------------------------------
# C. non-tag-only validation
# ---------------------------------------------------------------------------


def test_C_execution_report_not_tag_only() -> None:
    rep = nd.nonlinear_design_report_fields("A_rightmul")
    rep["stage"] = "qwen7b_folded_package_prefill_probe"
    # capability-only (no measured run yet) -> flagged tag-only
    assert nd.nonlinear_tag_only(rep) is True
    assert nd.report_has_right_multiply_execution(rep) is False

    # stamp measured execution evidence (what the worker/probe does)
    r = make_folded_nonlinear_runner("A_rightmul")
    r.silu(torch.randn(2, 8, dtype=torch.float64))
    rep.update(r.execution_evidence())
    assert nd.report_has_right_multiply_execution(rep) is True
    assert nd.report_has_real_nonlinear_execution(rep) is True
    assert nd.nonlinear_tag_only(rep) is False
    # a build/setup report (no execution stage) is never flagged tag-only
    build_rep = nd.nonlinear_design_report_fields("A_rightmul")
    build_rep["stage"] = "folded_package_build"
    assert nd.nonlinear_tag_only(build_rep) is False


def test_C_tampered_trusted_count_is_tag_only() -> None:
    # a report claiming A_rightmul but with a nonzero trusted nonlinear count is
    # NOT genuine right-multiply execution
    rep = nd.nonlinear_design_report_fields("A_rightmul")
    rep["stage"] = "qwen7b_folded_package_decode_probe"
    rep.update({"right_multiply_nonlinear_executed": True,
                "right_multiply_nonlinear_ops_count": 3,
                "trusted_nonlinear_ops_count": 2, "nonlinear_trusted_calls": 2})
    assert nd.report_has_right_multiply_execution(rep) is False
    assert nd.nonlinear_tag_only(rep) is True


# ---------------------------------------------------------------------------
# E. real Qwen folded worker (1 layer) emits A_rightmul evidence + is exact
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


def test_E_worker_emits_a_rightmul_evidence_and_is_exact() -> None:
    be, y_arm = _run_layer("A_rightmul")
    ev = be.nonlinear_execution_evidence()
    assert ev["nonlinear_op_backend"] == "compatible_right_multiply"
    assert ev["right_multiply_nonlinear_executed"] is True
    assert ev["right_multiply_nonlinear_ops_count"] > 0
    assert ev["trusted_nonlinear_ops_count"] == 0
    assert ev["nonlinear_trusted_calls"] == 0
    assert ev["nonlinear_execution_status"] == "right_multiply_on_accelerator"
    desc = be.describe()
    assert desc["nonlinear_backend"] == "A_rightmul"
    assert desc["nonlinear_op_backend"] == "compatible_right_multiply"
    assert desc["tee_used_on_gpu"] is False

    # A_rightmul output is bit-identical to current (only accounting differs)
    _be_c, y_cur = _run_layer("current")
    assert torch.allclose(y_arm.double(), y_cur.double(), atol=1e-10, rtol=1e-10)
