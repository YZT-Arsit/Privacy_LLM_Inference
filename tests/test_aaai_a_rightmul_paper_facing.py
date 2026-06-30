"""AAAI A_rightmul + Qwen2.5-7B paper-facing contract.

Covers the fixes that make the A_rightmul mainline genuinely paper-facing (not a
metadata tag):

* the compatible-mask verifier runs on REAL generated masks: a
  pairwise_complex_scaling / dense mask FAILS, a signed-permutation +
  pairwise_rotation mask PASSES and binds an audit into the package manifest;
* the worker refuses A_rightmul without a verified compatible-mask package, and
  the runner refuses to execute A_rightmul when compatible_masks_verified is False;
* the e2e validator fails an A_rightmul report missing compatible_masks_verified;
* the generation paper-facing gate rejects max_new_tokens!=512, --disable-eos-stop,
  and non-A_rightmul designs, accepts a fully-satisfying report, and the e2e TDX
  checks still reject simulated evidence + accept real bound evidence;
* GSM8K answer extraction / exact match + MT-Bench two-turn loading.

Run: python -m pytest tests/test_aaai_a_rightmul_paper_facing.py -q
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from pllo.ops.compatible_mask_verify import (  # noqa: E402
    CompatibleMaskViolation, verify_session_compatible_masks)


def _load_e2e():
    spec = importlib.util.spec_from_file_location(
        "e2e_val", REPO_ROOT / "scripts" / "validate_tee_gpu_e2e.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _session(*, mask_mode, attn_family, n_layers=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
    from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig
    mc = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=n_layers, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=128,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(mc).eval()
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=8, max_new_tokens=1,
        device="cpu", dtype="float32", folding_dtype="float32",
        folded_weight_device="cpu", mlp_down_chunk_size=64, seed=2035,
        use_linear_boundary_pad=True, linear_pad_scale=0.3,
        mask_mode=mask_mode, attention_mask_family=attn_family)
    return MaskedQwenSession(model, mc, cfg), mc


# ---------------------------------------------------------------------------
# 1 + 2. compatible-mask verifier on REAL generated masks
# ---------------------------------------------------------------------------

def test_complex_scaling_attention_mask_fails_verification() -> None:
    sess, _ = _session(mask_mode="signed_permutation",
                       attn_family="pairwise_complex_scaling")
    with pytest.raises(CompatibleMaskViolation):
        verify_session_compatible_masks(sess)


def test_dense_residual_mask_fails_verification() -> None:
    sess, _ = _session(mask_mode="dense_orthogonal",
                       attn_family="pairwise_rotation")
    with pytest.raises(CompatibleMaskViolation):
        verify_session_compatible_masks(sess)


def test_compatible_family_passes_with_all_required_fields() -> None:
    sess, _ = _session(mask_mode="signed_permutation",
                       attn_family="pairwise_rotation")
    audit = verify_session_compatible_masks(sess)
    for k in ("compatible_masks_verified", "residual_mask_is_signed_permutation",
              "attention_qk_scores_preserved", "swiglu_shared_channel_permutation",
              "arbitrary_dense_mask_rejected"):
        assert audit[k] is True


# ---------------------------------------------------------------------------
# worker + runner enforcement
# ---------------------------------------------------------------------------

def _pkg(nonlinear_backend, *, compatible):
    from pllo.deployment import (FoldedPackageWriter, build_manifest,
                                 write_manifest)
    sess, _ = _session(
        mask_mode="signed_permutation",
        attn_family=("pairwise_rotation" if compatible
                     else "pairwise_complex_scaling"))
    cmf = cma = None
    if compatible:
        cma = sess.verify_compatible_masks()
        cmf = "signed_permutation_residual+pairwise_rotation+shared_swiglu"
    pkg = tempfile.mkdtemp()
    w = FoldedPackageWriter(pkg)
    for ell in range(2):
        w.add_shard(f"layer_{ell:03d}", sess.export_folded_layer_tensors(ell))
    w.add_shard("head", sess.export_folded_head_tensors())
    write_manifest(build_manifest(
        package_type="base_model", model_name="tiny", model_path_or_id=None,
        num_layers=2, dtype="float32", nonlinear_backend=nonlinear_backend,
        created_by="test", shard_index=w.shard_index, hidden_size=64,
        vocab_size=128, mask_schedule_id="s",
        compatible_mask_family=cmf, compatible_mask_audit=cma), pkg)
    return pkg


def _init_worker(pkg, nonlinear_backend):
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import BoundaryInitRequest
    be = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=pkg, device="cpu", dtype="float32",
        nonlinear_backend=nonlinear_backend)
    be.init(BoundaryInitRequest(
        session_id="t", hidden_size=64, vocab_size=128, num_layers=2,
        dtype="float32", gpu_backend="qwen7b_folded_package"))
    return be


def test_worker_refuses_a_rightmul_without_verified_package() -> None:
    pkg = _pkg("A_rightmul", compatible=False)
    with pytest.raises(RuntimeError, match="compatible_masks_verified"):
        _init_worker(pkg, "A_rightmul")


def test_worker_accepts_a_rightmul_with_verified_package() -> None:
    pkg = _pkg("A_rightmul", compatible=True)
    be = _init_worker(pkg, "A_rightmul")
    assert be.compatible_masks_verified is True
    desc = be.describe()
    assert desc["compatible_masks_verified"] is True


def test_runner_refuses_a_rightmul_when_not_verified() -> None:
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    r = make_folded_nonlinear_runner("A_rightmul", compatible_masks_verified=False)
    with pytest.raises(CompatibleMaskViolation):
        r.silu(torch.randn(2, 4))


def test_runner_executes_a_rightmul_when_verified_and_stamps_fields() -> None:
    from pllo.deployment.folded_nonlinear import make_folded_nonlinear_runner
    audit = {"residual_mask_is_signed_permutation": True,
             "attention_qk_scores_preserved": True,
             "swiglu_shared_channel_permutation": True,
             "arbitrary_dense_mask_rejected": True}
    r = make_folded_nonlinear_runner(
        "A_rightmul", compatible_masks_verified=True, compatible_mask_audit=audit)
    out = r.silu(torch.randn(2, 4))
    assert out.shape == (2, 4)
    ev = r.execution_evidence()
    assert ev["compatible_masks_verified"] is True
    assert ev["residual_mask_is_signed_permutation"] is True
    assert ev["arbitrary_dense_mask_rejected"] is True
    assert ev["nonlinear_trusted_calls"] == 0


# ---------------------------------------------------------------------------
# 3 + 7 + 8. e2e validator
# ---------------------------------------------------------------------------

def _arm_report(**over):
    r = {"stage": "qwen7b_folded_package_prefill_probe",
         "nonlinear_backend": "A_rightmul",
         "nonlinear_op_backend": "compatible_right_multiply",
         "nonlinear_real_path_executed": True,
         "nonlinear_execution_status": "right_multiply_on_accelerator",
         "right_multiply_nonlinear_executed": True,
         "right_multiply_nonlinear_ops_count": 4,
         "trusted_nonlinear_ops_count": 0, "nonlinear_trusted_calls": 0,
         "nonlinear_single_tee_entry_exit": True,
         "compatible_masks_verified": True,
         "residual_mask_is_signed_permutation": True,
         "attention_qk_scores_preserved": True,
         "swiglu_shared_channel_permutation": True,
         "arbitrary_dense_mask_rejected": True}
    r.update(over)
    return r


def test_e2e_fails_a_rightmul_missing_compatible_masks_verified() -> None:
    e2e = _load_e2e()
    rep = e2e.validate([{"file": "x.json", "report": {
        k: v for k, v in _arm_report().items()
        if k != "compatible_masks_verified"}}],
        expected_mr_td=None, require=["nonlinear_exec"])
    assert rep["passed"] is False
    assert any("compatible_masks_verified" in c["check"]
               for c in rep["failed_checks"])


def test_e2e_passes_a_rightmul_with_compatible_masks_verified() -> None:
    e2e = _load_e2e()
    rep = e2e.validate([{"file": "x.json", "report": _arm_report()}],
                       expected_mr_td=None, require=["nonlinear_exec"])
    assert rep["passed"] is True


def test_e2e_rejects_simulated_tdx() -> None:
    e2e = _load_e2e()
    sim = {"file": "sim.json", "report": {
        "tee": "tdx", "simulated_unsigned": True, "paper_facing": False,
        "report_data": "ab", "runtime_hash": "ab", "jwt": "a.b.c",
        "tdx": {"td_attributes": {"debug": False}}}}
    rep = e2e.validate([sim], expected_mr_td=None, require=["tdx_quote"])
    assert rep["passed"] is False


def test_e2e_accepts_real_bound_tdx_evidence() -> None:
    e2e = _load_e2e()
    real = {"file": "tdx.json", "report": {
        "tee": "tdx", "nonlinear_backend": "A_rightmul",
        "runtime_hash_binds_nonlinear_backend": True,
        "report_data": "abcd", "runtime_hash": "abcd", "jwt": "a.b.c",
        "mr_td": "MRTD", "tdx": {"td_attributes": {"debug": False}}}}
    rep = e2e.validate([real], expected_mr_td="MRTD", require=["tdx_quote"])
    assert rep["passed"] is True


# ---------------------------------------------------------------------------
# 4 + 5 + 6. generation paper-facing gate
# ---------------------------------------------------------------------------

def _full_gen_report(**over):
    r = {"backend": "folded_remote", "nonlinear_backend": "A_rightmul",
         "seq_len": 1024, "max_new_tokens": 512, "stop_on_eos": True,
         "dry_run": False, "tdx_boundary_client": True,
         "full_model_weights_loaded_in_trusted_runtime": False,
         "attestation_evidence_attached": True,
         "attestation_runtime_hash_binds_nonlinear_backend": True,
         "h800_worker_health": {"ok": True},
         "h800_worker_tee_used_on_gpu": False,
         "nonlinear_trusted_calls": 0, "compatible_masks_verified": True,
         "schedule_full_coverage_verified": True}
    r.update(over)
    return r


def test_paper_facing_generation_full_report_passes() -> None:
    from pllo.benchmarks.paper_facing_generation import (
        is_paper_facing_generation, paper_facing_generation_violations)
    assert paper_facing_generation_violations(_full_gen_report()) == []
    assert is_paper_facing_generation(_full_gen_report()) is True


@pytest.mark.parametrize("over,needle", [
    ({"max_new_tokens": 256}, "max_new_tokens"),
    ({"stop_on_eos": False}, "stop_on_eos"),
    ({"nonlinear_backend": "current"}, "nonlinear_backend"),
    ({"nonlinear_backend": "amulet_secure_R"}, "nonlinear_backend"),
    ({"nonlinear_backend": "trusted_shortcut"}, "nonlinear_backend"),
    ({"compatible_masks_verified": False}, "compatible_masks_verified"),
    ({"nonlinear_trusted_calls": 2}, "nonlinear_trusted_calls"),
    ({"schedule_full_coverage_verified": False}, "schedule_full_coverage"),
    ({"attestation_runtime_hash_binds_nonlinear_backend": False},
     "attestation_runtime_hash_binds_nonlinear_backend"),
])
def test_paper_facing_generation_rejects(over, needle) -> None:
    from pllo.benchmarks.paper_facing_generation import (
        paper_facing_generation_violations)
    viol = paper_facing_generation_violations(_full_gen_report(**over))
    assert any(needle in v for v in viol), viol


def _run_ifeval(extra_args):
    """Invoke run_ifeval_generation.py and return (rc, stderr)."""
    tmp = tempfile.mkdtemp()
    ife = Path(tmp) / "in.jsonl"
    ife.write_text('{"id":"a","prompt":"hi"}\n', encoding="utf-8")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_ifeval_generation.py"),
           "--input-jsonl", str(ife),
           "--output-response-jsonl", str(Path(tmp) / "r.jsonl"),
           "--output-report-json", str(Path(tmp) / "rep.json")] + extra_args
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    import os
    env = {**os.environ, **env}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


_BASE = ["--backend", "folded_remote", "--nonlinear-backend", "A_rightmul",
         "--seq-len", "1024", "--max-new-tokens", "512", "--require-real",
         "--tdx-boundary-client", "--attestation-evidence-json", "/x.json",
         "--paper-facing-generation"]


def test_ifeval_gate_rejects_wrong_max_new_tokens() -> None:
    rc, err = _run_ifeval([a if a != "512" else "256" for a in _BASE])
    assert rc == 3 and "max_new_tokens" in err


def test_ifeval_gate_rejects_disable_eos_stop() -> None:
    rc, err = _run_ifeval(_BASE + ["--disable-eos-stop"])
    assert rc == 3 and "disable-eos-stop" in err


@pytest.mark.parametrize("design", ["current", "amulet_secure_R",
                                    "trusted_shortcut"])
def test_ifeval_gate_rejects_non_a_rightmul_design(design) -> None:
    args = [(design if a == "A_rightmul" else a) for a in _BASE]
    rc, err = _run_ifeval(args)
    assert rc == 3 and "A_rightmul" in err


# ---------------------------------------------------------------------------
# GSM8K / MT-Bench dataset helpers
# ---------------------------------------------------------------------------

def test_gsm8k_extract_and_exact_match() -> None:
    from pllo.benchmarks.generation_datasets import (
        extract_gsm8k_answer, gsm8k_exact_match)
    assert extract_gsm8k_answer("the answer is #### 42") == "42"
    assert extract_gsm8k_answer("first 7 then finally 18") == "18"
    assert extract_gsm8k_answer("$1,234.0 total") == "1234"
    assert gsm8k_exact_match("reasoning ... #### 4", "4") is True
    assert gsm8k_exact_match("nope 5", "4") is False


def test_gsm8k_loader_parses_gold(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_gsm8k
    p = tmp_path / "g.jsonl"
    p.write_text('{"question":"2+2?","answer":"so 4\\n#### 4"}\n',
                 encoding="utf-8")
    rows = load_gsm8k(p)
    assert rows[0]["reference"] == "4"
    assert "####" in rows[0]["prompt"]      # instruction prepended


def test_mt_bench_loader_two_turn(tmp_path) -> None:
    from pllo.benchmarks.generation_datasets import load_mt_bench
    p = tmp_path / "m.jsonl"
    p.write_text('{"question_id":"q1","category":"math",'
                 '"turns":["a","b"]}\n', encoding="utf-8")
    rows = load_mt_bench(p)
    assert rows[0]["turns"] == ["a", "b"]
    assert rows[0]["prompt"] == "a"
    assert rows[0]["category"] == "math"
