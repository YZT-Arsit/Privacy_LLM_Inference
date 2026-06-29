"""Linear-boundary additive padding in the PRODUCTION Qwen folded-package path.

These tests FAIL if the production Qwen path remains mask-only: they build a tiny
Qwen2 folded package WITH ``--linear-boundary-pad`` using the SAME production
functions (``MaskedQwenSession.export_folded_layer_tensors`` /
``export_folded_head_tensors``) the real builder uses, load it in the real worker
backend (``Qwen7BFoldedPackageGpuBackend``), execute the real folded runtime
(``folded_worker.apply_folded_prefill/decode/head`` -> ``_masked_attention`` /
``_masked_mlp`` / ``_linear``), and assert the audit invariants + recovered-logit /
token correctness.

CPU + float64 (strict tolerance); no CUDA, no real 7B weights.

    PYTHONPATH=$PWD/src pytest tests/test_qwen_linear_boundary_pad.py -q
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from pllo.deployment import (
    FoldedPackageWriter,
    build_manifest,
    verify_package,
    write_manifest,
)
from pllo.deployment.folded_package import forbidden_tensor_names
from pllo.deployment.folded_worker import (
    apply_folded_decode,
    apply_folded_head,
    apply_folded_prefill,
)
from pllo.deployment.linear_boundary_pad import (
    ALL_PAD_MODULES,
    fold_linear_with_input_pad,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig, _cfg_to
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest

_RAW_SECRET_KEYS = (
    "raw_pad", "raw_t", "raw_t_in", "raw_mask", "raw_n", "raw_n_in", "raw_n_out",
    "plaintext_x", "plaintext_hidden", "raw_input", "labels", "optimizer_state",
    "raw_lora",
)


# ---------------------------------------------------------------------------
# Test 1: algebraic Linear correctness (raw T / N_in / N_out)
# ---------------------------------------------------------------------------


def _rand_orth(n, dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=dtype))
    return q


@pytest.mark.parametrize("bs,din,dout", [(4, 16, 16), (4, 16, 32), (4, 32, 16)])
@pytest.mark.parametrize("bias", [True, False])
def test_1_algebraic_linear_correctness(bs, din, dout, bias):
    torch.manual_seed(bs + din + dout + int(bias))
    X = torch.randn(bs, din, dtype=torch.float64)
    W = torch.randn(din, dout, dtype=torch.float64)
    b = torch.randn(dout, dtype=torch.float64) if bias else None
    n_in = _rand_orth(din)
    n_out = _rand_orth(dout)
    n_in_inv = torch.linalg.inv(n_in)
    n_out_inv = torch.linalg.inv(n_out)
    T = 0.3 * torch.randn(din, dtype=torch.float64)

    w_tilde, b_tilde, c_pad = fold_linear_with_input_pad(W, b, n_in_inv, n_out, T)
    x_tilde = (X - T) @ n_in                       # GPU-visible Linear input view
    y_tilde = x_tilde @ w_tilde + c_pad
    if b_tilde is not None:
        y_tilde = y_tilde + b_tilde
    y_ref_masked = (X @ W + (0.0 if b is None else b)) @ n_out

    assert torch.allclose(y_tilde, y_ref_masked, atol=1e-10, rtol=1e-10)
    recovered = y_tilde @ n_out_inv                 # recover() == Y N_out @ N_out^-1
    assert torch.allclose(recovered, X @ W + (0.0 if b is None else b),
                          atol=1e-10, rtol=1e-10)
    # the pad genuinely perturbs the matmul operand (not a no-op view)
    assert not torch.allclose(x_tilde, X @ n_in, atol=1e-6)


# ---------------------------------------------------------------------------
# Tiny production folded package (with / without pad) -- the real functions
# ---------------------------------------------------------------------------


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=3, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _build_pkg(use_pad: bool, *, n_layers=3, seq=8, scale=0.3):
    """Build a tiny folded package via the PRODUCTION session export + writer.
    Returns (session, pkg_dir, ids, model, mc, all_tensor_names)."""
    model, mc = _tiny_qwen()
    ids = torch.randint(0, mc.vocab_size, (1, seq))
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=seq, max_new_tokens=4,
        device="cpu", dtype="float32", folding_dtype="float64",
        folded_weight_device="cpu", mlp_down_chunk_size=64, seed=2035,
        use_linear_boundary_pad=use_pad, linear_pad_scale=scale)
    sess = MaskedQwenSession(model, mc, cfg)
    pkg = tempfile.mkdtemp()
    writer = FoldedPackageWriter(pkg)
    names: list[str] = []
    for ell in range(n_layers):
        t = sess.export_folded_layer_tensors(ell)
        names += list(t)
        writer.add_shard(f"layer_{ell:03d}", t)
    ht = sess.export_folded_head_tensors()
    names += list(ht)
    writer.add_shard("head", ht)
    man = build_manifest(
        package_type="base_model", model_name="tiny", model_path_or_id=None,
        num_layers=n_layers, dtype="float64", nonlinear_backend="current",
        created_by="test", shard_index=writer.shard_index, hidden_size=128,
        vocab_size=256, mask_schedule_id="session-seed2035-n%d" % n_layers)
    write_manifest(man, pkg)
    return sess, pkg, ids, model, mc, names


def _load_backend(pkg, n_layers, mc):
    be = Qwen7BFoldedPackageGpuBackend(folded_package_path=pkg, device="cpu",
                                       dtype="float64")
    be.init(BoundaryInitRequest(
        session_id="t", hidden_size=int(mc.hidden_size),
        vocab_size=int(mc.vocab_size), num_layers=n_layers, dtype="float64",
        gpu_backend="qwen7b_folded_package"))
    return be


def _report_with_pad():
    sess, pkg, ids, model, mc, names = _build_pkg(use_pad=True)
    be = _load_backend(pkg, 3, mc)
    return be.describe(), names


# ---------------------------------------------------------------------------
# Test 2: no pad enters nonlinear core
# ---------------------------------------------------------------------------


def test_2_no_pad_in_nonlinear_core():
    report, _ = _report_with_pad()
    assert report["pad_enters_rmsnorm_core"] is False
    assert report["pad_enters_rope_core"] is False
    assert report["pad_enters_softmax"] is False
    assert report["pad_enters_swiglu_core"] is False
    assert report["persistent_residual_additive_pad"] is False
    assert report["nonlinear_masking_mode"] == \
        "compatible_right_multiply_or_permutation"


# ---------------------------------------------------------------------------
# Test 3: no intermediate TEE boundary crossing
# ---------------------------------------------------------------------------


def test_3_no_intermediate_tee_boundary():
    report, _ = _report_with_pad()
    assert report["intermediate_tee_boundary_calls_per_layer"] == 0
    assert report["semantic_input_boundary_calls"] == 1
    assert report["semantic_final_logits_boundary_calls"] == 1
    assert report["pad_scope"] == "linear_boundary_local"


# ---------------------------------------------------------------------------
# Test 4: production Qwen pad audit
# ---------------------------------------------------------------------------


def test_4_production_pad_audit():
    report, _ = _report_with_pad()
    assert report["qwen_production_path_uses_linear_input_pad"] is True
    assert report["linear_boundary_pad_enabled"] is True
    assert report["linear_input_form"] == "(X - T) N_in"
    assert report["linear_output_form"] == "Y N_out"
    assert report["linear_pad_compensation_formula"] == "C_pad = T W N_out"
    assert report["online_extra_matmul_for_pad"] == 0
    assert report["c_pad_materialization"] == "precomputed"


def test_4b_mask_only_path_reports_disabled():
    # a package built WITHOUT pad must NOT claim pad (guards against tag-only)
    _sess, pkg, _ids, _model, mc, names = _build_pkg(use_pad=False)
    be = _load_backend(pkg, 3, mc)
    report = be.describe()
    assert report["linear_boundary_pad_enabled"] is False
    assert report["qwen_production_path_uses_linear_input_pad"] is False
    assert not any("xpad" in n or "cpad" in n for n in names)


# ---------------------------------------------------------------------------
# Test 5: per-module coverage
# ---------------------------------------------------------------------------


def test_5_per_module_coverage():
    report, names = _report_with_pad()
    for name in ALL_PAD_MODULES:
        assert report["linear_pad_coverage"][name] is True, name
    assert report["linear_pad_all_modules_covered"] is True
    # the actual shard tensors back the claim (xpad+cpad present per family)
    for fam in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown", "w_lm"):
        assert f"{fam}_xpad_tilde" in names, fam
        assert f"{fam}_cpad_tilde" in names, fam


# ---------------------------------------------------------------------------
# Test 6: no raw secret visible to GPU
# ---------------------------------------------------------------------------


def test_6_no_raw_secret_visible_to_gpu():
    report, names = _report_with_pad()
    # GPU-visible package tensor names pass the forbidden-substring screen
    # (mask/perm/n_in/n_out/scale/...) -- this is what actually crosses to the GPU.
    assert forbidden_tensor_names(names) == []
    # no explicit raw-secret key appears among the GPU-visible TENSOR names
    # (audit FIELD names like ``raw_pad_visible_to_gpu`` are flags, not artifacts).
    lowered = " ".join(names).lower()
    for key in _RAW_SECRET_KEYS:
        assert key not in lowered, key
    assert report["raw_pad_visible_to_gpu"] is False
    assert report["raw_mask_visible_to_gpu"] is False
    assert report["c_pad_visible_to_gpu"] is True       # C_pad IS allowed visible
    # the package itself re-verifies clean (no secret tensor names on disk)
    # (reuse a freshly built package dir)
    _s, pkg, _i, _m, mc, _n = _build_pkg(use_pad=True)
    vrep = verify_package(pkg)
    assert vrep["package_valid"] is True
    assert vrep["forbidden_fields_found"] == []
    assert vrep["contains_mask_secrets"] is False


# ---------------------------------------------------------------------------
# Test 7: logits / token correctness (recovered == plaintext, tokens exact)
# ---------------------------------------------------------------------------


def _folded_greedy(sess, pkg, mc, ids, steps):
    n = sess.n
    cfg_c = _cfg_to(sess.layer_configs[0], torch.device("cpu"))
    h = sess.mask_embeddings(ids).double()
    out = apply_folded_prefill(h, pkg, n, cfg_c, sess._cos, sess._sin,
                               float(sess.eps))
    kv = out["kv"]
    rec = sess.recover(apply_folded_head(out["y_tilde"], pkg, float(sess.eps)))
    tok = rec[:, -1, :].argmax(-1)
    gen = [int(tok.item())]
    pos = int(ids.shape[1])
    for _ in range(steps - 1):
        xt = sess.mask_token_embedding(tok).double()
        out = apply_folded_decode(xt, pkg, kv, pos, n, cfg_c, sess._cos,
                                  sess._sin, float(sess.eps))
        kv = out["kv"]
        rec = sess.recover(apply_folded_head(out["y_tilde"], pkg,
                                             float(sess.eps)))
        tok = rec[:, -1, :].argmax(-1)
        gen.append(int(tok.item()))
        pos += 1
    return gen


def _plaintext_greedy(model, ids, steps):
    cur = ids.clone()
    gen = []
    with torch.no_grad():
        for _ in range(steps):
            lg = model(cur).logits[:, -1, :]
            nxt = lg.argmax(-1, keepdim=True)
            gen.append(int(nxt.item()))
            cur = torch.cat([cur, nxt], dim=1)
    return gen


def test_7_logits_and_token_correctness():
    sess, pkg, ids, model, mc, _names = _build_pkg(use_pad=True)
    # prefill recovered logits == plaintext (strict fp64 tolerance)
    with torch.no_grad():
        ref_logits = model(ids).logits.double()
    n = sess.n
    cfg_c = _cfg_to(sess.layer_configs[0], torch.device("cpu"))
    h = sess.mask_embeddings(ids).double()
    out = apply_folded_prefill(h, pkg, n, cfg_c, sess._cos, sess._sin,
                               float(sess.eps))
    rec = sess.recover(apply_folded_head(out["y_tilde"], pkg, float(sess.eps)))
    assert torch.allclose(rec.double(), ref_logits, atol=1e-4, rtol=1e-4)
    # all-position top-1 tokens match
    assert bool((rec[0].argmax(-1) == ref_logits[0].argmax(-1)).all())
    # generated tokens exact (prefill + decode loop, all padded Linears)
    folded_gen = _folded_greedy(sess, pkg, mc, ids, steps=4)
    plain_gen = _plaintext_greedy(model, ids, steps=4)
    assert folded_gen == plain_gen, (folded_gen, plain_gen)


def test_8_folded_lora_over_pad_base_matches_reference():
    """folded base(+pad) + merged folded-LoRA == folded(base+rawLoRA)(+pad) for
    EVERY tensor incl. the recomputed cpad -- the LoRA-merge cpad fix. Mirrors
    tests/test_lora_folded_e6.py::test_folding_matches_reference_all_modules but
    with the Linear-boundary pad enabled."""
    import copy
    from pllo.deployment.lora_folded_package import (
        ALL_TARGET_MODULES, apply_lora_to_model, fold_lora_for_layer,
        lora_scaling, merge_folded_lora, synthetic_lora_adapter)
    from pllo.experiments.folded_probe_common import tiny_model

    N, RANK, ALPHA, SEED = 2, 4, 8.0, 2035
    torch.manual_seed(0)
    model, mc = tiny_model()
    model_lora = copy.deepcopy(model)
    tm = list(ALL_TARGET_MODULES)
    scaling = lora_scaling(ALPHA, RANK)
    lora = synthetic_lora_adapter(mc, N, tm, RANK, seed=SEED)

    def cfg():
        return MemoryOptimizedConfig(
            num_layers=N, batch_size=1, seq_len=8, max_new_tokens=2, device="cpu",
            dtype="float32", folding_dtype="float64", folded_weight_device="cpu",
            seed=SEED, use_linear_boundary_pad=True, linear_pad_scale=0.3)

    sess = MaskedQwenSession(model, mc, cfg())
    apply_lora_to_model(model_lora, lora, tm, scaling)
    sess_lora = MaskedQwenSession(model_lora, mc, cfg())

    max_err = 0.0
    saw_cpad = False
    for ell in range(N):
        base = {k: v.clone()
                for k, v in sess.export_folded_layer_tensors(ell).items()}
        fl = fold_lora_for_layer(sess, ell, lora[ell], scaling=scaling,
                                 rank=RANK, rank_seed=SEED, target_modules=tm)
        merged = merge_folded_lora(base, fl, tm)
        ref = sess_lora.export_folded_layer_tensors(ell)
        for k in ref:
            max_err = max(max_err, (merged[k] - ref[k]).abs().max().item())
            if k.endswith("_cpad_tilde"):
                saw_cpad = True
    assert saw_cpad, "expected cpad tensors in the pad-enabled folded layer"
    # cpad recomputed against the merged weight matches the reference exactly
    assert max_err < 1e-6, max_err


def test_7b_pad_changes_operand_but_not_output():
    # WITH-pad and WITHOUT-pad packages (same seed/model) recover to the SAME
    # logits (pad is compensated), proving the pad is output-neutral while it is
    # really executed (xpad tensors are present + non-zero).
    import numpy as np  # noqa: F401
    s0, pkg0, ids, model, mc, names0 = _build_pkg(use_pad=False)
    s1, pkg1, ids1, _m, _mc, names1 = _build_pkg(use_pad=True)
    n = 3
    c0 = _cfg_to(s0.layer_configs[0], torch.device("cpu"))
    c1 = _cfg_to(s1.layer_configs[0], torch.device("cpu"))
    # same ids for both (rebuild uses same torch seed -> same randint)
    h0 = s0.mask_embeddings(ids).double()
    h1 = s1.mask_embeddings(ids1).double()
    r0 = s0.recover(apply_folded_head(
        apply_folded_prefill(h0, pkg0, n, c0, s0._cos, s0._sin,
                             float(s0.eps))["y_tilde"], pkg0, float(s0.eps)))
    r1 = s1.recover(apply_folded_head(
        apply_folded_prefill(h1, pkg1, n, c1, s1._cos, s1._sin,
                             float(s1.eps))["y_tilde"], pkg1, float(s1.eps)))
    assert torch.allclose(r0.double(), r1.double(), atol=1e-4, rtol=1e-4)
    # pad really present + non-trivial in the with-pad package
    from pllo.deployment.folded_worker import load_folded_layer
    lt = load_folded_layer(pkg1, 0)
    assert "wq_xpad_tilde" in lt and float(lt["wq_xpad_tilde"].abs().max()) > 0
    assert "wq_cpad_tilde" in lt and float(lt["wq_cpad_tilde"].abs().max()) > 0


# ---------------------------------------------------------------------------
# Test A / B: pad is the DEFAULT main scheme; --no-linear-boundary-pad is legacy
# (exercises the real build script + package_linear_boundary_pad_status helper)
# ---------------------------------------------------------------------------


def _run_build(tmp_path, *extra):
    import importlib.util
    import json
    import sys

    repo = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "bp_main", repo / "scripts" / "build_qwen7b_folded_package.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "pkg"
    js = tmp_path / "build.json"
    old = sys.argv
    try:
        sys.argv = ["x", "--dry-run", "--output-dir", str(out),
                    "--num-layers", "2", "--output-json", str(js), *extra]
        assert mod.main() == 0
    finally:
        sys.argv = old
    return out, json.loads(js.read_text())


def test_A_pad_is_default_main_scheme(tmp_path):
    """A build WITHOUT --no-linear-boundary-pad is pad-enabled + paper-ready."""
    from pllo.deployment.linear_boundary_pad import (
        package_linear_boundary_pad_status)
    pkg, rep = _run_build(tmp_path)
    assert rep["main_scheme"] == "linear_boundary_additive_pad"
    assert rep["linear_boundary_pad_enabled"] is True
    assert rep["qwen_production_path_uses_linear_input_pad"] is True
    assert rep.get("paper_ready", True) is not False
    assert all(rep["linear_pad_coverage"][m] for m in ALL_PAD_MODULES)
    # helper reads coverage from the REAL written shard tensor names
    status = package_linear_boundary_pad_status(pkg)
    assert status["base_linear_boundary_pad_enabled"] is True
    assert status["base_linear_pad_all_modules_covered"] is True


def test_B_no_pad_is_legacy_not_paper_ready(tmp_path):
    """--no-linear-boundary-pad -> mask_only_legacy + paper_ready False."""
    from pllo.deployment.linear_boundary_pad import (
        package_linear_boundary_pad_status)
    pkg, rep = _run_build(tmp_path, "--no-linear-boundary-pad")
    assert rep["main_scheme"] == "mask_only_legacy"
    assert rep["paper_ready"] is False
    assert "legacy" in rep["paper_ready_blocker"].lower()
    assert rep["linear_boundary_pad_enabled"] is False
    status = package_linear_boundary_pad_status(pkg)
    assert status["base_linear_boundary_pad_enabled"] is False
    assert status["base_linear_pad_all_modules_covered"] is False
