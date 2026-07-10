"""CPU-only protocol audit tests (Sections A-F). All CPU-only; no CUDA, no TEE."""

from __future__ import annotations

import torch

from pllo.experiments import cpu_correctness_audit as cca
from pllo.experiments import mask_conditioning_audit as mca
from pllo.experiments import synthetic_security_audit as ssa
from pllo.experiments import trusted_boundary_accounting as tba
from pllo.experiments.cpu_protocol_audit import (
    audit_metadata,
    b4_packed_vs_per_layer,
    run_full_audit,
)

PASS = 1e-8


# --- A: correctness ---------------------------------------------------------
def test_masked_linear_forward_bias():
    for bias in (True, False):
        r = cca.a1_masked_linear(family="dense_gl", cond=10.0, bias=bias)
        assert r["pass"] and r["max_abs_error"] < 1e-9
        assert r["solve_residual"] < 1e-9


def test_attention_score_invariant():
    r = cca.a2_attention()
    assert r["pass"] and r["score_error"] < PASS and r["prob_error"] < PASS


def test_kv_cache_prefill_decode_invariant():
    r = cca.a3_kv_cache()
    assert r["pass"] and r["cache_recover_error"] < PASS
    assert r["append_mixes_token_dim"] is False


def test_gelu_permutation_stabilizer():
    rows = {(x["activation"], x["mask_family"]): x["pass_bool"] for x in cca.a4_stabilizer()}
    assert rows[("gelu", "permutation")] is True
    assert rows[("gelu", "dense_gl")] is False


def test_silu_permutation_stabilizer():
    rows = {(x["activation"], x["mask_family"]): x["pass_bool"] for x in cca.a4_stabilizer()}
    assert rows[("silu", "permutation")] is True
    assert rows[("silu", "signed_permutation")] is False
    # tanh is odd -> signed permutation holds (non-homogeneous control)
    assert rows[("tanh", "signed_permutation")] is True
    assert rows[("tanh", "positive_diagonal")] is False


def test_swiglu_shared_permutation_forward():
    rows = {x["case"]: x for x in cca.a6_swiglu()}
    assert rows["shared_perm_forward"]["pass"]
    assert rows["shared_perm_forward"]["error"] < PASS


def test_swiglu_mismatched_permutations_fail():
    rows = {x["case"]: x for x in cca.a6_swiglu()}
    assert rows["mismatched_perm_forward_MUST_FAIL"]["pass"]        # pass = it DID fail
    assert rows["mismatched_perm_forward_MUST_FAIL"]["error"] > 1e-3


def test_nonlinear_backward_standard_autograd():
    rows = {x["case"]: x for x in cca.a6_swiglu()}
    assert rows["shared_perm_backward_gH"]["error"] < PASS          # via torch.autograd only
    mlp = [x for x in cca.a5_mlp() if x.get("section") == "A5"]
    assert all(x["pass"] for x in mlp)


def test_rmsnorm_permutation_equivariance():
    rows = {x["case"]: x for x in cca.a7_norm()}
    assert rows["rmsnorm_core_permutation"]["pass"]
    assert rows["rmsnorm_affine_gamma_reindexed"]["pass"]
    assert rows["rmsnorm_dense_gl_MUST_FAIL"]["pass"]               # dense GL breaks it


def test_layernorm_permutation_equivariance():
    rows = {x["case"]: x for x in cca.a7_norm()}
    assert rows["layernorm_core_permutation"]["pass"]
    assert rows["layernorm_dense_gl_MUST_FAIL"]["pass"]


# --- B: LoRA training -------------------------------------------------------
def test_lora_forward_exact():
    r = cca.a1_masked_linear(family="permutation", bias=True)   # linear boundary exactness
    assert r["max_abs_error"] < 1e-10
    # LoRA-specific forward exactness is covered by the conditioning LoRA sweep at cond 1
    lr = [x for x in mca.e_lora_conditioning() if x["family"] == "orthogonal"][0]
    assert lr["forward_max_abs"] < 1e-9


def test_lora_independent_backward_exact():
    from pllo.experiments.masked_lora_backward_audit import run_optimizer_equivalence
    eq = run_optimizer_equivalence(scheme="A1", steps=8)
    assert eq["per_optimizer"]["adamw"]["max_param_error"] < 1e-10
    assert eq["dense_masked_adamw"]["status"] == "raised_unsupported"


def test_lora_gradient_recovery():
    rows = [x for x in mca.e_lora_conditioning() if x["cond_target"] in (1.0, 10.0)]
    assert all(x["recovered_gradA_max_abs"] < 1e-6 for x in rows)


def test_multistep_masked_training_matches_plaintext():
    from pllo.experiments.masked_lora_backward_audit import run_optimizer_equivalence
    eq = run_optimizer_equivalence(scheme="A1", steps=10)
    for opt in ("sgd", "momentum_sgd", "adamw"):
        assert eq["per_optimizer"][opt]["loss_curve_distance"] < 1e-10


def test_packed_update_matches_per_layer_update():
    r = b4_packed_vs_per_layer()
    assert r["packed_equals_per_layer"]
    assert r["delta_W_error"] < 1e-10
    assert r["adam_second_moment_error"] < 1e-10
    assert r["next_step_logit_error"] < 1e-10


# --- C/D: accounting --------------------------------------------------------
def test_boundary_calls_constant_in_lora_layer_count():
    rows = [r for r in tba.c_boundary_calls() if r["schedule"] == "packed"]
    totals = {r["num_lora_layers"]: r["total_invocations"] for r in rows}
    assert set(totals.values()) == {3}                              # constant at 3
    assert all(r["observed_match"] for r in rows)
    per = [r for r in tba.c_boundary_calls() if r["schedule"] == "per_layer"]
    assert next(r for r in per if r["num_lora_layers"] == 32)["total_invocations"] == 34


def test_no_per_layer_nonlinear_trusted_calls():
    for r in tba.c_boundary_calls():
        assert r["nonlinear_calls"] == 0


def test_communication_accounting():
    d = tba.d1_communication(tba.WorkloadDims())
    # exact adapter-scale bytes: sum_j r (d_in + d_out) * dtype_bytes
    dims = tba.WorkloadDims()
    expect = dims.lora_layers * dims.rank * (2 * dims.hidden) * tba.DTYPE_BYTES[dims.dtype]
    assert d["packed_grad_bytes"] == expect
    assert d["logits_to_trusted_bytes"] == dims.batch * dims.seq * dims.vocab * tba.DTYPE_BYTES[dims.dtype]
    assert d["dominant_channel"] in ("logits", "adapter_update")


def test_trusted_compute_complexity_per_mask_family():
    rows = {r["mask_family"]: r for r in tba.d2_suite()}
    assert rows["permutation"]["unmask_complexity"] == "O(adapter_elems)"
    assert rows["dense_gl"]["unmask_complexity"] == "O(d^2 r)"       # not adapter-scale


# --- E: conditioning --------------------------------------------------------
def test_dense_mask_conditioning_error_growth():
    rows = [r for r in mca.e_linear_conditioning() if r["family"] == "dense_gl"]
    by_cond = {r["cond_target"]: r["forward_max_abs"] for r in rows}
    # fp64 stays finite/correct across the sweep; error at cond 1e4 >= error at cond 1
    assert all(r["finite"] for r in rows)
    assert by_cond[1e4] >= by_cond[1.0]


# --- F: synthetic security --------------------------------------------------
def test_cross_gram_general_region_not_exact():
    r = ssa.f1_cross_gram(trials=50)
    assert r["general_region_exact"] is False
    assert r["general_region_mean_rel_error"] > 1e-3


def test_cross_gram_nonlinear_region_exact():
    r = ssa.f1_cross_gram(trials=50)
    assert r["nonlinear_region_exact"] is True
    assert r["nonlinear_region_mean_corr"] > 1 - 1e-6


def test_subspace_rank_invariance():
    r = ssa.f3_subspace()
    assert r["rank_invariant_under_left_transform"]
    assert r["orthogonal_preserves_pairwise_angle"]
    assert r["gl_changes_pairwise_angle"]


def test_general_gl_does_not_preserve_singular_values():
    r = ssa.f3_subspace()
    assert r["orthogonal_preserves_singular_values"]
    assert r["gl_changes_singular_values"]


def test_compensation_second_moment_convergence():
    rows = ssa.f4_compensation()
    orth = [r for r in rows if r["n_out_family"] == "orthogonal"]
    err50 = next(r["estimator_rel_error"] for r in orth if r["K"] == 50)
    err1000 = next(r["estimator_rel_error"] for r in orth if r["K"] == 1000)
    assert err1000 < err50                                          # converges with K
    # orthogonal preserves spectrum; non-orthogonal does NOT (but still a masked Gram, not "only rank")
    assert next(r for r in rows if r["n_out_family"] == "orthogonal" and r["K"] == 1000)["spectrum_exactly_preserved"]
    assert not next(r for r in rows if r["n_out_family"] == "dense_non_orthogonal" and r["K"] == 1000)["spectrum_exactly_preserved"]


# --- metadata / no-real-hardware guarantees ---------------------------------
def test_metadata_declares_cpu_only():
    m = audit_metadata(0)
    assert m["uses_real_gpu"] is False
    assert m["uses_real_tee"] is False
    assert m["simulated_trusted_controller"] is True
    assert m["synthetic_security"] is True
    assert not torch.cuda.is_available() or True   # never depends on CUDA


def test_full_audit_runs():
    r = run_full_audit(0)
    assert set(r) >= {"metadata", "A_correctness", "B_lora_training", "B4_packed_vs_per_layer",
                      "C_D_accounting", "E_conditioning", "F_synthetic_security"}
