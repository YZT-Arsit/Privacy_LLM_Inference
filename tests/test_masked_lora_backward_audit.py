"""Tests for the Scheme A masked-domain LoRA backward audit.

Small dims, fp64. Verifies A0 correctness + exact cross-Gram leak, A1
correctness + broken exact cross-Gram equality, gradient recovery, the
dense-masked-AdamW refusal, the nonlinear-backward blocked status, and that the
weight-alignment audit runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pllo.experiments.masked_lora_backward_audit import (  # noqa: E402
    DenseMaskedAdamWUnsupported,
    attempt_dense_masked_adamw,
    run_cross_gram_statistics,
    run_nonlinear_backward_status,
    run_synthetic_linear,
    run_weight_alignment_audit,
)

DIMS = dict(m=8, d_in=16, d_out=12, r=4)


def test_dual_backward_correctness_linear_lora():
    res = run_synthetic_linear(**DIMS, seed=0)
    a0 = res["A0"]
    assert a0["forward_max_abs"] < 1e-9
    assert a0["grad_X_recover_max_abs"] < 1e-9
    assert a0["grad_A_recover_max_abs"] < 1e-9
    assert a0["grad_B_recover_max_abs"] < 1e-9


def test_dual_backward_cross_gram_leakage():
    res = run_synthetic_linear(**DIMS, seed=0)
    a0 = res["A0"]
    # A0 leaks the exact cross-Gram: equality to machine precision, corr == 1.
    assert a0["cross_gram_max_abs"] < 1e-9
    assert a0["cross_gram_corr"] == pytest.approx(1.0, abs=1e-6)


def test_independent_backward_correctness_linear_lora():
    res = run_synthetic_linear(**DIMS, seed=0)
    a1 = res["A1"]
    assert a1["grad_X_recover_max_abs"] < 1e-9
    assert a1["grad_A_recover_max_abs"] < 1e-9
    assert a1["grad_B_recover_max_abs"] < 1e-9
    # GPU-visible input gradient equals G_X M_in (independent backward mask).
    assert a1["grad_X_masked_relation_max_abs"] < 1e-9


def test_independent_backward_breaks_exact_cross_gram():
    res = run_synthetic_linear(**DIMS, seed=0)
    a0, a1 = res["A0"], res["A1"]
    # exact equality is broken: rel error is O(1), not machine precision.
    assert a1["cross_gram_rel_error"] > 0.1
    assert a1["cross_gram_max_abs"] > 1e-6
    # over many draws the A1 correlation mean collapses toward 0 while A0 stays 1.
    stats = run_cross_gram_statistics(ms=(32,), trials=100, seed=1)["per_m"][0]
    assert stats["A0_corr_mean"] == pytest.approx(1.0, abs=1e-6)
    assert abs(stats["A1_corr_mean"]) < 0.1
    assert stats["A1_abs_corr_mean"] < a0["cross_gram_corr"]


def test_lora_grad_recovery():
    # recovery holds for both orthogonal and signed-permutation masks.
    for kind in ("orthogonal", "signed_permutation"):
        res = run_synthetic_linear(**DIMS, seed=2, mask_kind=kind)
        for sc in ("A0", "A1"):
            assert res[sc]["grad_A_recover_max_abs"] < 1e-8
            assert res[sc]["grad_B_recover_max_abs"] < 1e-8


def test_dense_masked_adamw_unsupported():
    with pytest.raises(DenseMaskedAdamWUnsupported):
        attempt_dense_masked_adamw()


def test_nonlinear_backward_status_reports_blocked_if_missing():
    rep = run_nonlinear_backward_status(seed=3)
    names = {r["nonlinear"] for r in rep["rows"]}
    assert {"GELU", "SiLU", "SwiGLU"} <= names
    for r in rep["rows"]:
        # nonlinear masked-domain backward must NOT be reported as passed.
        assert r["masked_backward_status"] == "not_implemented"
        assert r["masked_forward_status"] != "implemented_and_passed"
        # pointwise nonlinearity does not commute with a dense right mask.
        if r["masked_forward_recovery_rel_err_dense"] is not None:
            assert r["masked_forward_recovery_rel_err_dense"] > 0.1


def test_weight_alignment_audit_runs():
    a = run_weight_alignment_audit(d_in=16, d_out=12, seed=6)
    # spectrum is shared (orthogonal-mask invariance) but no trivial W recovery.
    assert a["spectrum_shared"] is True
    assert a["naive_W_recovery_rel_error"] > 0.1
    assert isinstance(a["verdict"], str) and a["verdict"]


def test_gpu_visibility_no_plaintext_or_mask_published():
    res = run_synthetic_linear(**DIMS, seed=0)
    for sc in ("A0", "A1"):
        aud = res[sc]["gpu_visibility_audit"]
        assert aud["no_plaintext_gradients_or_data_published"] is True
        assert aud["no_raw_mask_or_switch_matrix_published"] is True
