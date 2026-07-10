"""Tests for the Scheme A nonlinear-backward primitive audit.

Small dims, fp64. Verifies: trusted_shortcut forward/backward exact; the Amulet
dense candidate forward works but its A1 backward is honestly reported as
blocked_by_hadamard (never mislabelled as passing); permutation backward works
but leaks the activation Gram; and the cross-Gram report is generated with A0
leaking exactly while A1 does not.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pllo.experiments.nonlinear_backward_primitive_audit import (  # noqa: E402
    run_cross_gram_after_nonlinear,
    run_silu,
    run_swiglu,
)


def test_silu_trusted_shortcut_forward_backward():
    r = run_silu(m=4, d=8, k=3, seed=0)
    ts = r["trusted_shortcut"]
    assert ts["forward"]["rel"] < 1e-9
    assert ts["backward"]["rel"] < 1e-9
    # honest label: it recovers plaintext inside the TEE, not a GPU primitive.
    assert ts["gpu_side"] is False
    assert ts["exposes_plaintext_in_tee"] is True


def test_silu_candidate_reports_honest_status():
    r = run_silu(m=4, d=8, k=3, seed=0)
    am = r["amulet_dense"]
    # forward is a genuine GPU-side primitive (exact); backward is blocked.
    assert am["forward"]["rel"] < 1e-9
    assert am["gpu_side"] is True
    assert am["backward_status"] == "blocked_by_hadamard"
    # even with a perfect masked derivative, the Hadamard step fails (O(1) error).
    assert am["masked_derivative_err"]["rel"] < 1e-9
    assert am["backward_hadamard_attempt"]["rel"] > 0.1


def test_swiglu_trusted_shortcut_forward_backward():
    r = run_swiglu(m=4, d=8, k=3, seed=0)
    ts = r["trusted_shortcut"]
    assert ts["forward"]["rel"] < 1e-9
    assert ts["backward_dG"]["rel"] < 1e-9
    assert ts["backward_dU"]["rel"] < 1e-9


def test_swiglu_candidate_reports_honest_status():
    r = run_swiglu(m=4, d=8, k=3, seed=0)
    am = r["amulet_dense"]
    assert am["forward"]["rel"] < 1e-9
    assert am["backward_status"] == "blocked_by_hadamard"
    assert am["backward_dU_hadamard_attempt"]["rel"] > 0.1
    assert am["backward_dG_hadamard_attempt"]["rel"] > 0.1


def test_no_standard_autograd_mislabel_as_A1():
    # No dense GPU-side variant may be reported as a passing A1 backward.
    for r in (run_silu(seed=0), run_swiglu(seed=0)):
        assert r["amulet_dense"]["backward_status"] != "backward_passed"
        assert r["amulet_dense"]["backward_status"] == "blocked_by_hadamard"
        # the only 'backward_passed' variant is permutation, and it must carry the
        # activation-Gram leak so it cannot masquerade as dense-mask A1 privacy.
        perm = r["permutation"]
        assert perm["backward_status"] == "backward_passed"
        assert perm["activation_gram_corr"] == pytest.approx(1.0, abs=1e-6)


def test_cross_gram_report_generated():
    cg = run_cross_gram_after_nonlinear(m=16, d=8, trials=60, seed=0)
    assert cg["amulet_dense"]["has_backward"] is False
    assert cg["amulet_dense"]["exact_cross_gram_leak"] == "not_applicable"
    # A0 dual leaks the exact cross-Gram; A1 dense does not.
    assert cg["trusted_shortcut_A0_dual"]["exact_cross_gram_leak"] is True
    assert cg["trusted_shortcut_A0_dual"]["corr_mean"] == pytest.approx(1.0, abs=1e-3)
    assert cg["trusted_shortcut_A1_dense"]["exact_cross_gram_leak"] is False
    assert abs(cg["trusted_shortcut_A1_dense"]["corr_mean"]) < 0.2
    # permutation A1 breaks exact cross-Gram but leaks the activation Gram.
    assert cg["permutation_A1"]["activation_gram_leak"] is True
