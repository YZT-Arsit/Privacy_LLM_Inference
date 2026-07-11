"""O1 optimizer-equivalence + preconditioner-leakage tests (private-base)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.experiments.private_base_optimizer import (  # noqa: E402
    O1_PROFILES,
    preconditioner_visibility,
    verify_matrix,
)


def test_preconditioned_update_recovers_at_all_conditions():
    rows = verify_matrix()
    for r in rows:
        assert r["precond_recovery_err"] < 1e-10, r


def test_naive_update_fails_for_nonorthogonal_only():
    rows = {r["cond"]: r for r in verify_matrix()}
    # orthogonal (cond=1): naive == correct
    assert rows[1.0]["naive_recovery_err"] < 1e-10
    # non-orthogonal: naive is meaningfully wrong
    for c in (1.5, 2.0, 4.0, 8.0, 16.0, 32.0):
        assert rows[c]["naive_recovery_err"] > 1e-2, c


def test_rmsnorm_invariance_only_orthogonal():
    rows = {r["cond"]: r for r in verify_matrix()}
    assert rows[1.0]["rmsnorm_invariance_err"] < 1e-10
    for c in (2.0, 8.0, 32.0):
        assert rows[c]["rmsnorm_invariance_err"] > 1e-1, c


def test_gram_grows_with_condition():
    rows = verify_matrix()
    grams = [r["gram_dist_from_I"] for r in rows]
    assert grams[0] < 1e-10               # orthogonal
    assert all(grams[i] <= grams[i + 1] + 1e-9 for i in range(len(grams) - 1))


def test_profile_leakage_flags():
    assert O1_PROFILES["O1-A"].gram_visible_to_gpu is False
    assert O1_PROFILES["O1-A"].paper_safe_eligible is True
    # O1-B leaks the Gram and is NOT paper-safe
    assert O1_PROFILES["O1-B"].gram_visible_to_gpu is True
    assert O1_PROFILES["O1-B"].paper_safe_eligible is False
    # O1-C hides it behind the boundary at +1 crossing
    assert O1_PROFILES["O1-C"].gram_visible_to_gpu is False
    assert O1_PROFILES["O1-C"].trusted_invocations_per_step == 3
    vis = preconditioner_visibility()
    assert vis["O1-B"]["gram_visible_to_gpu"] is True
