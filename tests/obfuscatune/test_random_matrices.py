"""Tests for ObfuscaTune random-matrix generators (arXiv:2407.02960, App. B)."""

from __future__ import annotations

import torch

from pllo.baselines.obfuscatune.random_matrices import (
    condition_number,
    gaussian_invertible_matrix,
    matrix_with_condition_number,
    orthogonal_matrix,
)


def test_orthogonal_is_orthonormal():
    q, q_inv = orthogonal_matrix(16, seed=0, dtype=torch.float64)
    eye = torch.eye(16, dtype=torch.float64)
    assert torch.allclose(q.transpose(0, 1) @ q, eye, atol=1e-10)
    assert torch.allclose(q @ q_inv, eye, atol=1e-10)          # inverse == transpose
    assert torch.allclose(q_inv, q.transpose(0, 1), atol=1e-12)


def test_orthogonal_condition_number_is_one():
    q, _ = orthogonal_matrix(24, seed=3, dtype=torch.float64)
    assert abs(condition_number(q) - 1.0) < 1e-6


def test_matrix_with_condition_number_matches_target():
    for kappa in (8.0, 32.0, 128.0, 160.0):
        r, r_inv = matrix_with_condition_number(20, kappa, seed=1, dtype=torch.float64)
        assert abs(condition_number(r) - kappa) / kappa < 0.05
        assert torch.allclose(r @ r_inv, torch.eye(20, dtype=torch.float64), atol=1e-8)


def test_generators_are_deterministic_in_seed():
    a1, _ = orthogonal_matrix(12, seed=7, dtype=torch.float64)
    a2, _ = orthogonal_matrix(12, seed=7, dtype=torch.float64)
    b, _ = orthogonal_matrix(12, seed=8, dtype=torch.float64)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b)

    c1, _ = matrix_with_condition_number(12, 32.0, seed=5, dtype=torch.float64)
    c2, _ = matrix_with_condition_number(12, 32.0, seed=5, dtype=torch.float64)
    assert torch.equal(c1, c2)


def test_gaussian_invertible_recovers_inverse():
    r, r_inv = gaussian_invertible_matrix(16, seed=2, dtype=torch.float64, jitter=1e-3)
    assert torch.allclose(r @ r_inv, torch.eye(16, dtype=torch.float64), atol=1e-8)
    # a naive Gaussian matrix is (almost surely) not orthogonal -> kappa > 1
    assert condition_number(r) > 1.0
