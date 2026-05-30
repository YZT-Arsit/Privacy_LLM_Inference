"""Tests for the Stage 5.2 operator-compatible mask generators."""

from __future__ import annotations

import torch

from pllo.ops.compatible_masks import (
    apply_permutation_columns,
    center_matrix,
    centered_orthogonality_error,
    generate_dense_invertible,
    generate_mean_preserving_orthogonal,
    generate_orthogonal,
    generate_permutation,
    matrix_fingerprint,
    mean_preservation_error,
    orthogonal_error,
)


HIDDEN = 32


def test_orthogonal_mask_satisfies_NTN_equals_I() -> None:
    torch.manual_seed(0)
    N = generate_orthogonal(HIDDEN, torch.float32, "cpu")
    assert orthogonal_error(N) < 1e-5


def test_mean_preserving_orthogonal_satisfies_N_at_ones_equals_ones() -> None:
    torch.manual_seed(0)
    N = generate_mean_preserving_orthogonal(HIDDEN, torch.float32, "cpu")
    assert mean_preservation_error(N) < 1e-5


def test_mean_preserving_orthogonal_satisfies_centered_orthogonality() -> None:
    torch.manual_seed(0)
    N = generate_mean_preserving_orthogonal(HIDDEN, torch.float32, "cpu")
    assert centered_orthogonality_error(N) < 1e-5
    # And the mask itself is orthogonal.
    assert orthogonal_error(N) < 1e-5


def test_permutation_and_inverse_compose_to_identity() -> None:
    torch.manual_seed(0)
    info = generate_permutation(HIDDEN, torch.float32, "cpu")
    perm = info["perm"]
    inv = info["inv_perm"]
    identity = torch.arange(HIDDEN)
    assert torch.equal(perm[inv], identity)
    assert torch.equal(inv[perm], identity)
    # Dense matrix form agrees with index form.
    x = torch.randn(2, HIDDEN)
    assert torch.allclose(apply_permutation_columns(x, perm), x @ info["matrix"])


def test_dense_invertible_mask_has_valid_inverse() -> None:
    torch.manual_seed(0)
    N, N_inv = generate_dense_invertible(HIDDEN, torch.float32, "cpu")
    err = (N @ N_inv - torch.eye(HIDDEN)).abs().max().item()
    assert err < 1e-4


def test_center_matrix_is_symmetric_idempotent_projection() -> None:
    C = center_matrix(HIDDEN, torch.float64, "cpu")
    assert torch.allclose(C, C.T)
    assert torch.allclose(C @ C, C, atol=1e-10)
    # C @ 1 = 0 (centering projects out the all-ones direction).
    ones = torch.ones(HIDDEN, dtype=torch.float64)
    assert (C @ ones).abs().max().item() < 1e-10


def test_matrix_fingerprint_is_deterministic_and_unique() -> None:
    torch.manual_seed(0)
    a = torch.randn(HIDDEN, HIDDEN)
    b = torch.randn(HIDDEN, HIDDEN)
    assert matrix_fingerprint(a) == matrix_fingerprint(a)
    assert matrix_fingerprint(a) != matrix_fingerprint(b)


def test_orthogonal_mask_preserves_row_norms() -> None:
    """``||X N||_2 = ||X||_2`` row-wise for orthogonal ``N``."""
    torch.manual_seed(0)
    N = generate_orthogonal(HIDDEN, torch.float32, "cpu")
    X = torch.randn(4, HIDDEN)
    err = (X.norm(dim=-1) - (X @ N).norm(dim=-1)).abs().max().item()
    assert err < 1e-5


def test_mean_preserving_orthogonal_preserves_row_mean() -> None:
    """``mean(X N) = mean(X)`` row-wise when ``N @ 1 = 1``."""
    torch.manual_seed(0)
    N = generate_mean_preserving_orthogonal(HIDDEN, torch.float32, "cpu")
    X = torch.randn(4, HIDDEN)
    err = (X.mean(dim=-1) - (X @ N).mean(dim=-1)).abs().max().item()
    assert err < 1e-5
