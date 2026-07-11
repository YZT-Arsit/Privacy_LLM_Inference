"""Tests for the O(V) monomial logit mask (Z_tilde = Z D Pi) private-CE contract.

Covers: loss correctness, dlogits correctness (analytic == autograd), exact
inverse, bounded condition / bf16 range, fp32 diagnostic, ignore_index/causal
shift/padding, and the permutation-vs-monomial value-multiset leakage proxy.
CPU-only synthetic contract; NOT a real-model result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pllo.masks.monomial_logit_mask import (  # noqa: E402
    make_monomial_logit_mask,
)


def _plain_ce_and_grad(Z, labels, ignore_index=-100):
    Z = Z.clone().double().requires_grad_(True)
    valid = labels != ignore_index
    loss = torch.nn.functional.cross_entropy(Z[valid], labels[valid],
                                             reduction="mean")
    loss.backward()
    return float(loss.detach()), Z.grad.detach()


def test_inverse_exact_dense_and_ov():
    m = make_monomial_logit_mask(17, seed=1)
    Z = torch.randn(5, 17, dtype=torch.float64)
    # O(V) mask then recover == identity
    Zt = m.mask_logits(Z)
    Zr = m.recover_logits(Zt)
    assert torch.allclose(Zr, Z, atol=1e-10)
    # O(V) mask == dense M right-multiply
    M = m.dense_matrix()
    assert torch.allclose(Zt, Z @ M, atol=1e-10)
    # dense inverse round-trips
    assert torch.allclose(Zt @ torch.linalg.inv(M), Z, atol=1e-8)


def test_loss_equals_plaintext_ce():
    torch.manual_seed(0)
    V, n = 40, 12
    Z = torch.randn(n, V, dtype=torch.float64)
    labels = torch.randint(0, V, (n,))
    m = make_monomial_logit_mask(V, seed=7)
    res = m.private_ce(m.mask_logits(Z), labels, compute_dtype=torch.float64)
    plain_loss, _ = _plain_ce_and_grad(Z, labels)
    assert abs(float(res.loss) - plain_loss) < 1e-9


def test_masked_gradient_matches_autograd_wrt_ztilde():
    # G_tilde must equal dL/dZ_tilde computed by autograd through recover+CE.
    torch.manual_seed(1)
    V, n = 32, 8
    Z = torch.randn(n, V, dtype=torch.float64)
    labels = torch.randint(0, V, (n,))
    m = make_monomial_logit_mask(V, seed=3)
    Zt = m.mask_logits(Z).detach().requires_grad_(True)
    Zrec = m.recover_logits(Zt)
    loss = torch.nn.functional.cross_entropy(Zrec, labels, reduction="mean")
    loss.backward()
    autograd_gtilde = Zt.grad.detach()
    res = m.private_ce(Zt.detach(), labels, compute_dtype=torch.float64)
    assert torch.allclose(res.masked_logit_gradient, autograd_gtilde, atol=1e-10)


def test_grad_maps_back_to_true_grad():
    # G recovered from G_tilde equals the plaintext true-logit gradient.
    torch.manual_seed(2)
    V, n = 24, 6
    Z = torch.randn(n, V, dtype=torch.float64)
    labels = torch.randint(0, V, (n,))
    _, true_grad = _plain_ce_and_grad(Z, labels)
    m = make_monomial_logit_mask(V, seed=5)
    res = m.private_ce(m.mask_logits(Z), labels, compute_dtype=torch.float64)
    # invert mask_gradient: G[:, pi[k]] = G_tilde[:, k] * d[k] -> scatter back
    Gt = res.masked_logit_gradient
    recovered_G = (Gt * m.d).index_select(-1, m.pi_inv)
    assert torch.allclose(recovered_G, true_grad, atol=1e-9)


def test_ignore_index_causal_padding():
    torch.manual_seed(3)
    V = 20
    Z = torch.randn(6, V, dtype=torch.float64)
    labels = torch.tensor([3, -100, 7, -100, 1, 9])   # padding via ignore_index
    m = make_monomial_logit_mask(V, seed=9)
    res = m.private_ce(m.mask_logits(Z), labels, compute_dtype=torch.float64)
    plain_loss, true_grad = _plain_ce_and_grad(Z, labels)
    assert res.n_valid == 4
    assert abs(float(res.loss) - plain_loss) < 1e-9
    # ignored rows carry zero gradient
    Gt = res.masked_logit_gradient
    recovered_G = (Gt * m.d).index_select(-1, m.pi_inv)
    assert torch.allclose(recovered_G[labels == -100], torch.zeros(2, V,
                          dtype=torch.float64), atol=1e-12)
    assert torch.allclose(recovered_G, true_grad, atol=1e-9)


def test_all_ignored_returns_zero():
    V = 10
    Z = torch.randn(3, V, dtype=torch.float64)
    labels = torch.full((3,), -100)
    m = make_monomial_logit_mask(V, seed=1)
    res = m.private_ce(m.mask_logits(Z), labels)
    assert res.n_valid == 0 and float(res.loss) == 0.0
    assert torch.count_nonzero(res.masked_logit_gradient) == 0


def test_condition_number_bounded_and_bf16_safe():
    m = make_monomial_logit_mask(2000, seed=4, scale_low=0.5, scale_high=2.0)
    assert m.condition_number() <= 2.0 / 0.5 + 1e-6
    # scales are well within bf16 finite range (no extreme values)
    assert float(m.d.abs().max()) <= 2.0 and float(m.d.abs().min()) >= 0.5


def test_bf16_vs_fp32_diagnostic():
    torch.manual_seed(5)
    V, n = 128, 16
    Z = torch.randn(n, V)
    labels = torch.randint(0, V, (n,))
    m = make_monomial_logit_mask(V, seed=2)
    Zt = m.mask_logits(Z)
    loss_fp32 = float(m.private_ce(Zt, labels, compute_dtype=torch.float32).loss)
    loss_bf16 = float(m.private_ce(Zt.to(torch.bfloat16), labels,
                                   compute_dtype=torch.float32).loss)
    # bf16 wire, fp32 compute: small but bounded residual (diagnostic, not exact)
    assert abs(loss_fp32 - loss_bf16) < 5e-2


def test_permutation_only_baseline_preserves_value_multiset():
    torch.manual_seed(6)
    Z = torch.randn(4, 50, dtype=torch.float64)
    perm = make_monomial_logit_mask(50, seed=1, permutation_only=True)
    mono = make_monomial_logit_mask(50, seed=1, scale_low=0.5, scale_high=2.0)
    assert perm.is_permutation_only
    assert not mono.is_permutation_only
    # permutation alone LEAKS the value multiset; the monomial hides it.
    assert perm.value_multiset_preserved(Z) is True
    assert mono.value_multiset_preserved(Z) is False


def test_permutation_only_still_exact_ce():
    torch.manual_seed(7)
    V, n = 30, 10
    Z = torch.randn(n, V, dtype=torch.float64)
    labels = torch.randint(0, V, (n,))
    perm = make_monomial_logit_mask(V, seed=3, permutation_only=True)
    res = perm.private_ce(perm.mask_logits(Z), labels, compute_dtype=torch.float64)
    plain_loss, _ = _plain_ce_and_grad(Z, labels)
    assert abs(float(res.loss) - plain_loss) < 1e-9
