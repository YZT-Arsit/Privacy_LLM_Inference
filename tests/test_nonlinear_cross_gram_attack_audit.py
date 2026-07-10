"""Tests for the nonlinear cross-Gram real-attack audit.

These assert the *shape* of the honest verdict: public base weights make the
co-leaked value multiset dangerous (token/adapter recovery), while under private
weights the pure cross-Gram is auxiliary structural leakage — except membership,
which the gradient-norm signal enables in both regimes.
"""

from __future__ import annotations

import numpy as np

from pllo.experiments.nonlinear_cross_gram_attack_audit import (
    attack_adapter_extraction,
    attack_future_token,
    attack_membership,
    attack_target_prediction,
    attack_token_reconstruction,
    build_scenario,
    forward_backward,
    leakage_features,
    silu,
    silu_prime,
)


def _scn():
    return build_scenario(seed=0)


def test_leakage_is_permutation_invariant():
    """ZGZ^T / gradnorm computed in the permuted domain equal the plaintext ones."""
    scn = _scn()
    ids = scn["member_ids"][:6]
    fb = forward_backward(scn, ids)
    Z, GZ = fb["Z"], fb["GZ"]
    rng = np.random.default_rng(1)
    Pi = np.eye(Z.shape[2])[:, rng.permutation(Z.shape[2])]
    g_plain = np.einsum("nmf,nkf->nmk", Z, GZ)
    g_perm = np.einsum("nmf,nkf->nmk", Z @ Pi, GZ @ Pi)
    assert np.allclose(g_plain, g_perm, atol=1e-10)
    assert np.allclose(np.linalg.norm(GZ, axis=2), np.linalg.norm(GZ @ Pi, axis=2), atol=1e-10)


def test_silu_derivative_matches_finite_difference():
    z = np.linspace(-4, 4, 50)
    fd = (silu(z + 1e-6) - silu(z - 1e-6)) / 2e-6
    assert np.max(np.abs(fd - silu_prime(z))) < 1e-6


def test_token_reconstruction_public_succeeds_private_fails():
    rows = attack_token_reconstruction(_scn())
    pub = next(r for r in rows if r["regime"] == "public_weights")
    priv = next(r for r in rows if r["regime"] == "private_weights")
    gram = next(r for r in rows if r["regime"] == "gram_only")
    assert pub["success"] and pub["token_top1"] > 0.9      # public weights -> tokens recovered
    assert not priv["success"]                              # private -> identification fails
    assert not gram["success"]                              # Gram alone -> no vocab identity


def test_target_prediction_public_beats_private():
    rows = attack_target_prediction(_scn())
    pub = next(r for r in rows if r["regime"] == "public_weights")
    priv = next(r for r in rows if r["regime"] == "private_weights")
    assert pub["auc"] >= priv["auc"]


def test_membership_above_random():
    rows = attack_membership(_scn())
    best = max(r["auc"] for r in rows)
    assert best > 0.6   # gradient-norm signal gives non-trivial membership advantage


def test_adapter_extraction_public_succeeds_private_fails():
    rows = attack_adapter_extraction(_scn())
    pub = next(r for r in rows if r["regime"] == "public_weights")
    priv = next(r for r in rows if r["regime"] == "private_weights")
    assert pub["success"] and pub["dW_cosine"] > 0.9       # ΔW recovered under public weights
    assert not priv["success"] and priv["dW_cosine"] < 0.5  # failure under private weights


def test_future_token_public_beats_bigram_baseline():
    rows = attack_future_token(_scn())
    base = next(r for r in rows if r["regime"] == "baseline_bigram")
    pub = next(r for r in rows if r["regime"] == "public_weights")
    priv = next(r for r in rows if r["regime"] == "private_weights")
    assert pub["next_token_top1"] > base["next_token_top1"] + 0.1
    assert priv["next_token_top1"] <= base["next_token_top1"] + 1e-9
